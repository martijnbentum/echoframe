'''LMDB-backed index for echoframe metadata.'''

import json
from pathlib import Path
from uuid import uuid4

from .metadata import Metadata, normalize_tags, utc_now


class LmdbIndex:
    '''Store and query echoframe metadata in LMDB.'''

    def __init__(self, path, map_size=1 << 30, env=None, shards_root=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.shards_root = None if shards_root is None else Path(shards_root)
        self.env = env or self._open_env(map_size=map_size)
        self.entries_db = self.env.open_db(b'entries')
        self.by_object_db = self.env.open_db(b'by_object')
        self.live_by_object_db = self.env.open_db(b'live_by_object')
        self.by_shard_db = self.env.open_db(b'by_shard')
        self.by_tag_db = self.env.open_db(b'by_tag')
        self.shard_meta_db = self.env.open_db(b'shard_meta')
        self.compaction_db = self.env.open_db(b'compaction')

    def _open_env(self, map_size):
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError('lmdb is required to use Store') from exc
        return lmdb.open(str(self.path), create=True, max_dbs=12,
            map_size=map_size, subdir=True)

    def upsert(self, metadata):
        '''Insert or replace one metadata record.'''
        return self.upsert_many([metadata])[0]

    def upsert_many(self, metadata_list):
        '''Insert or replace multiple metadata records in one transaction.'''
        if not metadata_list:
            return []

        previous_by_id = {metadata.entry_id: self.get(metadata.entry_id)
            for metadata in metadata_list}
        touched = set()
        with self.env.begin(write=True) as txn:
            for metadata in metadata_list:
                entry_id = metadata.entry_id
                payload = self._dump_json(metadata.to_dict())
                object_key = metadata.object_key.encode('utf-8')
                previous = previous_by_id[entry_id]

                txn.put(entry_id.encode('utf-8'), payload, db=self.entries_db)
                txn.put(object_key, entry_id.encode('utf-8'),
                    db=self.by_object_db)
                if metadata.storage_status == 'live':
                    txn.put(object_key, entry_id.encode('utf-8'),
                        db=self.live_by_object_db)
                else:
                    txn.delete(object_key, db=self.live_by_object_db)
                if metadata.shard_id:
                    shard_key = self._shard_key(metadata.shard_id, entry_id)
                    txn.put(shard_key, entry_id.encode('utf-8'),
                        db=self.by_shard_db)
                    touched.add(metadata.shard_id)
                if previous is not None:
                    self._delete_shard_keys(txn, previous, metadata)
                    self._delete_tag_keys(txn, previous)
                    if previous.shard_id:
                        touched.add(previous.shard_id)
                self._put_tag_keys(txn, metadata)

            for shard_id in sorted(touched):
                self._refresh_shard_stats(txn, shard_id)
        return metadata_list

    def get(self, entry_id):
        '''Load one metadata record by entry id.'''
        with self.env.begin() as txn:
            return self._get_in_txn(txn, entry_id)

    def get_many(self, entry_ids):
        '''Load multiple metadata records by entry id.'''
        if not entry_ids:
            return []
        with self.env.begin() as txn:
            return [self._get_in_txn(txn, entry_id) for entry_id in entry_ids]

    def find(self, phraser_key, model_name=None, output_type=None,
        layer=None, include_deleted=False):
        '''Find metadata records for one phraser key.'''
        with self.env.begin() as txn:
            return self._find_in_txn(txn, phraser_key=phraser_key,
                model_name=model_name, output_type=output_type, layer=layer,
                include_deleted=include_deleted)

    def find_many(self, queries):
        '''Find multiple records using collar matching rules.'''
        if not queries:
            return []
        with self.env.begin() as txn:
            results = []
            for query in queries:
                results.append(self._find_one_in_txn(txn, **query))
            return results

    def find_one(self, phraser_key, model_name, output_type, layer,
        collar, match='exact'):
        '''Find one record using collar matching rules.'''
        with self.env.begin() as txn:
            return self._find_one_in_txn(txn, phraser_key=phraser_key,
                model_name=model_name, output_type=output_type, layer=layer,
                collar=collar, match=match)

    def delete(self, metadata):
        '''Tombstone a metadata record and remove it from live indexes.'''
        deleted = metadata.mark_deleted()
        return self.upsert(deleted)

    def entries_for_shard(self, shard_id, include_deleted=False):
        '''List entries that point to one shard.'''
        with self.env.begin() as txn:
            return self._entries_for_shard_in_txn(txn, shard_id=shard_id,
                include_deleted=include_deleted)

    def list_entries(self, include_deleted=False):
        '''List metadata records across the full entries store.'''
        with self.env.begin() as txn:
            return self._list_entries_in_txn(txn,
                include_deleted=include_deleted)

    def list_shards(self):
        '''List shard identifiers present in the shard index.'''
        shard_ids = []
        seen = set()
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.by_shard_db)
            if not cursor.set_range(b'shard:'):
                return shard_ids
            for key, _ in cursor:
                if not key.startswith(b'shard:'):
                    break
                parts = key.decode('utf-8').split(':', 2)
                shard_id = parts[1]
                if shard_id in seen:
                    continue
                seen.add(shard_id)
                shard_ids.append(shard_id)
        return shard_ids

    def get_shard_metadata(self, shard_id):
        '''Return shard-level stats.'''
        with self.env.begin() as txn:
            payload = txn.get(shard_id.encode('utf-8'), db=self.shard_meta_db)
        if payload is None:
            return None
        return json.loads(payload.decode('utf-8'))

    def list_shard_metadata(self):
        '''Return shard stats for all known shards.'''
        rows = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.shard_meta_db)
            if not cursor.set_range(b''):
                return rows
            for key, value in cursor:
                data = json.loads(value.decode('utf-8'))
                data['shard_id'] = key.decode('utf-8')
                rows.append(data)
        rows.sort(key=lambda row: row['shard_id'])
        return rows

    def find_by_tag(self, tag, include_deleted=False):
        '''List entries for one tag.'''
        return self.find_by_tags([tag], match='all',
            include_deleted=include_deleted)

    def find_by_tags(self, tags, match='all', include_deleted=False):
        '''List entries that match one or more tags.'''
        tags = normalize_tags(tags)
        if match not in {'all', 'any'}:
            message = "match must be 'all' or 'any'"
            raise ValueError(message)
        if not tags:
            return []

        with self.env.begin() as txn:
            groups = [set(self._scan_prefix_in_txn(txn, self.by_tag_db,
                self._tag_key(tag, ''))) for tag in tags]
            if not groups:
                return []
            entry_ids = set.intersection(*groups) if match == 'all' else (
                set().union(*groups))
            records = [self._get_in_txn(txn, entry_id)
                for entry_id in sorted(entry_ids)]
        items = [record for record in records if record is not None]
        if include_deleted:
            return items
        return [item for item in items if item.storage_status == 'live']

    def list_tags(self, include_deleted=False):
        '''List all known tags.'''
        return sorted(self.tag_counts(
            include_deleted=include_deleted).keys())

    def tag_counts(self, include_deleted=False):
        '''Count entries per tag.'''
        counts = {}
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.by_tag_db)
            if not cursor.set_range(b'tag:'):
                return counts
            for key, value in cursor:
                if not key.startswith(b'tag:'):
                    break
                parts = key.decode('utf-8').split(':', 2)
                tag = parts[1]
                if include_deleted:
                    counts[tag] = counts.get(tag, 0) + 1
                    continue
                metadata = self._get_in_txn(txn, value.decode('utf-8'))
                if metadata is None:
                    continue
                if metadata.storage_status != 'live':
                    continue
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def add_tags(self, entry_id, tags):
        '''Add tags to one metadata record.'''
        updated = self.add_tags_many([entry_id], tags)
        if not updated:
            return None
        return updated[0]

    def add_tags_many(self, entry_ids, tags):
        '''Add tags to multiple metadata records.'''
        return self._update_tags_many(entry_ids, tags=tags, mode='add')

    def remove_tags(self, entry_id, tags):
        '''Remove tags from one metadata record.'''
        updated = self.remove_tags_many([entry_id], tags)
        if not updated:
            return None
        return updated[0]

    def remove_tags_many(self, entry_ids, tags):
        '''Remove tags from multiple metadata records.'''
        return self._update_tags_many(entry_ids, tags=tags, mode='remove')

    def remove_shard_entries(self, shard_id, entry_ids):
        '''Remove shard index entries for one shard.'''
        with self.env.begin(write=True) as txn:
            for entry_id in entry_ids:
                txn.delete(self._shard_key(shard_id, entry_id),
                    db=self.by_shard_db)
            self._refresh_shard_stats(txn, shard_id)

    def create_compaction_journal(self, shard_id, source_entry_ids,
        live_entry_ids, target_shard_id):
        '''Create a compaction journal record.'''
        journal_id = (
            f'{utc_now()}:{shard_id}:{target_shard_id}:{uuid4().hex}'
        )
        record = {
            'journal_id': journal_id,
            'shard_id': shard_id,
            'target_shard_id': target_shard_id,
            'source_entry_ids': list(source_entry_ids),
            'live_entry_ids': list(live_entry_ids),
            'status': 'running',
            'started_at': utc_now(),
            'finished_at': None,
            'error': None,
        }
        with self.env.begin(write=True) as txn:
            txn.put(journal_id.encode('utf-8'), self._dump_json(record),
                db=self.compaction_db)
        return record

    def update_compaction_journal(self, journal_id, status, error=None):
        '''Update one compaction journal record.'''
        with self.env.begin(write=True) as txn:
            key = journal_id.encode('utf-8')
            payload = txn.get(key, db=self.compaction_db)
            if payload is None:
                return None
            record = json.loads(payload.decode('utf-8'))
            record['status'] = status
            record['error'] = error
            record['finished_at'] = utc_now()
            txn.put(key, self._dump_json(record), db=self.compaction_db)
        return record

    def list_compaction_journal(self, status=None):
        '''List compaction journal records.'''
        rows = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.compaction_db)
            if not cursor.set_range(b''):
                return rows
            for _, value in cursor:
                record = json.loads(value.decode('utf-8'))
                if status is not None and record['status'] != status:
                    continue
                rows.append(record)
        rows.sort(key=lambda row: row['journal_id'])
        return rows

    def _find_in_txn(self, txn, phraser_key, model_name=None,
        output_type=None, layer=None, include_deleted=False):
        prefix = f'obj:{phraser_key}:'.encode('utf-8')
        db = self.by_object_db if include_deleted else self.live_by_object_db
        entry_ids = self._scan_prefix_in_txn(txn, db, prefix)
        entries = [self._get_in_txn(txn, entry_id) for entry_id in entry_ids]
        records = [entry for entry in entries if entry is not None]
        if model_name is not None:
            records = [entry for entry in records
                if entry.model_name == model_name]
        if output_type is not None:
            records = [entry for entry in records
                if entry.output_type == output_type]
        if layer is not None:
            records = [entry for entry in records if entry.layer == layer]
        records.sort(key=lambda entry: entry.collar)
        return records

    def _find_one_in_txn(self, txn, phraser_key, model_name, output_type,
        layer, collar, match='exact'):
        if match not in {'exact', 'min', 'max', 'nearest'}:
            message = (
                "match must be one of 'exact', 'min', 'max', 'nearest'"
            )
            raise ValueError(message)

        records = self._find_in_txn(txn, phraser_key=phraser_key,
            model_name=model_name, output_type=output_type, layer=layer)
        if not records:
            return None
        if match == 'exact':
            for entry in records:
                if entry.collar == collar:
                    return entry
            return None
        if match == 'min':
            for entry in records:
                if entry.collar >= collar:
                    return entry
            return None
        if match == 'max':
            for entry in reversed(records):
                if entry.collar <= collar:
                    return entry
            return None
        return min(records, key=lambda entry: abs(entry.collar - collar))

    def _entries_for_shard_in_txn(self, txn, shard_id, include_deleted=False):
        prefix = self._shard_key(shard_id, '')
        entry_ids = self._scan_prefix_in_txn(txn, self.by_shard_db, prefix)
        records = [self._get_in_txn(txn, entry_id) for entry_id in entry_ids]
        items = [record for record in records if record is not None]
        if include_deleted:
            return items
        return [item for item in items if item.storage_status == 'live']

    def _list_entries_in_txn(self, txn, include_deleted=False):
        entries = []
        cursor = txn.cursor(db=self.entries_db)
        if not cursor.set_range(b''):
            return entries
        for _, value in cursor:
            metadata = Metadata.from_dict(json.loads(value.decode('utf-8')))
            if (not include_deleted and
                metadata.storage_status != 'live'):
                continue
            entries.append(metadata)
        return entries

    def _update_tags_many(self, entry_ids, tags, mode):
        if not entry_ids:
            return []
        if mode not in {'add', 'remove'}:
            raise ValueError("mode must be 'add' or 'remove'")

        tag_values = normalize_tags(tags)
        if not tag_values:
            return [metadata for metadata in self.get_many(entry_ids)
                if metadata is not None]

        updates = []
        for metadata in self.get_many(entry_ids):
            if metadata is None:
                continue
            if mode == 'add':
                next_tags = list(metadata.tags) + tag_values
            else:
                blocked = set(tag_values)
                next_tags = [tag for tag in metadata.tags
                    if tag not in blocked]
            updates.append(metadata.with_tags(next_tags))
        return self.upsert_many(updates)

    def _refresh_shard_stats(self, txn, shard_id):
        prefix = self._shard_key(shard_id, '')
        entry_ids = self._scan_prefix_in_txn(txn, self.by_shard_db, prefix)
        if not entry_ids:
            txn.delete(shard_id.encode('utf-8'), db=self.shard_meta_db)
            return

        live_count = 0
        deleted_count = 0
        for entry_id in entry_ids:
            metadata = self._get_in_txn(txn, entry_id)
            if metadata is None:
                continue
            if metadata.storage_status == 'live':
                live_count += 1
            else:
                deleted_count += 1
        payload = {
            'live_entry_count': live_count,
            'deleted_entry_count': deleted_count,
            **self._shard_file_size(txn, shard_id),
            'updated_at': utc_now(),
        }
        txn.put(shard_id.encode('utf-8'), self._dump_json(payload),
            db=self.shard_meta_db)

    def _scan_prefix(self, db, prefix):
        with self.env.begin() as txn:
            return self._scan_prefix_in_txn(txn, db, prefix)

    def _scan_prefix_in_txn(self, txn, db, prefix):
        entry_ids = []
        cursor = txn.cursor(db=db)
        if not cursor.set_range(prefix):
            return entry_ids
        for key, value in cursor:
            if not key.startswith(prefix):
                break
            entry_ids.append(value.decode('utf-8'))
        return entry_ids

    def _get_in_txn(self, txn, entry_id):
        payload = txn.get(entry_id.encode('utf-8'), db=self.entries_db)
        if payload is None:
            return None
        return Metadata.from_dict(json.loads(payload.decode('utf-8')))

    def _shard_file_size(self, txn, shard_id):
        if self.shards_root is None:
            return {
                'byte_size': 0,
                'byte_size_is_estimated': False,
                'byte_size_error': None,
            }
        file_path = self.shards_root / f'{shard_id}.h5'
        try:
            return {
                'byte_size': file_path.stat().st_size,
                'byte_size_is_estimated': False,
                'byte_size_error': None,
            }
        except FileNotFoundError:
            return {
                'byte_size': 0,
                'byte_size_is_estimated': False,
                'byte_size_error': None,
            }
        except OSError as exc:
            payload = txn.get(shard_id.encode('utf-8'), db=self.shard_meta_db)
            fallback_size = 0
            if payload is not None:
                previous_size = json.loads(payload.decode('utf-8')).get(
                    'byte_size')
                if previous_size is not None:
                    fallback_size = previous_size
            return {
                'byte_size': fallback_size,
                'byte_size_is_estimated': True,
                'byte_size_error': str(exc),
            }

    def _dump_json(self, value):
        return json.dumps(value, sort_keys=True).encode('utf-8')

    def _shard_key(self, shard_id, entry_id):
        return f'shard:{shard_id}:{entry_id}'.encode('utf-8')

    def _tag_key(self, tag, entry_id):
        return f'tag:{tag}:{entry_id}'.encode('utf-8')

    def _put_tag_keys(self, txn, metadata):
        for tag in metadata.tags:
            txn.put(self._tag_key(tag, metadata.entry_id),
                metadata.entry_id.encode('utf-8'),
                db=self.by_tag_db)

    def _delete_tag_keys(self, txn, metadata):
        for tag in metadata.tags:
            txn.delete(self._tag_key(tag, metadata.entry_id),
                db=self.by_tag_db)

    def _delete_shard_keys(self, txn, previous, metadata):
        if previous.shard_id is None:
            return
        if previous.shard_id == metadata.shard_id:
            return
        txn.delete(self._shard_key(previous.shard_id, previous.entry_id),
            db=self.by_shard_db)
