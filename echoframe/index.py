'''LMDB-backed index for echoframe metadata.'''

import json
from pathlib import Path

from . import compaction, lmdb_helper
from .metadata import EchoframeMetadata, filter_metadata, normalize_tags, utc_now


class LmdbIndex:
    '''Store and query echoframe metadata in LMDB.'''

    def __init__(self, path, map_size=1 << 30, env=None, shards_root=None):
        self.path = Path(path)
        self.shards_root = None if shards_root is None else Path(shards_root)
        self.env = env or lmdb_helper.open_env(self.path, map_size=map_size)
        self.db_handles = lmdb_helper.open_databases(self.env)
        self.entries_db = self.db_handles['entries_db']
        self.by_phraser_db = self.db_handles['by_phraser_db']
        self.live_by_phraser_db = self.db_handles['live_by_phraser_db']
        self.by_shard_db = self.db_handles['by_shard_db']
        self.by_tag_db = self.db_handles['by_tag_db']
        self.shard_meta_db = self.db_handles['shard_meta_db']
        self.compaction_db = self.db_handles['compaction_db']

    def upsert(self, metadata):
        '''Insert or replace one metadata record.'''
        return self.upsert_many([metadata])[0]

    def upsert_many(self, metadata_list):
        '''Insert or replace multiple metadata records in one transaction.'''
        if not metadata_list:
            return []

        previous_by_id = {metadata.echoframe_key: self.get(
            metadata.echoframe_key)
            for metadata in metadata_list}
        touched = set()
        with lmdb_helper.write_txn(self.env) as txn:
            for metadata in metadata_list:
                payload = self._dump_json(metadata.to_dict())
                previous = previous_by_id[metadata.echoframe_key]
                if previous is not None:
                    lmdb_helper.delete_phraser_keys(txn, self.db_handles,
                        previous)
                    self._delete_shard_keys(txn, previous, metadata)
                    self._delete_tag_keys(txn, previous)
                    if previous.shard_id:
                        touched.add(previous.shard_id)
                touched.update(lmdb_helper.write_metadata(txn,
                    self.db_handles, metadata, payload))

            for shard_id in sorted(touched):
                self._refresh_shard_stats(txn, shard_id)
        return metadata_list

    def get(self, echoframe_key):
        '''Load one metadata record by echoframe key.'''
        return lmdb_helper.get_metadata(self.env, self.entries_db,
            EchoframeMetadata, echoframe_key)

    def get_many(self, echoframe_keys):
        '''Load multiple metadata records by echoframe key.'''
        return lmdb_helper.get_many_metadata(self.env, self.entries_db,
            EchoframeMetadata, echoframe_keys)

    def find_phraser(self, phraser_key, include_deleted=False):
        '''Find metadata records for one phraser key.'''
        with lmdb_helper.read_txn(self.env) as txn:
            prefix = lmdb_helper.phraser_scan_prefix(phraser_key)
            db = self.by_phraser_db if include_deleted else (
                self.live_by_phraser_db)
            echoframe_keys = self._scan_prefix_in_txn(txn, db, prefix)
            records = [self._get_in_txn(txn, echoframe_key)
                for echoframe_key in echoframe_keys]
        return filter_metadata(records)

    def delete(self, metadata):
        '''Tombstone a metadata record and remove it from live indexes.'''
        deleted = metadata.mark_deleted()
        return self.upsert(deleted)

    def entries_for_shard(self, shard_id, include_deleted=False):
        '''List entries that point to one shard.'''
        with lmdb_helper.read_txn(self.env) as txn:
            return self._entries_for_shard_in_txn(txn, shard_id=shard_id,
                include_deleted=include_deleted)

    def list_entries(self, include_deleted=False):
        '''List metadata records across the full entries store.'''
        return lmdb_helper.list_metadata(self.env, self.entries_db,
            EchoframeMetadata, include_deleted=include_deleted)

    def list_shards(self):
        '''List shard identifiers present in the shard index.'''
        shard_ids = []
        seen = set()
        with lmdb_helper.read_txn(self.env) as txn:
            cursor = txn.cursor(db=self.by_shard_db)
            if not cursor.set_range(b'shard:'):
                return shard_ids
            for key, _ in cursor:
                if not key.startswith(b'shard:'):
                    break
                _, remainder = key.split(b'shard:', 1)
                shard_id, _, _ = remainder.partition(b':')
                shard_id = shard_id.decode('utf-8')
                if shard_id in seen:
                    continue
                seen.add(shard_id)
                shard_ids.append(shard_id)
        return shard_ids

    def get_shard_metadata(self, shard_id):
        '''Return shard-level stats.'''
        with lmdb_helper.read_txn(self.env) as txn:
            payload = txn.get(shard_id.encode('utf-8'), db=self.shard_meta_db)
        if payload is None:
            return None
        return json.loads(payload.decode('utf-8'))

    def list_shard_metadata(self):
        '''Return shard stats for all known shards.'''
        rows = []
        with lmdb_helper.read_txn(self.env) as txn:
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

        with lmdb_helper.read_txn(self.env) as txn:
            groups = [set(self._scan_prefix_in_txn(txn, self.by_tag_db,
                self._tag_key(tag, ''))) for tag in tags]
            if not groups:
                return []
            echoframe_keys = set.intersection(*groups) if match == 'all' else (
                set().union(*groups))
            records = [self._get_in_txn(txn, echoframe_key)
                for echoframe_key in sorted(echoframe_keys)]
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
        with lmdb_helper.read_txn(self.env) as txn:
            cursor = txn.cursor(db=self.by_tag_db)
            if not cursor.set_range(b'tag:'):
                return counts
            for key, value in cursor:
                if not key.startswith(b'tag:'):
                    break
                _, remainder = key.split(b'tag:', 1)
                tag, _, _ = remainder.partition(b':')
                tag = tag.decode('utf-8')
                if include_deleted:
                    counts[tag] = counts.get(tag, 0) + 1
                    continue
                metadata = self._get_in_txn(txn, value)
                if metadata is None:
                    continue
                if metadata.storage_status != 'live':
                    continue
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def add_tags(self, echoframe_key, tags):
        '''Add tags to one metadata record.'''
        updated = self.add_tags_many([echoframe_key], tags)
        if not updated:
            return None
        return updated[0]

    def add_tags_many(self, echoframe_keys, tags):
        '''Add tags to multiple metadata records.'''
        return self._update_tags_many(echoframe_keys, tags=tags, mode='add')

    def remove_tags(self, echoframe_key, tags):
        '''Remove tags from one metadata record.'''
        updated = self.remove_tags_many([echoframe_key], tags)
        if not updated:
            return None
        return updated[0]

    def remove_tags_many(self, echoframe_keys, tags):
        '''Remove tags from multiple metadata records.'''
        return self._update_tags_many(echoframe_keys, tags=tags, mode='remove')

    def remove_shard_entries(self, shard_id, echoframe_keys):
        '''Remove shard index entries for one shard.'''
        with lmdb_helper.write_txn(self.env) as txn:
            for echoframe_key in echoframe_keys:
                txn.delete(self._shard_key(shard_id, echoframe_key),
                    db=self.by_shard_db)
            self._refresh_shard_stats(txn, shard_id)

    def create_compaction_journal(self, shard_id, source_echoframe_keys,
        live_echoframe_keys, target_shard_id):
        '''Create a compaction journal record.'''
        return compaction.create_compaction_journal(self, shard_id,
            source_echoframe_keys=source_echoframe_keys,
            live_echoframe_keys=live_echoframe_keys,
            target_shard_id=target_shard_id)

    def update_compaction_journal(self, journal_id, status, error=None):
        '''Update one compaction journal record.'''
        return compaction.update_compaction_journal(self, journal_id,
            status=status, error=error)

    def list_compaction_journal(self, status=None):
        '''List compaction journal records.'''
        return compaction.list_compaction_journal(self, status=status)

    def _entries_for_shard_in_txn(self, txn, shard_id, include_deleted=False):
        prefix = self._shard_key(shard_id, b'')
        echoframe_keys = self._scan_prefix_in_txn(txn, self.by_shard_db, prefix)
        records = [self._get_in_txn(txn, echoframe_key)
            for echoframe_key in echoframe_keys]
        items = [record for record in records if record is not None]
        if include_deleted:
            return items
        return [item for item in items if item.storage_status == 'live']

    def _list_entries_in_txn(self, txn, include_deleted=False):
        return lmdb_helper.list_metadata_in_txn(txn, self.entries_db,
            EchoframeMetadata, include_deleted=include_deleted)

    def _update_tags_many(self, echoframe_keys, tags, mode):
        if not echoframe_keys:
            return []
        if mode not in {'add', 'remove'}:
            raise ValueError("mode must be 'add' or 'remove'")

        tag_values = normalize_tags(tags)
        if not tag_values:
            return [metadata for metadata in self.get_many(echoframe_keys)
                if metadata is not None]

        updates = []
        for metadata in self.get_many(echoframe_keys):
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
        prefix = self._shard_key(shard_id, b'')
        echoframe_keys = self._scan_prefix_in_txn(txn, self.by_shard_db, prefix)
        if not echoframe_keys:
            txn.delete(shard_id.encode('utf-8'), db=self.shard_meta_db)
            return

        live_count = 0
        deleted_count = 0
        for echoframe_key in echoframe_keys:
            metadata = self._get_in_txn(txn, echoframe_key)
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
        with lmdb_helper.read_txn(self.env) as txn:
            return self._scan_prefix_in_txn(txn, db, prefix)

    def _scan_prefix_in_txn(self, txn, db, prefix):
        return lmdb_helper.scan_prefix_in_txn(txn, db, prefix)

    def _get_in_txn(self, txn, echoframe_key):
        return lmdb_helper.get_metadata_in_txn(txn, self.entries_db,
            EchoframeMetadata, echoframe_key)

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

    def _shard_key(self, shard_id, echoframe_key):
        return lmdb_helper.shard_key(shard_id, echoframe_key)

    def _tag_key(self, tag, echoframe_key):
        return lmdb_helper.tag_key(tag, echoframe_key)

    def _delete_tag_keys(self, txn, metadata):
        lmdb_helper.delete_tag_keys(txn, self.by_tag_db, metadata)

    def _delete_shard_keys(self, txn, previous, metadata):
        lmdb_helper.delete_shard_keys(txn, self.by_shard_db, previous,
            metadata)
