'''LMDB-backed index for echoframe metadata.'''

import json
from pathlib import Path

from .metadata import Metadata


class LmdbIndex:
    '''Store and query echoframe metadata in LMDB.'''

    def __init__(self, path, map_size=1 << 30, env=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.env = env or self._open_env(map_size=map_size)
        self.entries_db = self.env.open_db(b'entries')
        self.by_object_db = self.env.open_db(b'by_object')
        self.live_by_object_db = self.env.open_db(b'live_by_object')
        self.by_shard_db = self.env.open_db(b'by_shard')
        self.by_tag_db = self.env.open_db(b'by_tag')

    def _open_env(self, map_size):
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError('lmdb is required to use Store') from exc
        return lmdb.open(str(self.path), create=True, max_dbs=8,
            map_size=map_size, subdir=True)

    def upsert(self, metadata):
        '''Insert or replace one metadata record.'''
        entry_id = metadata.entry_id
        payload = json.dumps(metadata.to_dict(), sort_keys=True).encode('utf-8')
        object_key = metadata.object_key.encode('utf-8')
        previous = self.get(entry_id)

        with self.env.begin(write=True) as txn:
            txn.put(entry_id.encode('utf-8'), payload, db=self.entries_db)
            txn.put(object_key, entry_id.encode('utf-8'), db=self.by_object_db)
            if metadata.storage_status == 'live':
                txn.put(object_key, entry_id.encode('utf-8'),
                    db=self.live_by_object_db)
            else:
                txn.delete(object_key, db=self.live_by_object_db)
            if metadata.shard_id:
                shard_key = self._shard_key(metadata.shard_id, entry_id)
                txn.put(shard_key, entry_id.encode('utf-8'),
                    db=self.by_shard_db)
            if previous is not None:
                self._delete_shard_keys(txn, previous, metadata)
                self._delete_tag_keys(txn, previous)
            self._put_tag_keys(txn, metadata)
        return metadata

    def get(self, entry_id):
        '''Load one metadata record by entry id.'''
        with self.env.begin() as txn:
            payload = txn.get(entry_id.encode('utf-8'), db=self.entries_db)
        if payload is None:
            return None
        return Metadata.from_dict(json.loads(payload.decode('utf-8')))

    def find(self, phraser_key, model_name=None, output_type=None,
        layer=None, include_deleted=False):
        '''Find metadata records for one phraser key.'''
        prefix = f'obj:{phraser_key}:'.encode('utf-8')
        db = self.by_object_db if include_deleted else self.live_by_object_db
        entry_ids = self._scan_prefix(db, prefix)
        entries = [self.get(entry_id) for entry_id in entry_ids]
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

    def find_one(self, phraser_key, model_name, output_type, layer,
        collar, match='exact'):
        '''Find one record using collar matching rules.'''
        if match not in {'exact', 'min', 'max', 'nearest'}:
            message = (
                "match must be one of 'exact', 'min', 'max', 'nearest'"
            )
            raise ValueError(message)

        records = self.find(phraser_key=phraser_key,
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
        return min(records, key=lambda entry:
            abs(entry.collar - collar))

    def delete(self, metadata):
        '''Tombstone a metadata record and remove it from live indexes.'''
        deleted = metadata.mark_deleted()
        return self.upsert(deleted)

    def entries_for_shard(self, shard_id, include_deleted=False):
        '''List entries that point to one shard.'''
        prefix = self._shard_key(shard_id, '')
        entry_ids = self._scan_prefix(self.by_shard_db, prefix)
        records = [self.get(entry_id) for entry_id in entry_ids]
        items = [record for record in records if record is not None]
        if include_deleted:
            return items
        return [item for item in items if item.storage_status == 'live']

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

    def find_by_tag(self, tag, include_deleted=False):
        '''List entries for one tag.'''
        prefix = self._tag_key(tag, '')
        entry_ids = self._scan_prefix(self.by_tag_db, prefix)
        records = [self.get(entry_id) for entry_id in entry_ids]
        items = [record for record in records if record is not None]
        if include_deleted:
            return items
        return [item for item in items if item.storage_status == 'live']

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
                metadata = self.get(value.decode('utf-8'))
                if metadata is None:
                    continue
                if metadata.storage_status != 'live':
                    continue
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def add_tags(self, entry_id, tags):
        '''Add tags to one metadata record.'''
        metadata = self.get(entry_id)
        if metadata is None:
            return None
        next_tags = list(metadata.tags) + list(tags)
        metadata = metadata.with_tags(next_tags)
        return self.upsert(metadata)

    def remove_tags(self, entry_id, tags):
        '''Remove tags from one metadata record.'''
        metadata = self.get(entry_id)
        if metadata is None:
            return None
        blocked = set(tags)
        next_tags = [tag for tag in metadata.tags if tag not in blocked]
        metadata = metadata.with_tags(next_tags)
        return self.upsert(metadata)

    def remove_shard_entries(self, shard_id, entry_ids):
        '''Remove shard index entries for one shard.'''
        with self.env.begin(write=True) as txn:
            for entry_id in entry_ids:
                txn.delete(self._shard_key(shard_id, entry_id),
                    db=self.by_shard_db)

    def _scan_prefix(self, db, prefix):
        entry_ids = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=db)
            if not cursor.set_range(prefix):
                return entry_ids
            for key, value in cursor:
                if not key.startswith(prefix):
                    break
                entry_ids.append(value.decode('utf-8'))
        return entry_ids

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
