'''LMDB-backed index for echoframe metadata.'''

from __future__ import annotations

import json
from pathlib import Path

from .metadata import Metadata


class LmdbIndex:
    '''Store and query echoframe metadata in LMDB.'''

    def __init__(self, path: str | Path, map_size: int=1 << 30,
        env: object | None=None) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.env = env or self._open_env(map_size=map_size)
        self.entries_db = self.env.open_db(b'entries')
        self.by_object_db = self.env.open_db(b'by_object')
        self.live_by_object_db = self.env.open_db(b'live_by_object')
        self.by_shard_db = self.env.open_db(b'by_shard')

    def _open_env(self, map_size: int) -> object:
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError('lmdb is required to use Store') from exc
        return lmdb.open(str(self.path), create=True, max_dbs=8,
            map_size=map_size, subdir=True)

    def upsert(self, metadata: Metadata) -> Metadata:
        '''Insert or replace one metadata record.'''
        entry_id = metadata.entry_id
        payload = json.dumps(metadata.to_dict(), sort_keys=True).encode('utf-8')
        object_key = metadata.object_key.encode('utf-8')

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
                txn.put(shard_key, b'', db=self.by_shard_db)
        return metadata

    def get(self, entry_id: str) -> Metadata | None:
        '''Load one metadata record by entry id.'''
        with self.env.begin() as txn:
            payload = txn.get(entry_id.encode('utf-8'), db=self.entries_db)
        if payload is None:
            return None
        return Metadata.from_dict(json.loads(payload.decode('utf-8')))

    def find(self, phraser_key: str, model_name: str | None=None,
        output_type: str | None=None, layer: int | None=None,
        include_deleted: bool=False) -> list[Metadata]:
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
        records.sort(key=lambda entry: entry.collar_ms)
        return records

    def find_one(self, phraser_key: str, model_name: str,
        output_type: str, layer: int, collar_ms: int,
        match: str='exact') -> Metadata | None:
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
                if entry.collar_ms == collar_ms:
                    return entry
            return None
        if match == 'min':
            for entry in records:
                if entry.collar_ms >= collar_ms:
                    return entry
            return None
        if match == 'max':
            for entry in reversed(records):
                if entry.collar_ms <= collar_ms:
                    return entry
            return None
        return min(records, key=lambda entry:
            abs(entry.collar_ms - collar_ms))

    def delete(self, metadata: Metadata) -> Metadata:
        '''Tombstone a metadata record and remove it from live indexes.'''
        deleted = metadata.mark_deleted()
        return self.upsert(deleted)

    def entries_for_shard(self, shard_id: str,
        include_deleted: bool=False) -> list[Metadata]:
        '''List entries that point to one shard.'''
        prefix = self._shard_key(shard_id, '')
        entry_ids = self._scan_prefix(self.by_shard_db, prefix)
        records = [self.get(entry_id) for entry_id in entry_ids]
        items = [record for record in records if record is not None]
        if include_deleted:
            return items
        return [item for item in items if item.storage_status == 'live']

    def _scan_prefix(self, db: object, prefix: bytes) -> list[str]:
        entry_ids: list[str] = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=db)
            if not cursor.set_range(prefix):
                return entry_ids
            for key, value in cursor:
                if not key.startswith(prefix):
                    break
                entry_ids.append(value.decode('utf-8'))
        return entry_ids

    def _shard_key(self, shard_id: str, entry_id: str) -> bytes:
        return f'shard:{shard_id}:{entry_id}'.encode('utf-8')
