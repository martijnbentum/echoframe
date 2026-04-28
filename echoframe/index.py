'''LMDB-backed index for raw echoframe metadata values.'''

import json
from pathlib import Path

from . import compaction, lmdb_helper
from .metadata import EchoframeMetadata, normalize_tags, utc_now


class LmdbIndex:
    '''Store and query raw LMDB values for echoframe metadata.'''

    def __init__(self, path, map_size=1 << 30, env=None, shards_root=None):
        self.path = Path(path)
        self.shards_root = None if shards_root is None else Path(shards_root)
        self.env = env or lmdb_helper.open_env(self.path, map_size=map_size)
        self.db_handles = lmdb_helper.open_databases(self.env)
        self.entries_db = self.db_handles['entries_db']
        self.by_phraser_db = self.db_handles['by_phraser_db']
        self.by_shard_db = self.db_handles['by_shard_db']
        self.by_tag_db = self.db_handles['by_tag_db']
        self.shard_meta_db = self.db_handles['shard_meta_db']
        self.compaction_db = self.db_handles['compaction_db']

    def load(self, echoframe_key, store = None):
        value = lmdb_helper.load(self.env, self.entries_db, echoframe_key)
        if value is None:
            return None
        data = json.loads(value.decode('utf-8'))
        return EchoframeMetadata.from_dict(data, echoframe_key=echoframe_key,
            store=store)

    def load_many(self, echoframe_keys, store = None, keep_missing=False):
        '''Load multiple raw metadata values by echoframe key.
        echoframe_keys:     list of identifiers for metadata records to load
        store:             optional Store instance to attach to loaded metadata
        keep_missing:      if True, preserve input alignment with None for misses
        '''
        values = lmdb_helper.load_many(self.env, self.entries_db, echoframe_keys)
        metadata_list = []
        for key, value in zip(echoframe_keys,values):
            if value is None and not keep_missing: 
                print(f'Warning: missing metadata for echoframe key: {key}')
                continue
            elif value is None and keep_missing:
                metadata_list.append(None)
                continue
            data = json.loads(value.decode('utf-8'))
            md = EchoframeMetadata.from_dict(data,echoframe_key=key,store=store)
            metadata_list.append(md)
        return metadata_list

    def all_metadatas(self, store = None):
        '''Load all raw metadata values from the entries store.'''
        return self.load_many(self.all_echoframe_keys, store=store)

    @property
    def all_echoframe_keys(self):
        return lmdb_helper.load_all_keys(self.env, self.entries_db)

    def find_phraser(self, phraser_key, store=None):
        '''Find metadata records for one phraser key.'''
        with lmdb_helper.read_txn(self.env) as txn:
            prefix = lmdb_helper.phraser_scan_prefix(phraser_key)
            echoframe_keys = self._scan_prefix_in_txn(txn, self.by_phraser_db,
                prefix)
        return self.load_many(echoframe_keys, store=store)

    def find_by_tag(self, tag, store=None):
        '''List metadata records for one tag.'''
        records = self.find_by_tags([tag], match='all', store=store)
        return records

    def find_by_tags(self, tags, match='all', store=None):
        '''List metadata records that match one or more tags.'''
        tags = normalize_tags(tags)
        if match not in {'all', 'any'}:
            message = "match must be 'all' or 'any'"
            raise ValueError(message)
        if not tags:
            return []

        with lmdb_helper.read_txn(self.env) as txn:
            groups = [set(self._scan_prefix_in_txn(txn, self.by_tag_db,
                self._tag_key(tag, b''))) for tag in tags]
        if not groups:
            return []
        echoframe_keys = set.intersection(*groups) if match == 'all' else (
            set().union(*groups))
        return self.load_many(sorted(echoframe_keys), store=store)

    def list_tags(self):
        '''List all known tags.'''
        return sorted(self.tag_counts().keys())

    def tag_counts(self):
        '''Count metadata records per tag.'''
        counts = {}
        with lmdb_helper.read_txn(self.env) as txn:
            cursor = txn.cursor(db=self.by_tag_db)
            if not cursor.set_range(b'tag:'):
                return counts
            for key, _ in cursor:
                if not key.startswith(b'tag:'):
                    break
                _, remainder = key.split(b'tag:', 1)
                tag, _, _ = remainder.partition(b':')
                tag = tag.decode('utf-8')
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def add_tags(self, echoframe_key, tags, store=None):
        '''Add tags to one metadata record.'''
        updated = self.add_tags_many([echoframe_key], tags, store=store)
        if not updated:
            return None
        return updated[0]

    def add_tags_many(self, echoframe_keys, tags, store=None):
        '''Add tags to multiple metadata records.'''
        tags = normalize_tags(tags)
        if not echoframe_keys:
            return []

        updated = []
        for metadata in self.load_many(echoframe_keys, store=store):
            merged = sorted(set(metadata.tags).union(tags))
            updated.append(metadata.with_tags(merged))
        self.save_many(updated)
        return updated

    def remove_tags(self, echoframe_key, tags, store=None):
        '''Remove tags from one metadata record.'''
        updated = self.remove_tags_many([echoframe_key], tags, store=store)
        if not updated:
            return None
        return updated[0]

    def remove_tags_many(self, echoframe_keys, tags, store=None):
        '''Remove tags from multiple metadata records.'''
        tags = normalize_tags(tags)
        if not echoframe_keys:
            return []

        updated = []
        for metadata in self.load_many(echoframe_keys, store=store):
            remaining = sorted(tag for tag in metadata.tags if tag not in tags)
            updated.append(metadata.with_tags(remaining))
        self.save_many(updated)
        return updated

    def save(self, metadata):
        '''Insert or replace one metadata record.'''
        self.save_many([metadata])

    def save_many(self, metadata_list):
        '''Insert or replace multiple metadata records in one transaction.'''
        if not metadata_list: return 
        touched_shard_ids = set()
        with lmdb_helper.write_txn(self.env) as txn:
            for metadata in metadata_list:
                sids = self._save_many_helper(txn, metadata)
                touched_shard_ids.update(sids)

            for shard_id in sorted(touched_shard_ids):
                self._refresh_shard_stats(txn, shard_id)

    def delete(self, metadata):
        '''Delete one metadata record and its secondary index entries.'''
        self.delete_many([metadata])

    def delete_many(self, metadata_list):
        '''Delete multiple metadata records and their secondary indexes.'''
        if not metadata_list: return
        touched_shard_ids = set()
        with lmdb_helper.write_txn(self.env) as txn:
            for metadata in metadata_list:
                txn.delete(metadata.echoframe_key, db=self.entries_db)
                lmdb_helper.delete_phraser_keys(txn, self.db_handles, metadata)
                self._delete_tag_keys(txn, metadata)
                if metadata.shard_id:
                    txn.delete(self._shard_key(metadata.shard_id,
                        metadata.echoframe_key), db=self.by_shard_db)
                    touched_shard_ids.add(metadata.shard_id)

            for shard_id in sorted(touched_shard_ids):
                self._refresh_shard_stats(txn, shard_id)

    def _save_many_helper(self, txn, metadata):
        touched_shard_ids = set()
        previous = self.load(metadata.echoframe_key)
        if previous is not None:
            lmdb_helper.delete_phraser_keys(txn, self.db_handles, previous)
            self._delete_shard_keys(txn, previous, metadata)
            self._delete_tag_keys(txn, previous)
            if previous.shard_id: touched_shard_ids.add(previous.shard_id)

        shard_ids = lmdb_helper.write_metadata(txn, self.db_handles, metadata)
        touched_shard_ids.update(shard_ids)
        return touched_shard_ids

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
                if shard_id in seen: continue
                seen.add(shard_id)
                shard_ids.append(shard_id)
        return shard_ids

    def load_shard_metadata(self, shard_id):
        '''Return shard-level stats.'''
        with lmdb_helper.read_txn(self.env) as txn:
            value = txn.get(shard_id.encode('utf-8'), db=self.shard_meta_db)
        if value is None:
            return None
        return json.loads(value.decode('utf-8'))

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

    def remove_shard_entries(self, shard_id, echoframe_keys):
        '''Remove shard index entries for one shard.'''
        with lmdb_helper.write_txn(self.env) as txn:
            for echoframe_key in echoframe_keys:
                txn.delete(self._shard_key(shard_id, echoframe_key),
                    db=self.by_shard_db)
            self._refresh_shard_stats(txn, shard_id)

    def create_compaction_journal(self, shard_id, echoframe_keys,
        target_shard_id):
        '''Create a compaction journal record.'''
        return compaction.create_compaction_journal(self, shard_id,
            echoframe_keys=echoframe_keys,
            target_shard_id=target_shard_id)

    def update_compaction_journal(self, journal_id, status, error=None):
        '''Update one compaction journal record.'''
        return compaction.update_compaction_journal(self, journal_id,
            status=status, error=error)

    def list_compaction_journal(self, status=None):
        '''List compaction journal records.'''
        return compaction.list_compaction_journal(self, status=status)

    def _scan_prefix(self, db, prefix):
        with lmdb_helper.read_txn(self.env) as txn:
            return self._scan_prefix_in_txn(txn, db, prefix)

    def _scan_prefix_in_txn(self, txn, db, prefix):
        return lmdb_helper.scan_prefix_in_txn(txn, db, prefix)

    def _shard_key(self, shard_id, echoframe_key):
        return lmdb_helper.shard_key(shard_id, echoframe_key)

    def _tag_key(self, tag, echoframe_key):
        return lmdb_helper.tag_key(tag, echoframe_key)

    def _delete_tag_keys(self, txn, metadata):
        lmdb_helper.delete_tag_keys(txn, self.by_tag_db, metadata)

    def _delete_shard_keys(self, txn, previous, metadata):
        lmdb_helper.delete_shard_keys(txn, self.by_shard_db, previous,
            metadata)

    def _refresh_shard_stats(self, txn, shard_id):
        prefix = self._shard_key(shard_id, b'')
        echoframe_keys = self._scan_prefix_in_txn(txn, self.by_shard_db, prefix)
        if not echoframe_keys:
            txn.delete(shard_id.encode('utf-8'), db=self.shard_meta_db)
            return
        entry_count = 0
        for echoframe_key in echoframe_keys:
            value = txn.get(echoframe_key, db=self.entries_db)
            if value is None: continue
            entry_count += 1
        shard_size = self._shard_file_size(shard_id)
        value = {'entry_count': entry_count, 'byte_size': shard_size,
            'updated_at': utc_now()}
        d = json.dumps(value, sort_keys=True).encode('utf-8')
        txn.put(shard_id.encode('utf-8'), d, db=self.shard_meta_db)
            

    def _shard_file_size(self, shard_id):
        if self.shards_root is None:
            return 0
        file_path = self.shards_root / f'{shard_id}.h5'
        try: return file_path.stat().st_size
        except FileNotFoundError: return 0
        except OSError: return 0
