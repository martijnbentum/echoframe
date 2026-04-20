'''End-to-end store tests with real LMDB and HDF5 backends.'''

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from echoframe.index import LmdbIndex
from echoframe.store import Store
from tests.helpers import (
    delete as _delete,
    find_many as _find_many,
    find_one as _find_one,
    hex_key as _hex,
    iter_object_frames as _iter_object_frames,
    load_many_queries as _load_many_queries,
    load_object_frames as _load_object_frames,
    load_query as _load_query,
    make_real_store,
    payload_to_list,
    pk as _pk,
    put as _put,
    put_many as _put_many,
)


@unittest.skipUnless(importlib.util.find_spec('lmdb'),
    'lmdb is not installed')
@unittest.skipUnless(importlib.util.find_spec('h5py'),
    'h5py is not installed')
class TestStoreEndToEnd(unittest.TestCase):

    def test_real_put_delete_compact_and_tag_flow(self) -> None:
        tmpdir, store = make_real_store()
        with tmpdir:
            records = _put_many(store, [
                {'phraser_key': 'phrase-1', 'collar': 100,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 3, 'data': [[1.0]],
                 'tags': ['exp-a', 'speaker-1']},
                {'phraser_key': 'phrase-2', 'collar': 100,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 3, 'data': [[2.0]],
                 'tags': ['exp-a', 'speaker-2']},
            ])
            self.assertEqual(len(records), 2)
            loaded = _load_query(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=3)
            self.assertEqual(payload_to_list(loaded), [[1.0]])
            updated = store.add_tags_many(
                [record.echoframe_key for record in records], ['batch'])
            self.assertEqual(len(updated), 2)
            self.assertEqual(sorted(store.list_tags()),
                ['batch', 'exp-a', 'speaker-1', 'speaker-2'])
            self.assertEqual(len(store.find_by_tags(['exp-a', 'batch'])), 2)
            self.assertEqual(len(store.find_by_tags(['speaker-1', 'speaker-2'],
                match='any')), 2)
            deleted = _delete(store, phraser_key='phrase-2', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=3)
            self.assertEqual(deleted.storage_status, 'deleted')
            shard_id = records[0].shard_id
            dry_run = store.compact_shards(dry_run=True)
            self.assertEqual(len(dry_run), 1)
            self.assertEqual(dry_run[0]['shard_id'], shard_id)
            self.assertEqual(dry_run[0]['deleted_entry_count'], 1)
            compacted = store.compact_shards()
            self.assertEqual(compacted, [shard_id])
            live = _find_one(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=3)
            self.assertNotEqual(live.shard_id, shard_id)
            self.assertEqual(payload_to_list(_load_query(store,
                phraser_key='phrase-1', collar=100, model_name='wav2vec2',
                output_type='hidden_state', layer=3)), [[1.0]])

    def test_real_integrity_checks_and_shard_stats(self) -> None:
        tmpdir, store = make_real_store()
        with tmpdir:
            metadata = _put(store, phraser_key='phrase-3', collar=120,
                model_name='hubert', output_type='attention', layer=2,
                data=[[[1, 2], [3, 4]]], tags=['exp-b'])
            _put(store, phraser_key='phrase-4', collar=120,
                model_name='hubert', output_type='attention', layer=2,
                data=[[[5, 6], [7, 8]]], tags=['exp-b', 'subset-1'])
            _delete(store, phraser_key='phrase-4', collar=120,
                model_name='hubert', output_type='attention', layer=2)
            stats = store.shard_stats()
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]['live_entry_count'], 1)
            self.assertEqual(stats[0]['deleted_entry_count'], 1)
            self.assertGreater(stats[0]['byte_size'], 0)
            store.storage.delete(metadata)
            report = store.verify_integrity()
            self.assertFalse(report['ok'])
            self.assertEqual(report['checked_metadata_count'], 1)
            self.assertEqual(len(report['broken_metadata_references']), 1)
            self.assertEqual(
                report['broken_metadata_references'][0]['echoframe_key_hex'],
                _hex(metadata))

    def test_real_find_many_and_tag_queries(self) -> None:
        tmpdir, store = make_real_store()
        with tmpdir:
            _put_many(store, [
                {'phraser_key': 'phrase-10', 'collar': 80,
                 'model_name': 'encodec',
                 'output_type': 'codebook_indices', 'layer': 0,
                 'data': [1, 2, 3], 'tags': ['exp-a', 'run-1']},
                {'phraser_key': 'phrase-11', 'collar': 90,
                 'model_name': 'encodec',
                 'output_type': 'codebook_indices', 'layer': 0,
                 'data': [4, 5, 6], 'tags': ['exp-a', 'run-2']},
                {'phraser_key': 'phrase-12', 'collar': 90,
                 'model_name': 'encodec',
                 'output_type': 'codebook_indices', 'layer': 0,
                 'data': [7, 8, 9], 'tags': ['exp-b', 'run-2']},
            ])
            results = _find_many(store, [
                {'phraser_key': 'phrase-10', 'collar': 80,
                 'model_name': 'encodec', 'output_type': 'codebook_indices',
                 'layer': 0},
                {'phraser_key': 'phrase-11', 'collar': 95,
                 'model_name': 'encodec', 'output_type': 'codebook_indices',
                 'layer': 0, 'match': 'max'},
            ])
            self.assertEqual([result.phraser_key for result in results], [
                _pk('phrase-10'), _pk('phrase-11')])
            self.assertEqual(sorted(store.list_tags()),
                ['exp-a', 'exp-b', 'run-1', 'run-2'])
            all_match = store.find_by_tags(['exp-a', 'run-2'], match='all')
            any_match = store.find_by_tags(['exp-b', 'run-1'], match='any')
            self.assertEqual([item.phraser_key for item in all_match],
                [_pk('phrase-11')])
            self.assertEqual(sorted(item.phraser_key for item in any_match), [
                _pk('phrase-10'), _pk('phrase-12')])

    def test_real_retrieval_helpers(self) -> None:
        tmpdir, store = make_real_store()
        with tmpdir:
            _put(store, phraser_key='phrase-20', collar=500,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0, 2.0]])
            _put(store, phraser_key='phrase-20', collar=750,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[3.0, 4.0]])
            _put(store, phraser_key='phrase-21', collar=500,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[5.0, 6.0]])
            payloads = _load_many_queries(store, [
                {'phraser_key': 'phrase-21', 'collar': 500,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 7},
                {'phraser_key': 'missing', 'collar': 500,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 7},
                {'phraser_key': 'phrase-20', 'collar': 500,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 7},
            ])
            exact = _load_object_frames(store, phraser_key='phrase-20',
                model_name='wav2vec2', layer=7, collar=500)
            nearest = _load_object_frames(store, phraser_key='phrase-20',
                model_name='wav2vec2', layer=7, collar=700,
                match='nearest')
            all_collars = _load_object_frames(store, phraser_key='phrase-20',
                model_name='wav2vec2', layer=7, collar=None)
            rows = list(_iter_object_frames(store, phraser_key='phrase-20',
                model_name='wav2vec2', layer=7))
            self.assertEqual([payload_to_list(item) if item is not None else None
                for item in payloads], [[[5.0, 6.0]], None, [[1.0, 2.0]]])
            self.assertEqual(payload_to_list(exact), [[1.0, 2.0]])
            self.assertEqual(payload_to_list(nearest), [[3.0, 4.0]])
            self.assertEqual(list(all_collars.keys()), [500, 750])
            self.assertEqual({collar: payload_to_list(payload)
                for collar, payload in all_collars.items()},
                {500: [[1.0, 2.0]], 750: [[3.0, 4.0]]})
            self.assertEqual(
                [(metadata.collar, payload_to_list(payload))
                    for metadata, payload in rows],
                [(500, [[1.0, 2.0]]), (750, [[3.0, 4.0]])])
            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                _load_many_queries(store, [{
                    'phraser_key': 'missing', 'collar': 500,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state', 'layer': 7,
                }], strict=True)

    def test_repeated_store_construction_reuses_lmdb_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Store(tmpdir, max_shard_size_bytes=1024 * 1024)
            second = Store(tmpdir, max_shard_size_bytes=1024 * 1024)
            self.assertIs(first.index.env, second.index.env)
            first_record = _put(first, phraser_key='phrase-cache-1',
                collar=100, model_name='wav2vec2',
                output_type='hidden_state', layer=3,
                data=[[1.0, 2.0]])
            self.assertEqual(payload_to_list(second.load(
                first_record.echoframe_key)), [[1.0, 2.0]])
            second_record = _put(second, phraser_key='phrase-cache-2',
                collar=100, model_name='wav2vec2',
                output_type='hidden_state', layer=3,
                data=[[3.0, 4.0]])
            self.assertEqual(payload_to_list(first.load(
                second_record.echoframe_key)), [[3.0, 4.0]])

    def test_equivalent_paths_reuse_the_same_lmdb_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            equivalent_root = root / '.'
            first = Store(root, max_shard_size_bytes=1024 * 1024)
            second = Store(equivalent_root, max_shard_size_bytes=1024 * 1024)
            self.assertIs(first.index.env, second.index.env)

    def test_different_roots_do_not_reuse_the_same_lmdb_env(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmpdir:
            with tempfile.TemporaryDirectory() as second_tmpdir:
                first = Store(first_tmpdir, max_shard_size_bytes=1024 * 1024)
                second = Store(second_tmpdir, max_shard_size_bytes=1024 * 1024)
                self.assertIsNot(first.index.env, second.index.env)

    def test_same_root_with_different_map_size_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Store(tmpdir, max_shard_size_bytes=1024 * 1024)
            with self.assertRaisesRegex(ValueError, 'map_size'):
                LmdbIndex(Path(tmpdir) / 'index.lmdb', map_size=2 << 30)
