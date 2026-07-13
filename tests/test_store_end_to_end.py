'''End-to-end store tests with real LMDB and HDF5 backends.'''

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from echoframe.index import LmdbIndex
from echoframe.metadata import EchoframeMetadata
from echoframe.output_storage import Hdf5ShardStore
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

    def test_real_put_delete_compact_and_tag_flow(self) -> None:
        # payloads must dwarf HDF5 file overhead, and the deleted record must
        # not sit at the file tail (h5py truncates tail space on delete), so
        # the delete leaves enough garbage to trigger the compaction heuristic
        first_data = [[1.0] * 64] * 64
        second_data = [[2.0] * 64] * 64
        tmpdir, store = make_real_store()
        with tmpdir:
            first = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=first_data)
            second = _put(store, phraser_key='phrase-2', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=second_data)
            loaded = store.load(second.echoframe_key)
            store.add_tags_many([second.echoframe_key], ['review'])
            store.delete(first.echoframe_key)
            plans = store.compact_shards(dry_run=True)
            compacted = store.compact_shards()
            refreshed = store.load_metadata(second.echoframe_key)
            plans_after_compaction = store.compact_shards(dry_run=True)

        self.assertEqual(payload_to_list(loaded), second_data)
        self.assertEqual(store.load_metadata(first.echoframe_key), None)
        self.assertTrue(plans)
        self.assertTrue(compacted)
        self.assertIsNotNone(refreshed)
        self.assertEqual(plans_after_compaction, [])

    def test_real_find_many_and_tag_queries(self) -> None:
        tmpdir, store = make_real_store()
        with tmpdir:
            first = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]], tags=['exp-a', 'shared'])
            second = _put(store, phraser_key='phrase-2', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]], tags=['exp-b', 'shared'])
            third = _put(store, phraser_key='phrase-3', collar=100,
                model_name='hubert', output_type='hidden_state',
                layer=1, data=[[3.0]], tags=['exp-b'])
            matches = _find_many(store, [
                {'phraser_key': 'phrase-1', 'collar': 100,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 1},
                {'phraser_key': 'phrase-3', 'collar': 100,
                 'model_name': 'hubert', 'output_type': 'hidden_state',
                 'layer': 1},
            ])
            all_tags = store.find_by_tags(['shared'], match='all')
            any_tags = store.find_by_tags(['shared', 'exp-b'], match='any')

        self.assertEqual([_hex(item) for item in matches],
            [_hex(first), _hex(third)])
        self.assertEqual(sorted(_hex(item) for item in all_tags),
            sorted([_hex(first), _hex(second)]))
        self.assertEqual(sorted(_hex(item) for item in any_tags),
            sorted([_hex(first), _hex(second), _hex(third)]))

    def test_storage_bytes_follows_injected_storage_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'store'
            shards_elsewhere = Path(tmpdir) / 'elsewhere'
            storage = Hdf5ShardStore(shards_elsewhere,
                max_shard_size_bytes=1024 * 1024)
            store = Store(root, storage=storage)
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])

            self.assertGreater(store._storage_bytes(), 0)
            self.assertEqual(store._storage_bytes(), storage.storage_bytes())
            self.assertEqual(store.overview()['storage_bytes'],
                storage.storage_bytes())

    def test_save_many_duplicate_key_cleans_first_version_entries(self) -> None:
        tmpdir, store = make_real_store()
        with tmpdir:
            store.register_model('wav2vec2')
            echoframe_key = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=_pk('phrase-1'),
                layer=1, collar=100)
            first = EchoframeMetadata(echoframe_key, model_name='wav2vec2',
                tags=['exp-a'])
            second = EchoframeMetadata(echoframe_key, model_name='wav2vec2',
                tags=['exp-b'])

            store.index.save_many([first, second])

            self.assertEqual(store.find_by_tag('exp-a'), [])
            self.assertEqual([_hex(item) for item in
                store.find_by_tag('exp-b')], [_hex(second)])
            self.assertEqual(store.index.load(echoframe_key).tags, ['exp-b'])

    def test_real_integrity_checks_and_shard_stats(self) -> None:
        tmpdir, store = make_real_store()
        with tmpdir:
            created = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0], [2.0]])
            shard_rows = store.shard_stats()
            self.assertEqual(len(shard_rows), 1)
            self.assertEqual(shard_rows[0]['shard_id'], created.shard_id)
            self.assertEqual(shard_rows[0]['entry_count'], 1)
            self.assertGreater(shard_rows[0]['byte_size'], 0)

            store.storage.delete(created)
            integrity = store.verify_integrity()

        self.assertFalse(integrity['ok'])
        self.assertEqual(integrity['checked_metadata_count'], 1)
        self.assertEqual(len(integrity['broken_metadata_references']), 1)
        self.assertEqual(integrity['broken_metadata_references'][0]['reason'],
            'missing dataset')

    def test_real_retrieval_helpers(self) -> None:
        tmpdir, store = make_real_store()
        with tmpdir:
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            _put(store, phraser_key='phrase-1', collar=200,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]])
            payloads = _load_many_queries(store, [
                {'phraser_key': 'phrase-1', 'collar': 100,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 1},
                {'phraser_key': 'missing', 'collar': 100,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 1},
            ])
            exact = _load_object_frames(store, phraser_key='phrase-1',
                model_name='wav2vec2', layer=1, collar=100)
            nearest = _load_object_frames(store, phraser_key='phrase-1',
                model_name='wav2vec2', layer=1, collar=150,
                collar_match='nearest')
            all_frames = _load_object_frames(store, phraser_key='phrase-1',
                model_name='wav2vec2', layer=1, collar=None)
            iterated = list(_iter_object_frames(store, phraser_key='phrase-1',
                model_name='wav2vec2', layer=1, collar=None))

        self.assertEqual([payload_to_list(item) if item is not None else None
            for item in payloads], [[[1.0]], None])
        self.assertEqual(payload_to_list(exact), [[1.0]])
        self.assertEqual(payload_to_list(nearest), [[1.0]])
        self.assertEqual(sorted(all_frames), [100, 200])
        self.assertEqual(len(iterated), 2)

    def test_repeated_store_construction_reuses_lmdb_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Store(tmpdir, max_shard_size_bytes=1024 * 1024)
            second = Store(tmpdir, max_shard_size_bytes=1024 * 1024)
            _put(first, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            found = _find_one(second, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=1)

        self.assertIs(first.index.env, second.index.env)
        self.assertIsNotNone(found)
