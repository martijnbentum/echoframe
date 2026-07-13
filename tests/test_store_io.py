'''Tests for store I/O and public API behavior.'''

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest import mock

import echoframe
import numpy as np
from echoframe.metadata import EchoframeMetadata, filter_metadata
from echoframe.store import Store
from tests.helpers import (
    delete as _delete,
    exists as _exists,
    find as _find,
    find_many as _find_many,
    find_one as _find_one,
    hex_key as _hex,
    iter_object_frames as _iter_object_frames,
    load_many_queries as _load_many_queries,
    load_object_frames as _load_object_frames,
    load_query as _load_query,
    make_fake_store,
    pk as _pk,
    put as _put,
)


class TestStoreIo(unittest.TestCase):

    def test_public_exports(self) -> None:
        self.assertIn('Store', echoframe.__all__)
        self.assertIn('Embedding', echoframe.__all__)
        self.assertIn('Embeddings', echoframe.__all__)
        self.assertIn('EchoframeMetadata', echoframe.__all__)
        self.assertIn('Codevector', echoframe.__all__)
        self.assertIn('Codevectors', echoframe.__all__)
        self.assertIn('STABLE_METADATA_FIELDS', echoframe.__all__)
        self.assertNotIn('LmdbIndex', echoframe.__all__)
        self.assertNotIn('__version__', echoframe.__all__)
        self.assertFalse(hasattr(echoframe, '__version__'))

    def test_model_registry_str_includes_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            store.register_model('wav2vec2', language='en', size='base',
                architecture='wav2vec2')
            text = str(store.model_registry)
        self.assertIn('ModelRegistry', text)
        self.assertIn('architectures', text)
        self.assertIn('wav2vec2', text)

    def test_find_by_label_validation_and_missing_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            with self.assertRaisesRegex(ValueError,
                'label must be a non-empty string'):
                store.find_by_label('')
            with mock.patch.dict(sys.modules, {'phraser': None}):
                self.assertIsNone(store.find_by_label('hello'))

    def test_put_find_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            created = _put(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0, 2.0], [3.0, 4.0]], tags=['exp-a'])
            found = _find_one(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=7)
            payload = _load_query(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=7)

        self.assertEqual(found.echoframe_key, created.echoframe_key)
        self.assertEqual(found.tags, ['exp-a'])
        self.assertTrue(_exists(store, 'phrase-1', 120, 'wav2vec2',
            'hidden_state', 7))
        self.assertEqual(payload, [[1.0, 2.0], [3.0, 4.0]])

    def test_retrieval_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0], [2.0]])
            _put(store, phraser_key='phrase-1', collar=200,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[3.0], [4.0]])

            payloads = _load_many_queries(store, [
                {'phraser_key': 'phrase-1', 'collar': 100,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 1},
                {'phraser_key': 'phrase-1', 'collar': 200,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 1},
            ])
            metadata = _find(store, 'phrase-1', model_name='wav2vec2',
                output_type='hidden_state', layer=1)[0]
            payload = store.metadata_to_payload(metadata)
            many = store.metadatas_to_payloads([metadata])
            nearest = _load_object_frames(store, phraser_key='phrase-1',
                collar=150, model_name='wav2vec2',
                output_type='hidden_state', layer=1, collar_match='nearest')
            all_frames = _load_object_frames(store, phraser_key='phrase-1',
                collar=None, model_name='wav2vec2',
                output_type='hidden_state', layer=1)
            iterated = list(_iter_object_frames(store, phraser_key='phrase-1',
                collar=None, model_name='wav2vec2',
                output_type='hidden_state', layer=1))

        self.assertEqual(payloads,
            [[[1.0], [2.0]], [[3.0], [4.0]]])
        self.assertEqual(payload, [[1.0], [2.0]])
        self.assertEqual(many, [[[1.0], [2.0]]])
        self.assertEqual(nearest, [[1.0], [2.0]])
        self.assertEqual(sorted(all_frames), [100, 200])
        self.assertEqual(len(iterated), 2)

    def test_load_many_batches_payload_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            first = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            second = _put(store, phraser_key='phrase-2', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]])
            h5_module = store.storage.h5
            original_file = h5_module.File
            read_paths = []

            def counting_file(path, mode):
                if mode == 'r':
                    read_paths.append(str(path))
                return original_file(path, mode)

            keys = (key for key in [first.echoframe_key,
                second.echoframe_key])
            with mock.patch.object(h5_module, 'File',
                side_effect=counting_file):
                payloads = store.load_many(keys)

        self.assertEqual(payloads, [[[1.0]], [[2.0]]])
        self.assertEqual(len(read_paths), 1)

    def test_load_many_can_preserve_missing_payload_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            created = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            missing_key = b'\xff' * len(created.echoframe_key)
            skipped = store.load_many([created.echoframe_key, missing_key])
            kept = store.load_many([created.echoframe_key, missing_key],
                keep_missing=True)

        self.assertEqual(skipped, [[[1.0]]])
        self.assertEqual(kept, [[[1.0]], None])

    def test_load_frame_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            created = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1, 2], [3, 4], [5, 6], [7, 8]])

            first = store.load_frame(created.echoframe_key, frame='first')
            center = store.load_frame(created.echoframe_key, frame='center')
            last = store.load_frame(created.echoframe_key, frame='last')
            mean = store.load_frame(created.echoframe_key, frame='mean')

        self.assertEqual(first, [1, 2])
        self.assertEqual(center, [5, 6])
        self.assertEqual(last, [7, 8])
        self.assertEqual(mean.tolist(), [4.0, 5.0])
        self.assertEqual(mean.dtype.kind, 'f')

    def test_load_many_frames_batches_reads_and_keeps_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            first = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0], [2.0], [3.0]])
            second = _put(store, phraser_key='phrase-2', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[4.0], [5.0], [6.0]])
            missing_key = b'\xff' * len(first.echoframe_key)
            h5_module = store.storage.h5
            original_file = h5_module.File
            read_paths = []

            def counting_file(path, mode):
                if mode == 'r':
                    read_paths.append(str(path))
                return original_file(path, mode)

            with mock.patch.object(h5_module, 'File',
                side_effect=counting_file):
                rows = store.load_many_frames([first.echoframe_key,
                    missing_key, second.echoframe_key],
                    frame='last', keep_missing=True)

        self.assertEqual(rows, [[3.0], None, [6.0]])
        self.assertEqual(len(read_paths), 1)

    def test_load_frame_validates_mode_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            vector = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[1.0, 2.0])

            with self.assertRaisesRegex(ValueError, 'frame must be one of'):
                store.load_frame(vector.echoframe_key, frame='middle')
            with self.assertRaisesRegex(ValueError, '2D matrix payload'):
                store.load_frame(vector.echoframe_key)

    def test_delete_and_delete_many_by_echoframe_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            first = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            second = _put(store, phraser_key='phrase-2', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]])

            store.delete(first.echoframe_key)
            store.delete_many([second.echoframe_key])

        self.assertEqual(_find(store, 'phrase-1'), [])
        self.assertEqual(_find(store, 'phrase-2'), [])

    def test_collar_matching_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            _put(store, phraser_key='phrase-1', collar=200,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]])
            _put(store, phraser_key='phrase-1', collar=300,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[3.0]])

            min_match = _find_one(store, phraser_key='phrase-1', collar=150,
                model_name='wav2vec2', output_type='hidden_state', layer=1,
                collar_match='min')
            max_match = _find_one(store, phraser_key='phrase-1', collar=250,
                model_name='wav2vec2', output_type='hidden_state', layer=1,
                collar_match='max')
            nearest = _find_one(store, phraser_key='phrase-1', collar=240,
                model_name='wav2vec2', output_type='hidden_state', layer=1,
                collar_match='nearest')
            _delete(store, phraser_key='phrase-1', collar=200,
                model_name='wav2vec2', output_type='hidden_state', layer=1)

        self.assertEqual(min_match.collar, 200)
        self.assertEqual(max_match.collar, 100)
        self.assertEqual(nearest.collar, 200)
        self.assertIsNone(_find_one(store, phraser_key='phrase-1', collar=200,
            model_name='wav2vec2', output_type='hidden_state', layer=1))

    def test_tag_queries_and_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            created = _put(store, phraser_key='phrase-10', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[1.0]], tags=['exp-a', 'subset-1'])

            found = store.find_by_tag('exp-a')
            updated = store.add_tags(created.echoframe_key, ['review'])
            updated = store.remove_tags(created.echoframe_key, ['exp-a'])

        self.assertEqual([_hex(item) for item in found], [_hex(created)])
        self.assertEqual(updated.tags, ['review', 'subset-1'])
        self.assertEqual(store.find_by_tag('exp-a'), [])

    def test_tag_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-10', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[1.0]], tags=['exp-a', 'subset-1'])
            _put(store, phraser_key='phrase-11', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[2.0]], tags=['exp-a', 'subset-2'])

        self.assertEqual(store.tag_counts(), {
            'exp-a': 2, 'subset-1': 1, 'subset-2': 1})

    def test_index_validation_and_overview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            first = _put(store, phraser_key='phrase-10', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[1.0]], tags=['exp-a', 'subset-1'])
            second = _put(store, phraser_key='phrase-11', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[2.0]], tags=['exp-a', 'subset-2'])
            overview = store.overview()
            shard_rows = store.shard_stats()

        self.assertEqual(overview['metadata_count'], 2)
        self.assertEqual(overview['shard_count'], 1)
        self.assertIn('include_integrity=True', overview['integrity'])
        self.assertIn('include_garbage=True', overview['garbage_bytes'])
        self.assertEqual(overview['storage_bytes'],
            store.storage.storage_bytes())
        self.assertEqual(sorted(overview['tags']),
            ['exp-a', 'subset-1', 'subset-2'])
        self.assertEqual(len(overview['metadatas']), 2)
        self.assertEqual(len(overview['shards']), 1)
        self.assertEqual(shard_rows, overview['shards'])
        self.assertEqual(shard_rows[0]['entry_count'], 2)
        self.assertEqual(shard_rows[0]['byte_size'], 0)
        self.assertEqual(shard_rows[0]['shard_id'], first.shard_id)
        self.assertEqual(first.shard_id, second.shard_id)

    def test_find_by_shard_lists_only_shard_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            first = _put(store, phraser_key='phrase-10', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[1.0]])
            second = _put(store, phraser_key='phrase-11', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[2.0]])
            other = _put(store, phraser_key='phrase-12', collar=90,
                model_name='hubert', output_type='hidden_state',
                layer=5, data=[[3.0]])
            entries = store.index.find_by_shard(first.shard_id, store=store)
            other_entries = store.index.find_by_shard(other.shard_id,
                store=store)

        self.assertEqual(sorted(_hex(item) for item in entries),
            sorted([_hex(first), _hex(second)]))
        self.assertEqual([_hex(item) for item in other_entries],
            [_hex(other)])

    def test_find_by_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            first = _put(store, phraser_key='phrase-10', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[1.0]])
            second = _put(store, phraser_key='phrase-10', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=6, data=[[2.0]])
            _put(store, phraser_key='phrase-11', collar=90,
                model_name='hubert', output_type='hidden_state',
                layer=5, data=[[3.0]])
            phraser_store = types.SimpleNamespace(
                path='/phraser/cgn-main',
                load=mock.Mock(side_effect=lambda key: {
                    _pk('phrase-10'): types.SimpleNamespace(label='hello'),
                    _pk('phrase-11'): types.SimpleNamespace(label='world'),
                }[key]))
            store.attach_phraser_store('cgn-main', phraser_store)
            store.backfill_phraser_source_id('cgn-main')
            records = store.find_by_label('hello')
            filtered = store.find_by_label('hello',
                model_name='wav2vec2', layer=6)

        self.assertEqual(sorted(_hex(item) for item in records),
            sorted([_hex(first), _hex(second)]))
        self.assertEqual([_hex(item) for item in filtered], [_hex(second)])

    def test_find_by_label_requires_unambiguous_phraser_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-10', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[1.0]])
            with self.assertRaisesRegex(ValueError,
                'unknown phraser_source_id'):
                store.find_by_label('hello')

    def test_missing_and_validation_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            self.assertEqual(_find_many(store, []), [])
            self.assertEqual(_load_many_queries(store, []), [])
            self.assertEqual(store.save_many([]), 0)
            self.assertEqual(store.add_tags_many([], ['exp-a']), [])
            self.assertEqual(store.remove_tags_many([], ['exp-a']), [])
            self.assertIsNone(_delete(store, phraser_key='missing',
                collar=10, model_name='wav2vec2',
                output_type='hidden_state', layer=1))
            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                _load_query(store, phraser_key='missing', collar=10,
                    model_name='wav2vec2', output_type='hidden_state',
                    layer=1)
            with self.assertRaises(ValueError):
                EchoframeMetadata(b'bad-key', tags=['bad:tag'])

    def test_missing_retrieval_behaviors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            payloads = _load_many_queries(store, [
                {'phraser_key': 'phrase-1', 'collar': 100,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 1},
                {'phraser_key': 'missing', 'collar': 100,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 1},
            ])
            with self.assertRaisesRegex(ValueError, 'no stored output matched'):
                _load_many_queries(store, [{
                    'phraser_key': 'missing',
                    'collar': 100,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 1,
                }], strict=True)
            with self.assertRaisesRegex(ValueError, 'no stored output matched'):
                _load_object_frames(store, phraser_key='missing', collar=500,
                    model_name='wav2vec2', output_type='hidden_state',
                    layer=7)
            self.assertEqual(_load_object_frames(store, phraser_key='missing',
                collar=None, model_name='wav2vec2',
                output_type='hidden_state', layer=7), {})
            with self.assertRaisesRegex(ValueError, 'no stored output matched'):
                list(_iter_object_frames(store, phraser_key='missing',
                    collar=500, model_name='wav2vec2',
                    output_type='hidden_state', layer=7))
            self.assertEqual(list(_iter_object_frames(store,
                phraser_key='missing', collar=None, model_name='wav2vec2',
                output_type='hidden_state', layer=7)), [])
        self.assertEqual(payloads, [[[1.0]], None])

    def test_store_metadata_are_bound_and_can_load_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            created = _put(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0, 2.0], [3.0, 4.0]], tags=['exp-a'])
            found = _find_one(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=7)
            listed = store.metadatas

        self.assertIsNone(created.store)
        with self.assertRaisesRegex(ValueError, 'store is not attached'):
            created.load_payload()
        self.assertIs(found.store, store)
        self.assertEqual(len(listed), 1)
        self.assertIs(listed[0].store, store)

    def test_metadata_round_trips_phraser_source_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            store.register_model('wav2vec2')
            echoframe_key = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=_pk('phrase-1'),
                layer=7, collar=120)
            metadata = EchoframeMetadata(echoframe_key, store=store,
                model_name='wav2vec2', phraser_source_id='cgn-main')
            stored = store.save(echoframe_key, metadata, [[1.0]])
            loaded = store.load_metadata(stored.echoframe_key)
        self.assertEqual(loaded.phraser_source_id, 'cgn-main')

    def test_backfill_phraser_source_id_updates_legacy_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            store.register_phraser_store('cgn-main', '/data/cgn')
            created = _put(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0]])
            legacy = store.load_metadata(created.echoframe_key).copy(
                phraser_source_id=None)
            store.index.save(legacy)
            matrix_key = store.make_echoframe_key('codebook_matrix',
                model_name='wav2vec2')
            matrix_metadata = EchoframeMetadata(matrix_key, store=store,
                model_name='wav2vec2')
            matrix = store.save(matrix_key, matrix_metadata, [[1.0, 2.0]])
            count = store.backfill_phraser_source_id('cgn-main')
            loaded = store.load_metadata(created.echoframe_key)
            loaded_matrix = store.load_metadata(matrix.echoframe_key)
        self.assertEqual(count, 1)
        self.assertEqual(loaded.phraser_source_id, 'cgn-main')
        self.assertIsNone(loaded_matrix.phraser_source_id)

    def test_metadata_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            created = _put(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0, 2.0], [3.0, 4.0]], tags=['exp-a'])
            found = _find_one(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=7)
            data = found.to_dict()
            legacy_data = dict(data)
            legacy_data['accessed_at'] = '2026-01-01T00:00:00+00:00'
            round_tripped = EchoframeMetadata.from_dict(legacy_data,
                found.echoframe_key, store=store)
            retagged = found.with_tags(['review', 'subset-1'])
            self.assertEqual(data['model_name'], 'wav2vec2')
            self.assertEqual(data['phraser_source_id'], 'cgn-main')
            self.assertEqual(data['shard_id'], found.shard_id)
            self.assertEqual(data['dataset_path'], found.dataset_path)
            self.assertEqual(tuple(data['shape']), found.shape)
            self.assertEqual(data['tags'], ['exp-a'])
            self.assertNotIn('accessed_at', data)
            self.assertEqual(round_tripped.echoframe_key, found.echoframe_key)
            self.assertIs(round_tripped.store, store)
            self.assertEqual(round_tripped.to_dict(), data)
            self.assertEqual(retagged.tags, ['review', 'subset-1'])
            self.assertEqual(retagged.shard_id, found.shard_id)
            self.assertEqual(found.load_payload(), [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(created.echoframe_key, found.echoframe_key)


class TestMetadatasCache(unittest.TestCase):

    def test_metadatas_reflect_later_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0]])
            self.assertEqual(len(store.metadatas), 1)
            _put(store, phraser_key='phrase-2', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[2.0]])
            self.assertEqual(len(store.metadatas), 2)

    def test_metadatas_are_cached_between_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0]])
            first = store.metadatas
            self.assertIs(store.metadatas, first)

    def test_tag_update_invalidates_metadatas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            created = _put(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0]])
            before = store.metadatas
            store.add_tags(created.echoframe_key, ['exp-a'])
            after = store.metadatas
            self.assertIsNot(after, before)
            self.assertEqual(after[0].tags, ['exp-a'])


class TestFilterMetadataNoneCollar(unittest.TestCase):

    def _records_with_codebook_matrix(self, store):
        _put(store, phraser_key='phrase-1', collar=100,
            model_name='wav2vec2', output_type='hidden_state',
            layer=1, data=[[1.0]])
        _put(store, phraser_key='phrase-1', collar=300,
            model_name='wav2vec2', output_type='hidden_state',
            layer=1, data=[[2.0]])
        _put(store, phraser_key='phrase-1', collar=None,
            model_name='wav2vec2', output_type='codebook_matrix',
            layer=None, data=[[3.0]])
        return store.metadatas

    def test_collar_match_skips_records_without_collar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            records = self._records_with_codebook_matrix(store)

            self.assertEqual(len(filter_metadata(records)), 3)
            min_match = filter_metadata(records, collar=150,
                collar_match='min')
            max_match = filter_metadata(records, collar=150,
                collar_match='max')
            nearest = filter_metadata(records, collar=240,
                collar_match='nearest')

        self.assertEqual([record.collar for record in min_match], [300])
        self.assertEqual([record.collar for record in max_match], [100])
        self.assertEqual([record.collar for record in nearest], [300])

    def test_nearest_without_collared_records_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-1', collar=None,
                model_name='wav2vec2', output_type='codebook_matrix',
                layer=None, data=[[1.0]])
            records = store.metadatas

            nearest = filter_metadata(records, collar=100,
                collar_match='nearest')

        self.assertEqual(nearest, [])
