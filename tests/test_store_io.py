'''Tests for store I/O and public API behavior.'''

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest import mock

import echoframe
from echoframe.metadata import EchoframeMetadata
from echoframe.store import Store
from tests.helpers import (
    delete as _delete,
    exists as _exists,
    find as _find,
    find_many as _find_many,
    find_one as _find_one,
    find_or_compute as _find_or_compute,
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
        self.assertIn('EchoframeMetadata', echoframe.__all__)
        self.assertIn('Codebook', echoframe.__all__)
        self.assertIn('STABLE_METADATA_FIELDS', echoframe.__all__)
        self.assertNotIn('LmdbIndex', echoframe.__all__)
        self.assertNotIn('__version__', echoframe.__all__)
        self.assertFalse(hasattr(echoframe, '__version__'))

    def test_put_find_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            metadata = _put(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0, 2.0], [3.0, 4.0]], tags=['exp-a'])
            self.assertEqual(metadata.phraser_key, _pk('phrase-1'))
            self.assertEqual(metadata.layer, 7)
            self.assertEqual(metadata.shard_id, 'wav2vec2_hidden_state_0001')
            self.assertEqual(metadata.tags, ['exp-a'])
            self.assertTrue(_exists(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=7))
            self.assertEqual(_load_query(store, phraser_key='phrase-1',
                collar=120, model_name='wav2vec2',
                output_type='hidden_state', layer=7),
                [[1.0, 2.0], [3.0, 4.0]])

    def test_retrieval_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            first = _put(store, phraser_key='phrase-1', collar=500,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0, 2.0]])
            second = _put(store, phraser_key='phrase-1', collar=750,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[3.0, 4.0]])
            payloads = _load_many_queries(store, [
                {'phraser_key': 'phrase-1', 'collar': 750,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 7},
                {'phraser_key': 'phrase-1', 'collar': 500,
                 'model_name': 'wav2vec2', 'output_type': 'hidden_state',
                 'layer': 7},
            ])
            self.assertEqual(payloads, [[[3.0, 4.0]], [[1.0, 2.0]]])
            payload = store.metadata_to_payload(first)
            payloads = store.metadatas_to_payloads([second, first])
            self.assertEqual(payload, [[1.0, 2.0]])
            self.assertEqual(payloads, [[[3.0, 4.0]], [[1.0, 2.0]]])
            exact = _load_object_frames(store, phraser_key='phrase-1',
                collar=500, model_name='wav2vec2', output_type='hidden_state',
                layer=7)
            nearest = _load_object_frames(store, phraser_key='phrase-1',
                collar=700, model_name='wav2vec2', output_type='hidden_state',
                layer=7, match='nearest')
            rows = list(_iter_object_frames(store, phraser_key='phrase-1',
                model_name='wav2vec2', layer=7))
            self.assertEqual(exact, [[1.0, 2.0]])
            self.assertEqual(nearest, [[3.0, 4.0]])
            self.assertEqual([(metadata.collar, payload)
                for metadata, payload in rows],
                [(500, [[1.0, 2.0]]), (750, [[3.0, 4.0]])])
            self.assertEqual(_hex(rows[0][0]), _hex(first))
            self.assertEqual(_hex(rows[1][0]), _hex(second))

    def test_collar_matching_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            for collar in (100, 200, 350):
                _put(store, phraser_key='word-1', collar=collar,
                    model_name='hubert', output_type='attention', layer=3,
                    data=[[[1, 2], [3, 4]]])
            minimum = _find_one(store, phraser_key='word-1', collar=150,
                model_name='hubert', output_type='attention', layer=3,
                match='min')
            maximum = _find_one(store, phraser_key='word-1', collar=150,
                model_name='hubert', output_type='attention', layer=3,
                match='max')
            nearest = _find_one(store, phraser_key='word-1', collar=180,
                model_name='hubert', output_type='attention', layer=3,
                match='nearest')
            self.assertEqual(minimum.collar, 200)
            self.assertEqual(maximum.collar, 100)
            self.assertEqual(nearest.collar, 200)
            deleted = _delete(store, phraser_key=_pk('word-1'), collar=200,
                model_name='hubert', output_type='attention', layer=3)
            self.assertEqual(deleted.storage_status, 'deleted')
            self.assertFalse(_exists(store, phraser_key='word-1', collar=200,
                model_name='hubert', output_type='attention', layer=3))

    def test_find_or_compute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            calls: list[str] = []

            def compute() -> list[int]:
                calls.append('compute')
                return [1, 2, 3]

            metadata, created = _find_or_compute(store, phraser_key=_pk('p1'),
                collar=50, model_name='encodec',
                output_type='codebook_indices', layer=0, compute=compute)
            again, created_again = _find_or_compute(store, phraser_key=_pk('p1'),
                collar=50, model_name='encodec',
                output_type='codebook_indices', layer=0, compute=compute)
            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(_hex(metadata), _hex(again))
            self.assertEqual(calls, ['compute'])

    def test_find_or_compute_can_add_tags_on_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)

            def compute() -> list[int]:
                return [1, 2, 3]

            metadata, created = _find_or_compute(store, phraser_key=_pk('p2'),
                collar=50, model_name='encodec',
                output_type='codebook_indices', layer=0, compute=compute,
                tags=['exp-a'])
            again, created_again = _find_or_compute(store, phraser_key=_pk('p2'),
                collar=50, model_name='encodec',
                output_type='codebook_indices', layer=0, compute=compute,
                tags=['exp-b'], add_tags_on_hit=True)
            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(_hex(metadata), _hex(again))
            self.assertEqual(again.tags, ['exp-a', 'exp-b'])

    def test_tag_queries_and_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            metadata = _put(store, phraser_key='phrase-2', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=5, data=[[1.0]], tags=['exp-a', 'subset-1'])
            entries = store.find_by_tag('exp-a')
            self.assertEqual(len(entries), 1)
            self.assertEqual(_hex(entries[0]), _hex(metadata))
            updated = store.add_tags(metadata.echoframe_key, ['review'])
            self.assertEqual(updated.tags, ['exp-a', 'review', 'subset-1'])
            entries = store.find_by_tag('review')
            self.assertEqual(len(entries), 1)
            updated = store.remove_tags(metadata.echoframe_key, ['exp-a'])
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
            load = mock.Mock(side_effect=lambda key: {
                _pk('phrase-10'): types.SimpleNamespace(label='hello'),
                _pk('phrase-11'): types.SimpleNamespace(label='world'),
            }[key])
            fake_models = types.SimpleNamespace(
                cache=types.SimpleNamespace(load=load))
            fake_phraser = types.SimpleNamespace(models=fake_models)
            with mock.patch.dict(sys.modules, {'phraser': fake_phraser}):
                records = store.find_by_label('hello')
                filtered = store.find_by_label('hello',
                    model_name='wav2vec2', layer=6)
            self.assertEqual(sorted(_hex(item) for item in records),
                sorted([_hex(first), _hex(second)]))
            self.assertEqual([_hex(item) for item in filtered], [_hex(second)])

    def test_find_by_label_validation_and_missing_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            with self.assertRaisesRegex(ValueError,
                'label must be a non-empty string'):
                store.find_by_label('')
            with mock.patch.dict(sys.modules, {'phraser': None}):
                with self.assertRaisesRegex(ImportError,
                    'phraser is required to find entries by label'):
                    store.find_by_label('hello')

    def test_missing_and_validation_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            self.assertEqual(_find_many(store, []), [])
            self.assertEqual(_load_many_queries(store, []), [])
            self.assertEqual(store.put_many([]), [])
            self.assertEqual(store.add_tags_many([], ['exp-a']), [])
            self.assertEqual(store.remove_tags_many([], ['exp-a']), [])
            self.assertIsNone(_delete(store, phraser_key=_pk('missing'),
                collar=10, model_name='wav2vec2',
                output_type='hidden_state', layer=1))
            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                _load_query(store, phraser_key='missing', collar=10,
                    model_name='wav2vec2',
                    output_type='hidden_state', layer=1)
            with self.assertRaises(ValueError):
                EchoframeMetadata(phraser_key='phrase-1', collar=10,
                    model_name='wav2vec2', output_type='hidden_state',
                    layer=1, tags=['bad:tag'])
            with self.assertRaises(ValueError):
                EchoframeMetadata(phraser_key='phrase-1', collar=10,
                    model_name='wav2vec2', output_type='hidden_state',
                    layer=1, tags=['   '])
            with self.assertRaises(ValueError):
                EchoframeMetadata(phraser_key='phrase-1', collar=10,
                    model_name='wav2vec2', output_type='hidden_state',
                    layer=1, tags=[123])

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
            self.assertEqual(payloads, [[[1.0]], None])
            with self.assertRaisesRegex(ValueError, 'no stored output matched'):
                _load_many_queries(store, [
                    {'phraser_key': 'missing', 'collar': 100,
                     'model_name': 'wav2vec2',
                     'output_type': 'hidden_state', 'layer': 1},
                ], strict=True)
            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                _load_object_frames(store, phraser_key=_pk('missing'),
                    collar=500, model_name='wav2vec2',
                    output_type='hidden_state', layer=7)
            self.assertEqual(_load_object_frames(store, phraser_key=_pk('missing'),
                collar=None, model_name='wav2vec2',
                output_type='hidden_state', layer=7), {})
            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                list(_iter_object_frames(store, phraser_key=_pk('missing'),
                    collar=500, model_name='wav2vec2',
                    output_type='hidden_state', layer=7))
            self.assertEqual(list(_iter_object_frames(store,
                phraser_key=_pk('missing'), collar=None,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7)), [])

    def test_include_deleted_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            live = _put(store, phraser_key='phrase-live', collar=80,
                model_name='wav2vec2', output_type='hidden_state',
                layer=2, data=[[1.0]], tags=['exp-a', 'shared'])
            deleted = _put(store, phraser_key='phrase-deleted', collar=90,
                model_name='wav2vec2', output_type='hidden_state',
                layer=2, data=[[2.0]], tags=['exp-b', 'shared'])
            _delete(store, phraser_key=_pk('phrase-deleted'), collar=90,
                model_name='wav2vec2', output_type='hidden_state', layer=2)
            self.assertEqual([_hex(item) for item in _find(store,
                'phrase-deleted', include_deleted=True)], [_hex(deleted)])
            self.assertEqual(_find(store, 'phrase-deleted'), [])
            self.assertEqual([_hex(item) for item in store.find_by_tag(
                'exp-b', include_deleted=True)], [_hex(deleted)])
            self.assertEqual(store.find_by_tag('exp-b'), [])
            self.assertEqual(sorted(_hex(item) for item in
                store.find_by_tags(['shared'], include_deleted=True)),
                [_hex(deleted), _hex(live)])
            self.assertEqual([_hex(item) for item in
                store.find_by_tags(['shared'])], [_hex(live)])
            self.assertEqual(store.list_tags(), ['exp-a', 'shared'])

    def test_index_validation_and_overview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            first = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]], tags=['exp-a'])
            second = _put(store, phraser_key='phrase-2', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]], tags=['exp-b'])
            _delete(store, phraser_key=_pk('phrase-2'), collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=1)
            with self.assertRaisesRegex(ValueError, 'match must be one of'):
                _find_one(store, phraser_key='phrase-1', collar=100,
                    model_name='wav2vec2',
                    output_type='hidden_state', layer=1, match='bad')
            with self.assertRaisesRegex(ValueError, "match must be 'all'"):
                store.find_by_tags(['exp-a'], match='bad')
            shard_stats = store.index.get_shard_metadata(first.shard_id)
            self.assertIsNotNone(shard_stats)
            self.assertEqual(shard_stats['live_entry_count'], 1)
            self.assertEqual(shard_stats['deleted_entry_count'], 1)
            live_entries = store.list_entries()
            all_entries = store.list_entries(include_deleted=True)
            overview = store.overview(include_deleted=True,
                health_event_limit=5)
            self.assertEqual([_hex(item) for item in live_entries],
                [_hex(first)])
            self.assertEqual([_hex(item) for item in all_entries], [
                _hex(first), _hex(second)])
            self.assertEqual(overview['entry_count'], 2)
            self.assertEqual(overview['shard_count'], 1)
            self.assertEqual(sorted(overview['tags']), ['exp-a', 'exp-b'])
            self.assertIsNone(overview['integrity'])

    def test_metadata_helpers(self) -> None:
        metadata = EchoframeMetadata(phraser_key='phrase-1', collar=120,
            model_name='wav2vec2', output_type='hidden_state', layer=7,
            shard_id='wav2vec2_hidden_state_0001',
            dataset_path='/layer_0007/entry', shape=[2, 3], dtype='float32',
            tags=[' b ', 'a', 'a'])
        self.assertFalse(metadata.has_echoframe_key)
        with self.assertRaisesRegex(ValueError,
            'metadata does not have an echoframe_key'):
            metadata.echoframe_key
        with self.assertRaisesRegex(ValueError,
            'metadata does not have an echoframe_key'):
            metadata.format_echoframe_key()
        self.assertEqual(metadata.tags, ['a', 'b'])
        self.assertEqual(metadata.shape, (2, 3))
        self.assertIsNotNone(metadata.created_at)
        self.assertIsNone(metadata.deleted_at)
        self.assertEqual(repr(metadata),
            'MD(model=wav2vec2, layer=7, status=live, tags=a,b)')
        with self.assertRaisesRegex(ValueError,
            'metadata is not bound to a store'):
            metadata.load_payload()

        class FakePhraserObject:
            label = 'hello'

            def __repr__(self):
                return "PhraserObject(label='hello')"

        phraser_object = FakePhraserObject()
        fake_models = types.SimpleNamespace(
            cache=types.SimpleNamespace(load=mock.Mock(
                return_value=phraser_object)))
        fake_phraser = types.SimpleNamespace(models=fake_models)
        with mock.patch.dict(sys.modules, {'phraser': fake_phraser}):
            self.assertIs(metadata.phraser_object, phraser_object)
            self.assertEqual(metadata.label, 'hello')
        serialized = metadata.to_dict()
        self.assertNotIn('echoframe_key_hex', serialized)
        restored = EchoframeMetadata.from_dict(serialized)
        self.assertEqual(restored.to_dict(), serialized)
        self.assertFalse(restored.has_echoframe_key)
        updated = metadata.with_tags(['z', 'a'])
        self.assertEqual(updated.tags, ['a', 'z'])
        self.assertEqual(updated.created_at, metadata.created_at)
        self.assertFalse(updated.has_echoframe_key)
        deleted = metadata.mark_deleted()
        self.assertEqual(deleted.storage_status, 'deleted')
        self.assertEqual(deleted.created_at, metadata.created_at)
        self.assertIsNotNone(deleted.deleted_at)
        self.assertFalse(deleted.has_echoframe_key)

    def test_store_metadata_are_bound_and_can_load_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            created = _put(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=7, data=[[1.0, 2.0], [3.0, 4.0]], tags=['exp-a'])
            self.assertIs(created._store, store)
            self.assertEqual(created.load_payload(),
                [[1.0, 2.0], [3.0, 4.0]])
            found = _find_one(store, phraser_key='phrase-1', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=7)
            self.assertIs(found._store, store)
            listed = store.metadata
            self.assertEqual(len(listed), 1)
            self.assertIs(listed[0]._store, store)
