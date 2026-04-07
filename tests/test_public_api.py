'''Public API tests for echoframe.'''

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import echoframe
from echoframe.index import LmdbIndex
from echoframe.metadata import Metadata
from echoframe.output_storage import Hdf5ShardStore, sanitize_name
from echoframe.store import Store


class FakeCursor:
    def __init__(self, store: dict[bytes, bytes]) -> None:
        self.store = store
        self.keys: list[bytes] = []
        self.index = 0

    def set_range(self, prefix: bytes) -> bool:
        self.keys = sorted(key for key in self.store if key >= prefix)
        self.index = 0
        return bool(self.keys)

    def __iter__(self) -> 'FakeCursor':
        return self

    def __next__(self) -> tuple[bytes, bytes]:
        if self.index >= len(self.keys):
            raise StopIteration
        key = self.keys[self.index]
        self.index += 1
        return key, self.store[key]


class FakeTxn:
    def __init__(self, env: 'FakeEnv', write: bool) -> None:
        self.env = env
        self.write = write

    def __enter__(self) -> 'FakeTxn':
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def put(self, key: bytes, value: bytes, db: bytes) -> None:
        self.env.dbs[db][key] = value

    def get(self, key: bytes, db: bytes) -> bytes | None:
        return self.env.dbs[db].get(key)

    def delete(self, key: bytes, db: bytes) -> None:
        self.env.dbs[db].pop(key, None)

    def cursor(self, db: bytes) -> FakeCursor:
        return FakeCursor(self.env.dbs[db])


class FakeEnv:
    def __init__(self) -> None:
        self.dbs: dict[bytes, dict[bytes, bytes]] = {}

    def open_db(self, name: bytes) -> bytes:
        self.dbs.setdefault(name, {})
        return name

    def begin(self, write: bool = False) -> FakeTxn:
        return FakeTxn(self, write=write)


class FakeDataset:
    def __init__(self, data: object) -> None:
        self.data = data
        self.shape = self._shape(data)
        self.dtype = type(self._leaf(data)).__name__

    def __getitem__(self, item: object) -> object:
        if item == ():
            return self.data
        raise KeyError(item)

    def _shape(self, data: object) -> tuple[int, ...]:
        if isinstance(data, list) and data:
            return (len(data),) + self._shape(data[0])
        if isinstance(data, list):
            return (0,)
        return ()

    def _leaf(self, data: object) -> object:
        if isinstance(data, list) and data:
            return self._leaf(data[0])
        return data


class FakeGroup(dict[str, FakeDataset]):
    def create_dataset(self, name: str, data: object) -> FakeDataset:
        dataset = FakeDataset(data)
        self[name] = dataset
        return dataset


class FakeH5File:
    def __init__(self, files: dict[str, dict[str, FakeGroup]],
        path: Path, mode: str) -> None:
        self.files = files
        self.path = str(path)
        if 'r' in mode and not path.exists():
            raise FileNotFoundError(path)
        if 'r' not in mode:
            path.touch(exist_ok=True)
        self.groups = self.files.setdefault(self.path, {})

    def __enter__(self) -> 'FakeH5File':
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def require_group(self, path: str) -> FakeGroup:
        return self.groups.setdefault(path, FakeGroup())

    def __contains__(self, path: str) -> bool:
        group_path, name = path.rsplit('/', 1)
        return name in self.groups.get(group_path, {})

    def __getitem__(self, path: str) -> FakeDataset:
        group_path, name = path.rsplit('/', 1)
        return self.groups[group_path][name]

    def __delitem__(self, path: str) -> None:
        group_path, name = path.rsplit('/', 1)
        del self.groups[group_path][name]


class FakeH5Module:
    def __init__(self) -> None:
        self.files: dict[str, dict[str, FakeGroup]] = {}

    def File(self, path: Path, mode: str) -> FakeH5File:
        return FakeH5File(self.files, path, mode)


class FailingCompactStorage(Hdf5ShardStore):
    def compact_shard_to(self, shard_id, entries, target_shard_id,
        delete_source=True):
        super().compact_shard_to(shard_id, entries,
            target_shard_id=target_shard_id,
            delete_source=delete_source)
        raise RuntimeError('compaction exploded')


class FlakySizeStorage(Hdf5ShardStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_stat_failure = False

    def _size_info_with_retries(self, shard_id, file_path, purpose):
        if self.force_stat_failure and shard_id.endswith('_0001'):
            return {
                'exists': True,
                'byte_size': None,
                'is_estimated': False,
                'error': 'simulated stat failure',
            }
        return super()._size_info_with_retries(shard_id, file_path,
            purpose=purpose)


class EchoFrameTests(unittest.TestCase):
    def _make_fake_store(self, tmpdir: str) -> Store:
        index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
            shards_root=Path(tmpdir) / 'shards')
        storage = Hdf5ShardStore(
            Path(tmpdir) / 'shards',
            h5_module=FakeH5Module(),
        )
        return Store(
            tmpdir,
            index=index,
            storage=storage,
        )

    def test_public_exports(self) -> None:
        self.assertIn('Store', echoframe.__all__)
        self.assertIn('Metadata', echoframe.__all__)
        self.assertIn('STABLE_METADATA_FIELDS', echoframe.__all__)
        self.assertNotIn('LmdbIndex', echoframe.__all__)
        self.assertNotIn('__version__', echoframe.__all__)
        self.assertFalse(hasattr(echoframe, '__version__'))

    def test_put_find_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            metadata = store.put(
                phraser_key='phrase-1',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0], [3.0, 4.0]],
                tags=['exp-a'],
                to_vector_version='abc123',
            )

            self.assertEqual(metadata.phraser_key, 'phrase-1')
            self.assertEqual(metadata.layer, 7)
            self.assertEqual(metadata.shard_id, 'wav2vec2_hidden_state_0001')
            self.assertEqual(metadata.tags, ['exp-a'])
            self.assertTrue(store.exists(
                phraser_key='phrase-1',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            ))
            self.assertEqual(
                store.load(
                    phraser_key='phrase-1',
                    collar=120,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=7,
                ),
                [[1.0, 2.0], [3.0, 4.0]],
            )

    def test_load_many_returns_payloads_in_query_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            store.put(
                phraser_key='phrase-1',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0]],
            )
            store.put(
                phraser_key='phrase-2',
                collar=130,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[2.0]],
            )

            payloads = store.load_many([
                {
                    'phraser_key': 'phrase-2',
                    'collar': 130,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 7,
                },
                {
                    'phraser_key': 'phrase-1',
                    'collar': 120,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 7,
                },
            ])

            self.assertEqual(payloads, [[[2.0]], [[1.0]]])

    def test_load_object_frames_single_and_all_collars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            store.put(
                phraser_key='phrase-1',
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0]],
            )
            store.put(
                phraser_key='phrase-1',
                collar=750,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[3.0, 4.0]],
            )

            exact = store.load_object_frames(
                phraser_key='phrase-1',
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            )
            nearest = store.load_object_frames(
                phraser_key='phrase-1',
                collar=700,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                match='nearest',
            )
            all_collars = store.load_object_frames(
                phraser_key='phrase-1',
                collar=None,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            )

            self.assertEqual(exact, [[1.0, 2.0]])
            self.assertEqual(nearest, [[3.0, 4.0]])
            self.assertEqual(list(all_collars.keys()), [500, 750])
            self.assertEqual(all_collars, {
                500: [[1.0, 2.0]],
                750: [[3.0, 4.0]],
            })

    def test_iter_object_frames_yields_metadata_and_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            first = store.put(
                phraser_key='phrase-1',
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0]],
            )
            second = store.put(
                phraser_key='phrase-1',
                collar=750,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[3.0, 4.0]],
            )

            rows = list(store.iter_object_frames(
                phraser_key='phrase-1',
                model_name='wav2vec2',
                layer=7,
            ))
            nearest = list(store.iter_object_frames(
                phraser_key='phrase-1',
                model_name='wav2vec2',
                layer=7,
                collar=700,
                match='nearest',
            ))

            self.assertEqual(
                [(metadata.collar, payload) for metadata, payload in rows],
                [
                    (500, [[1.0, 2.0]]),
                    (750, [[3.0, 4.0]]),
                ],
            )
            self.assertEqual(rows[0][0].entry_id, first.entry_id)
            self.assertEqual(rows[1][0].entry_id, second.entry_id)
            self.assertEqual(len(nearest), 1)
            self.assertEqual(nearest[0][0].collar, 750)
            self.assertEqual(nearest[0][1], [[3.0, 4.0]])

    def test_collar_matching_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            for collar in (100, 200, 350):
                store.put(
                    phraser_key='word-1',
                    collar=collar,
                    model_name='hubert',
                    output_type='attention',
                    layer=3,
                    data=[[[1, 2], [3, 4]]],
                )

            minimum = store.find_one(
                phraser_key='word-1',
                collar=150,
                model_name='hubert',
                output_type='attention',
                layer=3,
                match='min',
            )
            maximum = store.find_one(
                phraser_key='word-1',
                collar=150,
                model_name='hubert',
                output_type='attention',
                layer=3,
                match='max',
            )
            nearest = store.find_one(
                phraser_key='word-1',
                collar=180,
                model_name='hubert',
                output_type='attention',
                layer=3,
                match='nearest',
            )

            self.assertEqual(minimum.collar, 200)
            self.assertEqual(maximum.collar, 100)
            self.assertEqual(nearest.collar, 200)

            deleted = store.delete(
                phraser_key='word-1',
                collar=200,
                model_name='hubert',
                output_type='attention',
                layer=3,
            )
            self.assertEqual(deleted.storage_status, 'deleted')
            self.assertFalse(store.exists(
                phraser_key='word-1',
                collar=200,
                model_name='hubert',
                output_type='attention',
                layer=3,
            ))

    def test_find_or_compute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            calls: list[str] = []

            def compute() -> list[int]:
                calls.append('compute')
                return [1, 2, 3]

            metadata, created = store.find_or_compute(
                phraser_key='phone-1',
                collar=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=1,
                compute=compute,
            )
            again, created_again = store.find_or_compute(
                phraser_key='phone-1',
                collar=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=1,
                compute=compute,
            )

            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(metadata.entry_id, again.entry_id)
            self.assertEqual(calls, ['compute'])

    def test_find_or_compute_can_add_tags_on_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            def compute() -> list[int]:
                return [1, 2, 3]

            metadata, created = store.find_or_compute(
                phraser_key='phone-2',
                collar=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=1,
                compute=compute,
                tags=['exp-a'],
            )
            again, created_again = store.find_or_compute(
                phraser_key='phone-2',
                collar=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=1,
                compute=compute,
                tags=['exp-b'],
                add_tags_on_hit=True,
            )

            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(metadata.entry_id, again.entry_id)
            self.assertEqual(again.tags, ['exp-a', 'exp-b'])

    def test_tag_queries_and_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            metadata = store.put(
                phraser_key='phrase-2',
                collar=90,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=5,
                data=[[1.0]],
                tags=['exp-a', 'subset-1'],
            )

            entries = store.find_by_tag('exp-a')
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].entry_id, metadata.entry_id)

            updated = store.add_tags(metadata.entry_id, ['review'])
            self.assertEqual(updated.tags, ['exp-a', 'review', 'subset-1'])

            entries = store.find_by_tag('review')
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].entry_id, metadata.entry_id)

            updated = store.remove_tags(metadata.entry_id, ['exp-a'])
            self.assertEqual(updated.tags, ['review', 'subset-1'])
            self.assertEqual(store.find_by_tag('exp-a'), [])

    def test_tag_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            store.put(
                phraser_key='phrase-10',
                collar=90,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=5,
                data=[[1.0]],
                tags=['exp-a', 'subset-1'],
            )
            store.put(
                phraser_key='phrase-11',
                collar=90,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=5,
                data=[[2.0]],
                tags=['exp-a', 'subset-2'],
            )

            self.assertEqual(store.tag_counts(), {
                'exp-a': 2,
                'subset-1': 1,
                'subset-2': 1,
            })

    def test_invalid_tags_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Metadata(
                phraser_key='phrase-1',
                collar=10,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                tags=['bad:tag'],
            )
        with self.assertRaises(ValueError):
            Metadata(
                phraser_key='phrase-1',
                collar=10,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                tags=['   '],
            )
        with self.assertRaises(ValueError):
            Metadata(
                phraser_key='phrase-1',
                collar=10,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                tags=[123],
            )

    def test_compact_shards_removes_deleted_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            one = store.put(
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            two = store.put(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )

            old_shard = one.shard_id
            self.assertEqual(old_shard, two.shard_id)

            store.delete(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            compacted = store.compact_shards()
            self.assertEqual(compacted, [old_shard])

            live = store.find_one(
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            self.assertNotEqual(live.shard_id, old_shard)
            self.assertEqual(
                store.load(
                    phraser_key='phrase-a',
                    collar=100,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=3,
                ),
                [[1.0]],
            )
            self.assertEqual(store.index.entries_for_shard(old_shard), [])

    def test_empty_and_missing_store_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            self.assertEqual(store.find_many([]), [])
            self.assertEqual(store.load_many([]), [])
            self.assertEqual(store.put_many([]), [])
            self.assertEqual(store.add_tags_many([], ['exp-a']), [])
            self.assertEqual(store.remove_tags_many([], ['exp-a']), [])
            self.assertIsNone(store.delete(
                phraser_key='missing',
                collar=10,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
            ))
            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                store.load(
                    phraser_key='missing',
                    collar=10,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=1,
                )

    def test_load_many_returns_none_for_misses_and_can_be_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            store.put(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )

            payloads = store.load_many([
                {
                    'phraser_key': 'phrase-1',
                    'collar': 100,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 1,
                },
                {
                    'phraser_key': 'missing',
                    'collar': 100,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 1,
                },
            ])

            self.assertEqual(payloads, [[[1.0]], None])
            with self.assertRaisesRegex(ValueError, 'no stored output matched'):
                store.load_many([
                    {
                        'phraser_key': 'phrase-1',
                        'collar': 100,
                        'model_name': 'wav2vec2',
                        'output_type': 'hidden_state',
                        'layer': 1,
                    },
                    {
                        'phraser_key': 'missing',
                        'collar': 100,
                        'model_name': 'wav2vec2',
                        'output_type': 'hidden_state',
                        'layer': 1,
                    },
                ], strict=True)

    def test_load_object_frames_missing_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                store.load_object_frames(
                    phraser_key='missing',
                    collar=500,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=7,
                )

            self.assertEqual(store.load_object_frames(
                phraser_key='missing',
                collar=None,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            ), {})
            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                list(store.iter_object_frames(
                    phraser_key='missing',
                    collar=500,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=7,
                ))
            self.assertEqual(list(store.iter_object_frames(
                phraser_key='missing',
                collar=None,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            )), [])

    def test_include_deleted_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            live = store.put(
                phraser_key='phrase-live',
                collar=80,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=2,
                data=[[1.0]],
                tags=['exp-a', 'shared'],
            )
            deleted = store.put(
                phraser_key='phrase-deleted',
                collar=90,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=2,
                data=[[2.0]],
                tags=['exp-b', 'shared'],
            )
            store.delete(
                phraser_key='phrase-deleted',
                collar=90,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=2,
            )

            self.assertEqual(
                [item.entry_id for item in store.find(
                    'phrase-deleted',
                    include_deleted=True,
                )],
                [deleted.entry_id],
            )
            self.assertEqual(store.find('phrase-deleted'), [])
            self.assertEqual(
                [item.entry_id for item in store.find_by_tag(
                    'exp-b',
                    include_deleted=True,
                )],
                [deleted.entry_id],
            )
            self.assertEqual(store.find_by_tag('exp-b'), [])
            self.assertEqual(sorted(item.entry_id for item in
                store.find_by_tags(['shared'], include_deleted=True)), [
                deleted.entry_id,
                live.entry_id,
            ])
            self.assertEqual([item.entry_id for item in
                store.find_by_tags(['shared'])], [live.entry_id])
            self.assertEqual(store.list_tags(), ['exp-a', 'shared'])
            self.assertEqual(sorted(store.list_tags(include_deleted=True)), [
                'exp-a',
                'exp-b',
                'shared',
            ])

    def test_index_validation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            store.put(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
                tags=['exp-a'],
            )

            with self.assertRaisesRegex(ValueError, 'match must be one of'):
                store.find_one(
                    phraser_key='phrase-1',
                    collar=100,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=1,
                    match='bad',
                )
            with self.assertRaisesRegex(ValueError, 'match must be one of'):
                store.find_many([
                    {
                        'phraser_key': 'phrase-1',
                        'collar': 100,
                        'model_name': 'wav2vec2',
                        'output_type': 'hidden_state',
                        'layer': 1,
                        'match': 'bad',
                    },
                ])
            with self.assertRaisesRegex(ValueError, "match must be 'all'"):
                store.find_by_tags(['exp-a'], match='bad')
            with self.assertRaisesRegex(ValueError, "mode must be 'add'"):
                store.index._update_tags_many(['missing-entry'],
                    tags=['exp-a'],
                    mode='bad')

    def test_get_shard_metadata_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            metadata = store.put(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )

            shard_stats = store.index.get_shard_metadata(metadata.shard_id)
            self.assertIsNotNone(shard_stats)
            self.assertEqual(shard_stats['live_entry_count'], 1)
            self.assertEqual(shard_stats['deleted_entry_count'], 0)
            self.assertGreaterEqual(shard_stats['byte_size'], 0)
            self.assertIsNone(store.index.get_shard_metadata('missing_0001'))

    def test_public_overview_and_entry_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            first = store.put(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
                tags=['exp-a'],
            )
            second = store.put(
                phraser_key='phrase-2',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[2.0]],
                tags=['exp-b'],
            )
            store.delete(
                phraser_key='phrase-2',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
            )

            live_entries = store.list_entries()
            all_entries = store.list_entries(include_deleted=True)
            overview = store.overview(include_deleted=True,
                health_event_limit=5)

            self.assertEqual([item.entry_id for item in live_entries],
                [first.entry_id])
            self.assertEqual([item.entry_id for item in all_entries], [
                first.entry_id,
                second.entry_id,
            ])
            self.assertEqual(overview['entry_count'], 2)
            self.assertEqual(overview['shard_count'], 1)
            self.assertEqual([item['entry_id'] for item in
                overview['entries']], [first.entry_id, second.entry_id])
            self.assertEqual(sorted(overview['tags']), ['exp-a', 'exp-b'])
            self.assertIsNone(overview['integrity'])
            self.assertLessEqual(len(
                overview['recent_shard_health_events']), 5)

            overview_with_integrity = store.overview(include_deleted=True,
                include_integrity=True)
            self.assertIn('ok', overview_with_integrity['integrity'])

    def test_metadata_helpers(self) -> None:
        metadata = Metadata(
            phraser_key='phrase-1',
            collar=120,
            model_name='wav2vec2',
            output_type='hidden_state',
            layer=7,
            shard_id='wav2vec2_hidden_state_0001',
            dataset_path='/layer_0007/entry',
            shape=[2, 3],
            dtype='float32',
            tags=[' b ', 'a', 'a'],
            to_vector_version='abc123',
        )

        self.assertEqual(metadata.identity_key,
            'phrase-1:wav2vec2:hidden_state:0007:000000120')
        self.assertEqual(metadata.object_key,
            'obj:phrase-1:wav2vec2:hidden_state:0007:000000120')
        self.assertEqual(metadata.tags, ['a', 'b'])
        self.assertEqual(metadata.shape, (2, 3))
        self.assertIsNotNone(metadata.created_at)
        self.assertIsNone(metadata.deleted_at)

        restored = Metadata.from_dict(metadata.to_dict())
        self.assertEqual(restored.to_dict(), metadata.to_dict())

        updated = metadata.with_tags(['z', 'a'])
        self.assertEqual(updated.tags, ['a', 'z'])
        self.assertEqual(updated.created_at, metadata.created_at)
        self.assertEqual(updated.deleted_at, metadata.deleted_at)

        deleted = metadata.mark_deleted()
        self.assertEqual(deleted.storage_status, 'deleted')
        self.assertEqual(deleted.created_at, metadata.created_at)
        self.assertIsNotNone(deleted.deleted_at)

    def test_output_storage_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(
                Path(tmpdir),
                h5_module=FakeH5Module(),
            )
            metadata = Metadata(
                phraser_key='phrase-1',
                collar=100,
                model_name='model name',
                output_type='hidden_state',
                layer=2,
                created_at='2024-01-01T00:00:00+00:00',
            )

            stored = storage.store_with_shard(metadata, data=[[1, 2]],
                shard_id='manual_0001')
            self.assertEqual(stored.shard_id, 'manual_0001')
            self.assertEqual(stored.dataset_path,
                f'/layer_0002/{metadata.entry_id}')
            self.assertEqual(storage.shard_size('manual_0001'),
                (Path(tmpdir) / 'manual_0001.h5').stat().st_size)

            replaced = storage.compact_shard('manual_0001', [stored])
            self.assertEqual(len(replaced), 1)
            self.assertEqual(replaced[0].shard_id, 'manual_0002')
            self.assertFalse((Path(tmpdir) / 'manual_0001.h5').exists())

            copied = storage.compact_shard_to('manual_0002', replaced,
                target_shard_id='manual_0003', delete_source=False)
            self.assertEqual([item.shard_id for item in copied],
                ['manual_0003'])
            self.assertTrue((Path(tmpdir) / 'manual_0002.h5').exists())
            self.assertTrue((Path(tmpdir) / 'manual_0003.h5').exists())

            with self.assertRaisesRegex(ValueError, 'invalid shard_id'):
                storage._replacement_shard_id('bad-shard')

            self.assertEqual(sanitize_name(' model/name :: v1 '),
                'model_name_v1')
            self.assertEqual(sanitize_name('***'), 'unknown')

    def test_store_does_not_create_diagnostic_log_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_root = Path(tmpdir) / 'tests' / 'data' / 'store-root'
            store = self._make_fake_store(str(store_root))
            store.put(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )

            log_root = Path(tmpdir) / 'tests' / 'data' / 'echoframe_logs'
            self.assertFalse(log_root.exists())

    def test_stat_retries_include_long_backoff_delays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(
                Path(tmpdir),
                h5_module=FakeH5Module(),
            )
            file_path = Path(tmpdir) / 'manual_0001.h5'
            file_path.touch()
            with mock.patch.object(Path, 'stat', autospec=True,
                side_effect=OSError(5, 'io error')):
                with mock.patch('echoframe.output_storage.time.sleep'
                    ) as mocked_sleep:
                    info = storage._size_info_with_retries('manual_0001',
                        file_path, purpose='rollover')

            self.assertIsNone(info['byte_size'])
            self.assertEqual([call.args[0] for call in
                mocked_sleep.call_args_list], [0.1, 0.3, 0.6, 0.9, 3.0])

    def test_unreadable_active_shard_rotates_to_next_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
                shards_root=Path(tmpdir) / 'shards')
            storage = FlakySizeStorage(
                Path(tmpdir) / 'shards',
                h5_module=FakeH5Module(),
            )
            store = Store(tmpdir, index=index, storage=storage)

            first = store.put(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )
            storage.force_stat_failure = True

            second = store.put(
                phraser_key='phrase-2',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[2.0]],
            )

            self.assertEqual(second.shard_id, 'wav2vec2_hidden_state_0002')
            events = store.get_shard_health_events()
            self.assertTrue(any(event['event_type'] == 'skip_unreadable_shard'
                for event in events))

    def test_active_shard_id_rotates_past_failing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(
                Path(tmpdir),
                h5_module=FakeH5Module(),
            )

            def fake_state(shard_id, file_path, purpose):
                if shard_id.endswith('_0001'):
                    return {
                        'state': 'unreadable',
                        'byte_size': 0,
                        'error': 'simulated shard failure',
                    }
                return {
                    'state': 'missing',
                    'byte_size': None,
                    'error': None,
                }

            with mock.patch.object(storage, '_shard_path_state',
                side_effect=fake_state):
                shard_id = storage._active_shard_id('wav2vec2',
                    'hidden_state')

            self.assertEqual(shard_id, 'wav2vec2_hidden_state_0002')

    def test_active_shard_id_hard_fails_when_all_probes_are_unreadable(
        self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(
                Path(tmpdir),
                h5_module=FakeH5Module(),
            )
            storage.MAX_SCAN_SUFFIXES = 5

            def always_unreadable(shard_id, file_path, purpose):
                return {
                    'state': 'unreadable',
                    'byte_size': None,
                    'error': 'simulated shard failure',
                }

            with mock.patch.object(storage, '_shard_path_state',
                side_effect=always_unreadable):
                with mock.patch('echoframe.output_storage.time.sleep'
                    ) as mocked_sleep:
                    with self.assertRaisesRegex(RuntimeError,
                        'unable to probe shard path'):
                        storage._active_shard_id('wav2vec2',
                            'hidden_state')

            self.assertEqual([call.args[0] for call in
                mocked_sleep.call_args_list], [0.1, 0.3, 0.6, 0.9, 3.0])

    def test_replacement_shard_id_skips_unreadable_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(
                Path(tmpdir),
                h5_module=FakeH5Module(),
            )
            candidate_path = Path(tmpdir) / 'manual_0002.h5'
            candidate_path.touch()
            original_stat = Path.stat

            def flaky_stat(path_obj, *args, **kwargs):
                if str(path_obj).endswith('manual_0002.h5'):
                    raise OSError(5, 'io error')
                return original_stat(path_obj, *args, **kwargs)

            with mock.patch.object(Path, 'stat', autospec=True,
                side_effect=flaky_stat):
                with mock.patch('echoframe.output_storage.time.sleep'):
                    replacement = storage._replacement_shard_id(
                        'manual_0001')

            self.assertEqual(replacement, 'manual_0003')
            events = storage.get_shard_health_events()
            self.assertTrue(any(event['event_type'] ==
                'stat_failure' for event in events))
            self.assertTrue(any(event['event_type'] ==
                'replacement_skip_unreadable_candidates'
                for event in events))

    def test_replacement_shard_id_hard_fails_when_all_probes_are_unreadable(
        self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(
                Path(tmpdir),
                h5_module=FakeH5Module(),
            )
            storage.MAX_SCAN_SUFFIXES = 5

            def always_unreadable(shard_id, file_path, purpose):
                return {
                    'state': 'unreadable',
                    'byte_size': None,
                    'error': 'simulated shard failure',
                }

            with mock.patch.object(storage, '_shard_path_state',
                side_effect=always_unreadable):
                with mock.patch('echoframe.output_storage.time.sleep'
                    ) as mocked_sleep:
                    with self.assertRaisesRegex(RuntimeError,
                        'unable to probe shard path'):
                        storage._replacement_shard_id('manual_0001')

            self.assertEqual([call.args[0] for call in
                mocked_sleep.call_args_list], [0.1, 0.3, 0.6, 0.9, 3.0])

    def test_replacement_shard_id_handles_more_than_100_existing_shards(
        self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(
                Path(tmpdir),
                h5_module=FakeH5Module(),
            )
            for index in range(2, 103):
                (Path(tmpdir) / f'manual_{index:04d}.h5').touch()

            replacement = storage._replacement_shard_id('manual_0001')

            self.assertEqual(replacement, 'manual_0103')

    def test_replacement_shard_id_skips_unreadable_candidates_within_budget(
        self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(
                Path(tmpdir),
                h5_module=FakeH5Module(),
            )
            original_stat = Path.stat

            def flaky_stat(path_obj, *args, **kwargs):
                path_str = str(path_obj)
                if 'manual_' in path_str:
                    shard_name = Path(path_str).stem
                    if shard_name >= 'manual_0002' and shard_name <= (
                        'manual_0006'):
                        raise OSError(5, 'io error')
                return original_stat(path_obj, *args, **kwargs)

            with mock.patch.object(Path, 'stat', autospec=True,
                side_effect=flaky_stat):
                with mock.patch('echoframe.output_storage.time.sleep'):
                    replacement = storage._replacement_shard_id(
                        'manual_0001')

            self.assertEqual(replacement, 'manual_0007')

    def test_index_uses_zero_when_first_size_probe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
                shards_root=Path(tmpdir) / 'shards')
            shards_root = Path(tmpdir) / 'shards'
            shards_root.mkdir(parents=True, exist_ok=True)
            file_path = shards_root / 'manual_0001.h5'
            file_path.touch()
            original_stat = Path.stat

            def flaky_stat(path_obj, *args, **kwargs):
                if str(path_obj).endswith('manual_0001.h5'):
                    raise OSError(5, 'io error')
                return original_stat(path_obj, *args, **kwargs)

            with index.env.begin(write=True) as txn:
                with mock.patch.object(Path, 'stat', autospec=True,
                    side_effect=flaky_stat):
                    size_info = index._shard_file_size(txn, 'manual_0001')

            self.assertEqual(size_info['byte_size'], 0)
            self.assertTrue(size_info['byte_size_is_estimated'])
            self.assertIn('io error', size_info['byte_size_error'])

    def test_index_falls_back_to_last_known_shard_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            metadata = store.put(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
                tags=['exp-a'],
            )
            shard_stats = store.index.get_shard_metadata(metadata.shard_id)
            original_size = shard_stats['byte_size']
            original_stat = Path.stat

            def flaky_stat(path_obj, *args, **kwargs):
                if str(path_obj).endswith(f'{metadata.shard_id}.h5'):
                    raise OSError(5, 'io error')
                return original_stat(path_obj, *args, **kwargs)

            with mock.patch.object(Path, 'stat', autospec=True,
                side_effect=flaky_stat):
                updated = store.add_tags(metadata.entry_id, ['exp-b'])

            self.assertEqual(updated.tags, ['exp-a', 'exp-b'])
            shard_stats = store.index.get_shard_metadata(metadata.shard_id)
            self.assertEqual(shard_stats['byte_size'], original_size)
            self.assertTrue(shard_stats['byte_size_is_estimated'])
            self.assertIn('io error', shard_stats['byte_size_error'])

    def test_compaction_no_op_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            live = store.put(
                phraser_key='phrase-live',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )
            deleted = store.put(
                phraser_key='phrase-deleted',
                collar=100,
                model_name='hubert',
                output_type='attention',
                layer=2,
                data=[[[1, 2]]],
            )
            store.delete(
                phraser_key='phrase-deleted',
                collar=100,
                model_name='hubert',
                output_type='attention',
                layer=2,
            )

            self.assertEqual(store.compact_shards(
                shard_ids=[live.shard_id]), [])
            dry_run = store.compact_shards(shard_ids=[deleted.shard_id],
                dry_run=True)
            self.assertEqual([plan['shard_id'] for plan in dry_run],
                [deleted.shard_id])

            compacted = store.compact_shards(shard_ids=[deleted.shard_id])
            self.assertEqual(compacted, [deleted.shard_id])
            self.assertEqual(store.index.entries_for_shard(
                deleted.shard_id, include_deleted=True), [])
            self.assertFalse((Path(tmpdir) / 'shards' /
                f'{deleted.shard_id}.h5').exists())

    def test_list_entries_include_deleted_keeps_tombstones_after_compaction(
        self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            live = store.put(
                phraser_key='phrase-live',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )
            deleted = store.put(
                phraser_key='phrase-deleted',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[2.0]],
            )
            store.delete(
                phraser_key='phrase-deleted',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
            )

            store.compact_shards()

            live_entries = store.list_entries()
            all_entries = store.list_entries(include_deleted=True)
            overview = store.overview(include_deleted=True)

            self.assertEqual([item.entry_id for item in live_entries],
                [live.entry_id])
            self.assertEqual([item.entry_id for item in all_entries], [
                deleted.entry_id,
                live.entry_id,
            ])
            self.assertEqual([item['entry_id'] for item in
                overview['entries']], [deleted.entry_id, live.entry_id])

    def test_list_entries_ignores_stale_shard_index_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            metadata = store.put(
                phraser_key='phrase-live',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )

            duplicated = Metadata(
                phraser_key=metadata.phraser_key,
                collar=metadata.collar,
                model_name=metadata.model_name,
                output_type=metadata.output_type,
                layer=metadata.layer,
                storage_status=metadata.storage_status,
                shard_id='wav2vec2_hidden_state_0002',
                dataset_path=metadata.dataset_path,
                shape=metadata.shape,
                dtype=metadata.dtype,
                tags=metadata.tags,
                created_at=metadata.created_at,
                deleted_at=metadata.deleted_at,
                to_vector_version=metadata.to_vector_version,
            )
            store.index.upsert(duplicated)
            with store.index.env.begin(write=True) as txn:
                txn.put(store.index._shard_key(metadata.shard_id,
                    metadata.entry_id),
                    metadata.entry_id.encode('utf-8'),
                    db=store.index.by_shard_db)

            entries = store.list_entries()

            self.assertEqual([item.entry_id for item in entries],
                [metadata.entry_id])

    def test_compaction_marks_failed_journal_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
                shards_root=Path(tmpdir) / 'shards')
            storage = FailingCompactStorage(
                Path(tmpdir) / 'shards',
                h5_module=FakeH5Module(),
            )
            store = Store(tmpdir, index=index, storage=storage)

            one = store.put(
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            store.put(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )
            store.delete(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )

            with self.assertRaisesRegex(RuntimeError, 'compaction exploded'):
                store.compact_shards()

            records = store.compaction_journal(status='failed')
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['shard_id'], one.shard_id)
            self.assertEqual(records[0]['status'], 'failed')
            self.assertEqual(records[0]['error'], 'compaction exploded')
            self.assertIsNotNone(records[0]['finished_at'])

    def test_shard_health_report_excludes_deleted_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            live = store.put(
                phraser_key='phrase-live',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )
            store.put(
                phraser_key='phrase-deleted',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[2.0]],
            )
            store.delete(
                phraser_key='phrase-deleted',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
            )

            report = store._build_shard_health_report(
                live.shard_id, 'simulated shard failure')

            self.assertEqual(report['checked_entries'], 1)
            self.assertEqual(report['lost_items'], [])

    def test_resume_pending_runs_before_new_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            one = store.put(
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            store.put(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )
            store.delete(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            plan = store.compact_shards(dry_run=True)[0]
            journal = store.index.create_compaction_journal(
                plan['shard_id'],
                source_entry_ids=plan['source_entry_ids'],
                live_entry_ids=plan['live_entry_ids'],
                target_shard_id=plan['target_shard_id'],
            )

            compacted = store.compact_shards(
                shard_ids=[one.shard_id],
                resume_pending=True,
            )
            self.assertEqual(compacted, [])
            records = store.compaction_journal()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['journal_id'], journal['journal_id'])
            self.assertEqual(records[0]['status'], 'completed')

    def test_resume_compaction_removes_stale_source_shard_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            one = store.put(
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            store.put(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )
            store.delete(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            plan = store.compact_shards(dry_run=True)[0]
            journal = store.index.create_compaction_journal(
                plan['shard_id'],
                source_entry_ids=plan['source_entry_ids'],
                live_entry_ids=plan['live_entry_ids'],
                target_shard_id=plan['target_shard_id'],
            )

            live_entries = [store.index.get(entry_id)
                for entry_id in journal['live_entry_ids']]
            updated = store.storage.compact_shard_to(
                journal['shard_id'],
                [entry for entry in live_entries if entry is not None],
                target_shard_id=journal['target_shard_id'],
                delete_source=False,
            )
            store.index.upsert_many(updated)

            self.assertIn(one.shard_id, store.index.list_shards())
            self.assertEqual(store.resume_compactions(), [one.shard_id])
            self.assertNotIn(one.shard_id, store.index.list_shards())
            self.assertEqual(store.index.entries_for_shard(one.shard_id,
                include_deleted=True), [])
            self.assertEqual(store.compaction_journal()[0]['status'],
                'completed')

    def test_compaction_journal_ids_are_unique_with_same_second(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            with mock.patch('echoframe.index.utc_now',
                return_value='2026-04-01T12:00:00+00:00'):
                first = store.index.create_compaction_journal(
                    'wav2vec2_hidden_state_0001',
                    source_entry_ids=['a'],
                    live_entry_ids=['a'],
                    target_shard_id='wav2vec2_hidden_state_0002',
                )
                second = store.index.create_compaction_journal(
                    'wav2vec2_hidden_state_0001',
                    source_entry_ids=['a'],
                    live_entry_ids=['a'],
                    target_shard_id='wav2vec2_hidden_state_0002',
                )

            self.assertNotEqual(first['journal_id'], second['journal_id'])
            self.assertEqual(len(store.compaction_journal()), 2)

    def test_missing_optional_dependencies_raise_helpful_import_errors(self
        ) -> None:
        with mock.patch.dict(sys.modules, {'lmdb': None}):
            with self.assertRaisesRegex(ImportError,
                'lmdb is required to use Store'):
                LmdbIndex(Path('tests/data/unused-index'))

        with mock.patch.dict(sys.modules, {'h5py': None}):
            with self.assertRaisesRegex(ImportError,
                'h5py is required to use Store'):
                Hdf5ShardStore(Path('tests/data/unused-shards'))


@unittest.skipUnless(importlib.util.find_spec('lmdb'),
    'lmdb is not installed')
@unittest.skipUnless(importlib.util.find_spec('h5py'),
    'h5py is not installed')
class EchoFrameIntegrationTests(unittest.TestCase):
    def _make_store(self) -> tuple[tempfile.TemporaryDirectory[str], Store]:
        tmpdir = tempfile.TemporaryDirectory()
        store = Store(tmpdir.name, max_shard_size_bytes=1024 * 1024)
        return tmpdir, store

    def _payload_to_list(self, payload):
        return payload.tolist() if hasattr(payload, 'tolist') else payload

    def test_real_put_delete_compact_and_tag_flow(self) -> None:
        tmpdir, store = self._make_store()
        with tmpdir:
            records = store.put_many([
                {
                    'phraser_key': 'phrase-1',
                    'collar': 100,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 3,
                    'data': [[1.0]],
                    'tags': ['exp-a', 'speaker-1'],
                },
                {
                    'phraser_key': 'phrase-2',
                    'collar': 100,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 3,
                    'data': [[2.0]],
                    'tags': ['exp-a', 'speaker-2'],
                },
            ])

            self.assertEqual(len(records), 2)
            loaded = store.load(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            self.assertEqual(self._payload_to_list(loaded), [[1.0]])

            updated = store.add_tags_many([record.entry_id for record in records],
                ['batch'])
            self.assertEqual(len(updated), 2)
            self.assertEqual(sorted(store.list_tags()), [
                'batch', 'exp-a', 'speaker-1', 'speaker-2'])
            self.assertEqual(len(store.find_by_tags(['exp-a', 'batch'])), 2)
            self.assertEqual(len(store.find_by_tags(['speaker-1', 'speaker-2'],
                match='any')), 2)

            deleted = store.delete(
                phraser_key='phrase-2',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            self.assertEqual(deleted.storage_status, 'deleted')

            shard_id = records[0].shard_id
            dry_run = store.compact_shards(dry_run=True)
            self.assertEqual(len(dry_run), 1)
            self.assertEqual(dry_run[0]['shard_id'], shard_id)
            self.assertEqual(dry_run[0]['deleted_entry_count'], 1)

            compacted = store.compact_shards()
            self.assertEqual(compacted, [shard_id])
            live = store.find_one(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            self.assertNotEqual(live.shard_id, shard_id)
            self.assertEqual(self._payload_to_list(store.load(
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )), [[1.0]])

    def test_real_integrity_checks_and_shard_stats(self) -> None:
        tmpdir, store = self._make_store()
        with tmpdir:
            metadata = store.put(
                phraser_key='phrase-3',
                collar=120,
                model_name='hubert',
                output_type='attention',
                layer=2,
                data=[[[1, 2], [3, 4]]],
                tags=['exp-b'],
            )
            store.put(
                phraser_key='phrase-4',
                collar=120,
                model_name='hubert',
                output_type='attention',
                layer=2,
                data=[[[5, 6], [7, 8]]],
                tags=['exp-b', 'subset-1'],
            )
            store.delete(
                phraser_key='phrase-4',
                collar=120,
                model_name='hubert',
                output_type='attention',
                layer=2,
            )

            stats = store.shard_stats()
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]['live_entry_count'], 1)
            self.assertEqual(stats[0]['deleted_entry_count'], 1)
            self.assertGreater(stats[0]['byte_size'], 0)

            store.storage.delete(metadata)
            report = store.verify_integrity()
            self.assertFalse(report['ok'])
            self.assertEqual(report['checked_entries'], 1)
            self.assertEqual(len(report['broken_references']), 1)
            self.assertEqual(report['broken_references'][0]['entry_id'],
                metadata.entry_id)

    def test_real_find_many_and_tag_queries(self) -> None:
        tmpdir, store = self._make_store()
        with tmpdir:
            store.put_many([
                {
                    'phraser_key': 'phrase-10',
                    'collar': 80,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 1,
                    'data': [1, 2, 3],
                    'tags': ['exp-a', 'run-1'],
                },
                {
                    'phraser_key': 'phrase-11',
                    'collar': 90,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 1,
                    'data': [4, 5, 6],
                    'tags': ['exp-a', 'run-2'],
                },
                {
                    'phraser_key': 'phrase-12',
                    'collar': 90,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 1,
                    'data': [7, 8, 9],
                    'tags': ['exp-b', 'run-2'],
                },
            ])

            results = store.find_many([
                {
                    'phraser_key': 'phrase-10',
                    'collar': 80,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 1,
                },
                {
                    'phraser_key': 'phrase-11',
                    'collar': 95,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 1,
                    'match': 'max',
                },
            ])

            self.assertEqual([result.phraser_key for result in results], [
                'phrase-10',
                'phrase-11',
            ])
            self.assertEqual(sorted(store.list_tags()), [
                'exp-a', 'exp-b', 'run-1', 'run-2'])
            all_match = store.find_by_tags(['exp-a', 'run-2'], match='all')
            any_match = store.find_by_tags(['exp-b', 'run-1'], match='any')
            self.assertEqual([item.phraser_key for item in all_match],
                ['phrase-11'])
            self.assertEqual(sorted(item.phraser_key for item in any_match), [
                'phrase-10',
                'phrase-12',
            ])

    def test_real_retrieval_helpers(self) -> None:
        tmpdir, store = self._make_store()
        with tmpdir:
            store.put(
                phraser_key='phrase-20',
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0]],
            )
            store.put(
                phraser_key='phrase-20',
                collar=750,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[3.0, 4.0]],
            )
            store.put(
                phraser_key='phrase-21',
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[5.0, 6.0]],
            )

            payloads = store.load_many([
                {
                    'phraser_key': 'phrase-21',
                    'collar': 500,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 7,
                },
                {
                    'phraser_key': 'missing',
                    'collar': 500,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 7,
                },
                {
                    'phraser_key': 'phrase-20',
                    'collar': 500,
                    'model_name': 'wav2vec2',
                    'output_type': 'hidden_state',
                    'layer': 7,
                },
            ])
            exact = store.load_object_frames(
                phraser_key='phrase-20',
                model_name='wav2vec2',
                layer=7,
                collar=500,
            )
            nearest = store.load_object_frames(
                phraser_key='phrase-20',
                model_name='wav2vec2',
                layer=7,
                collar=700,
                match='nearest',
            )
            all_collars = store.load_object_frames(
                phraser_key='phrase-20',
                model_name='wav2vec2',
                layer=7,
                collar=None,
            )
            rows = list(store.iter_object_frames(
                phraser_key='phrase-20',
                model_name='wav2vec2',
                layer=7,
            ))

            self.assertEqual([self._payload_to_list(item)
                if item is not None else None for item in payloads], [
                [[5.0, 6.0]],
                None,
                [[1.0, 2.0]],
            ])
            self.assertEqual(self._payload_to_list(exact), [[1.0, 2.0]])
            self.assertEqual(self._payload_to_list(nearest), [[3.0, 4.0]])
            self.assertEqual(list(all_collars.keys()), [500, 750])
            self.assertEqual({
                collar: self._payload_to_list(payload)
                for collar, payload in all_collars.items()
            }, {
                500: [[1.0, 2.0]],
                750: [[3.0, 4.0]],
            })
            self.assertEqual(
                [(metadata.collar, self._payload_to_list(payload))
                    for metadata, payload in rows],
                [
                    (500, [[1.0, 2.0]]),
                    (750, [[3.0, 4.0]]),
                ],
            )

            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                store.load_many([
                    {
                        'phraser_key': 'missing',
                        'collar': 500,
                        'model_name': 'wav2vec2',
                        'output_type': 'hidden_state',
                        'layer': 7,
                    },
                ], strict=True)

    def test_resume_compaction_from_journal(self) -> None:
        tmpdir, store = self._make_store()
        with tmpdir:
            one = store.put(
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            store.put(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )
            store.delete(
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            plan = store.compact_shards(dry_run=True)[0]
            journal = store.index.create_compaction_journal(
                plan['shard_id'],
                source_entry_ids=plan['source_entry_ids'],
                live_entry_ids=plan['live_entry_ids'],
                target_shard_id=plan['target_shard_id'],
            )

            resumed = store.resume_compactions()
            self.assertEqual(resumed, [one.shard_id])
            records = store.compaction_journal()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['journal_id'], journal['journal_id'])
            self.assertEqual(records[0]['status'], 'completed')
