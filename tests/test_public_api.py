'''Public API tests for echoframe.'''

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import echoframe
from echoframe.index import LmdbIndex
from echoframe.metadata import (
    EchoframeMetadata,
    filter_metadata,
    metadata_class_for_output_type,
)
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


def _pk(value) -> bytes:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = str(value).encode('utf-8')
    if len(raw) > 22:
        raise ValueError('test phraser keys must be <= 22 bytes')
    return raw.ljust(22, b'\0')


def _hex(metadata) -> str:
    return metadata.format_echoframe_key()


def _ensure_model(store: Store, model_name: str) -> None:
    if store.get_model_metadata(model_name) is None:
        store.register_model(model_name)


def _make_key(store: Store, *, phraser_key, collar, model_name, output_type,
    layer):
    _ensure_model(store, model_name)
    phraser_key = _pk(phraser_key)
    kwargs = {
        'output_type': output_type,
        'model_name': model_name,
    }
    if output_type in {'hidden_state', 'attention'}:
        kwargs.update({
            'phraser_key': phraser_key,
            'layer': layer,
            'collar': collar,
        })
    elif output_type == 'codebook_indices':
        kwargs.update({
            'phraser_key': phraser_key,
            'collar': collar,
        })
    elif output_type == 'codebook_matrix':
        pass
    else:
        raise ValueError(f'unsupported output_type in test helper: {output_type}')
    return store.make_echoframe_key(**kwargs)


def _put(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, data, tags=None):
    phraser_key = _pk(phraser_key)
    echoframe_key = _make_key(store, phraser_key=phraser_key,
        collar=collar, model_name=model_name, output_type=output_type,
        layer=layer)
    metadata_cls = metadata_class_for_output_type(output_type)
    metadata = metadata_cls(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=layer, tags=tags,
        echoframe_key=echoframe_key)
    return store.put(echoframe_key, metadata, data)


def _put_item(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, data, tags=None):
    phraser_key = _pk(phraser_key)
    echoframe_key = _make_key(store, phraser_key=phraser_key,
        collar=collar, model_name=model_name, output_type=output_type,
        layer=layer)
    metadata_cls = metadata_class_for_output_type(output_type)
    metadata = metadata_cls(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=layer, tags=tags,
        echoframe_key=echoframe_key)
    return {'echoframe_key': echoframe_key, 'metadata': metadata, 'data': data}


def _find(store: Store, phraser_key, include_deleted=False, **filters):
    records = store.find_phraser(_pk(phraser_key),
        include_deleted=include_deleted)
    return filter_metadata(records, **filters)


def _find_one(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, match='exact'):
    matches = _find(store, phraser_key, model_name=model_name,
        output_type=output_type, layer=layer, collar=collar, match=match)
    if not matches:
        return None
    return matches[0]


def _exists(store: Store, phraser_key, collar, model_name, output_type, layer,
    match='exact'):
    return _find_one(store, phraser_key=phraser_key, collar=collar,
        model_name=model_name, output_type=output_type, layer=layer,
        match=match) is not None


def _load_query(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, match='exact'):
    metadata = _find_one(store, phraser_key=phraser_key, collar=collar,
        model_name=model_name, output_type=output_type, layer=layer,
        match=match)
    if metadata is None:
        raise ValueError('no stored output matched the requested criteria')
    return store.load(metadata.echoframe_key)


def _find_many(store: Store, queries):
    return [_find_one(store, **query) for query in queries]


def _load_many_queries(store: Store, queries, strict=False):
    payloads = []
    for query in queries:
        metadata = _find_one(store, **query)
        if metadata is None:
            if strict:
                raise ValueError(
                    'no stored output matched one of the requested queries')
            payloads.append(None)
        else:
            payloads.append(store.load(metadata.echoframe_key))
    return payloads


def _put_many(store: Store, items):
    prepared = []
    for item in items:
        if 'metadata' in item:
            prepared.append(item)
            continue
        prepared.append(_put_item(store, **item))
    return store.put_many(prepared)


def _delete(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, match='exact'):
    return store.delete(phraser_key=_pk(phraser_key), collar=collar,
        model_name=model_name, output_type=output_type, layer=layer,
        match=match)


def _load_object_frames(store: Store, *, phraser_key, model_name, layer,
    collar=500, output_type='hidden_state', match='exact'):
    return store.load_object_frames(phraser_key=_pk(phraser_key),
        model_name=model_name, layer=layer, collar=collar,
        output_type=output_type, match=match)


def _iter_object_frames(store: Store, *, phraser_key, model_name, layer,
    collar=None, output_type='hidden_state', match='exact'):
    return store.iter_object_frames(phraser_key=_pk(phraser_key),
        model_name=model_name, layer=layer, collar=collar,
        output_type=output_type, match=match)


def _find_or_compute(store: Store, *, phraser_key, collar, model_name,
    output_type, layer, compute, match='exact', tags=None,
    add_tags_on_hit=False):
    _ensure_model(store, model_name)
    return store.find_or_compute(phraser_key=_pk(phraser_key), collar=collar,
        model_name=model_name, output_type=output_type, layer=layer,
        compute=compute, match=match, tags=tags,
        add_tags_on_hit=add_tags_on_hit)


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
        self.assertIn('EchoframeMetadata', echoframe.__all__)
        self.assertIn('Codebook', echoframe.__all__)
        self.assertIn('STABLE_METADATA_FIELDS', echoframe.__all__)
        self.assertNotIn('LmdbIndex', echoframe.__all__)
        self.assertNotIn('__version__', echoframe.__all__)
        self.assertFalse(hasattr(echoframe, '__version__'))

    def test_put_find_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            metadata = _put(store, 
                phraser_key='phrase-1',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0], [3.0, 4.0]],
                tags=['exp-a'],
            )

            self.assertEqual(metadata.phraser_key, _pk('phrase-1'))
            self.assertEqual(metadata.layer, 7)
            self.assertEqual(metadata.shard_id, 'wav2vec2_hidden_state_0001')
            self.assertEqual(metadata.tags, ['exp-a'])
            self.assertTrue(_exists(store, 
                phraser_key='phrase-1',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            ))
            self.assertEqual(
                _load_query(store, 
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
            _put(store, 
                phraser_key='phrase-1',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0]],
            )
            _put(store, 
                phraser_key='phrase-2',
                collar=130,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[2.0]],
            )

            payloads = _load_many_queries(store, [
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

    def test_metadata_to_payload_and_metadatas_to_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            first = _put(store, 
                phraser_key='phrase-1',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0]],
            )
            second = _put(store, 
                phraser_key='phrase-2',
                collar=130,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[2.0]],
            )

            payload = store.metadata_to_payload(first)
            payloads = store.metadatas_to_payloads([second, first])

            self.assertEqual(payload, [[1.0]])
            self.assertEqual(payloads, [[[2.0]], [[1.0]]])

    def test_metadatas_to_payloads_returns_none_for_misses_and_can_be_strict(
        self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            metadata = _put(store, 
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )

            payloads = store.metadatas_to_payloads([metadata, None])

            self.assertEqual(payloads, [[[1.0]], None])
            with self.assertRaisesRegex(ValueError, 'no stored output matched'):
                store.metadatas_to_payloads([metadata, None],
                    strict=True)

    def test_metadata_property_matches_list_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            first = _put(store, 
                phraser_key='phrase-2',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )
            second = _put(store, 
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[2.0]],
            )

            self.assertEqual([_hex(item) for item in store.metadata], [
                _hex(second),
                _hex(first),
            ])
            self.assertEqual([_hex(item) for item in store.metadata], [
                _hex(item) for item in store.list_entries()
            ])

    def test_load_object_frames_single_and_all_collars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            _put(store, 
                phraser_key='phrase-1',
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0]],
            )
            _put(store, 
                phraser_key='phrase-1',
                collar=750,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[3.0, 4.0]],
            )

            exact = store.load_object_frames(
                phraser_key=_pk('phrase-1'),
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            )
            nearest = store.load_object_frames(
                phraser_key=_pk('phrase-1'),
                collar=700,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                match='nearest',
            )
            all_collars = store.load_object_frames(
                phraser_key=_pk('phrase-1'),
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
            first = _put(store, 
                phraser_key='phrase-1',
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0]],
            )
            second = _put(store, 
                phraser_key='phrase-1',
                collar=750,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[3.0, 4.0]],
            )

            rows = list(store.iter_object_frames(
                phraser_key=_pk('phrase-1'),
                model_name='wav2vec2',
                layer=7,
            ))
            nearest = list(store.iter_object_frames(
                phraser_key=_pk('phrase-1'),
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
            self.assertEqual(_hex(rows[0][0]), _hex(first))
            self.assertEqual(_hex(rows[1][0]), _hex(second))
            self.assertEqual(len(nearest), 1)
            self.assertEqual(nearest[0][0].collar, 750)
            self.assertEqual(nearest[0][1], [[3.0, 4.0]])

    def test_collar_matching_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            for collar in (100, 200, 350):
                _put(store, 
                    phraser_key='word-1',
                    collar=collar,
                    model_name='hubert',
                    output_type='attention',
                    layer=3,
                    data=[[[1, 2], [3, 4]]],
                )

            minimum = _find_one(store, 
                phraser_key='word-1',
                collar=150,
                model_name='hubert',
                output_type='attention',
                layer=3,
                match='min',
            )
            maximum = _find_one(store, 
                phraser_key='word-1',
                collar=150,
                model_name='hubert',
                output_type='attention',
                layer=3,
                match='max',
            )
            nearest = _find_one(store, 
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

            deleted = _delete(store,
                phraser_key=_pk('word-1'),
                collar=200,
                model_name='hubert',
                output_type='attention',
                layer=3,
            )
            self.assertEqual(deleted.storage_status, 'deleted')
            self.assertFalse(_exists(store, 
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

            metadata, created = _find_or_compute(store,
                phraser_key=_pk('phone-1'),
                collar=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=0,
                compute=compute,
            )
            again, created_again = _find_or_compute(store,
                phraser_key=_pk('phone-1'),
                collar=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=0,
                compute=compute,
            )

            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(_hex(metadata), _hex(again))
            self.assertEqual(calls, ['compute'])

    def test_find_or_compute_can_add_tags_on_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            def compute() -> list[int]:
                return [1, 2, 3]

            metadata, created = _find_or_compute(store,
                phraser_key=_pk('phone-2'),
                collar=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=0,
                compute=compute,
                tags=['exp-a'],
            )
            again, created_again = _find_or_compute(store,
                phraser_key=_pk('phone-2'),
                collar=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=0,
                compute=compute,
                tags=['exp-b'],
                add_tags_on_hit=True,
            )

            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(_hex(metadata), _hex(again))
            self.assertEqual(again.tags, ['exp-a', 'exp-b'])

    def test_tag_queries_and_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            metadata = _put(store, 
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
            self.assertEqual(_hex(entries[0]), _hex(metadata))

            updated = store.add_tags(metadata.echoframe_key, ['review'])
            self.assertEqual(updated.tags, ['exp-a', 'review', 'subset-1'])

            entries = store.find_by_tag('review')
            self.assertEqual(len(entries), 1)
            self.assertEqual(_hex(entries[0]), _hex(metadata))

            updated = store.remove_tags(metadata.echoframe_key, ['exp-a'])
            self.assertEqual(updated.tags, ['review', 'subset-1'])
            self.assertEqual(store.find_by_tag('exp-a'), [])

    def test_tag_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            _put(store, 
                phraser_key='phrase-10',
                collar=90,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=5,
                data=[[1.0]],
                tags=['exp-a', 'subset-1'],
            )
            _put(store, 
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

    def test_find_by_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            first = _put(store, 
                phraser_key='phrase-10',
                collar=90,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=5,
                data=[[1.0]],
            )
            second = _put(store, 
                phraser_key='phrase-10',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=6,
                data=[[2.0]],
            )
            _put(store, 
                phraser_key='phrase-11',
                collar=90,
                model_name='hubert',
                output_type='hidden_state',
                layer=5,
                data=[[3.0]],
            )

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
            self.assertEqual([_hex(item) for item in filtered], [
                _hex(second),
            ])
            self.assertEqual(load.call_args_list, [
                mock.call(_pk('phrase-10')),
                mock.call(_pk('phrase-11')),
                mock.call(_pk('phrase-10')),
            ])

    def test_find_by_label_validates_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            with self.assertRaisesRegex(ValueError,
                'label must be a non-empty string'):
                store.find_by_label('')

    def test_find_by_label_missing_phraser_dependency_raises_helpful_error(
        self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)

            with mock.patch.dict(sys.modules, {'phraser': None}):
                with self.assertRaisesRegex(ImportError,
                    'phraser is required to find entries by label'):
                    store.find_by_label('hello')

    def test_invalid_tags_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            EchoframeMetadata(
                phraser_key='phrase-1',
                collar=10,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                tags=['bad:tag'],
            )
        with self.assertRaises(ValueError):
            EchoframeMetadata(
                phraser_key='phrase-1',
                collar=10,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                tags=['   '],
            )
        with self.assertRaises(ValueError):
            EchoframeMetadata(
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

            one = _put(store, 
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            two = _put(store, 
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )

            old_shard = one.shard_id
            self.assertEqual(old_shard, two.shard_id)

            _delete(store,
                phraser_key=_pk('phrase-b'),
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            compacted = store.compact_shards()
            self.assertEqual(compacted, [old_shard])

            live = _find_one(store, 
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            self.assertNotEqual(live.shard_id, old_shard)
            self.assertEqual(
                _load_query(store, 
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

            self.assertEqual(_find_many(store, []), [])
            self.assertEqual(_load_many_queries(store, []), [])
            self.assertEqual(store.put_many([]), [])
            self.assertEqual(store.add_tags_many([], ['exp-a']), [])
            self.assertEqual(store.remove_tags_many([], ['exp-a']), [])
            self.assertIsNone(_delete(store,
                phraser_key=_pk('missing'),
                collar=10,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
            ))
            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                _load_query(store, 
                    phraser_key='missing',
                    collar=10,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=1,
                )

    def test_load_many_returns_none_for_misses_and_can_be_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            _put(store, 
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )

            payloads = _load_many_queries(store, [
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
                _load_many_queries(store, [
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
                _load_object_frames(store,
                    phraser_key=_pk('missing'),
                    collar=500,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=7,
                )

            self.assertEqual(_load_object_frames(store,
                phraser_key=_pk('missing'),
                collar=None,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            ), {})
            with self.assertRaisesRegex(ValueError,
                'no stored output matched'):
                list(_iter_object_frames(store,
                    phraser_key=_pk('missing'),
                    collar=500,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=7,
                ))
            self.assertEqual(list(_iter_object_frames(store,
                phraser_key=_pk('missing'),
                collar=None,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            )), [])

    def test_include_deleted_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            live = _put(store, 
                phraser_key='phrase-live',
                collar=80,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=2,
                data=[[1.0]],
                tags=['exp-a', 'shared'],
            )
            deleted = _put(store, 
                phraser_key='phrase-deleted',
                collar=90,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=2,
                data=[[2.0]],
                tags=['exp-b', 'shared'],
            )
            _delete(store,
                phraser_key=_pk('phrase-deleted'),
                collar=90,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=2,
            )

            self.assertEqual(
                [_hex(item) for item in _find(store, 
                    'phrase-deleted',
                    include_deleted=True,
                )],
                [_hex(deleted)],
            )
            self.assertEqual(_find(store, 'phrase-deleted'), [])
            self.assertEqual(
                [_hex(item) for item in store.find_by_tag(
                    'exp-b',
                    include_deleted=True,
                )],
                [_hex(deleted)],
            )
            self.assertEqual(store.find_by_tag('exp-b'), [])
            self.assertEqual(sorted(_hex(item) for item in
                store.find_by_tags(['shared'], include_deleted=True)), [
                _hex(deleted),
                _hex(live),
            ])
            self.assertEqual([_hex(item) for item in
                store.find_by_tags(['shared'])], [_hex(live)])
            self.assertEqual(store.list_tags(), ['exp-a', 'shared'])
            self.assertEqual(sorted(store.list_tags(include_deleted=True)), [
                'exp-a',
                'exp-b',
                'shared',
            ])

    def test_index_validation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            _put(store, 
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
                tags=['exp-a'],
            )

            with self.assertRaisesRegex(ValueError, 'match must be one of'):
                _find_one(store, 
                    phraser_key='phrase-1',
                    collar=100,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=1,
                    match='bad',
                )
            with self.assertRaisesRegex(ValueError, 'match must be one of'):
                _find_many(store, [
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
            metadata = _put(store, 
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
            first = _put(store, 
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
                tags=['exp-a'],
            )
            second = _put(store, 
                phraser_key='phrase-2',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[2.0]],
                tags=['exp-b'],
            )
            _delete(store,
                phraser_key=_pk('phrase-2'),
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
            )

            live_entries = store.list_entries()
            all_entries = store.list_entries(include_deleted=True)
            overview = store.overview(include_deleted=True,
                health_event_limit=5)

            self.assertEqual([_hex(item) for item in live_entries],
                [_hex(first)])
            self.assertEqual([_hex(item) for item in all_entries], [
                _hex(first),
                _hex(second),
            ])
            self.assertEqual(overview['entry_count'], 2)
            self.assertEqual(overview['shard_count'], 1)
            self.assertEqual([item['echoframe_key_hex'] for item in
                overview['entries']], [_hex(first), _hex(second)])
            self.assertEqual(sorted(overview['tags']), ['exp-a', 'exp-b'])
            self.assertIsNone(overview['integrity'])
            self.assertLessEqual(len(
                overview['recent_shard_health_events']), 5)

            overview_with_integrity = store.overview(include_deleted=True,
                include_integrity=True)
            self.assertIn('ok', overview_with_integrity['integrity'])

    def test_metadata_helpers(self) -> None:
        metadata = EchoframeMetadata(
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
        )

        self.assertEqual(metadata.format_echoframe_key(),
            metadata.echoframe_key.hex())
        self.assertEqual(metadata.tags, ['a', 'b'])
        self.assertEqual(metadata.shape, (2, 3))
        self.assertIsNotNone(metadata.created_at)
        self.assertIsNone(metadata.deleted_at)
        self.assertEqual(repr(metadata),
            'MD(model=wav2vec2, layer=7, status=live, tags=a,b)')
        self.assertLessEqual(len(repr(metadata)), 80)
        with self.assertRaisesRegex(ValueError,
            'metadata is not bound to a store'):
            metadata.load_payload()

        class FakePhraserObject:
            label = 'hello'

            def __repr__(self):
                return "PhraserObject(label='hello')"

        phraser_object = FakePhraserObject()
        fake_models = types.SimpleNamespace(
            cache=types.SimpleNamespace(
                load=mock.Mock(return_value=phraser_object),
            ),
        )
        fake_phraser = types.SimpleNamespace(models=fake_models)
        with mock.patch.dict(sys.modules, {'phraser': fake_phraser}):
            self.assertIs(metadata.phraser_object, phraser_object)
            self.assertIs(metadata.phraser_object, phraser_object)
            self.assertEqual(metadata.label, 'hello')
            self.assertEqual(str(metadata),
                f"{{'echoframe_key_hex': '{metadata.format_echoframe_key()}',\n"
                " 'phraser_key': 'phrase-1',\n"
                " 'collar': 120,\n"
                " 'model_name': 'wav2vec2',\n"
                " 'output_type': 'hidden_state',\n"
                " 'layer': 7,\n"
                " 'shard_id': 'wav2vec2_hidden_state_0001',\n"
                " 'dataset_path': '/layer_0007/entry',\n"
                " 'shape': (2, 3),\n"
                " 'dtype': 'float32',\n"
                " 'tags': ['a', 'b'],\n"
                " 'phraser_object': \"PhraserObject(label='hello')\",\n"
                f" 'created_at': '{metadata.created_at}'}}")
        self.assertEqual(fake_models.cache.load.call_count, 1)

        restored = EchoframeMetadata.from_dict(metadata.to_dict())
        self.assertEqual(restored.to_dict(), metadata.to_dict())

        updated = metadata.with_tags(['z', 'a'])
        self.assertEqual(updated.tags, ['a', 'z'])
        self.assertEqual(updated.created_at, metadata.created_at)
        self.assertEqual(updated.deleted_at, metadata.deleted_at)
        self.assertIsNone(updated._store)

        deleted = metadata.mark_deleted()
        self.assertEqual(deleted.storage_status, 'deleted')
        self.assertEqual(deleted.created_at, metadata.created_at)
        self.assertIsNotNone(deleted.deleted_at)

    def test_store_metadata_are_bound_and_can_load_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            created = _put(store, 
                phraser_key='phrase-1',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0], [3.0, 4.0]],
                tags=['exp-a'],
            )

            self.assertIs(created._store, store)
            self.assertEqual(created.load_payload(),
                [[1.0, 2.0], [3.0, 4.0]])

            found = _find_one(store, 
                phraser_key='phrase-1',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            )
            self.assertIs(found._store, store)
            self.assertEqual(found.load_payload(),
                [[1.0, 2.0], [3.0, 4.0]])

            listed = store.metadata
            self.assertEqual(len(listed), 1)
            self.assertIs(listed[0]._store, store)
            self.assertEqual(listed[0].load_payload(),
                [[1.0, 2.0], [3.0, 4.0]])

    def test_output_storage_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(
                Path(tmpdir),
                h5_module=FakeH5Module(),
            )
            metadata = EchoframeMetadata(
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
                f'/layer_0002/{metadata.format_echoframe_key()}')
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
            _put(store, 
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

            first = _put(store, 
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )
            storage.force_stat_failure = True

            second = _put(store, 
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
            metadata = _put(store, 
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
                updated = store.add_tags(metadata.echoframe_key, ['exp-b'])

            self.assertEqual(updated.tags, ['exp-a', 'exp-b'])
            shard_stats = store.index.get_shard_metadata(metadata.shard_id)
            self.assertEqual(shard_stats['byte_size'], original_size)
            self.assertTrue(shard_stats['byte_size_is_estimated'])
            self.assertIn('io error', shard_stats['byte_size_error'])

    def test_compaction_no_op_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            live = _put(store, 
                phraser_key='phrase-live',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )
            deleted = _put(store, 
                phraser_key='phrase-deleted',
                collar=100,
                model_name='hubert',
                output_type='attention',
                layer=2,
                data=[[[1, 2]]],
            )
            _delete(store,
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
            live = _put(store, 
                phraser_key='phrase-live',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )
            deleted = _put(store, 
                phraser_key='phrase-deleted',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[2.0]],
            )
            _delete(store,
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

            self.assertEqual([_hex(item) for item in live_entries],
                [_hex(live)])
            self.assertEqual([_hex(item) for item in all_entries], [
                _hex(deleted),
                _hex(live),
            ])
            self.assertEqual([item['echoframe_key_hex'] for item in
                overview['entries']], [_hex(deleted), _hex(live)])

    def test_list_entries_ignores_stale_shard_index_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_fake_store(tmpdir)
            metadata = _put(store, 
                phraser_key='phrase-live',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )

            duplicated = EchoframeMetadata(
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
                echoframe_key=metadata.echoframe_key,
            )
            store.index.upsert(duplicated)
            with store.index.env.begin(write=True) as txn:
                txn.put(store.index._shard_key(metadata.shard_id,
                    metadata.echoframe_key),
                    metadata.echoframe_key,
                    db=store.index.by_shard_db)

            entries = store.list_entries()

            self.assertEqual([_hex(item) for item in entries],
                [_hex(metadata)])

    def test_compaction_marks_failed_journal_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
                shards_root=Path(tmpdir) / 'shards')
            storage = FailingCompactStorage(
                Path(tmpdir) / 'shards',
                h5_module=FakeH5Module(),
            )
            store = Store(tmpdir, index=index, storage=storage)

            one = _put(store, 
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            _put(store, 
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )
            _delete(store,
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
            live = _put(store, 
                phraser_key='phrase-live',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[1.0]],
            )
            _put(store, 
                phraser_key='phrase-deleted',
                collar=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=1,
                data=[[2.0]],
            )
            _delete(store,
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
            one = _put(store, 
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            _put(store, 
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )
            _delete(store,
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            plan = store.compact_shards(dry_run=True)[0]
            journal = store.index.create_compaction_journal(
                plan['shard_id'],
                source_echoframe_keys=plan['source_echoframe_keys'],
                live_echoframe_keys=plan['live_echoframe_keys'],
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
            one = _put(store, 
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            _put(store, 
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )
            _delete(store,
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            plan = store.compact_shards(dry_run=True)[0]
            journal = store.index.create_compaction_journal(
                plan['shard_id'],
                source_echoframe_keys=plan['source_echoframe_keys'],
                live_echoframe_keys=plan['live_echoframe_keys'],
                target_shard_id=plan['target_shard_id'],
            )

            live_entries = [store.index.get(echoframe_key)
                for echoframe_key in journal['live_echoframe_keys']]
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
                    source_echoframe_keys=['a'],
                    live_echoframe_keys=['a'],
                    target_shard_id='wav2vec2_hidden_state_0002',
                )
                second = store.index.create_compaction_journal(
                    'wav2vec2_hidden_state_0001',
                    source_echoframe_keys=['a'],
                    live_echoframe_keys=['a'],
                    target_shard_id='wav2vec2_hidden_state_0002',
                )

            self.assertNotEqual(first['journal_id'], second['journal_id'])
            self.assertEqual(len(store.compaction_journal()), 2)

    def test_missing_h5py_dependency_raises_helpful_import_error(self
        ) -> None:
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
            records = _put_many(store, [
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
            loaded = _load_query(store, 
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            self.assertEqual(self._payload_to_list(loaded), [[1.0]])

            updated = store.add_tags_many(
                [record.echoframe_key for record in records],
                ['batch'])
            self.assertEqual(len(updated), 2)
            self.assertEqual(sorted(store.list_tags()), [
                'batch', 'exp-a', 'speaker-1', 'speaker-2'])
            self.assertEqual(len(store.find_by_tags(['exp-a', 'batch'])), 2)
            self.assertEqual(len(store.find_by_tags(['speaker-1', 'speaker-2'],
                match='any')), 2)

            deleted = _delete(store,
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
            live = _find_one(store, 
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            self.assertNotEqual(live.shard_id, shard_id)
            self.assertEqual(self._payload_to_list(_load_query(store, 
                phraser_key='phrase-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )), [[1.0]])

    def test_real_integrity_checks_and_shard_stats(self) -> None:
        tmpdir, store = self._make_store()
        with tmpdir:
            metadata = _put(store, 
                phraser_key='phrase-3',
                collar=120,
                model_name='hubert',
                output_type='attention',
                layer=2,
                data=[[[1, 2], [3, 4]]],
                tags=['exp-b'],
            )
            _put(store, 
                phraser_key='phrase-4',
                collar=120,
                model_name='hubert',
                output_type='attention',
                layer=2,
                data=[[[5, 6], [7, 8]]],
                tags=['exp-b', 'subset-1'],
            )
            _delete(store,
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
            self.assertEqual(
                report['broken_references'][0]['echoframe_key_hex'],
                _hex(metadata),
            )

    def test_real_find_many_and_tag_queries(self) -> None:
        tmpdir, store = self._make_store()
        with tmpdir:
            _put_many(store, [
                {
                    'phraser_key': 'phrase-10',
                    'collar': 80,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 0,
                    'data': [1, 2, 3],
                    'tags': ['exp-a', 'run-1'],
                },
                {
                    'phraser_key': 'phrase-11',
                    'collar': 90,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 0,
                    'data': [4, 5, 6],
                    'tags': ['exp-a', 'run-2'],
                },
                {
                    'phraser_key': 'phrase-12',
                    'collar': 90,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 0,
                    'data': [7, 8, 9],
                    'tags': ['exp-b', 'run-2'],
                },
            ])

            results = _find_many(store, [
                {
                    'phraser_key': 'phrase-10',
                    'collar': 80,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 0,
                },
                {
                    'phraser_key': 'phrase-11',
                    'collar': 95,
                    'model_name': 'encodec',
                    'output_type': 'codebook_indices',
                    'layer': 0,
                    'match': 'max',
                },
            ])

            self.assertEqual([result.phraser_key for result in results], [
                _pk('phrase-10'),
                _pk('phrase-11'),
            ])
            self.assertEqual(sorted(store.list_tags()), [
                'exp-a', 'exp-b', 'run-1', 'run-2'])
            all_match = store.find_by_tags(['exp-a', 'run-2'], match='all')
            any_match = store.find_by_tags(['exp-b', 'run-1'], match='any')
            self.assertEqual([item.phraser_key for item in all_match],
                [_pk('phrase-11')])
            self.assertEqual(sorted(item.phraser_key for item in any_match), [
                _pk('phrase-10'),
                _pk('phrase-12'),
            ])

    def test_real_retrieval_helpers(self) -> None:
        tmpdir, store = self._make_store()
        with tmpdir:
            _put(store, 
                phraser_key='phrase-20',
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0]],
            )
            _put(store, 
                phraser_key='phrase-20',
                collar=750,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[3.0, 4.0]],
            )
            _put(store, 
                phraser_key='phrase-21',
                collar=500,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[5.0, 6.0]],
            )

            payloads = _load_many_queries(store, [
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
            exact = _load_object_frames(store,
                phraser_key='phrase-20',
                model_name='wav2vec2',
                layer=7,
                collar=500,
            )
            nearest = _load_object_frames(store,
                phraser_key='phrase-20',
                model_name='wav2vec2',
                layer=7,
                collar=700,
                match='nearest',
            )
            all_collars = _load_object_frames(store,
                phraser_key='phrase-20',
                model_name='wav2vec2',
                layer=7,
                collar=None,
            )
            rows = list(_iter_object_frames(store,
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
                _load_many_queries(store, [
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
            one = _put(store, 
                phraser_key='phrase-a',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0]],
            )
            _put(store, 
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[2.0]],
            )
            _delete(store,
                phraser_key='phrase-b',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
            )
            plan = store.compact_shards(dry_run=True)[0]
            journal = store.index.create_compaction_journal(
                plan['shard_id'],
                source_echoframe_keys=plan['source_echoframe_keys'],
                live_echoframe_keys=plan['live_echoframe_keys'],
                target_shard_id=plan['target_shard_id'],
            )

            resumed = store.resume_compactions()
            self.assertEqual(resumed, [one.shard_id])
            records = store.compaction_journal()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['journal_id'], journal['journal_id'])
            self.assertEqual(records[0]['status'], 'completed')

    def test_repeated_store_construction_reuses_lmdb_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Store(tmpdir, max_shard_size_bytes=1024 * 1024)
            second = Store(tmpdir, max_shard_size_bytes=1024 * 1024)

            self.assertIs(first.index.env, second.index.env)

            first_record = _put(first,
                phraser_key='phrase-cache-1',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[1.0, 2.0]],
            )
            self.assertEqual(self._payload_to_list(second.load(
                first_record.echoframe_key,
            )), [[1.0, 2.0]])

            second_record = _put(second,
                phraser_key='phrase-cache-2',
                collar=100,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=3,
                data=[[3.0, 4.0]],
            )
            self.assertEqual(self._payload_to_list(first.load(
                second_record.echoframe_key,
            )), [[3.0, 4.0]])

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
                first = Store(first_tmpdir,
                    max_shard_size_bytes=1024 * 1024)
                second = Store(second_tmpdir,
                    max_shard_size_bytes=1024 * 1024)

                self.assertIsNot(first.index.env, second.index.env)

    def test_same_root_with_different_map_size_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Store(tmpdir, max_shard_size_bytes=1024 * 1024)

            with self.assertRaisesRegex(ValueError, 'map_size'):
                LmdbIndex(Path(tmpdir) / 'index.lmdb', map_size=2 << 30)
