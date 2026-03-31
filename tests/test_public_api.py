'''Public API tests for echoframe.'''

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import echoframe
from echoframe.index import LmdbIndex
from echoframe.output_storage import Hdf5ShardStore
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
        path: Path) -> None:
        self.files = files
        self.path = str(path)
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
        return FakeH5File(self.files, path)


class EchoFrameTests(unittest.TestCase):
    def test_public_exports(self) -> None:
        self.assertEqual(echoframe.__version__, '0.1.0')
        self.assertIn('Store', echoframe.__all__)
        self.assertIn('Metadata', echoframe.__all__)
        self.assertNotIn('LmdbIndex', echoframe.__all__)

    def test_put_find_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv())
            storage = Hdf5ShardStore(
                Path(tmpdir) / 'shards',
                h5_module=FakeH5Module(),
            )
            store = Store(
                tmpdir,
                index=index,
                storage=storage,
            )

            metadata = store.put(
                phraser_key='phrase-1',
                collar_ms=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
                data=[[1.0, 2.0], [3.0, 4.0]],
                to_vector_version='abc123',
            )

            self.assertEqual(metadata.phraser_key, 'phrase-1')
            self.assertEqual(metadata.layer, 7)
            self.assertEqual(metadata.shard_id, 'wav2vec2_hidden_state_0001')
            self.assertTrue(store.exists(
                phraser_key='phrase-1',
                collar_ms=120,
                model_name='wav2vec2',
                output_type='hidden_state',
                layer=7,
            ))
            self.assertEqual(
                store.load(
                    phraser_key='phrase-1',
                    collar_ms=120,
                    model_name='wav2vec2',
                    output_type='hidden_state',
                    layer=7,
                ),
                [[1.0, 2.0], [3.0, 4.0]],
            )

    def test_collar_matching_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv())
            storage = Hdf5ShardStore(
                Path(tmpdir) / 'shards',
                h5_module=FakeH5Module(),
            )
            store = Store(
                tmpdir,
                index=index,
                storage=storage,
            )

            for collar in (100, 200, 350):
                store.put(
                    phraser_key='word-1',
                    collar_ms=collar,
                    model_name='hubert',
                    output_type='attention',
                    layer=3,
                    data=[[[1, 2], [3, 4]]],
                )

            minimum = store.find_one(
                phraser_key='word-1',
                collar_ms=150,
                model_name='hubert',
                output_type='attention',
                layer=3,
                match='min',
            )
            maximum = store.find_one(
                phraser_key='word-1',
                collar_ms=150,
                model_name='hubert',
                output_type='attention',
                layer=3,
                match='max',
            )
            nearest = store.find_one(
                phraser_key='word-1',
                collar_ms=180,
                model_name='hubert',
                output_type='attention',
                layer=3,
                match='nearest',
            )

            self.assertEqual(minimum.collar_ms, 200)
            self.assertEqual(maximum.collar_ms, 100)
            self.assertEqual(nearest.collar_ms, 200)

            deleted = store.delete(
                phraser_key='word-1',
                collar_ms=200,
                model_name='hubert',
                output_type='attention',
                layer=3,
            )
            self.assertEqual(deleted.storage_status, 'deleted')
            self.assertFalse(store.exists(
                phraser_key='word-1',
                collar_ms=200,
                model_name='hubert',
                output_type='attention',
                layer=3,
            ))

    def test_find_or_compute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv())
            storage = Hdf5ShardStore(
                Path(tmpdir) / 'shards',
                h5_module=FakeH5Module(),
            )
            store = Store(
                tmpdir,
                index=index,
                storage=storage,
            )
            calls: list[str] = []

            def compute() -> list[int]:
                calls.append('compute')
                return [1, 2, 3]

            metadata, created = store.find_or_compute(
                phraser_key='phone-1',
                collar_ms=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=1,
                compute=compute,
            )
            again, created_again = store.find_or_compute(
                phraser_key='phone-1',
                collar_ms=50,
                model_name='encodec',
                output_type='codebook_indices',
                layer=1,
                compute=compute,
            )

            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(metadata.entry_id, again.entry_id)
            self.assertEqual(calls, ['compute'])
