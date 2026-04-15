'''Tests for accessed_at tracking and recency eviction.'''

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from echoframe.index import LmdbIndex
from echoframe.metadata import (
    EchoframeMetadata,
    metadata_class_for_output_type,
    utc_now,
)
from echoframe.output_storage import Hdf5ShardStore
from echoframe.store import Store
from tests.test_public_api import _ensure_model, _find_one, _load_many_queries, _load_query, _pk


# ---------------------------------------------------------------------------
# Minimal fakes (mirrored from test_public_api.py)
# ---------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, store):
        self.store = store
        self.keys = []
        self.index = 0

    def set_range(self, prefix):
        self.keys = sorted(k for k in self.store if k >= prefix)
        self.index = 0
        return bool(self.keys)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.keys):
            raise StopIteration
        key = self.keys[self.index]
        self.index += 1
        return key, self.store[key]


class FakeTxn:
    def __init__(self, env, write):
        self.env = env
        self.write = write

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def put(self, key, value, db):
        self.env.dbs[db][key] = value

    def get(self, key, db):
        return self.env.dbs[db].get(key)

    def delete(self, key, db):
        self.env.dbs[db].pop(key, None)

    def cursor(self, db):
        return FakeCursor(self.env.dbs[db])


class FakeEnv:
    def __init__(self):
        self.dbs = {}

    def open_db(self, name):
        self.dbs.setdefault(name, {})
        return name

    def begin(self, write=False):
        return FakeTxn(self, write=write)


class FakeDataset:
    def __init__(self, data):
        self.data = data
        self.shape = self._shape(data)
        self.dtype = type(self._leaf(data)).__name__

    def __getitem__(self, item):
        if item == ():
            return self.data
        raise KeyError(item)

    def _shape(self, data):
        if isinstance(data, list) and data:
            return (len(data),) + self._shape(data[0])
        if isinstance(data, list):
            return (0,)
        return ()

    def _leaf(self, data):
        if isinstance(data, list) and data:
            return self._leaf(data[0])
        return data


class FakeGroup(dict):
    def create_dataset(self, name, data):
        dataset = FakeDataset(data)
        self[name] = dataset
        return dataset


class FakeH5File:
    def __init__(self, files, path, mode):
        self.files = files
        self.path = str(path)
        if 'r' in mode and not Path(path).exists():
            raise FileNotFoundError(path)
        if 'r' not in mode:
            Path(path).touch(exist_ok=True)
        self.groups = self.files.setdefault(self.path, {})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def require_group(self, path):
        return self.groups.setdefault(path, FakeGroup())

    def __contains__(self, path):
        group_path, name = path.rsplit('/', 1)
        return name in self.groups.get(group_path, {})

    def __getitem__(self, path):
        group_path, name = path.rsplit('/', 1)
        return self.groups[group_path][name]

    def __delitem__(self, path):
        group_path, name = path.rsplit('/', 1)
        del self.groups[group_path][name]


class FakeH5Module:
    def __init__(self):
        self.files = {}

    def File(self, path, mode):
        return FakeH5File(self.files, path, mode)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_store(tmpdir):
    index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
        shards_root=Path(tmpdir) / 'shards')
    storage = Hdf5ShardStore(Path(tmpdir) / 'shards',
        h5_module=FakeH5Module())
    return Store(tmpdir, index=index, storage=storage)


def _put(store, *, phraser_key, collar, model_name, output_type, layer,
    data, tags=None):
    phraser_key = _pk(phraser_key)
    _ensure_model(store, model_name)
    key_kwargs = {
        'output_type': output_type,
        'model_name': model_name,
    }
    if output_type in {'hidden_state', 'attention'}:
        key_kwargs.update({
            'phraser_key': phraser_key,
            'layer': layer,
            'collar': collar,
        })
    elif output_type == 'codebook_indices':
        key_kwargs.update({
            'phraser_key': phraser_key,
            'collar': collar,
        })
    metadata_cls = metadata_class_for_output_type(output_type)
    metadata = metadata_cls(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=layer, tags=tags,
        echoframe_key=store.make_echoframe_key(**key_kwargs))
    return store.put(metadata.echoframe_key, metadata, data)


def _days_ago(n):
    ts = datetime.now(timezone.utc) - timedelta(days=n)
    return ts.replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAccessedAt(unittest.TestCase):
    def test_load_updates_accessed_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_fake_store(tmpdir)
            _put(store, phraser_key='p1', collar=0, model_name='m1',
                output_type='hidden_state', layer=0, data=[1.0])
            metadata_before = _find_one(store, phraser_key='p1', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            self.assertIsNone(metadata_before.accessed_at)

            _load_query(store, phraser_key='p1', collar=0, model_name='m1',
                output_type='hidden_state', layer=0)
            metadata_after = _find_one(store, phraser_key='p1', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            self.assertIsNotNone(metadata_after.accessed_at)

    def test_load_many_updates_accessed_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_fake_store(tmpdir)
            _put(store, phraser_key='p1', collar=0, model_name='m1',
                output_type='hidden_state', layer=0, data=[1.0])
            _load_many_queries(store, [{'phraser_key': 'p1', 'collar': 0,
                'model_name': 'm1', 'output_type': 'hidden_state',
                'layer': 0}])
            metadata = _find_one(store, phraser_key='p1', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            self.assertIsNotNone(metadata.accessed_at)


class TestEvictByRecency(unittest.TestCase):
    def _store_with_entry(self, tmpdir, phraser_key, accessed_at):
        store = _make_fake_store(tmpdir)
        _put(store, phraser_key=phraser_key, collar=0, model_name='m1',
            output_type='hidden_state', layer=0, data=[1.0])
        metadata = _find_one(store, phraser_key=phraser_key, collar=0,
            model_name='m1', output_type='hidden_state', layer=0)
        if accessed_at is not None:
            updated = metadata.with_accessed_at(accessed_at)
            store.index.upsert(updated)
        return store

    def test_entries_within_window_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._store_with_entry(tmpdir, 'p1', _days_ago(5))
            with mock.patch.dict('os.environ',
                {'ECHOFRAME_RECENCY_WINDOW_DAYS': '30',
                 'ECHOFRAME_STORAGE_BUDGET_GB': '0'}):
                evicted = store.evict_by_recency()
            self.assertEqual(evicted, [])

    def test_stale_entries_are_evicted_oldest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
                shards_root=Path(tmpdir) / 'shards')
            storage = Hdf5ShardStore(Path(tmpdir) / 'shards',
                h5_module=FakeH5Module())
            store = Store(tmpdir, index=index, storage=storage)

            _put(store, phraser_key='old', collar=0, model_name='m1',
                output_type='hidden_state', layer=0, data=[1.0])
            _put(store, phraser_key='older', collar=0, model_name='m1',
                output_type='hidden_state', layer=0, data=[2.0])

            m_old = _find_one(store, phraser_key='old', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            m_older = _find_one(store, phraser_key='older', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            store.index.upsert(m_old.with_accessed_at(_days_ago(40)))
            store.index.upsert(m_older.with_accessed_at(_days_ago(60)))

            # Budget forces one eviction; oldest should go first
            call_count = [0]
            real_storage_bytes = store._storage_bytes

            def fake_storage_bytes():
                call_count[0] += 1
                # First call: over budget; second: under budget
                if call_count[0] <= 2:
                    return 2_000_000_000
                return 0

            with mock.patch.dict('os.environ',
                {'ECHOFRAME_RECENCY_WINDOW_DAYS': '30',
                 'ECHOFRAME_STORAGE_BUDGET_GB': '1'}):
                with mock.patch.object(store, '_storage_bytes',
                    side_effect=fake_storage_bytes):
                    evicted = store.evict_by_recency()

            self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0].phraser_key, _pk('older'))

    def test_entries_without_accessed_at_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._store_with_entry(tmpdir, 'p1', None)
            with mock.patch.dict('os.environ',
                {'ECHOFRAME_RECENCY_WINDOW_DAYS': '30',
                 'ECHOFRAME_STORAGE_BUDGET_GB': '0'}):
                evicted = store.evict_by_recency()
            self.assertEqual(evicted, [])

    def test_no_eviction_when_under_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._store_with_entry(tmpdir, 'p1', _days_ago(60))
            with mock.patch.dict('os.environ',
                {'ECHOFRAME_RECENCY_WINDOW_DAYS': '30',
                 'ECHOFRAME_STORAGE_BUDGET_GB': '1000'}):
                with mock.patch.object(store, '_storage_bytes',
                    return_value=100):
                    evicted = store.evict_by_recency()
            self.assertEqual(evicted, [])


if __name__ == '__main__':
    unittest.main()
