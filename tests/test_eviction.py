'''Tests for accessed_at tracking and recency eviction.'''

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from echoframe.index import LmdbIndex
from echoframe.metadata import EchoframeMetadata, utc_now
from echoframe.output_storage import Hdf5ShardStore
from echoframe.store import Store
from tests.helpers import (
    FakeEnv,
    FakeH5Module,
    find_one as _find_one,
    load_many_queries as _load_many_queries,
    load_query as _load_query,
    make_fake_store,
    pk as _pk,
    put as _put,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_ago(n):
    ts = datetime.now(timezone.utc) - timedelta(days=n)
    return ts.replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAccessedAt(unittest.TestCase):
    def test_load_updates_accessed_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
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
            store = make_fake_store(tmpdir)
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
        store = make_fake_store(tmpdir)
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

    def test_tied_access_times_follow_store_entry_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
                shards_root=Path(tmpdir) / 'shards')
            storage = Hdf5ShardStore(Path(tmpdir) / 'shards',
                h5_module=FakeH5Module())
            store = Store(tmpdir, index=index, storage=storage)

            _put(store, phraser_key='a-first', collar=0, model_name='m1',
                output_type='hidden_state', layer=0, data=[1.0])
            _put(store, phraser_key='b-second', collar=0, model_name='m1',
                output_type='hidden_state', layer=0, data=[2.0])

            tied_time = _days_ago(60)
            first = _find_one(store, phraser_key='a-first', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            second = _find_one(store, phraser_key='b-second', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            store.index.upsert(first.with_accessed_at(tied_time))
            store.index.upsert(second.with_accessed_at(tied_time))

            call_count = [0]

            def fake_storage_bytes():
                call_count[0] += 1
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
        self.assertEqual(evicted[0].phraser_key, _pk('a-first'))

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

    def test_deleted_entries_are_not_re_evicted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='live', collar=0, model_name='m1',
                output_type='hidden_state', layer=0, data=[1.0])
            _put(store, phraser_key='deleted', collar=0, model_name='m1',
                output_type='hidden_state', layer=0, data=[2.0])

            live = _find_one(store, phraser_key='live', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            deleted = _find_one(store, phraser_key='deleted', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            store.index.upsert(live.with_accessed_at(_days_ago(60)))
            store.index.upsert(deleted.with_accessed_at(_days_ago(60)))
            store.delete_phraser_key(_pk('deleted'), 'm1', 'hidden_state', 0,
                0)

            call_count = [0]

            def fake_storage_bytes():
                call_count[0] += 1
                if call_count[0] <= 2:
                    return 2_000_000_000
                return 0

            with mock.patch.dict('os.environ',
                {'ECHOFRAME_RECENCY_WINDOW_DAYS': '30',
                 'ECHOFRAME_STORAGE_BUDGET_GB': '1'}):
                with mock.patch.object(store, '_storage_bytes',
                    side_effect=fake_storage_bytes):
                    evicted = store.evict_by_recency()

            self.assertEqual([m.phraser_key for m in evicted], [_pk('live')])
            deleted_entries = store.list_entries(include_deleted=True)
            deleted_keys = [m.phraser_key for m in deleted_entries
                if m.deleted_at is not None]
            self.assertEqual(sorted(deleted_keys),
                [_pk('deleted'), _pk('live')])

    def test_only_deleted_entries_returns_no_evictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='deleted', collar=0, model_name='m1',
                output_type='hidden_state', layer=0, data=[2.0])
            metadata = _find_one(store, phraser_key='deleted', collar=0,
                model_name='m1', output_type='hidden_state', layer=0)
            store.index.upsert(metadata.with_accessed_at(_days_ago(60)))
            store.delete_phraser_key(_pk('deleted'), 'm1', 'hidden_state', 0,
                0)

            with mock.patch.dict('os.environ',
                {'ECHOFRAME_RECENCY_WINDOW_DAYS': '30',
                 'ECHOFRAME_STORAGE_BUDGET_GB': '1'}):
                with mock.patch.object(store, '_storage_bytes',
                    return_value=2_000_000_000):
                    evicted = store.evict_by_recency()

            self.assertEqual(evicted, [])


if __name__ == '__main__':
    unittest.main()
