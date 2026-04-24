'''Tests for store compaction, journals, and resume flows.'''

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from echoframe import compaction
from echoframe.index import LmdbIndex
from echoframe.metadata import EchoframeMetadata
from echoframe.output_storage import Hdf5ShardStore
from echoframe.store import Store
from tests.helpers import (
    delete as _delete,
    FakeEnv,
    FakeH5Module,
    find_one as _find_one,
    hex_key as _hex,
    make_fake_store,
    put as _put,
)


class FailingCompactStorage(Hdf5ShardStore):
    def compact_shard_to(self, shard_id, entries, target_shard_id,
        delete_source=True):
        super().compact_shard_to(shard_id, entries,
            target_shard_id=target_shard_id,
            delete_source=delete_source)
        raise RuntimeError('compaction exploded')


class TestStoreCompactionResume(unittest.TestCase):
    def test_compaction_no_op_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            self.assertEqual(store.compact_shards(dry_run=True), [])
            self.assertEqual(store.compact_shards(), [])
            self.assertEqual(store.compact_shards(shard_ids=[]), [])

            created = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            with mock.patch.object(store.index, 'load_shard_metadata',
                return_value={'entry_count': 1, 'byte_size': 0}):
                plans = store.compact_shards(shard_ids=[created.shard_id],
                    dry_run=True)
                compacted = store.compact_shards(
                    shard_ids=[created.shard_id])

        self.assertEqual(plans, [])
        self.assertEqual(compacted, [])

    def test_resume_compaction_rewrites_surviving_entries_after_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            first = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            second = _put(store, phraser_key='phrase-2', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]])
            original_shard_id = first.shard_id
            store.delete(second.echoframe_key)
            with mock.patch.object(store.index, 'load_shard_metadata',
                return_value=None):
                with mock.patch.object(store.storage, 'shard_size',
                    return_value=1):
                    plan = store.compact_shards(dry_run=True)[0]
            store.index.create_compaction_journal(plan['shard_id'],
                plan['echoframe_keys'], plan['target_shard_id'])

            completed = store.resume_compactions()
            refreshed = store.load_metadata(first.echoframe_key)
            payload = store.load(first.echoframe_key)

        self.assertEqual(completed, [plan['shard_id']])
        self.assertEqual(payload, [[1.0]])
        self.assertEqual(store.load_metadata(second.echoframe_key), None)
        self.assertEqual(refreshed.shard_id, plan['target_shard_id'])
        self.assertNotEqual(refreshed.shard_id, original_shard_id)
        self.assertNotIn(original_shard_id, store.index.list_shards())

    def test_compaction_marks_failed_journal_on_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
                shards_root=Path(tmpdir) / 'shards')
            storage = FailingCompactStorage(Path(tmpdir) / 'shards',
                h5_module=FakeH5Module())
            store = Store(tmpdir, index=index, storage=storage)
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            with mock.patch.object(store.index, 'load_shard_metadata',
                return_value=None):
                with mock.patch.object(store.storage, 'shard_size',
                    return_value=1):
                    plan = store.compact_shards(dry_run=True)[0]
                    with self.assertRaisesRegex(RuntimeError,
                        'compaction exploded'):
                        store.compact_shards()

            records = store.compaction_journal(status='failed')
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['shard_id'], plan['shard_id'])
            self.assertIn('compaction exploded', records[0]['error'])

    def test_resume_pending_runs_before_new_compaction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            _put(store, phraser_key='phrase-2', collar=200,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]])
            with mock.patch.object(store.index, 'load_shard_metadata',
                return_value=None):
                with mock.patch.object(store.storage, 'shard_size',
                    return_value=1):
                    dry_run = store.compact_shards(dry_run=True)
            first_plan = dry_run[0]
            journal = store.index.create_compaction_journal(
                first_plan['shard_id'], first_plan['echoframe_keys'],
                first_plan['target_shard_id'])
            calls = []
            original = compaction.run_compaction_plan

            def recording_run(store_obj, plan, from_journal=False):
                calls.append((plan['journal_id'] if from_journal else
                    plan['shard_id'], from_journal))
                return original(store_obj, plan, from_journal=from_journal)

            with mock.patch.object(compaction, 'run_compaction_plan',
                side_effect=recording_run):
                store.compact_shards(resume_pending=True)

        self.assertTrue(calls)
        self.assertEqual(calls[0], (journal['journal_id'], True))

    def test_resume_compaction_removes_stale_source_shard_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            created = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            with mock.patch.object(store.index, 'load_shard_metadata',
                return_value=None):
                with mock.patch.object(store.storage, 'shard_size',
                    return_value=1):
                    plan = store.compact_shards(dry_run=True)[0]
            store.index.create_compaction_journal(plan['shard_id'],
                plan['echoframe_keys'], plan['target_shard_id'])

            store.resume_compactions()
            refreshed = store.load_metadata(created.echoframe_key)

        self.assertNotEqual(refreshed.shard_id, plan['shard_id'])
        self.assertNotIn(plan['shard_id'], store.index.list_shards())

    def test_compaction_journal_ids_are_unique_with_same_second(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            with mock.patch.object(compaction, 'utc_now',
                return_value='2026-01-01T00:00:00+00:00'):
                first = store.index.create_compaction_journal('s1',
                    ['aa'], 's2')
                second = store.index.create_compaction_journal('s1',
                    ['aa'], 's2')
        self.assertNotEqual(first['journal_id'], second['journal_id'])

    def test_resume_compaction_from_journal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            with mock.patch.object(store.index, 'load_shard_metadata',
                return_value=None):
                with mock.patch.object(store.storage, 'shard_size',
                    return_value=1):
                    plan = store.compact_shards(dry_run=True)[0]
            store.index.create_compaction_journal(plan['shard_id'],
                plan['echoframe_keys'], plan['target_shard_id'])

            completed = store.resume_compactions()
            records = store.compaction_journal(status='completed')

        self.assertEqual(completed, [plan['shard_id']])
        self.assertEqual(len(records), 1)
