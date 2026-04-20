'''Tests for store compaction, journals, and resume flows.'''

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

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

    def test_compact_shards_removes_deleted_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            one = _put(store, phraser_key='phrase-a', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[1.0]])
            two = _put(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[2.0]])
            old_shard = one.shard_id
            self.assertEqual(old_shard, two.shard_id)
            _delete(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3)
            compacted = store.compact_shards()
            self.assertEqual(compacted, [old_shard])
            live = _find_one(store, phraser_key='phrase-a', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=3)
            self.assertNotEqual(live.shard_id, old_shard)
            self.assertEqual(store.load(live.echoframe_key), [[1.0]])
            self.assertEqual(store.index.entries_for_shard(old_shard), [])

    def test_compaction_no_op_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            live = _put(store, phraser_key='phrase-live', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            deleted = _put(store, phraser_key='phrase-deleted', collar=100,
                model_name='hubert', output_type='attention', layer=2,
                data=[[[1, 2]]])
            _delete(store, phraser_key='phrase-deleted', collar=100,
                model_name='hubert', output_type='attention', layer=2)
            self.assertEqual(store.compact_shards(shard_ids=[live.shard_id]), [])
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
            store = make_fake_store(tmpdir)
            live = _put(store, phraser_key='phrase-live', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            deleted = _put(store, phraser_key='phrase-deleted', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]])
            _delete(store, phraser_key='phrase-deleted', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=1)
            store.compact_shards()
            live_entries = store.list_entries()
            all_entries = store.list_entries(include_deleted=True)
            overview = store.overview(include_deleted=True)
            self.assertEqual([_hex(item) for item in live_entries], [_hex(live)])
            self.assertEqual([_hex(item) for item in all_entries], [
                _hex(deleted), _hex(live)])
            self.assertEqual([item['echoframe_key_hex']
                for item in overview['metadatas']], [_hex(deleted), _hex(live)])

    def test_list_entries_ignores_stale_shard_index_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            metadata = _put(store, phraser_key='phrase-live', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            duplicated = EchoframeMetadata(
                phraser_key=metadata.phraser_key, collar=metadata.collar,
                model_name=metadata.model_name,
                output_type=metadata.output_type, layer=metadata.layer,
                storage_status=metadata.storage_status, shard_id='other_0002',
                dataset_path=metadata.dataset_path, shape=metadata.shape,
                dtype=metadata.dtype, tags=metadata.tags,
                created_at=metadata.created_at,
                deleted_at=metadata.deleted_at,
                echoframe_key=metadata.echoframe_key)
            store.index.upsert(duplicated)
            with store.index.env.begin(write=True) as txn:
                txn.put(store.index._shard_key(metadata.shard_id,
                    metadata.echoframe_key), metadata.echoframe_key,
                    db=store.index.by_shard_db)
            entries = store.list_entries()
            self.assertEqual([_hex(item) for item in entries], [_hex(metadata)])

    def test_compaction_marks_failed_journal_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
                shards_root=Path(tmpdir) / 'shards')
            storage = FailingCompactStorage(Path(tmpdir) / 'shards',
                h5_module=FakeH5Module())
            store = Store(tmpdir, index=index, storage=storage)
            one = _put(store, phraser_key='phrase-a', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[1.0]])
            _put(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[2.0]])
            _delete(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=3)
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
            store = make_fake_store(tmpdir)
            live = _put(store, phraser_key='phrase-live', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            _put(store, phraser_key='phrase-deleted', collar=120,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]])
            _delete(store, phraser_key='phrase-deleted', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=1)
            report = store._build_shard_health_report(
                live.shard_id, 'simulated shard failure')
            self.assertEqual(report['checked_entries'], 1)
            self.assertEqual(report['lost_items'], [])

    def test_resume_pending_runs_before_new_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            one = _put(store, phraser_key='phrase-a', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[1.0]])
            _put(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[2.0]])
            _delete(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=3)
            plan = store.compact_shards(dry_run=True)[0]
            journal = store.index.create_compaction_journal(plan['shard_id'],
                source_echoframe_keys=plan['source_echoframe_keys'],
                live_echoframe_keys=plan['live_echoframe_keys'],
                target_shard_id=plan['target_shard_id'])
            compacted = store.compact_shards(shard_ids=[one.shard_id],
                resume_pending=True)
            self.assertEqual(compacted, [])
            records = store.compaction_journal()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['journal_id'], journal['journal_id'])
            self.assertEqual(records[0]['status'], 'completed')

    def test_resume_compaction_removes_stale_source_shard_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            one = _put(store, phraser_key='phrase-a', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[1.0]])
            _put(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[2.0]])
            _delete(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=3)
            plan = store.compact_shards(dry_run=True)[0]
            journal = store.index.create_compaction_journal(plan['shard_id'],
                source_echoframe_keys=plan['source_echoframe_keys'],
                live_echoframe_keys=plan['live_echoframe_keys'],
                target_shard_id=plan['target_shard_id'])
            live_entries = [store.index.get(echoframe_key)
                for echoframe_key in journal['live_echoframe_keys']]
            updated = store.storage.compact_shard_to(journal['shard_id'],
                [entry for entry in live_entries if entry is not None],
                target_shard_id=journal['target_shard_id'],
                delete_source=False)
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
            store = make_fake_store(tmpdir)
            with mock.patch('echoframe.index.utc_now',
                return_value='2026-04-01T12:00:00+00:00'):
                first = store.index.create_compaction_journal(
                    'wav2vec2_hidden_state_0001', source_echoframe_keys=['a'],
                    live_echoframe_keys=['a'],
                    target_shard_id='wav2vec2_hidden_state_0002')
                second = store.index.create_compaction_journal(
                    'wav2vec2_hidden_state_0001', source_echoframe_keys=['a'],
                    live_echoframe_keys=['a'],
                    target_shard_id='wav2vec2_hidden_state_0002')
            self.assertNotEqual(first['journal_id'], second['journal_id'])
            self.assertEqual(len(store.compaction_journal()), 2)

    def test_resume_compaction_from_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            one = _put(store, phraser_key='phrase-a', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[1.0]])
            _put(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=3, data=[[2.0]])
            _delete(store, phraser_key='phrase-b', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=3)
            plan = store.compact_shards(dry_run=True)[0]
            journal = store.index.create_compaction_journal(plan['shard_id'],
                source_echoframe_keys=plan['source_echoframe_keys'],
                live_echoframe_keys=plan['live_echoframe_keys'],
                target_shard_id=plan['target_shard_id'])
            resumed = store.resume_compactions()
            self.assertEqual(resumed, [one.shard_id])
            records = store.compaction_journal()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['journal_id'], journal['journal_id'])
            self.assertEqual(records[0]['status'], 'completed')
