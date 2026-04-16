'''Tests for shard storage runtime edge cases and diagnostics.'''

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from echoframe.index import LmdbIndex
from echoframe.metadata import EchoframeMetadata
from echoframe.output_storage import Hdf5ShardStore, sanitize_name
from echoframe.store import Store
from tests.helpers import FakeEnv, FakeH5Module, make_fake_store, put as _put


class FlakySizeStorage(Hdf5ShardStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_stat_failure = False

    def _size_info_with_retries(self, shard_id, file_path, purpose):
        if self.force_stat_failure and shard_id.endswith('_0001'):
            return {'exists': True, 'byte_size': None,
                'is_estimated': False, 'error': 'simulated stat failure'}
        return super()._size_info_with_retries(shard_id, file_path,
            purpose=purpose)


class TestStorageRuntime(unittest.TestCase):

    def test_output_storage_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(Path(tmpdir), h5_module=FakeH5Module())
            metadata = EchoframeMetadata(phraser_key='phrase-1', collar=100,
                model_name='model name', output_type='hidden_state', layer=2,
                created_at='2024-01-01T00:00:00+00:00',
                echoframe_key=b'\x01\x02')
            stored = storage.store_with_shard(metadata, data=[[1, 2]],
                shard_id='manual_0001')
            self.assertEqual(stored.shard_id, 'manual_0001')
            self.assertEqual(stored.dataset_path,
                '/layer_0002/0102')
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
            store = make_fake_store(str(store_root))
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            log_root = Path(tmpdir) / 'tests' / 'data' / 'echoframe_logs'
            self.assertFalse(log_root.exists())

    def test_stat_retries_include_long_backoff_delays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(Path(tmpdir), h5_module=FakeH5Module())
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
            storage = FlakySizeStorage(Path(tmpdir) / 'shards',
                h5_module=FakeH5Module())
            store = Store(tmpdir, index=index, storage=storage)
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            storage.force_stat_failure = True
            second = _put(store, phraser_key='phrase-2', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[2.0]])
            self.assertEqual(second.shard_id, 'wav2vec2_hidden_state_0002')
            events = store.get_shard_health_events()
            self.assertTrue(any(event['event_type'] == 'skip_unreadable_shard'
                for event in events))

    def test_active_and_replacement_shard_probe_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(Path(tmpdir), h5_module=FakeH5Module())

            def fake_state(shard_id, file_path, purpose):
                if shard_id.endswith('_0001'):
                    return {'state': 'unreadable', 'byte_size': 0,
                        'error': 'simulated shard failure'}
                return {'state': 'missing', 'byte_size': None, 'error': None}

            with mock.patch.object(storage, '_shard_path_state',
                side_effect=fake_state):
                shard_id = storage._active_shard_id('wav2vec2',
                    'hidden_state')
            self.assertEqual(shard_id, 'wav2vec2_hidden_state_0002')

            storage.MAX_SCAN_SUFFIXES = 5

            def always_unreadable(shard_id, file_path, purpose):
                return {'state': 'unreadable', 'byte_size': None,
                    'error': 'simulated shard failure'}

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

    def test_replacement_shard_id_runtime_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(Path(tmpdir), h5_module=FakeH5Module())
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
                    replacement = storage._replacement_shard_id('manual_0001')
            self.assertEqual(replacement, 'manual_0003')
            events = storage.get_shard_health_events()
            self.assertTrue(any(event['event_type'] == 'stat_failure'
                for event in events))
            self.assertTrue(any(event['event_type'] ==
                'replacement_skip_unreadable_candidates' for event in events))

            storage.MAX_SCAN_SUFFIXES = 5

            def always_unreadable(shard_id, file_path, purpose):
                return {'state': 'unreadable', 'byte_size': None,
                    'error': 'simulated shard failure'}

            with mock.patch.object(storage, '_shard_path_state',
                side_effect=always_unreadable):
                with mock.patch('echoframe.output_storage.time.sleep'
                    ) as mocked_sleep:
                    with self.assertRaisesRegex(RuntimeError,
                        'unable to probe shard path'):
                        storage._replacement_shard_id('manual_0001')
            self.assertEqual([call.args[0] for call in
                mocked_sleep.call_args_list], [0.1, 0.3, 0.6, 0.9, 3.0])

    def test_replacement_shard_id_scaling_and_budget_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(Path(tmpdir), h5_module=FakeH5Module())
            for index in range(2, 103):
                (Path(tmpdir) / f'manual_{index:04d}.h5').touch()
            replacement = storage._replacement_shard_id('manual_0001')
            self.assertEqual(replacement, 'manual_0103')

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Hdf5ShardStore(Path(tmpdir), h5_module=FakeH5Module())
            original_stat = Path.stat

            def flaky_stat(path_obj, *args, **kwargs):
                path_str = str(path_obj)
                if 'manual_' in path_str:
                    shard_name = Path(path_str).stem
                    if 'manual_0002' <= shard_name <= 'manual_0006':
                        raise OSError(5, 'io error')
                return original_stat(path_obj, *args, **kwargs)

            with mock.patch.object(Path, 'stat', autospec=True,
                side_effect=flaky_stat):
                with mock.patch('echoframe.output_storage.time.sleep'):
                    replacement = storage._replacement_shard_id('manual_0001')
            self.assertEqual(replacement, 'manual_0007')

    def test_index_size_probe_fallbacks(self) -> None:
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

        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            metadata = _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]], tags=['exp-a'])
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

    def test_missing_h5py_dependency_raises_helpful_import_error(self) -> None:
        with mock.patch.dict('sys.modules', {'h5py': None}):
            with self.assertRaisesRegex(ImportError,
                'h5py is required to use Store'):
                Hdf5ShardStore(Path('tests/data/unused-shards'))
