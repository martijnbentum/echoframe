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
    def test_index_size_probe_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FlakySizeStorage(Path(tmpdir), h5_module=FakeH5Module())
            (Path(tmpdir) / 'manual_0001.h5').touch()
            storage.force_stat_failure = True

            state = storage._shard_path_state('manual_0001',
                Path(tmpdir) / 'manual_0001.h5', purpose='rollover')
            with mock.patch('echoframe.output_storage.time.sleep'):
                replacement = storage._replacement_shard_id('manual_0000')

        self.assertEqual(state['state'], 'unreadable')
        self.assertEqual(state['byte_size'], None)
        self.assertIn('simulated stat failure', state['error'])
        self.assertEqual(replacement, 'manual_0002')
        events = storage.get_shard_health_events()
        self.assertTrue(any(event['event_type'] ==
            'replacement_skip_unreadable_candidates' for event in events))

    def test_output_storage_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
                shards_root=Path(tmpdir) / 'shards')
            storage = Hdf5ShardStore(Path(tmpdir) / 'shards',
                h5_module=FakeH5Module())
            store = Store(tmpdir, index=index, storage=storage)
            store.register_model('wav2vec2')
            phraser_key = b'phrase'.ljust(22, b'\0')
            echoframe_key = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=phraser_key, layer=3,
                collar=100)
            metadata = EchoframeMetadata(echoframe_key, model_name='wav2vec2')

            stored = storage.store_with_shard(metadata, [[1.0, 2.0]],
                shard_id='manual_0001')
            copied = storage.compact_shard('manual_0001', [stored])

        self.assertEqual(stored.dataset_path,
            f'/layer_0003/{echoframe_key.hex()}')
        self.assertEqual(copied[0].shard_id, 'manual_0002')
        self.assertEqual(sanitize_name('wav2vec2 hidden/state'),
            'wav2vec2_hidden_state')

    def test_store_does_not_create_diagnostic_log_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _put(store, phraser_key='phrase-1', collar=100,
                model_name='wav2vec2', output_type='hidden_state',
                layer=1, data=[[1.0]])
            files = sorted(path.name for path in Path(tmpdir).rglob('*')
                if path.is_file())

        self.assertTrue(all(not name.endswith('.log') for name in files))

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

    def test_missing_h5py_dependency_raises_helpful_import_error(self) -> None:
        with mock.patch.dict('sys.modules', {'h5py': None}):
            with self.assertRaisesRegex(ImportError,
                'h5py is required to use Store'):
                Hdf5ShardStore(Path('tests/data/unused-shards'))
