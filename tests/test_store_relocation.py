'''Tests for same-filesystem Echoframe store relocation.'''

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from echoframe import Store, transfer
from tests import helpers


@unittest.skipUnless(importlib.util.find_spec('lmdb'),
    'lmdb is not installed')
@unittest.skipUnless(importlib.util.find_spec('h5py'),
    'h5py is not installed')
class TestStoreRelocation(unittest.TestCase):

    def _make_populated_store(self, root, external_root):
        store = Store(root, max_shard_size_bytes=1024 * 1024)
        model_path = external_root / 'models' / 'wav2vec2'
        phraser_path = external_root / 'phraser'
        model = store.register_model('wav2vec2', local_path=str(model_path),
            huggingface_id='example/wav2vec2', language='nl', size='base',
            architecture='wav2vec2')
        store.register_phraser_store('cgn-main', phraser_path)
        metadata = helpers.put(store, phraser_key='selected', collar=120,
            model_name='wav2vec2', output_type='hidden_state', layer=7,
            data=[[1.0, 2.0], [3.0, 4.0]], tags=['experiment-a'])
        return store, model, metadata

    def test_moves_complete_store_and_preserves_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / 'source'
            destination_path = root / 'destination'
            store, model, metadata = self._make_populated_store(
                source_path, root / 'external')
            config = store.model_registry.read_config_dict()
            shard_id = metadata.shard_id
            dataset_path = metadata.dataset_path
            echoframe_key = metadata.echoframe_key
            store.close()

            result = transfer.move_store(source_path, destination_path)

            self.assertFalse(source_path.exists())
            self.assertTrue(destination_path.exists())
            self.assertEqual(result['source_path'], str(source_path.resolve()))
            self.assertEqual(result['destination_path'],
                str(destination_path.resolve()))
            self.assertGreater(result['file_count'], 0)
            self.assertGreater(result['byte_count'], 0)
            self.assertTrue(result['integrity']['ok'])

            moved = Store(destination_path)
            try:
                copied = moved.load_metadata(echoframe_key)
                payload = helpers.payload_to_list(moved.load(echoframe_key))
                self.assertEqual(copied.shard_id, shard_id)
                self.assertEqual(copied.dataset_path, dataset_path)
                self.assertEqual(payload,
                    [[1.0, 2.0], [3.0, 4.0]])
                self.assertEqual(
                    moved.load_model_metadata('wav2vec2').model_id,
                    model.model_id)
                self.assertEqual(moved.model_registry.read_config_dict(),
                    config)
                self.assertEqual(moved.storage.root,
                    destination_path / 'shards')
                self.assertEqual(moved.index.shards_root,
                    destination_path / 'shards')
                added = helpers.put(moved, phraser_key='after-move',
                    collar=120, model_name='wav2vec2',
                    output_type='hidden_state', layer=7, data=[[5.0]])
                self.assertIsNotNone(moved.load(added.echoframe_key))
            finally:
                moved.close()

    def test_moves_empty_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / 'source'
            destination_path = root / 'destination'
            store = Store(source_path)
            store.close()

            result = transfer.move_store(source_path, destination_path)

            moved = Store(destination_path)
            try:
                self.assertEqual(moved.metadatas, [])
                self.assertTrue(moved.verify_integrity()['ok'])
                self.assertGreater(result['file_count'], 0)
            finally:
                moved.close()

    def test_rejects_store_open_in_current_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / 'source'
            destination_path = root / 'destination'
            store = Store(source_path)
            try:
                with self.assertRaisesRegex(ValueError, 'open'):
                    transfer.move_store(source_path, destination_path)
                self.assertTrue(source_path.exists())
                self.assertFalse(destination_path.exists())
            finally:
                store.close()

    def test_rejects_missing_and_malformed_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            destination_path = root / 'destination'
            with self.assertRaisesRegex(ValueError, 'source'):
                transfer.move_store(root / 'missing', destination_path)

            malformed = root / 'malformed'
            malformed.mkdir()
            with self.assertRaisesRegex(ValueError, 'store'):
                transfer.move_store(malformed, destination_path)

    def test_rejects_existing_or_nested_destination(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / 'source'
            store = Store(source_path)
            store.close()

            destination_path = root / 'destination'
            destination_path.mkdir()
            with self.assertRaisesRegex(ValueError, 'destination'):
                transfer.move_store(source_path, destination_path)

            nested_path = source_path / 'nested'
            with self.assertRaisesRegex(ValueError, 'inside'):
                transfer.move_store(source_path, nested_path)
            self.assertTrue(source_path.exists())

    def test_rejects_destination_with_missing_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / 'source'
            store = Store(source_path)
            store.close()
            destination_path = root / 'missing-parent' / 'destination'

            with self.assertRaisesRegex(ValueError, 'parent'):
                transfer.move_store(source_path, destination_path)

            self.assertTrue(source_path.exists())
            self.assertFalse(destination_path.exists())

    def test_preflight_integrity_failure_leaves_source_untouched(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / 'source'
            destination_path = root / 'destination'
            store, _, metadata = self._make_populated_store(
                source_path, root / 'external')
            shard_path = store.storage.root / f'{metadata.shard_id}.h5'
            store.close()
            shard_path.unlink()

            with self.assertRaisesRegex(RuntimeError, 'integrity'):
                transfer.move_store(source_path, destination_path)

            self.assertTrue(source_path.exists())
            self.assertFalse(destination_path.exists())

    def test_same_filesystem_move_does_not_copy_or_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / 'source'
            destination_path = root / 'destination'
            store = Store(source_path)
            store.close()

            with mock.patch.object(shutil, 'copytree') as copytree:
                with mock.patch.object(hashlib, 'sha256') as sha256:
                    transfer.move_store(source_path, destination_path)

            copytree.assert_not_called()
            sha256.assert_not_called()

    def test_destination_verification_failure_rolls_back_move(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / 'source'
            destination_path = root / 'destination'
            store = Store(source_path)
            store.close()
            valid = {'ok': True, 'broken_metadata_references': []}
            invalid = {'ok': False, 'broken_metadata_references': ['broken']}

            with mock.patch.object(Store, 'verify_integrity',
                side_effect=[valid, invalid]):
                with self.assertRaisesRegex(RuntimeError,
                    'destination.*integrity'):
                    transfer.move_store(source_path, destination_path)

            self.assertTrue(source_path.exists())
            self.assertFalse(destination_path.exists())
            reopened = Store(source_path)
            reopened.close()
