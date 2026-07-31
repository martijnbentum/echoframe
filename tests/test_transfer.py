'''Tests for model-selected hidden-state store transfers.'''

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from echoframe import transfer
from echoframe.metadata import EchoframeMetadata
from tests import helpers


class TestHiddenStateTransfer(unittest.TestCase):

    def _make_stores(self, root):
        source = helpers.make_fake_store(str(root / 'source'))
        destination = helpers.make_fake_store(str(root / 'destination'))
        source.register_model('other-model', huggingface_id='example/other')
        source.register_model('wav2vec2', huggingface_id='example/wav2vec2',
            language='nl', size='base', architecture='wav2vec2')
        source.register_phraser_store('cgn-main', root / 'phraser-cgn')
        source.register_phraser_store('unused-source',
            root / 'phraser-unused')
        return source, destination

    def _put_source_records(self, source):
        selected = helpers.put(source, phraser_key='selected', collar=120,
            model_name='wav2vec2', output_type='hidden_state', layer=7,
            data=[[1.0, 2.0], [3.0, 4.0]], tags=['experiment-a'])
        same_model_other_output = helpers.put(source, phraser_key='attention',
            collar=120, model_name='wav2vec2', output_type='attention',
            layer=7, data=[[5.0]])
        other_model = helpers.put(source, phraser_key='other', collar=120,
            model_name='other-model', output_type='hidden_state', layer=7,
            data=[[6.0]])
        return selected, same_model_other_output, other_model

    def test_copy_selects_model_hidden_states_and_rebuilds_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source, destination = self._make_stores(Path(tmpdir))
            selected, other_output, other_model = self._put_source_records(
                source)
            source_model_id = source.load_model_metadata(
                'wav2vec2').model_id

            result = transfer.copy_hidden_states_for_model(source, destination,
                'wav2vec2', batch_size=1)

            destination_model = destination.load_model_metadata('wav2vec2')
            copied_key = destination.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=selected.phraser_key,
                layer=selected.layer, collar=selected.collar)
            copied = destination.load_metadata(copied_key)
            copied_payload = helpers.payload_to_list(
                destination.load(copied_key))
            destination_sources = destination.phraser_registry.source_ids()

            self.assertNotEqual(source_model_id, destination_model.model_id)
            self.assertNotEqual(selected.echoframe_key, copied_key)
            self.assertEqual(len(destination.metadatas), 1)
            self.assertEqual(copied.model_name, selected.model_name)
            self.assertEqual(copied.phraser_source_id,
                selected.phraser_source_id)
            self.assertEqual(copied.tags, selected.tags)
            self.assertEqual(copied.created_at, selected.created_at)
            self.assertEqual(copied_payload,
                [[1.0, 2.0], [3.0, 4.0]])
            self.assertIsNotNone(source.load_metadata(
                selected.echoframe_key))
            self.assertIsNotNone(source.load_metadata(
                other_output.echoframe_key))
            self.assertIsNotNone(source.load_metadata(
                other_model.echoframe_key))
            self.assertEqual(destination_sources, ['cgn-main'])
            self.assertEqual(result['copied_count'], 1)
            self.assertEqual(result['deleted_count'], 0)
            self.assertEqual(result['model_name'], 'wav2vec2')

    def test_copy_preserves_model_registration_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source, destination = self._make_stores(Path(tmpdir))
            self._put_source_records(source)

            transfer.copy_hidden_states_for_model(source, destination,
                'wav2vec2')

            source_model = source.load_model_metadata('wav2vec2')
            destination_model = destination.load_model_metadata('wav2vec2')

        self.assertEqual(destination_model.local_path, source_model.local_path)
        self.assertEqual(destination_model.huggingface_id,
            source_model.huggingface_id)
        self.assertEqual(destination_model.language, source_model.language)
        self.assertEqual(destination_model.size, source_model.size)
        self.assertEqual(destination_model.architecture,
            source_model.architecture)

    def test_copy_copies_each_referenced_phraser_registration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, destination = self._make_stores(root)
            source.register_phraser_store('ifadv-main', root / 'phraser-ifadv')
            first = helpers.put(source, phraser_key='first', collar=100,
                model_name='wav2vec2', output_type='hidden_state', layer=3,
                data=[[1.0]])
            key = source.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=helpers.pk('second'),
                layer=3, collar=100)
            metadata = EchoframeMetadata(key, store=source,
                model_name='wav2vec2', phraser_source_id='ifadv-main')
            second = source.save(key, metadata, [[2.0]])

            transfer.copy_hidden_states_for_model(source, destination,
                'wav2vec2')

            self.assertEqual(destination.phraser_registry.source_ids(),
                ['cgn-main', 'ifadv-main'])
            for source_id in destination.phraser_registry.source_ids():
                self.assertEqual(
                    destination.phraser_registry.load_path(source_id),
                    source.phraser_registry.load_path(source_id))
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)

    def test_move_deletes_index_entries_and_complete_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source, destination = self._make_stores(Path(tmpdir))
            selected, other_output, other_model = self._put_source_records(
                source)
            selected_shard = selected.shard_id
            selected_file = source.storage.root / f'{selected_shard}.h5'
            other_files = {
                source.storage.root / f'{other_output.shard_id}.h5',
                source.storage.root / f'{other_model.shard_id}.h5',
            }

            with mock.patch.object(source, 'compact_shards') as compact:
                result = transfer.move_hidden_states_for_model(
                    source, destination, 'wav2vec2')

            compact.assert_not_called()
            self.assertIsNone(source.load_metadata(selected.echoframe_key))
            self.assertFalse(selected_file.exists())
            self.assertTrue(all(path.exists() for path in other_files))
            self.assertIsNotNone(source.load_metadata(
                other_output.echoframe_key))
            self.assertIsNotNone(source.load_metadata(
                other_model.echoframe_key))
            self.assertIsNotNone(source.load_model_metadata('wav2vec2'))
            self.assertEqual(source.phraser_registry.source_ids(),
                ['cgn-main', 'unused-source'])
            self.assertEqual(result['copied_count'], 1)
            self.assertEqual(result['deleted_count'], 1)
            self.assertEqual(result['deleted_shard_count'], 1)
            report = source.verify_integrity()
            self.assertEqual(report['unreferenced_shard_files'], [])

    def test_move_refuses_a_shard_with_an_unselected_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source, destination = self._make_stores(Path(tmpdir))
            selected, _, other_model = self._put_source_records(source)
            mixed = other_model.copy(shard_id=selected.shard_id)
            source.index.save(mixed)

            with self.assertRaisesRegex(ValueError, 'shard'):
                transfer.move_hidden_states_for_model(source, destination,
                    'wav2vec2')

            self.assertEqual(destination.metadatas, [])
            self.assertIsNotNone(source.load_metadata(
                selected.echoframe_key))
            self.assertIsNotNone(source.load_metadata(
                other_model.echoframe_key))

    def test_move_keeps_source_when_destination_verification_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source, destination = self._make_stores(Path(tmpdir))
            selected, _, _ = self._put_source_records(source)
            shard_file = source.storage.root / f'{selected.shard_id}.h5'

            report = {'ok': False, 'broken_metadata_references': ['broken']}
            with mock.patch.object(destination, 'verify_integrity',
                return_value=report):
                with self.assertRaisesRegex(RuntimeError, 'integrity'):
                    transfer.move_hidden_states_for_model(source, destination,
                        'wav2vec2')

            self.assertIsNotNone(source.load_metadata(
                selected.echoframe_key))
            self.assertTrue(shard_file.exists())

    def test_validation_rejects_invalid_transfer_requests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, destination = self._make_stores(root)
            helpers.put(source, phraser_key='selected', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=7,
                data=[[1.0]])

            with self.assertRaisesRegex(ValueError, 'different stores'):
                transfer.copy_hidden_states_for_model(
                    source, source, 'wav2vec2')
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                transfer.copy_hidden_states_for_model(
                    source, destination, 'wav2vec2', batch_size=0)
            with self.assertRaisesRegex(ValueError, 'not registered'):
                transfer.copy_hidden_states_for_model(
                    source, destination, 'missing')

            destination.register_model('existing')
            with self.assertRaisesRegex(ValueError, 'destination.*empty'):
                transfer.copy_hidden_states_for_model(
                    source, destination, 'wav2vec2')

    def test_copy_rejects_model_without_hidden_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source, destination = self._make_stores(Path(tmpdir))
            helpers.put(source, phraser_key='attention', collar=120,
                model_name='wav2vec2', output_type='attention', layer=7,
                data=[[1.0]])

            with self.assertRaisesRegex(ValueError, 'no hidden_state'):
                transfer.copy_hidden_states_for_model(
                    source, destination, 'wav2vec2')


@unittest.skipUnless(importlib.util.find_spec('lmdb'),
    'lmdb is not installed')
@unittest.skipUnless(importlib.util.find_spec('h5py'),
    'h5py is not installed')
class TestRealHiddenStateTransfer(unittest.TestCase):

    def test_move_transfers_payload_and_removes_real_source_shard(self):
        source_tmpdir, source = helpers.make_real_store()
        destination_tmpdir, destination = helpers.make_real_store()
        with source_tmpdir, destination_tmpdir:
            source.register_model('wav2vec2', architecture='wav2vec2')
            phraser_path = Path(source_tmpdir.name) / 'phraser'
            source.register_phraser_store('cgn-main', phraser_path)
            selected = helpers.put(source, phraser_key='selected', collar=120,
                model_name='wav2vec2', output_type='hidden_state', layer=7,
                data=[[1.0, 2.0]])
            source_shard = source.storage.root / f'{selected.shard_id}.h5'

            result = transfer.move_hidden_states_for_model(
                source, destination, 'wav2vec2')
            destination_key = destination.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=selected.phraser_key,
                layer=selected.layer, collar=selected.collar)

            self.assertEqual(result['copied_count'], 1)
            self.assertEqual(result['deleted_count'], 1)
            self.assertFalse(source_shard.exists())
            self.assertIsNone(source.load_metadata(selected.echoframe_key))
            payload = helpers.payload_to_list(
                destination.load(destination_key))
            self.assertEqual(payload, [[1.0, 2.0]])
            self.assertTrue(destination.verify_integrity()['ok'])
