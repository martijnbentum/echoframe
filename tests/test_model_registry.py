'''Tests for store-owned model registry (implementation feature 2).'''

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from echoframe.model_registry import (
    ModelMetadata,
    check_model_name_conflict,
    check_model_names_conflict,
    load_model_seed_file,
)
from echoframe.phraser_registry import normalise_phraser_store_path
from echoframe.store_config import config_from_dict, config_to_dict
from echoframe.store import Store
from tests.helpers import pk as _pk


def _make_store(tmp_dir):
    return Store(root=tmp_dir)


def _write_seed(tmp_dir, data):
    path = Path(tmp_dir) / 'seed.json'
    path.write_text(json.dumps(data))
    return path


def _write_model_file(tmp_dir, data):
    path = Path(tmp_dir) / 'models.json'
    path.write_text(json.dumps(data))
    return path


def _put(store, phraser_key, collar, model_name, output_type, layer, data,
    tags=None):
    phraser_key = _pk(phraser_key)
    echoframe_key = store.make_echoframe_key(output_type,
        model_name=model_name, phraser_key=phraser_key, layer=layer,
        collar=collar)
    from echoframe.metadata import EchoframeMetadata
    metadata = EchoframeMetadata(echoframe_key, model_name=model_name,
        phraser_source_id='cgn-main', tags=tags)
    return store.save(echoframe_key, metadata, data)



class TestRegisterModel(unittest.TestCase):

    def test_register_returns_record_with_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            record = store.register_model('bert-base-uncased')
        self.assertEqual(record.model_name, 'bert-base-uncased')

    def test_register_assigns_model_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            record = store.register_model('bert-base-uncased')
        self.assertIsInstance(record.model_id, int)

    def test_register_first_model_gets_id_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            record = store.register_model('bert-base-uncased')
        self.assertEqual(record.model_id, 0)

    def test_register_second_model_gets_next_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            record = store.register_model('wav2vec2-base')
        self.assertEqual(record.model_id, 1)

    def test_register_includes_default_metadata_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            record = store.register_model('bert-base-uncased')
        self.assertEqual(record.local_path, None)
        self.assertEqual(record.huggingface_id, None)
        self.assertEqual(record.language, None)
        self.assertEqual(record.size, None)
        self.assertEqual(record.architecture, None)

    def test_register_accepts_optional_metadata_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            record = store.register_model('bert-base-uncased',
                local_path='/models/bert-base-uncased',
                huggingface_id='bert-base-uncased', language='en',
                size='base', architecture='bert')
        self.assertEqual(record.local_path, '/models/bert-base-uncased')
        self.assertEqual(record.huggingface_id, 'bert-base-uncased')
        self.assertEqual(record.language, 'en')
        self.assertEqual(record.size, 'base')
        self.assertEqual(record.architecture, 'bert')

    def test_register_model_id_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            r1 = store.register_model('bert-base-uncased')
            retrieved = store.load_model_metadata('bert-base-uncased')
        self.assertEqual(r1.model_id, retrieved.model_id)

    def test_register_empty_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            with self.assertRaises(ValueError):
                store.register_model('')

    def test_register_whitespace_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            with self.assertRaises(ValueError):
                store.register_model('   ')

    def test_register_non_string_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            with self.assertRaises((ValueError, TypeError)):
                store.register_model(42)


class TestModelMetadata(unittest.TestCase):

    def test_round_trip_dict_preserves_fields(self):
        metadata = ModelMetadata('bert-base-uncased', model_id=3,
            local_path='/models/bert', huggingface_id='bert-base-uncased',
            language='en', size='base', architecture='bert')
        restored = ModelMetadata.from_dict(metadata.to_dict())
        self.assertEqual(restored.model_name, 'bert-base-uncased')
        self.assertEqual(restored.model_id, 3)
        self.assertEqual(restored.local_path, '/models/bert')
        self.assertEqual(restored.huggingface_id, 'bert-base-uncased')
        self.assertEqual(restored.language, 'en')
        self.assertEqual(restored.size, 'base')
        self.assertEqual(restored.architecture, 'bert')

    def test_to_dict_includes_architecture(self):
        metadata = ModelMetadata('bert-base-uncased', architecture='bert')
        data = metadata.to_dict()
        self.assertEqual(data['architecture'], 'bert')

    def test_invalid_model_id_raises(self):
        with self.assertRaises(ValueError):
            ModelMetadata('bert-base-uncased', model_id=-1)

    def test_invalid_optional_string_raises(self):
        with self.assertRaises(ValueError):
            ModelMetadata('bert-base-uncased', language='   ')

    def test_invalid_architecture_raises(self):
        with self.assertRaises(ValueError):
            ModelMetadata('bert-base-uncased', architecture='   ')

    def test_invalid_size_boolean_raises(self):
        with self.assertRaises(ValueError):
            ModelMetadata('bert-base-uncased', size=True)


class TestDuplicateRegistration(unittest.TestCase):

    def test_duplicate_model_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            with self.assertRaises(ValueError):
                store.register_model('bert-base-uncased')

    def test_duplicate_error_mentions_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            with self.assertRaises(ValueError) as ctx:
                store.register_model('bert-base-uncased')
        self.assertIn('bert-base-uncased', str(ctx.exception))

    def test_different_model_names_do_not_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            record = store.register_model('wav2vec2-base')
        self.assertEqual(record.model_name, 'wav2vec2-base')


class TestGetModelMetadata(unittest.TestCase):

    def test_get_returns_none_for_unknown_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            result = store.load_model_metadata('unknown-model')
        self.assertIsNone(result)

    def test_get_returns_record_after_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            record = store.load_model_metadata('bert-base-uncased')
        self.assertIsNotNone(record)
        self.assertEqual(record.model_name, 'bert-base-uncased')

    def test_get_returns_correct_model_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            registered = store.register_model('bert-base-uncased')
            retrieved = store.load_model_metadata('bert-base-uncased')
        self.assertEqual(registered.model_id, retrieved.model_id)

    def test_get_is_stable_across_lookups(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            r1 = store.load_model_metadata('bert-base-uncased')
            r2 = store.load_model_metadata('bert-base-uncased')
        self.assertEqual(r1.model_id, r2.model_id)


class TestPersistence(unittest.TestCase):

    def test_record_readable_from_second_store_on_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store1 = _make_store(tmp)
            store1.register_model('bert-base-uncased')
            # second store reuses the same LMDB env (cached by path)
            store2 = _make_store(tmp)
            record = store2.load_model_metadata('bert-base-uncased')
        self.assertIsNotNone(record)
        self.assertEqual(record.model_name, 'bert-base-uncased')

    def test_registry_creates_config_json_next_to_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            config_path = Path(tmp) / 'config.json'
            payload = json.loads(config_path.read_text())
        self.assertEqual(payload['models']['bert-base-uncased']['model_id'], 0)
        self.assertNotIn('output_type_ids', payload)
        self.assertEqual(payload['phraser_sources'], {})


class TestPhraserStores(unittest.TestCase):

    def test_register_phraser_store_persists_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            path = store.register_phraser_store('cgn-main',
                '/data/cgn/phraser.lmdb')
            payload = json.loads((Path(tmp) / 'config.json').read_text())
        expected = normalise_phraser_store_path('/data/cgn/phraser.lmdb')
        self.assertEqual(path, expected)
        self.assertEqual(payload['phraser_sources']['cgn-main'], expected)

    def test_load_phraser_path_from_second_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = _make_store(tmp)
            first.register_phraser_store('cgn-main', '/data/cgn')
            second = _make_store(tmp)
            path = second.phraser_registry.load_path('cgn-main')
        self.assertEqual(path, normalise_phraser_store_path('/data/cgn'))

    def test_same_id_different_path_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_phraser_store('cgn-main', '/data/cgn')
            with self.assertRaisesRegex(ValueError, 'already registered'):
                store.register_phraser_store('cgn-main', '/other')

    def test_same_id_same_path_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_phraser_store('cgn-main', '/data/cgn')
            path = store.register_phraser_store('cgn-main', '/data/cgn')
        self.assertEqual(path, normalise_phraser_store_path('/data/cgn'))

    def test_normalized_same_path_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            root = Path(tmp) / 'phraser'
            root.mkdir()
            store.register_phraser_store('cgn-main', root)
            path = store.register_phraser_store('cgn-main',
                root / '..' / 'phraser')
        self.assertEqual(path, normalise_phraser_store_path(root))

    def test_second_id_may_share_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_phraser_store('cgn-main', '/data/cgn')
            store.register_phraser_store('cgn-copy', '/data/cgn')
            shared = store.phraser_registry.load_path('cgn-copy')
        self.assertEqual(shared, normalise_phraser_store_path('/data/cgn'))

    def test_phraser_store_config_rejects_non_string_path(self):
        with self.assertRaisesRegex(ValueError, 'non-empty path strings'):
            config_from_dict({'phraser_sources': {'cgn-main': {}}})

    def test_load_phraser_object_rejects_none_source_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_phraser_store('cgn-main', '/data/cgn')
            with self.assertRaisesRegex(ValueError, 'unknown phraser_source_id'):
                store.load_phraser_object(b'phrase-key', None)

    def test_load_phraser_object_rejects_unknown_source_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_phraser_store('cgn-main', '/data/cgn')
            with self.assertRaisesRegex(ValueError, 'unknown phraser_source_id'):
                store.load_phraser_object(b'phrase-key', 'missing')

    def test_attach_phraser_store_allows_runtime_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            phraser_store = type('FakePhraserStore', (), {
                'path': '/data/cgn',
                'load': lambda self, key: ('loaded', key),
            })()
            store.attach_phraser_store('cgn-main', phraser_store)
            loaded = store.load_phraser_object(b'phrase-key', 'cgn-main')
        self.assertEqual(loaded, ('loaded', b'phrase-key'))

    def test_get_store_opens_registered_phraser_store(self):
        try:
            import phraser
        except ImportError:
            self.skipTest('phraser is not installed')
        with tempfile.TemporaryDirectory() as tmp:
            phraser_path = str(Path(tmp) / 'cgn.lmdb')
            writer = phraser.Store(path=phraser_path)
            if not getattr(writer, 'CLASS_MAP', None):
                self.skipTest('installed phraser Store does not self-register')
            speaker = writer.create(phraser.Speaker, name='spk-1',
                dataset='test')
            key = speaker.key
            writer.close()

            store = _make_store(str(Path(tmp) / 'ef'))
            store.register_phraser_store('cgn-main', phraser_path)
            # no attached store: this exercises the lazy phraser.Store open path
            loaded = store.load_phraser_object(key, 'cgn-main')
        self.assertEqual(loaded.name, 'spk-1')
        self.assertEqual(loaded.key, key)

    def test_read_config_dict_returns_default_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            config = store.model_registry.read_config_dict()
        self.assertEqual(config, {'models': {}, 'phraser_sources': {}})


class TestModelMetadataListing(unittest.TestCase):

    def test_model_metadatas_returns_all_registered_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            store.register_model('wav2vec2-base')
            model_metadatas = store.model_registry.model_metadatas
        names = sorted(metadata.model_name for metadata in model_metadatas)
        self.assertEqual(names, ['bert-base-uncased', 'wav2vec2-base'])


class TestConflictHelpers(unittest.TestCase):

    def test_check_model_name_conflict_returns_true_for_existing_model(self):
        config = {'models': {
            'bert-base-uncased': ModelMetadata('bert-base-uncased', model_id=0),
        }}
        metadata = ModelMetadata('bert-base-uncased')
        self.assertTrue(check_model_name_conflict(config, metadata))

    def test_check_model_name_conflict_returns_false_for_new_model(self):
        config = {'models': {
            'bert-base-uncased': ModelMetadata('bert-base-uncased', model_id=0),
        }}
        metadata = ModelMetadata('wav2vec2-base')
        self.assertFalse(check_model_name_conflict(config, metadata))

    def test_check_model_names_conflict_returns_none_without_conflicts(self):
        config = {'models': {
            'bert-base-uncased': ModelMetadata('bert-base-uncased', model_id=0),
        }}
        metadata_list = [ModelMetadata('wav2vec2-base')]
        self.assertIsNone(check_model_names_conflict(config, metadata_list))

    def test_check_model_names_conflict_returns_conflicting_metadata(self):
        config = {'models': {
            'bert-base-uncased': ModelMetadata('bert-base-uncased', model_id=0),
        }}
        metadata = ModelMetadata('bert-base-uncased')
        conflicts = check_model_names_conflict(config, [metadata])
        self.assertEqual(conflicts, [metadata])


class TestConfigHelpers(unittest.TestCase):

    def test_config_round_trip_preserves_metadata(self):
        data = {'models': {
            'bert-base-uncased': {
                'model_id': 2,
                'local_path': '/models/bert',
                'huggingface_id': 'bert-base-uncased',
                'language': 'en',
                'size': 'base',
            },
        }}
        config = config_from_dict(data)
        restored = config_to_dict(config)
        record = restored['models']['bert-base-uncased']
        self.assertEqual(record['model_id'], 2)
        self.assertEqual(record['size'], 'base')

    def test_config_from_dict_rejects_non_dict_models(self):
        with self.assertRaises(ValueError):
            config_from_dict({'models': []})


class TestImportModels(unittest.TestCase):

    def test_import_valid_seed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'wav2vec2-base'},
            ])
            records = store.import_models(seed_path)
        self.assertEqual(len(records), 2)

    def test_import_assigns_sequential_model_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'wav2vec2-base'},
            ])
            records = store.import_models(seed_path)
        ids = [r.model_id for r in records]
        self.assertEqual(sorted(ids), list(range(len(ids))))

    def test_import_records_readable_via_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased',
                 'huggingface_id': 'bert-base-uncased',
                 'language': 'en',
                 'size': 'base',
                 'architecture': 'bert'},
            ])
            store.import_models(seed_path)
            record = store.load_model_metadata('bert-base-uncased')
        self.assertIsNotNone(record)
        self.assertEqual(record.model_name, 'bert-base-uncased')
        self.assertEqual(record.huggingface_id, 'bert-base-uncased')
        self.assertEqual(record.language, 'en')
        self.assertEqual(record.size, 'base')
        self.assertEqual(record.architecture, 'bert')

    def test_import_ids_continue_from_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('existing-model')
            seed_path = _write_seed(tmp, [{'model_name': 'new-model'}])
            records = store.import_models(seed_path)
        self.assertEqual(records[0].model_id, 1)

    def test_import_checked_in_seed_file(self):
        seed_path = Path(__file__).parent.parent / 'data' / 'models.json'
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            records = store.import_models(seed_path)
        self.assertGreater(len(records), 0)


class TestImportModelConflicts(unittest.TestCase):

    def test_import_duplicate_name_in_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'bert-base-uncased'},
            ])
            with self.assertRaises(ValueError):
                store.import_models(seed_path)

    def test_import_name_already_in_store_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
            ])
            with self.assertRaises(ValueError):
                store.import_models(seed_path)

    def test_import_conflict_writes_no_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('existing-model')
            seed_path = _write_seed(tmp, [
                {'model_name': 'new-model'},
                {'model_name': 'existing-model'},
            ])
            with self.assertRaises(ValueError):
                store.import_models(seed_path)
            # new-model must not have been written
            result = store.load_model_metadata('new-model')
        self.assertIsNone(result)

    def test_import_conflict_error_mentions_conflicting_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
            ])
            with self.assertRaises(ValueError) as ctx:
                store.import_models(seed_path)
        self.assertIn('bert-base-uncased', str(ctx.exception))


class TestExistingStoreUnchanged(unittest.TestCase):
    def test_put_and_find_still_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert')
            data = np.zeros((4, 8), dtype='float32')
            _put(store, 'pk_abc', 0, 'bert', 'hidden_state', 0, data)
            results = store.find_phraser(_pk('pk_abc'))
        self.assertEqual(len(results), 1)

    def test_registry_does_not_affect_artifact_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert')
            data = np.zeros((4, 8), dtype='float32')
            _put(store, 'pk_abc', 0, 'bert', 'hidden_state', 0, data)
            entries = store.metadatas
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].model_name, 'bert')



class TestLoadModelSeedFile(unittest.TestCase):

    def test_load_valid_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_model_file(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'wav2vec2-base'},
            ])
            records = load_model_seed_file(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].model_name, 'bert-base-uncased')
        self.assertEqual(records[1].model_name, 'wav2vec2-base')

    def test_load_uses_fixed_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_model_file(tmp, [{
                'model_name': 'bert-base-uncased',
                'local_path': '/models/bert',
                'huggingface_id': 'bert-base-uncased',
                'language': 'en',
                'size': 'base',
                'architecture': 'bert',
            }])
            records = load_model_seed_file(path)
        self.assertEqual(records[0].local_path, '/models/bert')
        self.assertEqual(records[0].huggingface_id, 'bert-base-uncased')
        self.assertEqual(records[0].language, 'en')
        self.assertEqual(records[0].size, 'base')
        self.assertEqual(records[0].architecture, 'bert')

    def test_load_checked_in_model_file(self):
        path = Path(__file__).parent.parent / 'data' / 'models.json'
        records = load_model_seed_file(path)
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIsNotNone(record.model_name)

    def test_not_a_list_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_model_file(tmp, {'model_name': 'bert-base-uncased'})
            with self.assertRaises(ValueError):
                load_model_seed_file(path)

    def test_record_not_a_dict_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_model_file(tmp, ['bert-base-uncased'])
            with self.assertRaises(ValueError):
                load_model_seed_file(path)

    def test_missing_model_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_model_file(tmp, [{'language': 'en'}])
            with self.assertRaises(ValueError):
                load_model_seed_file(path)

    def test_invalid_optional_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_model_file(tmp, [{
                'model_name': 'bert-base-uncased',
                'language': '',
            }])
            with self.assertRaises(ValueError):
                load_model_seed_file(path)

    def test_duplicate_model_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_model_file(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'bert-base-uncased'},
            ])
            with self.assertRaises(ValueError):
                load_model_seed_file(path)


if __name__ == '__main__':
    unittest.main()
