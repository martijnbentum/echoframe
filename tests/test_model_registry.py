'''Tests for store-owned model registry (implementation feature 2).'''

import json
import tempfile
import unittest
from pathlib import Path

from echoframe.metadata import metadata_class_for_output_type
from echoframe.store import Store


def _make_store(tmp_dir):
    return Store(root=tmp_dir)


def _write_seed(tmp_dir, data):
    path = Path(tmp_dir) / 'seed.json'
    path.write_text(json.dumps(data))
    return path


def _put(store, phraser_key, collar, model_name, output_type, layer, data,
    tags=None):
    metadata_cls = metadata_class_for_output_type(output_type)
    metadata = metadata_cls(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=layer, tags=tags)
    return store.put(metadata.echoframe_key, metadata, data)


class TestRegisterModel(unittest.TestCase):

    def test_register_returns_record_with_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            record = store.register_model('bert-base-uncased')
        self.assertEqual(record['model_name'], 'bert-base-uncased')

    def test_register_assigns_model_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            record = store.register_model('bert-base-uncased')
        self.assertIn('model_id', record)
        self.assertIsInstance(record['model_id'], int)

    def test_register_first_model_gets_id_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            record = store.register_model('bert-base-uncased')
        self.assertEqual(record['model_id'], 0)

    def test_register_second_model_gets_next_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            record = store.register_model('wav2vec2-base')
        self.assertEqual(record['model_id'], 1)

    def test_register_includes_created_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            record = store.register_model('bert-base-uncased')
        self.assertIn('created_at', record)
        self.assertIsNotNone(record['created_at'])

    def test_register_model_id_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            r1 = store.register_model('bert-base-uncased')
            retrieved = store.get_model_metadata('bert-base-uncased')
        self.assertEqual(r1['model_id'], retrieved['model_id'])

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
        self.assertEqual(record['model_name'], 'wav2vec2-base')


class TestGetModelMetadata(unittest.TestCase):

    def test_get_returns_none_for_unknown_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            result = store.get_model_metadata('unknown-model')
        self.assertIsNone(result)

    def test_get_returns_record_after_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            record = store.get_model_metadata('bert-base-uncased')
        self.assertIsNotNone(record)
        self.assertEqual(record['model_name'], 'bert-base-uncased')

    def test_get_returns_correct_model_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            registered = store.register_model('bert-base-uncased')
            retrieved = store.get_model_metadata('bert-base-uncased')
        self.assertEqual(registered['model_id'], retrieved['model_id'])

    def test_get_is_stable_across_lookups(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            r1 = store.get_model_metadata('bert-base-uncased')
            r2 = store.get_model_metadata('bert-base-uncased')
        self.assertEqual(r1['model_id'], r2['model_id'])


class TestPersistence(unittest.TestCase):

    def test_record_readable_from_second_store_on_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store1 = _make_store(tmp)
            store1.register_model('bert-base-uncased')
            # second store reuses the same LMDB env (cached by path)
            store2 = _make_store(tmp)
            record = store2.get_model_metadata('bert-base-uncased')
        self.assertIsNotNone(record)
        self.assertEqual(record['model_name'], 'bert-base-uncased')


class TestImportSeeds(unittest.TestCase):

    def test_import_valid_seed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'wav2vec2-base'},
            ])
            records = store.import_model_seeds(seed_path)
        self.assertEqual(len(records), 2)

    def test_import_assigns_sequential_model_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'wav2vec2-base'},
            ])
            records = store.import_model_seeds(seed_path)
        ids = [r['model_id'] for r in records]
        self.assertEqual(sorted(ids), list(range(len(ids))))

    def test_import_records_readable_via_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
            ])
            store.import_model_seeds(seed_path)
            record = store.get_model_metadata('bert-base-uncased')
        self.assertIsNotNone(record)
        self.assertEqual(record['model_name'], 'bert-base-uncased')

    def test_import_ids_continue_from_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('existing-model')
            seed_path = _write_seed(tmp, [{'model_name': 'new-model'}])
            records = store.import_model_seeds(seed_path)
        self.assertEqual(records[0]['model_id'], 1)

    def test_import_checked_in_seed_file(self):
        seed_path = Path(__file__).parent.parent / 'data' / 'models.json'
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            records = store.import_model_seeds(seed_path)
        self.assertGreater(len(records), 0)


class TestImportSeedConflicts(unittest.TestCase):

    def test_import_duplicate_name_in_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'bert-base-uncased'},
            ])
            with self.assertRaises(ValueError):
                store.import_model_seeds(seed_path)

    def test_import_name_already_in_store_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
            ])
            with self.assertRaises(ValueError):
                store.import_model_seeds(seed_path)

    def test_import_conflict_writes_no_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('existing-model')
            seed_path = _write_seed(tmp, [
                {'model_name': 'new-model'},
                {'model_name': 'existing-model'},
            ])
            with self.assertRaises(ValueError):
                store.import_model_seeds(seed_path)
            # new-model must not have been written
            result = store.get_model_metadata('new-model')
        self.assertIsNone(result)

    def test_import_conflict_error_mentions_conflicting_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            seed_path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
            ])
            with self.assertRaises(ValueError) as ctx:
                store.import_model_seeds(seed_path)
        self.assertIn('bert-base-uncased', str(ctx.exception))


class TestExistingStoreUnchanged(unittest.TestCase):
    '''Verify that existing store behavior is unaffected by F2.'''

    def test_put_and_find_still_work(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            data = np.zeros((4, 8), dtype='float32')
            _put(store, 'pk_abc', 0, 'bert', 'hidden_state', 0, data)
            results = store.find_phraser('pk_abc')
        self.assertEqual(len(results), 1)

    def test_registry_does_not_affect_artifact_entries(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            store.register_model('bert-base-uncased')
            data = np.zeros((4, 8), dtype='float32')
            _put(store, 'pk_abc', 0, 'bert', 'hidden_state', 0, data)
            entries = store.list_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].model_name, 'bert')


if __name__ == '__main__':
    unittest.main()
