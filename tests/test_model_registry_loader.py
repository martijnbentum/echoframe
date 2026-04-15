'''Tests for model_registry_loader seed file loading and validation.'''

import json
import tempfile
import unittest
from pathlib import Path

from echoframe.model_registry_loader import load_seed_file


def _write_seed(tmp_dir, data):
    path = Path(tmp_dir) / 'models.json'
    path.write_text(json.dumps(data))
    return path


class TestLoadSeedFileValid(unittest.TestCase):

    def test_load_valid_seed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'wav2vec2-base'},
            ])
            records = load_seed_file(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['model_name'], 'bert-base-uncased')
        self.assertEqual(records[1]['model_name'], 'wav2vec2-base')

    def test_load_returns_list_of_dicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [{'model_name': 'bert-base-uncased'}])
            records = load_seed_file(path)
        self.assertIsInstance(records, list)
        self.assertIsInstance(records[0], dict)

    def test_load_preserves_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased', 'description': 'BERT'}
            ])
            records = load_seed_file(path)
        self.assertEqual(records[0]['description'], 'BERT')

    def test_load_returns_copies(self):
        original = [{'model_name': 'bert-base-uncased', 'extra': 1}]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, original)
            records = load_seed_file(path)
        records[0]['extra'] = 99
        self.assertEqual(original[0]['extra'], 1)

    def test_load_checked_in_seed_file(self):
        seed_path = Path(__file__).parent.parent / 'data' / 'models.json'
        records = load_seed_file(seed_path)
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIn('model_name', record)


class TestLoadSeedFileInvalidStructure(unittest.TestCase):

    def test_not_a_list_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, {'model_name': 'bert-base-uncased'})
            with self.assertRaises(ValueError):
                load_seed_file(path)

    def test_record_not_a_dict_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, ['bert-base-uncased'])
            with self.assertRaises(ValueError):
                load_seed_file(path)

    def test_missing_model_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [{'description': 'no name'}])
            with self.assertRaises(ValueError):
                load_seed_file(path)

    def test_empty_model_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [{'model_name': ''}])
            with self.assertRaises(ValueError):
                load_seed_file(path)

    def test_whitespace_only_model_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [{'model_name': '   '}])
            with self.assertRaises(ValueError):
                load_seed_file(path)

    def test_non_string_model_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [{'model_name': 42}])
            with self.assertRaises(ValueError):
                load_seed_file(path)


class TestLoadSeedFileDuplicates(unittest.TestCase):

    def test_duplicate_model_name_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'bert-base-uncased'},
            ])
            with self.assertRaises(ValueError):
                load_seed_file(path)

    def test_duplicate_error_message_contains_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'bert-base-uncased'},
            ])
            with self.assertRaises(ValueError) as ctx:
                load_seed_file(path)
        self.assertIn('bert-base-uncased', str(ctx.exception))

    def test_unique_model_names_do_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_seed(tmp, [
                {'model_name': 'bert-base-uncased'},
                {'model_name': 'wav2vec2-base'},
            ])
            records = load_seed_file(path)
        self.assertEqual(len(records), 2)


if __name__ == '__main__':
    unittest.main()
