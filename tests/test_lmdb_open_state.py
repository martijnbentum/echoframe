'''Tests for reporting cached LMDB environment state.'''

from pathlib import Path
import tempfile
import unittest

from echoframe import lmdb_helper


class TestLmdbOpenState(unittest.TestCase):

    def test_reports_false_before_opening_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'index.lmdb'

            self.assertFalse(lmdb_helper.env_is_open(path))

    def test_reports_true_until_final_shared_reference_closes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'index.lmdb'
            first = lmdb_helper.open_env(path, map_size=1 << 20)
            self.addCleanup(lmdb_helper.close_env, path)
            second = lmdb_helper.open_env(path, map_size=1 << 20)
            self.addCleanup(lmdb_helper.close_env, path)

            self.assertIs(first, second)
            self.assertTrue(lmdb_helper.env_is_open(path))

            lmdb_helper.close_env(path)
            self.assertTrue(lmdb_helper.env_is_open(path))

            lmdb_helper.close_env(path)
            self.assertFalse(lmdb_helper.env_is_open(path))

    def test_resolves_path_like_environment_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'index.lmdb'
            lmdb_helper.open_env(path, map_size=1 << 20)
            self.addCleanup(lmdb_helper.close_env, path)
            equivalent_path = path.parent / '..' / path.parent.name / path.name

            self.assertTrue(lmdb_helper.env_is_open(equivalent_path))

            lmdb_helper.close_env(equivalent_path)
            self.assertFalse(lmdb_helper.env_is_open(path))


if __name__ == '__main__':
    unittest.main()
