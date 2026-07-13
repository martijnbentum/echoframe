'''Tests for formatting helpers and store summary builders.'''

import unittest
from unittest import mock

from echoframe.util_formatting import build_store_summary


class TestBuildStoreSummary(unittest.TestCase):

    def _make_store(self):
        store = mock.Mock()
        store.index.list_shard_metadata.return_value = [{
            'entry_count': 3,
            'byte_size': 128,
        }]
        store.list_tags.return_value = ['alpha', 'beta']
        store.model_registry.model_metadatas = [object(), object()]
        return store

    def test_summary_uses_store_list_tags_signature(self):
        store = self._make_store()
        with mock.patch('echoframe.util_formatting._db_entry_count',
            return_value=3):
            summary = build_store_summary(store)

        store.list_tags.assert_called_once_with()
        self.assertEqual(summary['record_count'], 3)
        self.assertEqual(summary['tag_count'], 2)
        self.assertEqual(summary['tags'], ['alpha', 'beta'])

    def test_record_count_falls_back_to_shard_entry_counts(self):
        store = self._make_store()
        with mock.patch('echoframe.util_formatting._db_entry_count',
            return_value=None):
            summary = build_store_summary(store)

        self.assertEqual(summary['record_count'], 3)
