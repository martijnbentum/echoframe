'''Tests for formatting helpers and store summary builders.'''

import unittest
from unittest import mock

from echoframe.util_formatting import build_store_summary


class TestBuildStoreSummary(unittest.TestCase):

    def test_summary_uses_store_list_tags_signature(self):
        store = mock.Mock()
        store.index.list_shard_metadata.return_value = [{
            'live_entry_count': 1,
            'deleted_entry_count': 0,
            'byte_size': 128,
        }]
        store.list_tags.return_value = ['alpha', 'beta']
        store.registry.model_metadatas = [object(), object()]
        with mock.patch('echoframe.util_formatting._db_entry_count',
            return_value=1):
            summary = build_store_summary(store)

        store.list_tags.assert_called_once_with()
        self.assertEqual(summary['record_count'], 1)
        self.assertEqual(summary['live_record_count'], 1)
        self.assertEqual(summary['deleted_record_count'], 0)
        self.assertEqual(summary['tag_count'], 2)
        self.assertEqual(summary['tags'], ['alpha', 'beta'])
