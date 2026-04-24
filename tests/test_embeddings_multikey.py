'''Tests for multi-key embedding collection behavior.'''

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from echoframe.embeddings import Embeddings
from tests.helpers import (
    ensure_model,
    make_fake_store,
    pk as _pk,
    put as _put,
)


class TestEmbeddingsMultiKey(unittest.TestCase):
    def _make_store_with_hidden_states(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        store = make_fake_store(tmpdir.name)
        ensure_model(store, 'wav2vec2')
        data_1 = np.arange(6).reshape(2, 3).astype(float)
        data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
        data_3 = np.arange(12, 18).reshape(2, 3).astype(float)
        item_1 = _put(store, phraser_key='phrase-1', collar=500,
            model_name='wav2vec2', output_type='hidden_state', layer=4,
            data=data_1)
        item_2 = _put(store, phraser_key='phrase-2', collar=500,
            model_name='wav2vec2', output_type='hidden_state', layer=4,
            data=data_2)
        item_3 = _put(store, phraser_key='phrase-3', collar=500,
            model_name='wav2vec2', output_type='hidden_state', layer=4,
            data=data_3)
        return store, (item_1, item_2, item_3), (data_1, data_2, data_3)

    def test_preserves_key_order(self):
        store, items, data = self._make_store_with_hidden_states()

        result = Embeddings.from_echoframe_keys(store, [
            items[2].echoframe_key,
            items[0].echoframe_key,
            items[1].echoframe_key,
        ])

        self.assertEqual(result.phraser_keys, (
            _pk('phrase-3'),
            _pk('phrase-1'),
            _pk('phrase-2'),
        ))
        np.testing.assert_array_equal(result.to_numpy(), np.stack([
            data[2],
            data[0],
            data[1],
        ], axis=0))

    def test_all_invalid_value_error_keys_raise_with_skip_count(self):
        metadata = mock.Mock()
        metadata.echoframe_key = b'bad'
        metadata.phraser_key = _pk('phrase-bad')
        metadata.model_name = 'wav2vec2'
        metadata.output_type = 'hidden_state'
        metadata.layer = 4
        store = mock.Mock()
        store.load_metadata.return_value = metadata
        store.metadata_to_payload.return_value = np.zeros((2, 3, 4))

        with self.assertRaisesRegex(ValueError,
            'no embeddings were loaded skipped keys 2'):
            Embeddings.from_echoframe_keys(store, [b'bad-1', b'bad-2'])

    def test_skips_invalid_value_error_keys_and_keeps_valid_order(self):
        store, items, data = self._make_store_with_hidden_states()
        valid_store = store
        invalid_metadata = mock.Mock()
        invalid_metadata.echoframe_key = b'invalid'
        invalid_metadata.phraser_key = _pk('phrase-bad')
        invalid_metadata.model_name = 'wav2vec2'
        invalid_metadata.output_type = 'hidden_state'
        invalid_metadata.layer = 4
        store = mock.Mock()
        store.load_metadata.side_effect = lambda key: (
            invalid_metadata if key == b'invalid'
            else valid_store.load_metadata(key))
        store.metadata_to_payload.side_effect = lambda metadata: (
            np.zeros((2, 3, 4))
            if metadata.echoframe_key == b'invalid'
            else valid_store.metadata_to_payload(metadata))

        with mock.patch('builtins.print') as print_mock:
            result = Embeddings.from_echoframe_keys(store, [
                b'invalid',
                items[1].echoframe_key,
                items[0].echoframe_key,
            ])

        self.assertEqual(result.count, 2)
        self.assertEqual(result.phraser_keys, (_pk('phrase-2'), _pk('phrase-1')))
        self.assertEqual(print_mock.call_count, 1)
        np.testing.assert_array_equal(result.to_numpy(), np.stack([
            data[1],
            data[0],
        ], axis=0))


if __name__ == '__main__':
    unittest.main()
