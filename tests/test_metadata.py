'''Tests for EchoframeMetadata helpers.'''

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from echoframe.embeddings import Embedding
from tests.helpers import make_fake_store, put as _put


class TestLoadEmbedding(unittest.TestCase):
    def _hidden_state_metadata(self, store, data, layer=4):
        item = _put(store, phraser_key='phrase-1', collar=500,
            model_name='wav2vec2', output_type='hidden_state', layer=layer,
            data=data)
        return store.load_metadata(item.echoframe_key)

    def test_returns_embedding_for_hidden_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            data = np.arange(6).reshape(2, 3).astype(float)
            metadata = self._hidden_state_metadata(store, data)

            result = metadata.load_embedding()

            self.assertIsInstance(result, Embedding)
            self.assertEqual(result.echoframe_key, metadata.echoframe_key)
            np.testing.assert_array_equal(result.data, data)

    def test_passes_self_through_as_embedding_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            data = np.arange(6).reshape(2, 3).astype(float)
            metadata = self._hidden_state_metadata(store, data)

            result = metadata.load_embedding()

            self.assertIs(result.metadata, metadata)

    def test_raises_if_store_is_not_attached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            data = np.arange(6).reshape(2, 3).astype(float)
            metadata = self._hidden_state_metadata(store, data)
            metadata.store = None

            with self.assertRaisesRegex(ValueError,
                'store is not attached to metadata'):
                metadata.load_embedding()

    def test_raises_early_for_non_hidden_state_output_type(self):
        cases = {
            'attention': dict(layer=4, data=np.zeros((1, 2, 2)).astype(float)),
            'codebook_indices': dict(layer=None,
                data=np.arange(4).reshape(2, 2).astype('int64')),
        }
        for output_type, kwargs in cases.items():
            with self.subTest(output_type=output_type):
                with tempfile.TemporaryDirectory() as tmpdir:
                    store = make_fake_store(tmpdir)
                    item = _put(store, phraser_key='phrase-1', collar=500,
                        model_name='wav2vec2', output_type=output_type,
                        layer=kwargs['layer'], data=kwargs['data'])
                    metadata = store.load_metadata(item.echoframe_key)

                    message = ('output_type must be hidden_state to load an '
                        f'Embedding, got {output_type}')
                    with self.assertRaisesRegex(ValueError, message):
                        metadata.load_embedding()


if __name__ == '__main__':
    unittest.main()
