'''Tests for typed Store loaders.'''

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from echoframe.embeddings import Embedding, Embeddings
from tests.helpers import (
    make_fake_store,
    pk as _pk,
    put as _put,
)


def _make_store():
    tmpdir = tempfile.TemporaryDirectory()
    store = make_fake_store(tmpdir.name)
    return tmpdir, store


def _put_hidden_state(store, phraser_key, collar, model_name, layer, data):
    _put(store, phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='hidden_state', layer=layer, data=data)


class TestLoadEmbedding(unittest.TestCase):
    def test_returns_embedding_for_one_echoframe_key(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data = np.arange(6).reshape(2, 3).astype(float)
            phraser_key = _pk('phrase-1')
            _put_hidden_state(store, phraser_key, 500, 'wav2vec2', 4, data)
            echoframe_key = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=phraser_key, layer=4,
                collar=500)

            result = store.load_embedding(echoframe_key)

            self.assertIsInstance(result, Embedding)
            self.assertEqual(result.phraser_key, phraser_key)
            self.assertEqual(result.model_name, 'wav2vec2')
            self.assertEqual(result.layer, 4)
            self.assertEqual(result.shape, (2, 3))
            np.testing.assert_array_equal(result.data, data)


class TestLoadEmbeddings(unittest.TestCase):
    def test_returns_embeddings_for_many_echoframe_keys(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data_1 = np.arange(6).reshape(2, 3).astype(float)
            data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
            phraser_key_1 = _pk('phrase-1')
            phraser_key_2 = _pk('phrase-2')
            _put_hidden_state(store, phraser_key_1, 500, 'wav2vec2', 3, data_1)
            _put_hidden_state(store, phraser_key_2, 500, 'wav2vec2', 3, data_2)
            echoframe_key_1 = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=phraser_key_1, layer=3,
                collar=500)
            echoframe_key_2 = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=phraser_key_2, layer=3,
                collar=500)

            result = store.load_embeddings([echoframe_key_1, echoframe_key_2])

            self.assertIsInstance(result, Embeddings)
            self.assertEqual(result.count, 2)
            self.assertEqual(result.phraser_keys,
                (phraser_key_1, phraser_key_2))
            self.assertEqual(result.model_name, 'wav2vec2')
            self.assertEqual(result.output_type, 'hidden_state')
            self.assertEqual(result.layer, 3)
            np.testing.assert_array_equal(result.to_numpy(), np.stack([
                data_1,
                data_2,
            ], axis=0))

    def test_preserves_requested_echoframe_key_order(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data_1 = np.arange(6).reshape(2, 3).astype(float)
            data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
            phraser_key_1 = _pk('phrase-1')
            phraser_key_2 = _pk('phrase-2')
            _put_hidden_state(store, phraser_key_1, 500, 'wav2vec2', 3, data_1)
            _put_hidden_state(store, phraser_key_2, 500, 'wav2vec2', 3, data_2)
            echoframe_key_1 = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=phraser_key_1, layer=3,
                collar=500)
            echoframe_key_2 = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=phraser_key_2, layer=3,
                collar=500)

            result = store.load_embeddings([echoframe_key_2, echoframe_key_1])

            self.assertEqual(result.count, 2)
            self.assertEqual(result.phraser_keys,
                (phraser_key_2, phraser_key_1))
            np.testing.assert_array_equal(result.to_numpy(), np.stack([
                data_2,
                data_1,
            ], axis=0))


if __name__ == '__main__':
    unittest.main()
