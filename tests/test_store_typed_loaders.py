'''Tests for typed Store loaders.'''

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from echoframe import Store
from echoframe.index import LmdbIndex
from echoframe.output_storage import Hdf5ShardStore
from tests.test_public_api import FakeEnv, FakeH5Module


def _make_store():
    tmpdir = tempfile.TemporaryDirectory()
    index = LmdbIndex(Path(tmpdir.name) / 'index', env=FakeEnv(),
        shards_root=Path(tmpdir.name) / 'shards')
    storage = Hdf5ShardStore(Path(tmpdir.name) / 'shards',
        h5_module=FakeH5Module())
    store = Store(tmpdir.name, index=index, storage=storage)
    return tmpdir, store


def _put_hidden_state(store, phraser_key, collar, model_name, layer, data):
    store.put(phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='hidden_state', layer=layer, data=data)


def _put_codebook(store, phraser_key, collar, model_name, indices, matrix):
    store.put(phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='codebook_indices', layer=0, data=indices)
    store.put(phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='codebook_matrix', layer=0, data=matrix)


class TestLoadEmbeddings(unittest.TestCase):
    def test_single_layer_returns_existing_shape(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data = np.arange(6).reshape(2, 3).astype(float)
            _put_hidden_state(store, 'phrase-1', 500, 'wav2vec2', 4, data)

            result = store.load_embeddings('phrase-1', 500, 'wav2vec2', 4)

            self.assertEqual(result.echoframe_keys, (result.echoframe_key,))
            self.assertEqual(result.dims, ('frames', 'embed_dim'))
            self.assertIsNone(result.layers)
            np.testing.assert_array_equal(result.to_numpy(), data)

    def test_multi_layer_returns_existing_shape(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data_1 = np.arange(6).reshape(2, 3).astype(float)
            data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
            _put_hidden_state(store, 'phrase-1', 500, 'wav2vec2', 3, data_1)
            _put_hidden_state(store, 'phrase-1', 500, 'wav2vec2', 7, data_2)

            result = store.load_embeddings('phrase-1', 500, 'wav2vec2',
                [3, 7])

            self.assertEqual(result.echoframe_keys, (
                result.key_for_layer(3),
                result.key_for_layer(7),
            ))
            self.assertEqual(result.dims, ('layers', 'frames', 'embed_dim'))
            self.assertEqual(result.layers, (3, 7))
            np.testing.assert_array_equal(result.to_numpy(), np.stack([
                data_1,
                data_2,
            ], axis=0))

    def test_single_layer_mean_aggregation(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data = np.array([
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0],
            ])
            _put_hidden_state(store, 'phrase-1', 500, 'wav2vec2', 4, data)

            result = store.load_embeddings('phrase-1', 500, 'wav2vec2', 4,
                frame_aggregation='mean')

            self.assertEqual(result.dims, ('embed_dim',))
            self.assertIsNone(result.layers)
            self.assertEqual(result.frame_aggregation, 'mean')
            np.testing.assert_array_equal(result.to_numpy(),
                np.array([3.0, 4.0, 5.0]))

    def test_multi_layer_centroid_aggregation(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data_1 = np.array([
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ])
            data_2 = np.array([
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ])
            _put_hidden_state(store, 'phrase-1', 500, 'wav2vec2', 3, data_1)
            _put_hidden_state(store, 'phrase-1', 500, 'wav2vec2', 9, data_2)

            result = store.load_embeddings('phrase-1', 500, 'wav2vec2',
                [3, 9], frame_aggregation='centroid')

            self.assertEqual(result.dims, ('layers', 'embed_dim'))
            self.assertEqual(result.layers, (3, 9))
            np.testing.assert_array_equal(result.to_numpy(), np.array([
                [3.0, 4.0],
                [9.0, 10.0],
            ]))


class TestLoadManyEmbeddings(unittest.TestCase):
    def test_deduplicates_before_loading_and_preserves_order(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data_1 = np.arange(6).reshape(2, 3).astype(float)
            data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
            _put_hidden_state(store, 'phrase-1', 500, 'wav2vec2', 3, data_1)
            _put_hidden_state(store, 'phrase-1', 500, 'wav2vec2', 7, data_2)
            _put_hidden_state(store, 'phrase-2', 500, 'wav2vec2', 3, data_1)
            _put_hidden_state(store, 'phrase-2', 500, 'wav2vec2', 7, data_2)

            calls = []
            original = store.load_with_echoframe_key

            def counting_load(echoframe_key):
                calls.append(echoframe_key)
                return original(echoframe_key)

            store.load_with_echoframe_key = counting_load

            result = store.load_many_embeddings([
                {
                    'phraser_key': 'phrase-1',
                    'collar': 500,
                    'model_name': 'wav2vec2',
                    'layers': [3, 7],
                },
                {
                    'phraser_key': 'phrase-2',
                    'collar': 500,
                    'model_name': 'wav2vec2',
                    'layers': [3, 7],
                },
                {
                    'phraser_key': 'phrase-1',
                    'collar': 500,
                    'model_name': 'wav2vec2',
                    'layers': [3, 7],
                },
            ])

            self.assertEqual(result.token_count, 2)
            self.assertEqual(result.echoframe_keys, (
                result.tokens[0].echoframe_key,
                result.tokens[1].echoframe_key,
            ))
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(set(calls)), 4)


class TestLoadCodebook(unittest.TestCase):
    def test_load_codebook_returns_store_bound_object(self):
        tmpdir, store = _make_store()
        with tmpdir:
            indices = np.array([[1, 0], [0, 1]])
            matrix = np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ])
            _put_codebook(store, 'phrase-1', 500, 'spidr', indices, matrix)

            result = store.load_codebook('phrase-1', 500, 'spidr')

            metadata = store.find_one(phraser_key='phrase-1', collar=500,
                model_name='spidr', output_type='codebook_indices', layer=0)
            self.assertEqual(result.echoframe_key, metadata.entry_id)
            self.assertIs(result.store, store)
            self.assertEqual(result.model_architecture, 'spidr')
            np.testing.assert_array_equal(result.to_numpy(), indices)
            np.testing.assert_array_equal(result.codevectors[0], np.array([
                [3.0, 4.0],
                [5.0, 6.0],
            ]))


class TestLoadManyCodebooks(unittest.TestCase):
    def test_deduplicates_before_loading_and_preserves_order(self):
        tmpdir, store = _make_store()
        with tmpdir:
            indices_1 = np.array([[0, 1]])
            indices_2 = np.array([[1, 0]])
            matrix_1 = np.array([[1.0, 2.0], [3.0, 4.0]])
            matrix_2 = np.array([[5.0, 6.0], [7.0, 8.0]])
            _put_codebook(store, 'phrase-1', 500, 'wav2vec2', indices_1,
                matrix_1)
            _put_codebook(store, 'phrase-2', 500, 'wav2vec2', indices_2,
                matrix_2)

            calls = []
            original = store.load_with_echoframe_key

            def counting_load(echoframe_key):
                calls.append(echoframe_key)
                return original(echoframe_key)

            store.load_with_echoframe_key = counting_load

            result = store.load_many_codebooks([
                {
                    'phraser_key': 'phrase-1',
                    'collar': 500,
                    'model_name': 'wav2vec2',
                },
                {
                    'phraser_key': 'phrase-2',
                    'collar': 500,
                    'model_name': 'wav2vec2',
                },
                {
                    'phraser_key': 'phrase-1',
                    'collar': 500,
                    'model_name': 'wav2vec2',
                },
            ])

            self.assertEqual(result.token_count, 2)
            self.assertEqual(result.echoframe_keys, (
                result.tokens[0].echoframe_key,
                result.tokens[1].echoframe_key,
            ))
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(set(calls)), 2)
            np.testing.assert_array_equal(result.tokens[0].to_numpy(),
                indices_1)
            np.testing.assert_array_equal(result.tokens[1].to_numpy(),
                indices_2)


if __name__ == '__main__':
    unittest.main()
