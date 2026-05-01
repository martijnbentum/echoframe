'''Tests for codevector containers and reconstruction.'''

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from echoframe import Codevector, Codevectors
from echoframe.metadata import EchoframeMetadata
from tests.helpers import make_fake_store, make_key, pk as _pk, put as _put


def _save_codebook_matrix(store, model_name, data):
    matrix_key = make_key(store, phraser_key='unused', collar=0,
        model_name=model_name, output_type='codebook_matrix', layer=0)
    metadata = EchoframeMetadata(matrix_key, model_name=model_name)
    store.save(matrix_key, metadata, np.asarray(data))
    return matrix_key


def _save_codebook_indices(store, phraser_key, model_name, data, collar=500):
    metadata = _put(store, phraser_key=phraser_key, collar=collar,
        model_name=model_name, output_type='codebook_indices', layer=0,
        data=np.asarray(data))
    return metadata.echoframe_key


class TestCodevector(unittest.TestCase):
    def test_wav2vec2_indices_and_vectors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _save_codebook_matrix(store, 'wav2vec2', [
                [1.0, 2.0],
                [3.0, 4.0],
            ])
            key = _save_codebook_indices(store, 'phrase-1', 'wav2vec2',
                [[0, 1], [1, 0]])

            result = store.load_codevector(key)
            vectors = result.vectors

        self.assertIsInstance(result, Codevector)
        self.assertEqual(result.phraser_key, _pk('phrase-1'))
        self.assertEqual(result.model_name, 'wav2vec2')
        self.assertEqual(result.output_type, 'codebook_indices')
        self.assertEqual(result.model_architecture, 'wav2vec2')
        np.testing.assert_array_equal(result.indices, np.array([
            [0, 1],
            [1, 0],
        ]))
        np.testing.assert_array_equal(result.to_numpy(), result.indices)
        np.testing.assert_array_equal(vectors, np.array([
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 1.0, 2.0],
        ]))

    def test_wav2vec2_single_frame_indices_are_2d(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _save_codebook_matrix(store, 'wav2vec2', [
                [1.0, 2.0],
                [3.0, 4.0],
            ])
            key = _save_codebook_indices(store, 'phrase-1', 'wav2vec2',
                [0, 1])

            result = store.load_codevector(key)

        np.testing.assert_array_equal(result.indices, np.array([[0, 1]]))

    def test_spidr_vectors_preserve_frame_major_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _save_codebook_matrix(store, 'spidr', [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ])
            key = _save_codebook_indices(store, 'phrase-1', 'spidr',
                [[1, 0], [0, 1]])

            result = store.load_codevector(key)
            vectors = result.vectors

        self.assertEqual(result.model_architecture, 'spidr')
        self.assertEqual(vectors.shape, (2, 2, 2))
        np.testing.assert_array_equal(vectors[0], np.array([
            [3.0, 4.0],
            [5.0, 6.0],
        ]))
        np.testing.assert_array_equal(vectors[1], np.array([
            [1.0, 2.0],
            [7.0, 8.0],
        ]))

    def test_matrix_payload_is_cached_for_single_codevector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _save_codebook_matrix(store, 'wav2vec2', [
                [1.0, 2.0],
                [3.0, 4.0],
            ])
            key = _save_codebook_indices(store, 'phrase-1', 'wav2vec2',
                [0, 1])
            result = store.load_codevector(key)
            original_load = store.load
            calls = []

            def counting_load(echoframe_key):
                calls.append(echoframe_key)
                return original_load(echoframe_key)

            store.load = counting_load
            first = result.codebook_matrix
            second = result.codebook_matrix

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(calls), 1)

    def test_invalid_indices_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _save_codebook_matrix(store, 'wav2vec2', [
                [1.0, 2.0],
                [3.0, 4.0],
            ])
            key = _save_codebook_indices(store, 'phrase-1', 'wav2vec2',
                [0, 1, 2])

            with self.assertRaisesRegex(ValueError,
                'wav2vec2 codebook indices must contain two indices'):
                store.load_codevector(key)


class TestCodevectors(unittest.TestCase):
    def test_to_numpy_and_indices_stack_uniform_codevectors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _save_codebook_matrix(store, 'wav2vec2', [
                [1.0],
                [2.0],
            ])
            first = _save_codebook_indices(store, 'phrase-1', 'wav2vec2',
                [[0, 1]])
            second = _save_codebook_indices(store, 'phrase-2', 'wav2vec2',
                [[1, 0]])

            result = store.load_codevectors([first, second])

        self.assertIsInstance(result, Codevectors)
        self.assertEqual(result.count, 2)
        self.assertEqual(result.phraser_keys,
            (_pk('phrase-1'), _pk('phrase-2')))
        expected = np.array([[[0, 1]], [[1, 0]]])
        np.testing.assert_array_equal(result.indices, expected)
        np.testing.assert_array_equal(result.data, expected)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_vectors_uses_shared_matrix_for_collection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _save_codebook_matrix(store, 'wav2vec2', [
                [1.0],
                [2.0],
            ])
            first = _save_codebook_indices(store, 'phrase-1', 'wav2vec2',
                [[0, 1]])
            second = _save_codebook_indices(store, 'phrase-2', 'wav2vec2',
                [[1, 0]])
            result = store.load_codevectors([first, second])
            original_load = store.load
            calls = []

            def counting_load(echoframe_key):
                calls.append(echoframe_key)
                return original_load(echoframe_key)

            store.load = counting_load
            vectors = result.vectors

        np.testing.assert_array_equal(vectors, np.array([
            [[1.0, 2.0]],
            [[2.0, 1.0]],
        ]))
        self.assertEqual(len(calls), 1)

    def test_duplicate_phraser_key_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            _save_codebook_matrix(store, 'wav2vec2', [
                [1.0],
                [2.0],
            ])
            first = _save_codebook_indices(store, 'phrase-1', 'wav2vec2',
                [[0, 1]], collar=100)
            second = _save_codebook_indices(store, 'phrase-1', 'wav2vec2',
                [[1, 0]], collar=200)

            with self.assertRaisesRegex(ValueError, 'duplicate phraser_key'):
                store.load_codevectors([first, second])

    def test_load_codevectors_requires_list_or_tuple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)

            with self.assertRaisesRegex(ValueError,
                'echoframe_keys must be a list or tuple'):
                store.load_codevectors(iter([]))


if __name__ == '__main__':
    unittest.main()
