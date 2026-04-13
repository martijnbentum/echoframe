'''Tests for the Embeddings container.'''

import unittest

import numpy as np

from echoframe.embeddings import Embeddings


def _make(shape, dims, layers=None):
    return Embeddings(data=np.zeros(shape), dims=dims, layers=layers)


class TestEmbeddingsInit(unittest.TestCase):
    def test_raises_if_dims_length_mismatch(self):
        with self.assertRaises(ValueError):
            _make((3, 4), ('frames',))

    def test_raises_if_layers_in_dims_but_layers_none(self):
        with self.assertRaises(ValueError):
            _make((2, 5, 8), ('layers', 'frames', 'embed_dim'), layers=None)

    def test_raises_if_layers_set_but_not_in_dims(self):
        with self.assertRaises(ValueError):
            _make((5, 8), ('frames', 'embed_dim'), layers=(0, 1, 2))

    def test_raises_if_layers_length_mismatch(self):
        with self.assertRaises(ValueError):
            _make((2, 5, 8), ('layers', 'frames', 'embed_dim'),
                layers=(3, 6, 12))

    def test_valid_three_dim(self):
        emb = _make((3, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12))
        self.assertEqual(emb.shape, (3, 5, 8))

    def test_valid_two_dim_frames(self):
        emb = _make((5, 8), ('frames', 'embed_dim'))
        self.assertEqual(emb.shape, (5, 8))

    def test_valid_one_dim(self):
        emb = _make((8,), ('embed_dim',))
        self.assertEqual(emb.shape, (8,))


class TestEmbeddingsLayer(unittest.TestCase):
    def setUp(self):
        self.emb = _make((3, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12))

    def test_layer_by_value(self):
        result = self.emb.layer(6)
        self.assertEqual(result.shape, (5, 8))
        self.assertEqual(result.dims, ('frames', 'embed_dim'))
        self.assertIsNone(result.layers)

    def test_layer_correct_data(self):
        data = np.arange(3 * 5 * 8).reshape(3, 5, 8).astype(float)
        emb = Embeddings(data=data, dims=('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12))
        result = emb.layer(3)
        np.testing.assert_array_equal(result.data, data[0])

    def test_layer_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.emb.layer(99)


class TestEmbeddingsConcat(unittest.TestCase):
    def test_concat_frames(self):
        a = _make((3, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12))
        b = _make((3, 7, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12))
        result = Embeddings.concat([a, b], axis='frames')
        self.assertEqual(result.shape, (3, 12, 8))
        self.assertEqual(result.dims, ('layers', 'frames', 'embed_dim'))
        self.assertEqual(result.layers, (3, 6, 12))

    def test_concat_layers_merges_tuples(self):
        a = _make((2, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6))
        b = _make((1, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(12,))
        result = Embeddings.concat([a, b], axis='layers')
        self.assertEqual(result.shape, (3, 5, 8))
        self.assertEqual(result.layers, (3, 6, 12))

    def test_concat_mismatched_dims_raises(self):
        a = _make((5, 8), ('frames', 'embed_dim'))
        b = _make((8,), ('embed_dim',))
        with self.assertRaises(ValueError):
            Embeddings.concat([a, b], axis='frames')

    def test_concat_axis_not_in_dims_raises(self):
        a = _make((5, 8), ('frames', 'embed_dim'))
        b = _make((5, 8), ('frames', 'embed_dim'))
        with self.assertRaises(ValueError):
            Embeddings.concat([a, b], axis='layers')

    def test_add_sugar_for_frames_concat(self):
        a = _make((5, 8), ('frames', 'embed_dim'))
        b = _make((3, 8), ('frames', 'embed_dim'))
        result = a + b
        self.assertEqual(result.shape, (8, 8))

    def test_add_raises_if_no_frames_dim(self):
        a = _make((8,), ('embed_dim',))
        b = _make((8,), ('embed_dim',))
        with self.assertRaises(ValueError):
            _ = a + b

    def test_round_trip_concat_then_layer(self):
        data = np.arange(3 * 5 * 8).reshape(3, 5, 8).astype(float)
        emb = Embeddings(data=data, dims=('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12))
        a = emb.layer(3)
        b = emb.layer(6)
        c = emb.layer(12)
        # Wrap each back into a layers dim for concat
        a3 = Embeddings(data=a.data[np.newaxis], dims=('layers', 'frames',
            'embed_dim'), layers=(3,))
        b3 = Embeddings(data=b.data[np.newaxis], dims=('layers', 'frames',
            'embed_dim'), layers=(6,))
        c3 = Embeddings(data=c.data[np.newaxis], dims=('layers', 'frames',
            'embed_dim'), layers=(12,))
        merged = Embeddings.concat([a3, b3, c3], axis='layers')
        self.assertEqual(merged.shape, (3, 5, 8))
        np.testing.assert_array_equal(merged.layer(6).data, data[1])


if __name__ == '__main__':
    unittest.main()
