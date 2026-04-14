'''Tests for multi-key support in Embeddings (F1–F5).'''

import unittest
import warnings

import numpy as np

from echoframe.embeddings import Embeddings, TokenEmbeddings


def _make_no_layers(echoframe_keys=('k1',), shape=(8,),
        dims=('embed_dim',), data=None):
    if data is None:
        data = np.zeros(shape)
    return Embeddings(echoframe_keys=echoframe_keys, data=data, dims=dims)


def _make_with_layers(echoframe_keys, layers, shape=None, data=None,
        frame_aggregation=None):
    n = len(layers)
    if shape is None:
        shape = (n, 4)
    if data is None:
        data = np.zeros(shape)
    dims = ('layers', 'embed_dim')
    return Embeddings(echoframe_keys=echoframe_keys, data=data, dims=dims,
        layers=layers, frame_aggregation=frame_aggregation)


class TestF1SingleKey(unittest.TestCase):
    def test_single_key_no_layers_ok(self):
        emb = _make_no_layers(echoframe_keys=('k1',))
        self.assertEqual(emb.echoframe_keys, ('k1',))

    def test_single_key_wrong_length_no_layers_raises(self):
        with self.assertRaises(ValueError):
            _make_no_layers(echoframe_keys=('k1', 'k2'))

    def test_multi_key_matches_layers_ok(self):
        emb = _make_with_layers(echoframe_keys=('k1', 'k2', 'k3'),
            layers=(0, 1, 2))
        self.assertEqual(emb.echoframe_keys, ('k1', 'k2', 'k3'))

    def test_multi_key_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _make_with_layers(echoframe_keys=('k1', 'k2'),
                layers=(0, 1, 2))

    def test_echoframe_key_property_returns_first(self):
        emb = _make_with_layers(echoframe_keys=('first', 'second', 'third'),
            layers=(0, 1, 2))
        self.assertEqual(emb.echoframe_key, 'first')

    def test_empty_string_key_raises(self):
        with self.assertRaises(ValueError):
            _make_no_layers(echoframe_keys=('',))

    def test_non_string_key_raises(self):
        with self.assertRaises(ValueError):
            Embeddings(echoframe_keys=(42,), data=np.zeros((8,)),
                dims=('embed_dim',))


class TestF2KeyForLayer(unittest.TestCase):
    def setUp(self):
        self.emb = _make_with_layers(
            echoframe_keys=('key-0', 'key-1', 'key-2'),
            layers=(10, 20, 30))

    def test_key_for_layer_returns_correct_key(self):
        self.assertEqual(self.emb.key_for_layer(10), 'key-0')
        self.assertEqual(self.emb.key_for_layer(20), 'key-1')
        self.assertEqual(self.emb.key_for_layer(30), 'key-2')

    def test_key_for_layer_no_layers_dim_raises(self):
        emb = _make_no_layers(echoframe_keys=('k1',))
        with self.assertRaises(ValueError):
            emb.key_for_layer(0)

    def test_key_for_layer_unknown_n_raises(self):
        with self.assertRaises(ValueError):
            self.emb.key_for_layer(99)


class TestF3LayerSlice(unittest.TestCase):
    def setUp(self):
        data = np.arange(3 * 4).reshape(3, 4).astype(float)
        self.emb = Embeddings(
            echoframe_keys=('ka', 'kb', 'kc'),
            data=data,
            dims=('layers', 'embed_dim'),
            layers=(10, 20, 30))

    def test_layer_slice_carries_correct_key(self):
        result = self.emb.layer(20)
        self.assertEqual(result.echoframe_keys, ('kb',))
        self.assertEqual(result.echoframe_key, 'kb')

    def test_layer_slice_echoframe_keys_length_one(self):
        result = self.emb.layer(30)
        self.assertEqual(len(result.echoframe_keys), 1)


class TestF4Repr(unittest.TestCase):
    def test_repr_contains_echoframe_keys(self):
        emb = _make_no_layers(echoframe_keys=('k1',))
        self.assertIn('echoframe_keys=', repr(emb))

    def test_repr_does_not_contain_singular_key(self):
        emb = _make_no_layers(echoframe_keys=('k1',))
        text = repr(emb)
        # 'echoframe_keys=' is present; 'echoframe_key=' (without s) must not
        # appear as a standalone field name
        self.assertNotIn('echoframe_key=', text.replace('echoframe_keys=', ''))


class TestF5TokenEmbeddings(unittest.TestCase):
    def _make_token(self, key, shape=(4,), dims=('embed_dim',)):
        return Embeddings(echoframe_keys=(key,), data=np.zeros(shape),
            dims=dims)

    def test_token_embeddings_dedup_with_multi_key_embeddings(self):
        t1 = self._make_token('tok-1')
        t2 = self._make_token('tok-2')
        t1_dup = self._make_token('tok-1')
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            obj = TokenEmbeddings(tokens=[t1, t2, t1_dup])
        self.assertEqual(obj.token_count, 2)
        self.assertEqual(len(records), 1)
        self.assertIs(records[0].category, UserWarning)

    def test_token_embeddings_echoframe_keys_returns_one_per_token(self):
        t1 = self._make_token('tok-1')
        t2 = self._make_token('tok-2')
        t3 = self._make_token('tok-3')
        obj = TokenEmbeddings(tokens=[t1, t2, t3])
        self.assertEqual(obj.echoframe_keys, ('tok-1', 'tok-2', 'tok-3'))
        self.assertEqual(len(obj.echoframe_keys), obj.token_count)

    def test_token_embeddings_repr_still_works(self):
        t1 = self._make_token('tok-1')
        t2 = self._make_token('tok-2')
        obj = TokenEmbeddings(tokens=[t1, t2])
        text = repr(obj)
        self.assertIn('TokenEmbeddings(', text)
        self.assertIn("echoframe_keys=('tok-1', 'tok-2')", text)


if __name__ == '__main__':
    unittest.main()
