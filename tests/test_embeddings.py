'''Tests for the Embeddings container.'''

from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings
from unittest import mock

import numpy as np

from echoframe.embeddings import Embeddings, TokenEmbeddings


def _make(shape, dims, layers=None, echoframe_key='token-1',
        frame_aggregation=None, data=None, path=None):
    if data is None:
        data = np.zeros(shape)
    if layers is not None and 'layers' in dims:
        echoframe_keys = tuple(echoframe_key for _ in layers)
    else:
        echoframe_keys = (echoframe_key,)
    return Embeddings(echoframe_keys=echoframe_keys, data=data, dims=dims,
        layers=layers, frame_aggregation=frame_aggregation, path=path)


def _make_token(shape, dims, layers=None, echoframe_key='token-1',
        frame_aggregation=None, data=None, path=None):
    return _make(shape, dims, layers=layers, echoframe_key=echoframe_key,
        frame_aggregation=frame_aggregation, data=data, path=path)


class TestEmbeddingsInit(unittest.TestCase):
    def test_raises_if_echoframe_key_missing(self):
        with self.assertRaises(TypeError):
            Embeddings(data=np.zeros((8,)), dims=('embed_dim',))

    def test_raises_if_echoframe_key_empty(self):
        with self.assertRaisesRegex(ValueError, 'echoframe_keys'):
            _make((8,), ('embed_dim',), echoframe_key='')

    def test_raises_if_dims_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'dims length mismatch'):
            _make((3, 4), ('frames',))

    def test_raises_if_layers_in_dims_but_layers_none(self):
        with self.assertRaisesRegex(ValueError,
                "'layers' in dims requires layers metadata"):
            _make((2, 5, 8), ('layers', 'frames', 'embed_dim'), layers=None)

    def test_raises_if_layers_set_but_not_in_dims(self):
        with self.assertRaisesRegex(ValueError,
                "layers metadata requires 'layers' in dims"):
            _make((5, 8), ('frames', 'embed_dim'), layers=(0, 1, 2))

    def test_raises_if_layers_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'layers length mismatch'):
            _make((2, 5, 8), ('layers', 'frames', 'embed_dim'),
                layers=(3, 6, 12))

    def test_valid_three_dim(self):
        emb = _make((3, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12))
        self.assertEqual(emb.shape, (3, 5, 8))

    def test_valid_three_dim_aggregated(self):
        emb = _make((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), frame_aggregation='mean')
        self.assertEqual(emb.shape, (3, 8))

    def test_valid_two_dim_frames(self):
        emb = _make((5, 8), ('frames', 'embed_dim'))
        self.assertEqual(emb.shape, (5, 8))

    def test_valid_one_dim(self):
        emb = _make((8,), ('embed_dim',))
        self.assertEqual(emb.shape, (8,))

    def test_raises_if_frame_aggregation_empty(self):
        with self.assertRaisesRegex(ValueError,
                'frame_aggregation must be a non-empty string'):
            _make((8,), ('embed_dim',), frame_aggregation='')

    def test_raises_if_frame_aggregation_with_frames(self):
        with self.assertRaisesRegex(ValueError,
                "frame_aggregation must be None when 'frames' is in dims"):
            _make((5, 8), ('frames', 'embed_dim'),
                frame_aggregation='mean')

    def test_repr_hides_array_preview(self):
        data = np.arange(6).reshape(2, 3)
        emb = Embeddings(echoframe_keys=('token-1',), data=data,
            dims=('frames', 'embed_dim'))
        text = repr(emb)
        self.assertIn("echoframe_keys=('token-1',)", text)
        self.assertIn("shape=(2, 3)", text)
        self.assertIn("dims=('frames', 'embed_dim')", text)
        self.assertIn('frame_aggregation=None', text)
        self.assertNotIn('data=', text)
        self.assertNotIn('array(', text)

    def test_to_numpy_returns_original_array(self):
        data = np.arange(6).reshape(2, 3)
        emb = Embeddings(echoframe_keys=('token-1',), data=data,
            dims=('frames', 'embed_dim'))
        result = emb.to_numpy()
        self.assertIs(result, data)
        np.testing.assert_array_equal(result, data)

    def test_store_raises_without_path_or_binding(self):
        emb = _make((8,), ('embed_dim',))
        with self.assertRaisesRegex(ValueError,
                'embeddings are not bound to a store and have no path'):
            _ = emb.store

    def test_bind_store_sets_path_and_returns_store(self):
        store = SimpleNamespace(root=Path('/tmp/embeddings-store'))
        emb = _make((8,), ('embed_dim',))

        result = emb.bind_store(store)

        self.assertIs(result, emb)
        self.assertIs(emb.store, store)
        self.assertEqual(emb.path, store.root)

    @mock.patch('echoframe.store.Store')
    def test_store_lazily_opens_and_caches_from_path(self, store_cls):
        fake_store = object()
        store_cls.return_value = fake_store
        emb = _make((8,), ('embed_dim',), path=Path('/tmp/embeddings-store'))

        first = emb.store
        second = emb.store

        self.assertIs(first, fake_store)
        self.assertIs(second, fake_store)
        store_cls.assert_called_once_with(Path('/tmp/embeddings-store'))


class TestEmbeddingsLayer(unittest.TestCase):
    def setUp(self):
        self.emb = _make((3, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12))

    def test_layer_by_value(self):
        result = self.emb.layer(6)
        self.assertEqual(result.shape, (5, 8))
        self.assertEqual(result.dims, ('frames', 'embed_dim'))
        self.assertIsNone(result.layers)
        self.assertEqual(result.echoframe_key, self.emb.echoframe_key)

    def test_layer_correct_data(self):
        data = np.arange(3 * 5 * 8).reshape(3, 5, 8).astype(float)
        emb = Embeddings(echoframe_keys=('token-1', 'token-1', 'token-1'),
            data=data, dims=('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12))
        result = emb.layer(3)
        np.testing.assert_array_equal(result.data, data[0])

    def test_layer_preserves_frame_aggregation(self):
        data = np.arange(3 * 8).reshape(3, 8).astype(float)
        emb = Embeddings(echoframe_keys=('token-1', 'token-1', 'token-1'),
            data=data, dims=('layers', 'embed_dim'), layers=(3, 6, 12),
            frame_aggregation='mean')
        result = emb.layer(6)
        self.assertEqual(result.dims, ('embed_dim',))
        self.assertEqual(result.frame_aggregation, 'mean')
        np.testing.assert_array_equal(result.data, data[1])

    def test_layer_preserves_path_without_binding_store(self):
        emb = _make((3, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12), path=Path('/tmp/embeddings-store'))

        result = emb.layer(6)

        self.assertEqual(result.path, emb.path)
        self.assertIsNone(result.__dict__.get('_store'))

    def test_layer_preserves_path_and_bound_store(self):
        store = SimpleNamespace(root=Path('/tmp/embeddings-store'))
        emb = _make((3, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12), path=Path('/tmp/embeddings-store'))
        emb.bind_store(store)

        result = emb.layer(6)

        self.assertEqual(result.path, emb.path)
        self.assertIs(result.store, store)

    def test_layer_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.emb.layer(99)


class TestTokenEmbeddingsInit(unittest.TestCase):
    def test_raises_if_input_empty(self):
        with self.assertRaisesRegex(ValueError,
                'tokens must contain at least one Embeddings'):
            TokenEmbeddings(tokens=[])

    def test_raises_if_input_has_non_embeddings(self):
        token = _make_token((8,), ('embed_dim',))
        with self.assertRaisesRegex(ValueError,
                'tokens must contain only Embeddings'):
            TokenEmbeddings(tokens=[token, object()])

    def test_deduplicates_duplicate_keys_and_warns_once(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2')
        token_1_dup = _make_token((8,), ('embed_dim',),
            echoframe_key='token-1')
        token_2_dup = _make_token((8,), ('embed_dim',),
            echoframe_key='token-2')
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            obj = TokenEmbeddings(tokens=[
                token_1, token_2, token_1_dup, token_2_dup])
        self.assertEqual(len(records), 1)
        self.assertIs(records[0].category, UserWarning)
        message = str(records[0].message)
        self.assertEqual(
            message,
            'duplicate echoframe_key values were removed\n'
            'token-1\n'
            'token-2')
        self.assertEqual(obj.token_count, 2)
        self.assertEqual(obj.echoframe_keys, ('token-1', 'token-2'))
        self.assertEqual(obj.tokens, [token_1, token_2])

    def test_raises_if_dims_mismatch(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-2')
        with self.assertRaisesRegex(ValueError, 'token dims mismatch'):
            TokenEmbeddings(tokens=[token_1, token_2])

    def test_raises_if_layers_mismatch(self):
        token_1 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-1')
        token_2 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(1, 2, 3), echoframe_key='token-2')
        with self.assertRaisesRegex(ValueError, 'token layers mismatch'):
            TokenEmbeddings(tokens=[token_1, token_2])

    def test_raises_if_frame_aggregation_mismatch(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1',
            frame_aggregation='mean')
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2')
        with self.assertRaisesRegex(ValueError,
                'token frame_aggregation mismatch'):
            TokenEmbeddings(tokens=[token_1, token_2])

    def test_exposes_shared_metadata(self):
        token_1 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-1',
            frame_aggregation='mean')
        token_2 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-2',
            frame_aggregation='mean')
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        self.assertEqual(obj.token_count, 2)
        self.assertEqual(obj.echoframe_keys, (
            'token-1', 'token-1', 'token-1',
            'token-2', 'token-2', 'token-2',
        ))
        self.assertEqual(obj.dims, ('layers', 'embed_dim'))
        self.assertEqual(obj.layers, (3, 6, 12))
        self.assertEqual(obj.frame_aggregation, 'mean')

    def test_accepts_failed_metadatas_tuple(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2')
        failed = ({'reason': 'missing file'},)

        obj = TokenEmbeddings(tokens=[token_1, token_2],
            _failed_metadatas=failed)

        self.assertEqual(obj._failed_metadatas, failed)

    def test_rejects_non_tuple_failed_metadatas(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2')

        with self.assertRaisesRegex(ValueError,
            '_failed_metadatas must be a tuple'):
            TokenEmbeddings(tokens=[token_1, token_2],
                _failed_metadatas=[{'reason': 'missing file'}])

    def test_infers_path_from_child_tokens(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1',
            path=Path('/tmp/shared-store'))
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2',
            path=Path('/tmp/shared-store'))

        obj = TokenEmbeddings(tokens=[token_1, token_2])

        self.assertEqual(obj.path, Path('/tmp/shared-store'))

    def test_rejects_conflicting_child_paths(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1',
            path=Path('/tmp/store-a'))
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2',
            path=Path('/tmp/store-b'))

        with self.assertRaisesRegex(ValueError, 'token path mismatch'):
            TokenEmbeddings(tokens=[token_1, token_2])

    def test_rejects_collection_path_conflicting_with_children(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1',
            path=Path('/tmp/store-a'))
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2',
            path=Path('/tmp/store-a'))

        with self.assertRaisesRegex(ValueError, 'token path mismatch'):
            TokenEmbeddings(tokens=[token_1, token_2],
                path=Path('/tmp/store-b'))

    def test_collection_path_allows_detached_child_tokens(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2')

        obj = TokenEmbeddings(tokens=[token_1, token_2],
            path=Path('/tmp/shared-store'))

        self.assertEqual(obj.path, Path('/tmp/shared-store'))

    def test_repr_hides_array_preview(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2')
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        text = repr(obj)
        self.assertIn('TokenEmbeddings(', text)
        self.assertIn('token_count=2', text)
        self.assertIn("echoframe_keys=('token-1', 'token-2')", text)
        self.assertIn("dims=('embed_dim',)", text)
        self.assertIn('layers=None', text)
        self.assertIn('frame_aggregation=None', text)
        self.assertNotIn('data=', text)
        self.assertNotIn('array(', text)

    def test_store_raises_without_path_or_binding(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2')
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        with self.assertRaisesRegex(ValueError,
                'token embeddings are not bound to a store and have no path'):
            _ = obj.store


class TestTokenEmbeddingsLayer(unittest.TestCase):
    def test_layer_returns_new_token_embeddings(self):
        token_1 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-1')
        token_2 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-2')
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        result = obj.layer(6)
        self.assertIsInstance(result, TokenEmbeddings)
        self.assertEqual(result.echoframe_keys, ('token-1', 'token-2'))
        self.assertEqual(result.token_count, 2)
        self.assertEqual(result.dims, ('embed_dim',))
        self.assertEqual(result.layers, None)
        np.testing.assert_array_equal(result.tokens[0].data, token_1.data[1])
        np.testing.assert_array_equal(result.tokens[1].data, token_2.data[1])

    def test_layer_preserves_path_and_bound_store(self):
        store = SimpleNamespace(root=Path('/tmp/token-embeddings-store'))
        token_1 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-1',
            path=Path('/tmp/token-embeddings-store'))
        token_2 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-2',
            path=Path('/tmp/token-embeddings-store'))
        obj = TokenEmbeddings(tokens=[token_1, token_2]).bind_store(store)

        result = obj.layer(6)

        self.assertEqual(result.path, obj.path)
        self.assertIs(result.store, store)

    def test_layer_preserves_path_without_binding_store(self):
        token_1 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-1',
            path=Path('/tmp/token-embeddings-store'))
        token_2 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-2',
            path=Path('/tmp/token-embeddings-store'))
        obj = TokenEmbeddings(tokens=[token_1, token_2],
            path=Path('/tmp/token-embeddings-store'))

        result = obj.layer(6)

        self.assertEqual(result.path, obj.path)
        self.assertIsNone(result.__dict__.get('_store'))
        self.assertEqual(result.tokens[0].path, obj.path)
        self.assertEqual(result.tokens[1].path, obj.path)

    def test_layer_preserves_failed_metadatas(self):
        token_1 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-1')
        token_2 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-2')
        failed = ({'reason': 'missing file'},)
        obj = TokenEmbeddings(tokens=[token_1, token_2],
            _failed_metadatas=failed)

        result = obj.layer(6)

        self.assertEqual(result._failed_metadatas, failed)

    def test_layer_preserves_frame_based_tokens(self):
        data_1 = np.arange(3 * 5 * 8).reshape(3, 5, 8).astype(float)
        data_2 = np.arange(3 * 7 * 8).reshape(3, 7, 8).astype(float)
        token_1 = _make_token((3, 5, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-1', data=data_1)
        token_2 = _make_token((3, 7, 8), ('layers', 'frames', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-2', data=data_2)
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        result = obj.layer(12)
        self.assertEqual(result.echoframe_keys, ('token-1', 'token-2'))
        self.assertEqual(result.dims, ('frames', 'embed_dim'))
        np.testing.assert_array_equal(result.tokens[0].data, data_1[2])
        np.testing.assert_array_equal(result.tokens[1].data, data_2[2])

    def test_layer_unknown_raises(self):
        token_1 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-1')
        token_2 = _make_token((3, 8), ('layers', 'embed_dim'),
            layers=(3, 6, 12), echoframe_key='token-2')
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        with self.assertRaises(ValueError):
            obj.layer(99)

    def test_layer_on_non_layer_embeddings_raises(self):
        token_1 = _make_token((8,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((8,), ('embed_dim',), echoframe_key='token-2')
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        with self.assertRaises(ValueError):
            obj.layer(3)


class TestTokenEmbeddingsAggregate(unittest.TestCase):
    def test_aggregate_embed_dim_tokens(self):
        token_1 = _make_token((3,), ('embed_dim',), echoframe_key='token-1',
            data=np.array([1.0, 2.0, 3.0]))
        token_2 = _make_token((3,), ('embed_dim',), echoframe_key='token-2',
            data=np.array([3.0, 4.0, 5.0]))
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        result = obj.aggregate()
        self.assertEqual(result.shape, (3,))
        np.testing.assert_array_equal(result, np.array([2.0, 3.0, 4.0]))

    def test_aggregate_layers_embed_dim_tokens(self):
        data_1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        data_2 = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        token_1 = _make_token((2, 3), ('layers', 'embed_dim'),
            layers=(3, 6), echoframe_key='token-1', data=data_1)
        token_2 = _make_token((2, 3), ('layers', 'embed_dim'),
            layers=(3, 6), echoframe_key='token-2', data=data_2)
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        result = obj.aggregate()
        self.assertEqual(result.shape, (2, 3))
        expected = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        np.testing.assert_array_equal(result, expected)

    def test_aggregate_unsupported_method_raises(self):
        token_1 = _make_token((3,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((3,), ('embed_dim',), echoframe_key='token-2')
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        with self.assertRaises(NotImplementedError) as ctx:
            obj.aggregate(method='median')
        self.assertEqual(
            str(ctx.exception),
            "TokenEmbeddings.aggregate() only supports method='mean'")

    def test_aggregate_with_frames_raises(self):
        token_1 = _make_token((2, 3), ('frames', 'embed_dim'),
            echoframe_key='token-1')
        token_2 = _make_token((2, 3), ('frames', 'embed_dim'),
            echoframe_key='token-2')
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        with self.assertRaises(NotImplementedError) as ctx:
            obj.aggregate()
        self.assertEqual(
            str(ctx.exception),
            "TokenEmbeddings.aggregate() does not support 'frames'")


class TestTokenEmbeddingsToNumpy(unittest.TestCase):
    def test_to_numpy_stacks_embed_dim_tokens(self):
        data_1 = np.arange(4)
        data_2 = np.arange(4, 8)
        token_1 = _make_token((4,), ('embed_dim',), echoframe_key='token-1',
            data=data_1)
        token_2 = _make_token((4,), ('embed_dim',), echoframe_key='token-2',
            data=data_2)
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        result = obj.to_numpy()
        self.assertEqual(result.shape, (2, 4))
        np.testing.assert_array_equal(result, np.stack([data_1, data_2], axis=0))

    def test_to_numpy_stacks_layered_tokens(self):
        data_1 = np.arange(6).reshape(2, 3)
        data_2 = np.arange(6, 12).reshape(2, 3)
        token_1 = _make_token((2, 3), ('layers', 'embed_dim'),
            layers=(3, 6), echoframe_key='token-1', data=data_1)
        token_2 = _make_token((2, 3), ('layers', 'embed_dim'),
            layers=(3, 6), echoframe_key='token-2', data=data_2)
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        result = obj.to_numpy()
        self.assertEqual(result.shape, (2, 2, 3))
        np.testing.assert_array_equal(result,
            np.stack([data_1, data_2], axis=0))

    def test_to_numpy_preserves_token_order(self):
        data_1 = np.array([1, 2, 3])
        data_2 = np.array([4, 5, 6])
        token_1 = _make_token((3,), ('embed_dim',), echoframe_key='token-1',
            data=data_1)
        token_2 = _make_token((3,), ('embed_dim',), echoframe_key='token-2',
            data=data_2)
        obj = TokenEmbeddings(tokens=[token_2, token_1])
        result = obj.to_numpy()
        np.testing.assert_array_equal(result[0], data_2)
        np.testing.assert_array_equal(result[1], data_1)

    def test_to_numpy_raises_for_non_uniform_shapes(self):
        token_1 = _make_token((3,), ('embed_dim',), echoframe_key='token-1')
        token_2 = _make_token((4,), ('embed_dim',), echoframe_key='token-2')
        obj = TokenEmbeddings(tokens=[token_1, token_2])
        with self.assertRaises(NotImplementedError) as context:
            obj.to_numpy()
        message = str(context.exception)
        self.assertIn('requires identical token shapes', message)


if __name__ == '__main__':
    unittest.main()
