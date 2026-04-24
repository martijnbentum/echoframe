'''Tests for store-backed embedding containers.'''

from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from echoframe.embeddings import Embedding, Embeddings
from tests.helpers import (
    ensure_model,
    make_fake_store,
    pk as _pk,
    put as _put,
)


def _make_metadata(echoframe_key=b'key', phraser_key=None, model_name='model',
    output_type='hidden_state', layer=3):
    if phraser_key is None:
        phraser_key = _pk('phrase-1')
    return SimpleNamespace(
        echoframe_key=echoframe_key,
        phraser_key=phraser_key,
        model_name=model_name,
        output_type=output_type,
        layer=layer,
    )


class TestEmbedding(unittest.TestCase):
    def test_loads_metadata_and_payload_from_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            ensure_model(store, 'wav2vec2')
            data = np.arange(6).reshape(2, 3).astype(float)
            phraser_key = _pk('phrase-1')
            metadata = _put(store, phraser_key=phraser_key, collar=500,
                model_name='wav2vec2', output_type='hidden_state', layer=4,
                data=data)

            result = Embedding(metadata.echoframe_key, store)

            self.assertEqual(result.echoframe_key, metadata.echoframe_key)
            self.assertEqual(result.phraser_key, phraser_key)
            self.assertEqual(result.model_name, 'wav2vec2')
            self.assertEqual(result.output_type, 'hidden_state')
            self.assertEqual(result.layer, 4)
            self.assertEqual(result.shape, (2, 3))
            np.testing.assert_array_equal(result.data, data)

    def test_accepts_preloaded_metadata_and_data(self):
        metadata = _make_metadata(echoframe_key=b'abc', layer=7)
        data = np.arange(4).astype(float)
        store = SimpleNamespace()

        result = Embedding(b'abc', store, metadata=metadata, data=data)

        self.assertEqual(result.metadata, metadata)
        self.assertEqual(result.layer, 7)
        np.testing.assert_array_equal(result.data, data)

    def test_repr_includes_shape_and_layer(self):
        metadata = _make_metadata(echoframe_key=b'abc', layer=7)
        data = np.arange(4).astype(float)
        result = Embedding(b'abc', SimpleNamespace(), metadata=metadata,
            data=data)

        self.assertEqual(repr(result), 'Embedding(shape=(4,), layer=7)')

    def test_raises_if_metadata_missing(self):
        store = SimpleNamespace(
            load_metadata=lambda key: None,
            metadata_to_payload=lambda metadata: np.arange(4).astype(float),
        )

        with self.assertRaisesRegex(ValueError,
            "no metadata found for echoframe_key b'abc'"):
            Embedding(b'abc', store)

    def test_raises_if_data_missing(self):
        metadata = _make_metadata(echoframe_key=b'abc')
        store = SimpleNamespace(
            load_metadata=lambda key: metadata,
            metadata_to_payload=lambda md: None,
        )

        with self.assertRaisesRegex(ValueError,
            "no embedding data found for echoframe_key b'abc'"):
            Embedding(b'abc', store)

    def test_raises_if_output_type_is_not_hidden_state(self):
        metadata = _make_metadata(echoframe_key=b'abc',
            output_type='codebook_indices')

        with self.assertRaisesRegex(ValueError,
            'metadata.output_type must be hidden_state'):
            Embedding(b'abc', SimpleNamespace(), metadata=metadata,
                data=np.arange(4).astype(float))

    def test_raises_if_layer_is_none(self):
        metadata = _make_metadata(echoframe_key=b'abc', layer=None)

        with self.assertRaisesRegex(ValueError,
            'embedding metadata.layer must not be None'):
            Embedding(b'abc', SimpleNamespace(), metadata=metadata,
                data=np.arange(4).astype(float))

    def test_raises_if_data_ndim_is_invalid(self):
        metadata = _make_metadata(echoframe_key=b'abc')

        with self.assertRaisesRegex(ValueError, 'data must be ndim 1 or 2'):
            Embedding(b'abc', SimpleNamespace(), metadata=metadata,
                data=np.zeros((2, 3, 4)))

    def test_raises_if_metadata_key_does_not_match(self):
        metadata = _make_metadata(echoframe_key=b'other')

        with self.assertRaisesRegex(ValueError,
            'metadata.echoframe_key did not match echoframe_key'):
            Embedding(b'abc', SimpleNamespace(), metadata=metadata,
                data=np.arange(4).astype(float))


class TestEmbeddings(unittest.TestCase):
    def _make_embedding(self, phraser_name, layer=3, model_name='wav2vec2',
        output_type='hidden_state', shape=(2, 3)):
        phraser_key = _pk(phraser_name)
        metadata = _make_metadata(echoframe_key=phraser_name.encode('utf-8'),
            phraser_key=phraser_key, model_name=model_name,
            output_type=output_type, layer=layer)
        data = np.arange(np.prod(shape)).reshape(shape).astype(float)
        return Embedding(metadata.echoframe_key, SimpleNamespace(),
            metadata=metadata, data=data)

    def test_requires_non_empty_embedding_list(self):
        with self.assertRaisesRegex(ValueError,
            'embeddings must contain at least one Embedding'):
            Embeddings([], SimpleNamespace())

    def test_rejects_non_embedding_items(self):
        with self.assertRaisesRegex(ValueError,
            'embeddings must contain only Embedding'):
            Embeddings([object()], SimpleNamespace())

    def test_exposes_shared_fields_and_stacks_numpy(self):
        emb_1 = self._make_embedding('phrase-1')
        emb_2 = self._make_embedding('phrase-2')
        store = SimpleNamespace(root='fake-root')

        result = Embeddings([emb_1, emb_2], store)

        self.assertIs(result.store, store)
        self.assertEqual(result.count, 2)
        self.assertEqual(result.phraser_keys, (_pk('phrase-1'), _pk('phrase-2')))
        self.assertEqual(result.model_name, 'wav2vec2')
        self.assertEqual(result.output_type, 'hidden_state')
        self.assertEqual(result.layer, 3)
        np.testing.assert_array_equal(result.data, np.stack([
            emb_1.data,
            emb_2.data,
        ], axis=0))

    def test_repr_includes_count_and_layer(self):
        emb_1 = self._make_embedding('phrase-1', layer=9)
        emb_2 = self._make_embedding('phrase-2', layer=9)

        result = Embeddings([emb_1, emb_2], SimpleNamespace())

        self.assertEqual(repr(result), 'Embeddings(# 2, layer=9)')

    def test_rejects_duplicate_phraser_keys(self):
        emb_1 = self._make_embedding('phrase-1')
        emb_2 = self._make_embedding('phrase-1')

        with self.assertRaisesRegex(ValueError, 'duplicate phraser_key'):
            Embeddings([emb_1, emb_2], SimpleNamespace())

    def test_rejects_mixed_model_names(self):
        emb_1 = self._make_embedding('phrase-1', model_name='wav2vec2')
        emb_2 = self._make_embedding('phrase-2', model_name='hubert')

        with self.assertRaisesRegex(ValueError, 'embedding model_name mismatch'):
            Embeddings([emb_1, emb_2], SimpleNamespace())

    def test_rejects_mixed_layers(self):
        emb_1 = self._make_embedding('phrase-1', layer=3)
        emb_2 = self._make_embedding('phrase-2', layer=7)

        with self.assertRaisesRegex(ValueError, 'embedding layer mismatch'):
            Embeddings([emb_1, emb_2], SimpleNamespace())

    def test_to_numpy_raises_for_mismatched_shapes(self):
        emb_1 = self._make_embedding('phrase-1', shape=(2, 3))
        emb_2 = self._make_embedding('phrase-2', shape=(4, 3))
        result = Embeddings([emb_1, emb_2], SimpleNamespace())

        with self.assertRaisesRegex(NotImplementedError,
            'Embeddings.to_numpy\\(\\) requires identical embedding shapes'):
            result.to_numpy()

    def test_from_echoframe_keys_loads_store_backed_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            ensure_model(store, 'wav2vec2')
            data_1 = np.arange(6).reshape(2, 3).astype(float)
            data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
            item_1 = _put(store, phraser_key='phrase-1', collar=500,
                model_name='wav2vec2', output_type='hidden_state', layer=4,
                data=data_1)
            item_2 = _put(store, phraser_key='phrase-2', collar=500,
                model_name='wav2vec2', output_type='hidden_state', layer=4,
                data=data_2)

            result = Embeddings.from_echoframe_keys(store,
                [item_1.echoframe_key, item_2.echoframe_key])

            self.assertEqual(result.count, 2)
            self.assertEqual(result.layer, 4)
            np.testing.assert_array_equal(result.to_numpy(), np.stack([
                data_1,
                data_2,
            ], axis=0))

    def test_from_echoframe_keys_skips_value_error_embeddings_and_prints(self):
        valid_metadata = _make_metadata(echoframe_key=b'valid',
            phraser_key=_pk('phrase-1'), model_name='wav2vec2', layer=4)
        invalid_metadata = _make_metadata(echoframe_key=b'invalid',
            phraser_key=_pk('phrase-2'), model_name='wav2vec2', layer=4)
        store = SimpleNamespace(
            load_metadata=lambda key: (
                valid_metadata if key == b'valid' else invalid_metadata),
            metadata_to_payload=lambda metadata: (
                np.arange(6).reshape(2, 3).astype(float)
                if metadata.echoframe_key == b'valid'
                else np.zeros((2, 3, 4))),
        )

        with mock.patch('builtins.print') as print_mock:
            result = Embeddings.from_echoframe_keys(store, [b'valid', b'invalid'])

        self.assertEqual(result.count, 1)
        print_mock.assert_called_once()
        self.assertIn('skipping echoframe_key', print_mock.call_args.args[0])

    def test_from_echoframe_keys_raises_if_only_value_error_embeddings_seen(self):
        invalid_metadata = _make_metadata(echoframe_key=b'invalid',
            phraser_key=_pk('phrase-2'), model_name='wav2vec2', layer=4)
        store = SimpleNamespace(
            load_metadata=lambda key: invalid_metadata,
            metadata_to_payload=lambda metadata: np.zeros((2, 3, 4)),
        )

        with self.assertRaisesRegex(ValueError,
            'no embeddings were loaded skipped keys 1'):
            Embeddings.from_echoframe_keys(store, [b'invalid'])


if __name__ == '__main__':
    unittest.main()
