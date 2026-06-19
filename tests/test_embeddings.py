'''Tests for store-backed embedding containers.'''

from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from echoframe.embeddings import Embedding, Embeddings, SlicedEmbedding
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
        phraser_object=SimpleNamespace(start_seconds=0.0,
            object_type='Phrase'),
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

        self.assertEqual(repr(result),
            'Embedding(shape=(4,), layer=7, class=Phrase)')

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


def _slice_embedding(start_seconds=0.0, n_frames=5, dim=2, layer=3):
    phraser_object = SimpleNamespace(start_seconds=start_seconds,
        object_type='Phrase')
    metadata = SimpleNamespace(
        echoframe_key=b'abc',
        phraser_key=_pk('phrase-1'),
        model_name='wav2vec2',
        output_type='hidden_state',
        layer=layer,
        phraser_object=phraser_object,
    )
    data = np.arange(n_frames * dim).reshape(n_frames, dim).astype(float)
    return Embedding(b'abc', SimpleNamespace(), metadata=metadata, data=data)


class TestEmbeddingToFrames(unittest.TestCase):
    def test_frame_count_matches_rows(self):
        embedding = _slice_embedding(n_frames=5)
        self.assertEqual(len(embedding.to_frames()), 5)

    def test_grid_anchored_at_segment_start(self):
        embedding = _slice_embedding(start_seconds=0.5, n_frames=5)
        frames = embedding.to_frames(stride=0.02)
        self.assertAlmostEqual(frames[0].start_time, 0.5)
        self.assertAlmostEqual(frames[3].start_time, 0.5 + 3 * 0.02)

    def test_frame_index_equals_row_index(self):
        frames = _slice_embedding(n_frames=5).to_frames()
        self.assertEqual([frame.index for frame in frames], [0, 1, 2, 3, 4])

    def test_custom_stride_and_field_propagate(self):
        frames = _slice_embedding(n_frames=5).to_frames(stride=0.01,
            field=0.02)
        self.assertAlmostEqual(frames.stride, 0.01)
        self.assertAlmostEqual(frames.field, 0.02)

    def test_raises_for_non_2d_payload(self):
        metadata = SimpleNamespace(
            echoframe_key=b'abc', phraser_key=_pk('phrase-1'),
            model_name='wav2vec2', output_type='hidden_state', layer=3,
            phraser_object=SimpleNamespace(start_seconds=0.0,
                object_type='Phrase'))
        embedding = Embedding(b'abc', SimpleNamespace(), metadata=metadata,
            data=np.arange(4).astype(float))

        with self.assertRaisesRegex(ValueError,
            'slicing requires a 2D \\(frames, dim\\) payload'):
            embedding.to_frames()


class TestEmbeddingSliceTime(unittest.TestCase):
    # grid (start=0, stride=0.02, field=0.025):
    # 0:[0.000,0.025] 1:[0.020,0.045] 2:[0.040,0.065]
    # 3:[0.060,0.085] 4:[0.080,0.105]
    def test_returns_rows_for_subinterval(self):
        embedding = _slice_embedding(n_frames=5)
        result = embedding.slice_time(0.02, 0.05)
        np.testing.assert_array_equal(result, embedding.data[[0, 1, 2]])

    def test_full_span_returns_all_rows(self):
        embedding = _slice_embedding(n_frames=5)
        result = embedding.slice_time(0.0, 0.105)
        np.testing.assert_array_equal(result, embedding.data)

    def test_single_frame_interval_returns_one_row(self):
        embedding = _slice_embedding(n_frames=5)
        result = embedding.slice_time(0.001, 0.015)
        self.assertEqual(result.shape, (1, 2))
        np.testing.assert_array_equal(result, embedding.data[[0]])

    def test_percentage_overlap_is_forwarded(self):
        embedding = _slice_embedding(n_frames=5)
        any_overlap = embedding.slice_time(0.02, 0.05)
        fully_contained = embedding.slice_time(0.02, 0.05,
            percentage_overlap=100)
        self.assertEqual(len(any_overlap), 3)
        np.testing.assert_array_equal(fully_contained, embedding.data[[1]])

    def test_raises_when_interval_outside_span(self):
        embedding = _slice_embedding(n_frames=5)
        with self.assertRaisesRegex(ValueError,
            'no frames overlap 1.000-1.100s'):
            embedding.slice_time(1.0, 1.1)


class TestEmbeddingSliceSegment(unittest.TestCase):
    def _segment(self, start_seconds, end_seconds):
        return SimpleNamespace(
            start=int(start_seconds * 1000),
            end=int(end_seconds * 1000),
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

    def test_returns_row_subset_for_descendant(self):
        embedding = _slice_embedding(n_frames=5)
        segment = self._segment(0.02, 0.05)
        result = embedding.slice_segment(segment)
        np.testing.assert_array_equal(result, embedding.data[[0, 1, 2]])

    def test_selects_by_seconds_not_milliseconds(self):
        embedding = _slice_embedding(n_frames=5)
        segment = self._segment(0.02, 0.05)
        result = embedding.slice_segment(segment)
        self.assertEqual(len(result), 3)

    def test_forwards_percentage_overlap(self):
        embedding = _slice_embedding(n_frames=5)
        segment = self._segment(0.02, 0.05)
        result = embedding.slice_segment(segment, percentage_overlap=100)
        np.testing.assert_array_equal(result, embedding.data[[1]])

    def test_descendant_shorter_than_field_returns_a_row(self):
        embedding = _slice_embedding(n_frames=5)
        segment = self._segment(0.001, 0.010)
        result = embedding.slice_segment(segment)
        self.assertGreaterEqual(len(result), 1)

    def test_raises_when_descendant_outside_span(self):
        embedding = _slice_embedding(n_frames=5)
        segment = self._segment(1.0, 1.1)
        with self.assertRaisesRegex(ValueError, 'no frames overlap'):
            embedding.slice_segment(segment)


def _phrase_embedding(start_seconds=0.0, n_frames=5, dim=2, layer=3,
    collar=500, object_type='Phrase'):
    phraser_object = SimpleNamespace(start_seconds=start_seconds,
        object_type=object_type)
    metadata = SimpleNamespace(
        echoframe_key=b'abc',
        phraser_key=_pk('phrase-1'),
        model_name='wav2vec2',
        output_type='hidden_state',
        layer=layer,
        collar=collar,
        phraser_object=phraser_object,
    )
    data = np.arange(n_frames * dim).reshape(n_frames, dim).astype(float)
    return Embedding(b'abc', SimpleNamespace(), metadata=metadata, data=data)


class TestEmbeddingSubEmbedding(unittest.TestCase):
    # grid (start=0, stride=0.02, field=0.025):
    # 0:[0.000,0.025] 1:[0.020,0.045] 2:[0.040,0.065]
    # 3:[0.060,0.085] 4:[0.080,0.105]
    def _phone(self, start_seconds, end_seconds):
        return SimpleNamespace(
            start=int(start_seconds * 1000),
            end=int(end_seconds * 1000),
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            object_type='Phone',
            label='p',
        )

    def test_returns_sliced_embedding_with_parent_metadata(self):
        parent = _phrase_embedding(n_frames=5, collar=500)
        phone = self._phone(0.02, 0.05)
        sliced = parent.sub_embedding(phone)
        self.assertIsInstance(sliced, SlicedEmbedding)
        self.assertIs(sliced.parent_embedding, parent)
        self.assertIs(sliced.phraser_object, phone)
        self.assertEqual(sliced.parent_class, 'Phrase')
        self.assertEqual(sliced.parent_collar, 500)
        self.assertEqual(sliced.parent_phraser_key, parent.phraser_key)
        self.assertEqual(sliced.model_name, 'wav2vec2')
        self.assertEqual(sliced.output_type, 'hidden_state')
        self.assertEqual(sliced.layer, parent.layer)

    def test_default_data_matches_descendant_rows(self):
        parent = _phrase_embedding(n_frames=5)
        phone = self._phone(0.02, 0.05)
        sliced = parent.sub_embedding(phone)
        self.assertEqual(sliced.rows, [0, 1, 2])
        np.testing.assert_array_equal(sliced.data, parent.data[[0, 1, 2]])
        np.testing.assert_array_equal(sliced.data, parent.slice_segment(phone))
        self.assertEqual(sliced.shape, sliced.data.shape)

    def test_aggregate_mean_returns_1d_vector(self):
        parent = _phrase_embedding(n_frames=5)
        phone = self._phone(0.02, 0.05)
        sliced = parent.sub_embedding(phone, aggregate='mean')
        expected = parent.data[[0, 1, 2]].mean(axis=0)
        np.testing.assert_array_equal(sliced.data, expected)
        self.assertEqual(sliced.data.ndim, 1)

    def test_aggregate_middle_returns_single_row(self):
        parent = _phrase_embedding(n_frames=5)
        phone = self._phone(0.02, 0.05)
        sliced = parent.sub_embedding(phone, aggregate='middle')
        self.assertEqual(len(sliced.rows), 1)
        np.testing.assert_array_equal(sliced.data, parent.data[sliced.rows[0]])

    def test_invalid_aggregate_raises(self):
        parent = _phrase_embedding(n_frames=5)
        phone = self._phone(0.02, 0.05)
        with self.assertRaisesRegex(ValueError,
            "aggregate must be None, 'mean', or 'middle'"):
            parent.sub_embedding(phone, aggregate='max')

    def test_raises_when_descendant_outside_span(self):
        parent = _phrase_embedding(n_frames=5)
        phone = self._phone(1.0, 1.1)
        with self.assertRaisesRegex(ValueError, 'no frames overlap'):
            parent.sub_embedding(phone)

    def test_repr_includes_class_and_parent_class(self):
        parent = _phrase_embedding(n_frames=5, layer=4)
        sliced = parent.sub_embedding(self._phone(0.02, 0.05))
        self.assertEqual(sliced.object_class, 'Phone')
        self.assertEqual(repr(sliced),
            f'SlicedEmbedding(shape={sliced.shape}, layer=4, '
            f'class=Phone, parent_class=Phrase)')


class TestEmbeddingSubEmbeddings(unittest.TestCase):
    # grid (start=0, stride=0.02, field=0.025):
    # 0:[0.000,0.025] 1:[0.020,0.045] 2:[0.040,0.065]
    # 3:[0.060,0.085] 4:[0.080,0.105]
    def _phone(self, start_seconds, end_seconds, label='p'):
        return SimpleNamespace(
            start=int(start_seconds * 1000),
            end=int(end_seconds * 1000),
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            object_type='Phone',
            label=label,
        )

    def _embedding(self, n_frames=5, dim=2, **descendants):
        phraser_object = SimpleNamespace(start_seconds=0.0,
            object_type='Phrase', **descendants)
        metadata = SimpleNamespace(
            echoframe_key=b'abc',
            phraser_key=_pk('phrase-1'),
            model_name='wav2vec2',
            output_type='hidden_state',
            layer=3,
            collar=500,
            phraser_object=phraser_object,
        )
        data = np.arange(n_frames * dim).reshape(n_frames, dim).astype(float)
        return Embedding(b'abc', SimpleNamespace(), metadata=metadata,
            data=data)

    def test_returns_one_sliced_embedding_per_descendant(self):
        phones = [self._phone(0.02, 0.05), self._phone(0.06, 0.09)]
        embedding = self._embedding(phones=phones)

        result = embedding.sub_embeddings('phone')

        self.assertEqual(len(result), 2)
        for sub, phone in zip(result, phones):
            self.assertIsInstance(sub, SlicedEmbedding)
            self.assertIs(sub.phraser_object, phone)
            expected = embedding.sub_embedding(phone)
            self.assertEqual(sub.rows, expected.rows)
            np.testing.assert_array_equal(sub.data, expected.data)

    def test_accepts_singular_plural_and_mixed_case(self):
        phone = self._phone(0.02, 0.05)
        embedding = self._embedding(phones=[phone])

        for name in ('phone', 'phones', 'Phone', 'PHONES'):
            result = embedding.sub_embeddings(name)
            self.assertEqual(len(result), 1)
            self.assertIs(result[0].phraser_object, phone)

    def test_aggregate_is_forwarded(self):
        phones = [self._phone(0.02, 0.05), self._phone(0.06, 0.09)]
        embedding = self._embedding(phones=phones)

        result = embedding.sub_embeddings('phone', aggregate='mean')

        for sub, phone in zip(result, phones):
            self.assertEqual(sub.data.ndim, 1)
            np.testing.assert_array_equal(sub.data,
                embedding.sub_embedding(phone, aggregate='mean').data)

    def test_percentage_overlap_is_forwarded(self):
        phone = self._phone(0.02, 0.05)
        embedding = self._embedding(phones=[phone])

        result = embedding.sub_embeddings('phone', percentage_overlap=100)

        np.testing.assert_array_equal(result[0].data,
            embedding.sub_embedding(phone, percentage_overlap=100).data)

    def test_empty_descendant_list_returns_empty_list(self):
        embedding = self._embedding(phones=[])
        self.assertEqual(embedding.sub_embeddings('phone'), [])

    def test_raises_for_missing_descendant_class(self):
        embedding = self._embedding(phones=[self._phone(0.02, 0.05)])

        with self.assertRaisesRegex(ValueError,
            "Phrase has no descendant 'word'"):
            embedding.sub_embeddings('word')

    def test_raises_when_a_descendant_is_outside_span(self):
        phones = [self._phone(0.02, 0.05), self._phone(1.0, 1.1)]
        embedding = self._embedding(phones=phones)

        with self.assertRaisesRegex(ValueError, 'no frames overlap'):
            embedding.sub_embeddings('phone')


def _embedding_with_data(data, start_seconds=0.0, layer=3):
    phraser_object = SimpleNamespace(start_seconds=start_seconds,
        object_type='Phrase')
    metadata = SimpleNamespace(
        echoframe_key=b'abc',
        phraser_key=_pk('phrase-1'),
        model_name='wav2vec2',
        output_type='hidden_state',
        layer=layer,
        phraser_object=phraser_object,
    )
    return Embedding(b'abc', SimpleNamespace(), metadata=metadata, data=data)


def _segment(start_seconds, end_seconds):
    return SimpleNamespace(
        start=int(start_seconds * 1000),
        end=int(end_seconds * 1000),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


class TestEmbeddingMiddleFrame(unittest.TestCase):
    # grid (start=0, stride=0.02, field=0.025):
    # 0:[0.000,0.025] 1:[0.020,0.045] 2:[0.040,0.065]
    # 3:[0.060,0.085] 4:[0.080,0.105]
    def test_middle_of_odd_selection(self):
        embedding = _slice_embedding(n_frames=5)
        result = embedding.middle_frame_time(0.02, 0.05)
        np.testing.assert_array_equal(result, embedding.data[1])

    def test_middle_of_even_selection(self):
        embedding = _slice_embedding(n_frames=5)
        result = embedding.middle_frame_time(0.0, 0.04)
        np.testing.assert_array_equal(result, embedding.data[0])

    def test_single_frame_selection(self):
        embedding = _slice_embedding(n_frames=5)
        result = embedding.middle_frame_time(0.001, 0.015)
        np.testing.assert_array_equal(result, embedding.data[0])

    def test_returns_1d_row(self):
        embedding = _slice_embedding(n_frames=5)
        result = embedding.middle_frame_time(0.02, 0.05)
        self.assertEqual(result.shape, (2,))

    def test_percentage_overlap_is_forwarded(self):
        embedding = _slice_embedding(n_frames=5)
        result = embedding.middle_frame_time(0.02, 0.05,
            percentage_overlap=100)
        np.testing.assert_array_equal(result, embedding.data[1])

    def test_raises_when_interval_outside_span(self):
        embedding = _slice_embedding(n_frames=5)
        with self.assertRaisesRegex(ValueError,
            'no frames overlap 1.000-1.100s'):
            embedding.middle_frame_time(1.0, 1.1)

    def test_segment_delegates(self):
        embedding = _slice_embedding(n_frames=5)
        result = embedding.middle_frame_segment(_segment(0.02, 0.05))
        np.testing.assert_array_equal(result,
            embedding.middle_frame_time(0.02, 0.05))


class TestEmbeddingAggregate(unittest.TestCase):
    # grid (start=0, stride=0.02, field=0.025):
    # 0:[0.000,0.025] 1:[0.020,0.045] 2:[0.040,0.065]
    # 3:[0.060,0.085] 4:[0.080,0.105]
    # data chosen so mean != middle row over selections
    def _embedding(self):
        data = np.array(
            [[0.0, 0.0], [10.0, 10.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        return _embedding_with_data(data)

    def test_mean_matches_manual_average(self):
        embedding = self._embedding()
        result = embedding.aggregate_time(0.0, 0.105, 'mean')
        np.testing.assert_array_equal(result, embedding.data.mean(axis=0))

    def test_mean_subinterval(self):
        embedding = self._embedding()
        result = embedding.aggregate_time(0.02, 0.05, 'mean')
        np.testing.assert_array_equal(result,
            embedding.data[[0, 1, 2]].mean(axis=0))

    def test_middle_delegates(self):
        embedding = self._embedding()
        result = embedding.aggregate_time(0.02, 0.05, 'middle')
        np.testing.assert_array_equal(result,
            embedding.middle_frame_time(0.02, 0.05))

    def test_default_method_is_mean(self):
        embedding = self._embedding()
        np.testing.assert_array_equal(
            embedding.aggregate_time(0.02, 0.05),
            embedding.aggregate_time(0.02, 0.05, 'mean'))

    def test_returns_1d_for_both_methods(self):
        embedding = self._embedding()
        self.assertEqual(embedding.aggregate_time(0.02, 0.05, 'mean').shape,
            (2,))
        self.assertEqual(embedding.aggregate_time(0.02, 0.05, 'middle').shape,
            (2,))

    def test_invalid_method_raises(self):
        embedding = self._embedding()
        with self.assertRaisesRegex(ValueError,
            "method must be 'mean' or 'middle'"):
            embedding.aggregate_time(0.02, 0.05, 'bad')

    def test_percentage_overlap_forwarded(self):
        embedding = self._embedding()
        result = embedding.aggregate_time(0.02, 0.05, 'mean',
            percentage_overlap=100)
        np.testing.assert_array_equal(result,
            embedding.data[[1]].mean(axis=0))

    def test_raises_when_interval_outside_span(self):
        embedding = self._embedding()
        for method in ('mean', 'middle'):
            with self.assertRaisesRegex(ValueError, 'no frames overlap'):
                embedding.aggregate_time(1.0, 1.1, method)

    def test_segment_delegates(self):
        embedding = self._embedding()
        segment = _segment(0.02, 0.05)
        for method in ('mean', 'middle'):
            np.testing.assert_array_equal(
                embedding.aggregate_segment(segment, method=method),
                embedding.aggregate_time(0.02, 0.05, method=method))


if __name__ == '__main__':
    unittest.main()
