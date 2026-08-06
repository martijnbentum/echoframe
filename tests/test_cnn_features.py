'''Tests for store-backed CNN feature-extractor containers.'''

from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from echoframe.cnn_features import CNNFeature, CNNFeatures, SlicedCNNFeature
from tests.helpers import (
    ensure_model,
    make_fake_store,
    pk as _pk,
    put as _put,
)


def _make_metadata(echoframe_key=b'key', phraser_key=None, model_name='model',
    output_type='cnn'):
    if phraser_key is None:
        phraser_key = _pk('phrase-1')
    return SimpleNamespace(
        echoframe_key=echoframe_key,
        phraser_key=phraser_key,
        model_name=model_name,
        output_type=output_type,
        phraser_object=SimpleNamespace(start_seconds=0.0,
            object_type='Phrase'),
    )


class TestCNNFeature(unittest.TestCase):
    def test_loads_metadata_and_payload_from_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            ensure_model(store, 'wav2vec2')
            data = np.arange(6).reshape(2, 3).astype(float)
            phraser_key = _pk('phrase-1')
            metadata = _put(store, phraser_key=phraser_key, collar=500,
                model_name='wav2vec2', output_type='cnn', layer=None,
                data=data)

            result = CNNFeature(metadata.echoframe_key, store)

            self.assertEqual(result.echoframe_key, metadata.echoframe_key)
            self.assertEqual(result.phraser_key, phraser_key)
            self.assertEqual(result.model_name, 'wav2vec2')
            self.assertEqual(result.output_type, 'cnn')
            self.assertEqual(result.shape, (2, 3))
            np.testing.assert_array_equal(result.data, data)

    def test_accepts_preloaded_metadata_and_data(self):
        metadata = _make_metadata(echoframe_key=b'abc')
        data = np.arange(4).astype(float)
        store = SimpleNamespace()

        result = CNNFeature(b'abc', store, metadata=metadata, data=data)

        self.assertEqual(result.metadata, metadata)
        np.testing.assert_array_equal(result.data, data)

    def test_repr_includes_shape_and_model(self):
        metadata = _make_metadata(echoframe_key=b'abc', model_name='wav2vec2')
        data = np.arange(4).astype(float)
        result = CNNFeature(b'abc', SimpleNamespace(), metadata=metadata,
            data=data)

        self.assertEqual(repr(result),
            'CNNFeature(shape=(4,), model=wav2vec2, class=Phrase)')

    def test_repr_shows_unknown_class_when_phraser_unreachable(self):
        class BrokenMetadata(SimpleNamespace):
            @property
            def phraser_object(self):
                raise ValueError('unknown phraser_source_id: None')
        metadata = BrokenMetadata(
            echoframe_key=b'abc',
            phraser_key=_pk('phrase-1'),
            model_name='wav2vec2',
            output_type='cnn',
        )
        data = np.arange(4).astype(float)
        result = CNNFeature(b'abc', SimpleNamespace(), metadata=metadata,
            data=data)

        self.assertEqual(repr(result),
            'CNNFeature(shape=(4,), model=wav2vec2, class=?)')

    def test_raises_if_metadata_missing(self):
        store = SimpleNamespace(
            load_metadata=lambda key: None,
            metadata_to_payload=lambda metadata: np.arange(4).astype(float),
        )

        with self.assertRaisesRegex(ValueError,
            "no metadata found for echoframe_key b'abc'"):
            CNNFeature(b'abc', store)

    def test_raises_if_data_missing(self):
        metadata = _make_metadata(echoframe_key=b'abc')
        store = SimpleNamespace(
            load_metadata=lambda key: metadata,
            metadata_to_payload=lambda md: None,
        )

        with self.assertRaisesRegex(ValueError,
            "no cnn feature data found for echoframe_key b'abc'"):
            CNNFeature(b'abc', store)

    def test_raises_if_output_type_is_not_cnn(self):
        metadata = _make_metadata(echoframe_key=b'abc',
            output_type='codebook_indices')

        with self.assertRaisesRegex(ValueError,
            'metadata.output_type must be cnn'):
            CNNFeature(b'abc', SimpleNamespace(), metadata=metadata,
                data=np.arange(4).astype(float))

    def test_raises_if_data_ndim_is_invalid(self):
        metadata = _make_metadata(echoframe_key=b'abc')

        with self.assertRaisesRegex(ValueError, 'data must be ndim 1 or 2'):
            CNNFeature(b'abc', SimpleNamespace(), metadata=metadata,
                data=np.zeros((2, 3, 4)))

    def test_raises_if_metadata_key_does_not_match(self):
        metadata = _make_metadata(echoframe_key=b'other')

        with self.assertRaisesRegex(ValueError,
            'metadata.echoframe_key did not match echoframe_key'):
            CNNFeature(b'abc', SimpleNamespace(), metadata=metadata,
                data=np.arange(4).astype(float))


class TestCNNFeatures(unittest.TestCase):
    def _make_cnn_feature(self, phraser_name, model_name='wav2vec2',
        output_type='cnn', shape=(2, 3)):
        phraser_key = _pk(phraser_name)
        metadata = _make_metadata(echoframe_key=phraser_name.encode('utf-8'),
            phraser_key=phraser_key, model_name=model_name,
            output_type=output_type)
        data = np.arange(np.prod(shape)).reshape(shape).astype(float)
        return CNNFeature(metadata.echoframe_key, SimpleNamespace(),
            metadata=metadata, data=data)

    def test_requires_non_empty_cnn_feature_list(self):
        with self.assertRaisesRegex(ValueError,
            'cnn_features must contain at least one CNNFeature'):
            CNNFeatures([], SimpleNamespace())

    def test_rejects_non_cnn_feature_items(self):
        with self.assertRaisesRegex(ValueError,
            'cnn_features must contain only CNNFeature'):
            CNNFeatures([object()], SimpleNamespace())

    def test_exposes_shared_fields_and_stacks_numpy(self):
        feat_1 = self._make_cnn_feature('phrase-1')
        feat_2 = self._make_cnn_feature('phrase-2')
        store = SimpleNamespace(root='fake-root')

        result = CNNFeatures([feat_1, feat_2], store)

        self.assertIs(result.store, store)
        self.assertEqual(result.count, 2)
        self.assertEqual(result.phraser_keys, (_pk('phrase-1'), _pk('phrase-2')))
        self.assertEqual(result.model_name, 'wav2vec2')
        self.assertEqual(result.output_type, 'cnn')
        np.testing.assert_array_equal(result.data, np.stack([
            feat_1.data,
            feat_2.data,
        ], axis=0))

    def test_repr_includes_count(self):
        feat_1 = self._make_cnn_feature('phrase-1')
        feat_2 = self._make_cnn_feature('phrase-2')

        result = CNNFeatures([feat_1, feat_2], SimpleNamespace())

        self.assertEqual(repr(result), 'CNNFeatures(# 2)')

    def test_rejects_duplicate_phraser_keys(self):
        feat_1 = self._make_cnn_feature('phrase-1')
        feat_2 = self._make_cnn_feature('phrase-1')

        with self.assertRaisesRegex(ValueError, 'duplicate phraser_key'):
            CNNFeatures([feat_1, feat_2], SimpleNamespace())

    def test_rejects_mixed_model_names(self):
        feat_1 = self._make_cnn_feature('phrase-1', model_name='wav2vec2')
        feat_2 = self._make_cnn_feature('phrase-2', model_name='hubert')

        with self.assertRaisesRegex(ValueError,
            'cnn_feature model_name mismatch'):
            CNNFeatures([feat_1, feat_2], SimpleNamespace())

    def test_to_numpy_raises_for_mismatched_shapes(self):
        feat_1 = self._make_cnn_feature('phrase-1', shape=(2, 3))
        feat_2 = self._make_cnn_feature('phrase-2', shape=(4, 3))
        result = CNNFeatures([feat_1, feat_2], SimpleNamespace())

        with self.assertRaisesRegex(NotImplementedError,
            'CNNFeatures.to_numpy\\(\\) requires identical feature shapes'):
            result.to_numpy()

    def test_from_echoframe_keys_loads_store_backed_cnn_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            ensure_model(store, 'wav2vec2')
            data_1 = np.arange(6).reshape(2, 3).astype(float)
            data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
            item_1 = _put(store, phraser_key='phrase-1', collar=500,
                model_name='wav2vec2', output_type='cnn', layer=None,
                data=data_1)
            item_2 = _put(store, phraser_key='phrase-2', collar=500,
                model_name='wav2vec2', output_type='cnn', layer=None,
                data=data_2)

            result = CNNFeatures.from_echoframe_keys(store,
                [item_1.echoframe_key, item_2.echoframe_key])

            self.assertEqual(result.count, 2)
            np.testing.assert_array_equal(result.to_numpy(), np.stack([
                data_1,
                data_2,
            ], axis=0))

    def test_from_echoframe_keys_skips_value_error_cnn_features_and_prints(
        self):
        valid_metadata = _make_metadata(echoframe_key=b'valid',
            phraser_key=_pk('phrase-1'), model_name='wav2vec2')
        invalid_metadata = _make_metadata(echoframe_key=b'invalid',
            phraser_key=_pk('phrase-2'), model_name='wav2vec2')
        store = SimpleNamespace(
            load_many_metadata=lambda keys, keep_missing=False: [
                valid_metadata if key == b'valid' else invalid_metadata
                for key in keys],
            metadatas_to_payloads=lambda metadatas: [
                np.arange(6).reshape(2, 3).astype(float)
                if metadata.echoframe_key == b'valid'
                else np.zeros((2, 3, 4))
                for metadata in metadatas],
        )

        with self.assertLogs('echoframe.cnn_features', level='WARNING') as logs:
            result = CNNFeatures.from_echoframe_keys(store,
                [b'valid', b'invalid'])

        self.assertEqual(result.count, 1)
        self.assertEqual(len(logs.output), 1)
        self.assertIn('skipping echoframe_key', logs.output[0])

    def test_from_echoframe_keys_raises_if_only_value_error_cnn_features_seen(
        self):
        invalid_metadata = _make_metadata(echoframe_key=b'invalid',
            phraser_key=_pk('phrase-2'), model_name='wav2vec2')
        store = SimpleNamespace(
            load_many_metadata=lambda keys, keep_missing=False: [
                invalid_metadata for key in keys],
            metadatas_to_payloads=lambda metadatas: [
                np.zeros((2, 3, 4)) for metadata in metadatas],
        )

        with self.assertRaisesRegex(ValueError,
            'no cnn features were loaded skipped keys 1'):
            CNNFeatures.from_echoframe_keys(store, [b'invalid'])


def _slice_cnn_feature(start_seconds=0.0, n_frames=5, dim=2):
    phraser_object = SimpleNamespace(start_seconds=start_seconds,
        object_type='Phrase')
    metadata = SimpleNamespace(
        echoframe_key=b'abc',
        phraser_key=_pk('phrase-1'),
        model_name='wav2vec2',
        output_type='cnn',
        phraser_object=phraser_object,
    )
    data = np.arange(n_frames * dim).reshape(n_frames, dim).astype(float)
    return CNNFeature(b'abc', SimpleNamespace(), metadata=metadata, data=data)


class TestCNNFeatureToFrames(unittest.TestCase):
    def test_frame_count_matches_rows(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        self.assertEqual(len(cnn_feature.to_frames()), 5)

    def test_grid_anchored_at_segment_start(self):
        cnn_feature = _slice_cnn_feature(start_seconds=0.5, n_frames=5)
        frames = cnn_feature.to_frames(stride=0.02)
        self.assertAlmostEqual(frames[0].start_time, 0.5)
        self.assertAlmostEqual(frames[3].start_time, 0.5 + 3 * 0.02)

    def test_frame_index_equals_row_index(self):
        frames = _slice_cnn_feature(n_frames=5).to_frames()
        self.assertEqual([frame.index for frame in frames], [0, 1, 2, 3, 4])

    def test_custom_stride_and_field_propagate(self):
        frames = _slice_cnn_feature(n_frames=5).to_frames(stride=0.01,
            field=0.02)
        self.assertAlmostEqual(frames.stride, 0.01)
        self.assertAlmostEqual(frames.field, 0.02)

    def test_raises_for_non_2d_payload(self):
        metadata = SimpleNamespace(
            echoframe_key=b'abc', phraser_key=_pk('phrase-1'),
            model_name='wav2vec2', output_type='cnn',
            phraser_object=SimpleNamespace(start_seconds=0.0,
                object_type='Phrase'))
        cnn_feature = CNNFeature(b'abc', SimpleNamespace(), metadata=metadata,
            data=np.arange(4).astype(float))

        with self.assertRaisesRegex(ValueError,
            'slicing requires a 2D \\(frames, dim\\) payload'):
            cnn_feature.to_frames()


class TestCNNFeatureSliceTime(unittest.TestCase):
    # grid (start=0, stride=0.02, field=0.025):
    # 0:[0.000,0.025] 1:[0.020,0.045] 2:[0.040,0.065]
    # 3:[0.060,0.085] 4:[0.080,0.105]
    def test_returns_rows_for_subinterval(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        result = cnn_feature.slice_time(0.02, 0.05)
        np.testing.assert_array_equal(result, cnn_feature.data[[0, 1, 2]])

    def test_full_span_returns_all_rows(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        result = cnn_feature.slice_time(0.0, 0.105)
        np.testing.assert_array_equal(result, cnn_feature.data)

    def test_single_frame_interval_returns_one_row(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        result = cnn_feature.slice_time(0.001, 0.015)
        self.assertEqual(result.shape, (1, 2))
        np.testing.assert_array_equal(result, cnn_feature.data[[0]])

    def test_percentage_overlap_is_forwarded(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        any_overlap = cnn_feature.slice_time(0.02, 0.05)
        fully_contained = cnn_feature.slice_time(0.02, 0.05,
            percentage_overlap=100)
        self.assertEqual(len(any_overlap), 3)
        np.testing.assert_array_equal(fully_contained, cnn_feature.data[[1]])

    def test_raises_when_interval_outside_span(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        with self.assertRaisesRegex(ValueError,
            'no frames overlap 1.000-1.100s'):
            cnn_feature.slice_time(1.0, 1.1)


class TestCNNFeatureSliceSegment(unittest.TestCase):
    def _segment(self, start_seconds, end_seconds):
        return SimpleNamespace(
            start=int(start_seconds * 1000),
            end=int(end_seconds * 1000),
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

    def test_returns_row_subset_for_descendant(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        segment = self._segment(0.02, 0.05)
        result = cnn_feature.slice_segment(segment)
        np.testing.assert_array_equal(result, cnn_feature.data[[0, 1, 2]])

    def test_selects_by_seconds_not_milliseconds(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        segment = self._segment(0.02, 0.05)
        result = cnn_feature.slice_segment(segment)
        self.assertEqual(len(result), 3)

    def test_forwards_percentage_overlap(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        segment = self._segment(0.02, 0.05)
        result = cnn_feature.slice_segment(segment, percentage_overlap=100)
        np.testing.assert_array_equal(result, cnn_feature.data[[1]])

    def test_descendant_shorter_than_field_returns_a_row(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        segment = self._segment(0.001, 0.010)
        result = cnn_feature.slice_segment(segment)
        self.assertGreaterEqual(len(result), 1)

    def test_raises_when_descendant_outside_span(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        segment = self._segment(1.0, 1.1)
        with self.assertRaisesRegex(ValueError, 'no frames overlap'):
            cnn_feature.slice_segment(segment)


def _phrase_cnn_feature(start_seconds=0.0, n_frames=5, dim=2, collar=500,
    object_type='Phrase'):
    phraser_object = SimpleNamespace(start_seconds=start_seconds,
        object_type=object_type)
    metadata = SimpleNamespace(
        echoframe_key=b'abc',
        phraser_key=_pk('phrase-1'),
        model_name='wav2vec2',
        output_type='cnn',
        collar=collar,
        phraser_object=phraser_object,
    )
    data = np.arange(n_frames * dim).reshape(n_frames, dim).astype(float)
    return CNNFeature(b'abc', SimpleNamespace(), metadata=metadata, data=data)


class TestCNNFeatureSubFeature(unittest.TestCase):
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

    def test_returns_sliced_cnn_feature_with_parent_metadata(self):
        parent = _phrase_cnn_feature(n_frames=5, collar=500)
        phone = self._phone(0.02, 0.05)
        sliced = parent.sub_feature(phone)
        self.assertIsInstance(sliced, SlicedCNNFeature)
        self.assertIs(sliced.parent_feature, parent)
        self.assertIs(sliced.phraser_object, phone)
        self.assertEqual(sliced.parent_class, 'Phrase')
        self.assertEqual(sliced.parent_collar, 500)
        self.assertEqual(sliced.parent_phraser_key, parent.phraser_key)
        self.assertEqual(sliced.model_name, 'wav2vec2')
        self.assertEqual(sliced.output_type, 'cnn')

    def test_default_data_matches_descendant_rows(self):
        parent = _phrase_cnn_feature(n_frames=5)
        phone = self._phone(0.02, 0.05)
        sliced = parent.sub_feature(phone)
        self.assertEqual(sliced.rows, [0, 1, 2])
        np.testing.assert_array_equal(sliced.data, parent.data[[0, 1, 2]])
        np.testing.assert_array_equal(sliced.data, parent.slice_segment(phone))
        self.assertEqual(sliced.shape, sliced.data.shape)

    def test_aggregate_mean_returns_1d_vector(self):
        parent = _phrase_cnn_feature(n_frames=5)
        phone = self._phone(0.02, 0.05)
        sliced = parent.sub_feature(phone, aggregate='mean')
        expected = parent.data[[0, 1, 2]].mean(axis=0)
        np.testing.assert_array_equal(sliced.data, expected)
        self.assertEqual(sliced.data.ndim, 1)

    def test_aggregate_middle_returns_single_row(self):
        parent = _phrase_cnn_feature(n_frames=5)
        phone = self._phone(0.02, 0.05)
        sliced = parent.sub_feature(phone, aggregate='middle')
        self.assertEqual(len(sliced.rows), 1)
        np.testing.assert_array_equal(sliced.data, parent.data[sliced.rows[0]])

    def test_invalid_aggregate_raises(self):
        parent = _phrase_cnn_feature(n_frames=5)
        phone = self._phone(0.02, 0.05)
        with self.assertRaisesRegex(ValueError,
            "aggregate must be None, 'mean', or 'middle'"):
            parent.sub_feature(phone, aggregate='max')

    def test_raises_when_descendant_outside_span(self):
        parent = _phrase_cnn_feature(n_frames=5)
        phone = self._phone(1.0, 1.1)
        with self.assertRaisesRegex(ValueError, 'no frames overlap'):
            parent.sub_feature(phone)

    def test_repr_includes_class_and_parent_class(self):
        parent = _phrase_cnn_feature(n_frames=5)
        sliced = parent.sub_feature(self._phone(0.02, 0.05))
        self.assertEqual(sliced.object_class, 'Phone')
        self.assertEqual(repr(sliced),
            f'SlicedCNNFeature(shape={sliced.shape}, '
            f'class=Phone, parent_class=Phrase, rows=[0..2])')

    def test_repr_rows_single_frame(self):
        parent = _phrase_cnn_feature(n_frames=5)
        sliced = parent.sub_feature(self._phone(0.02, 0.05),
            aggregate='middle')
        self.assertEqual(sliced.rows, [1])
        self.assertIn('rows=[1]', repr(sliced))


class TestCNNFeatureSubFeatures(unittest.TestCase):
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

    def _cnn_feature(self, n_frames=5, dim=2, **descendants):
        phraser_object = SimpleNamespace(start_seconds=0.0,
            object_type='Phrase', **descendants)
        metadata = SimpleNamespace(
            echoframe_key=b'abc',
            phraser_key=_pk('phrase-1'),
            model_name='wav2vec2',
            output_type='cnn',
            collar=500,
            phraser_object=phraser_object,
        )
        data = np.arange(n_frames * dim).reshape(n_frames, dim).astype(float)
        return CNNFeature(b'abc', SimpleNamespace(), metadata=metadata,
            data=data)

    def test_returns_one_sliced_feature_per_descendant(self):
        phones = [self._phone(0.02, 0.05), self._phone(0.06, 0.09)]
        cnn_feature = self._cnn_feature(phones=phones)

        result = cnn_feature.sub_features('phone')

        self.assertEqual(len(result), 2)
        for sub, phone in zip(result, phones):
            self.assertIsInstance(sub, SlicedCNNFeature)
            self.assertIs(sub.phraser_object, phone)
            expected = cnn_feature.sub_feature(phone)
            self.assertEqual(sub.rows, expected.rows)
            np.testing.assert_array_equal(sub.data, expected.data)

    def test_accepts_singular_plural_and_mixed_case(self):
        phone = self._phone(0.02, 0.05)
        cnn_feature = self._cnn_feature(phones=[phone])

        for name in ('phone', 'phones', 'Phone', 'PHONES'):
            result = cnn_feature.sub_features(name)
            self.assertEqual(len(result), 1)
            self.assertIs(result[0].phraser_object, phone)

    def test_aggregate_is_forwarded(self):
        phones = [self._phone(0.02, 0.05), self._phone(0.06, 0.09)]
        cnn_feature = self._cnn_feature(phones=phones)

        result = cnn_feature.sub_features('phone', aggregate='mean')

        for sub, phone in zip(result, phones):
            self.assertEqual(sub.data.ndim, 1)
            np.testing.assert_array_equal(sub.data,
                cnn_feature.sub_feature(phone, aggregate='mean').data)

    def test_percentage_overlap_is_forwarded(self):
        phone = self._phone(0.02, 0.05)
        cnn_feature = self._cnn_feature(phones=[phone])

        result = cnn_feature.sub_features('phone', percentage_overlap=100)

        np.testing.assert_array_equal(result[0].data,
            cnn_feature.sub_feature(phone, percentage_overlap=100).data)

    def test_empty_descendant_list_returns_empty_list(self):
        cnn_feature = self._cnn_feature(phones=[])
        self.assertEqual(cnn_feature.sub_features('phone'), [])

    def test_raises_for_missing_descendant_class(self):
        cnn_feature = self._cnn_feature(phones=[self._phone(0.02, 0.05)])

        with self.assertRaisesRegex(ValueError,
            "Phrase has no descendant 'word'"):
            cnn_feature.sub_features('word')

    def test_raises_when_a_descendant_is_outside_span(self):
        phones = [self._phone(0.02, 0.05), self._phone(1.0, 1.1)]
        cnn_feature = self._cnn_feature(phones=phones)

        with self.assertRaisesRegex(ValueError, 'no frames overlap'):
            cnn_feature.sub_features('phone')


def _cnn_feature_with_data(data, start_seconds=0.0):
    phraser_object = SimpleNamespace(start_seconds=start_seconds,
        object_type='Phrase')
    metadata = SimpleNamespace(
        echoframe_key=b'abc',
        phraser_key=_pk('phrase-1'),
        model_name='wav2vec2',
        output_type='cnn',
        phraser_object=phraser_object,
    )
    return CNNFeature(b'abc', SimpleNamespace(), metadata=metadata, data=data)


def _segment(start_seconds, end_seconds):
    return SimpleNamespace(
        start=int(start_seconds * 1000),
        end=int(end_seconds * 1000),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


class TestCNNFeatureMiddleFrame(unittest.TestCase):
    # grid (start=0, stride=0.02, field=0.025):
    # 0:[0.000,0.025] 1:[0.020,0.045] 2:[0.040,0.065]
    # 3:[0.060,0.085] 4:[0.080,0.105]
    def test_middle_of_odd_selection(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        result = cnn_feature.middle_frame_time(0.02, 0.05)
        np.testing.assert_array_equal(result, cnn_feature.data[1])

    def test_middle_of_even_selection(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        result = cnn_feature.middle_frame_time(0.0, 0.04)
        np.testing.assert_array_equal(result, cnn_feature.data[0])

    def test_single_frame_selection(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        result = cnn_feature.middle_frame_time(0.001, 0.015)
        np.testing.assert_array_equal(result, cnn_feature.data[0])

    def test_returns_1d_row(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        result = cnn_feature.middle_frame_time(0.02, 0.05)
        self.assertEqual(result.shape, (2,))

    def test_percentage_overlap_is_forwarded(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        result = cnn_feature.middle_frame_time(0.02, 0.05,
            percentage_overlap=100)
        np.testing.assert_array_equal(result, cnn_feature.data[1])

    def test_raises_when_interval_outside_span(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        with self.assertRaisesRegex(ValueError,
            'no frames overlap 1.000-1.100s'):
            cnn_feature.middle_frame_time(1.0, 1.1)

    def test_segment_delegates(self):
        cnn_feature = _slice_cnn_feature(n_frames=5)
        result = cnn_feature.middle_frame_segment(_segment(0.02, 0.05))
        np.testing.assert_array_equal(result,
            cnn_feature.middle_frame_time(0.02, 0.05))


class TestCNNFeatureAggregate(unittest.TestCase):
    # grid (start=0, stride=0.02, field=0.025):
    # 0:[0.000,0.025] 1:[0.020,0.045] 2:[0.040,0.065]
    # 3:[0.060,0.085] 4:[0.080,0.105]
    # data chosen so mean != middle row over selections
    def _cnn_feature(self):
        data = np.array(
            [[0.0, 0.0], [10.0, 10.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        return _cnn_feature_with_data(data)

    def test_mean_matches_manual_average(self):
        cnn_feature = self._cnn_feature()
        result = cnn_feature.aggregate_time(0.0, 0.105, 'mean')
        np.testing.assert_array_equal(result, cnn_feature.data.mean(axis=0))

    def test_mean_subinterval(self):
        cnn_feature = self._cnn_feature()
        result = cnn_feature.aggregate_time(0.02, 0.05, 'mean')
        np.testing.assert_array_equal(result,
            cnn_feature.data[[0, 1, 2]].mean(axis=0))

    def test_middle_delegates(self):
        cnn_feature = self._cnn_feature()
        result = cnn_feature.aggregate_time(0.02, 0.05, 'middle')
        np.testing.assert_array_equal(result,
            cnn_feature.middle_frame_time(0.02, 0.05))

    def test_default_method_is_mean(self):
        cnn_feature = self._cnn_feature()
        np.testing.assert_array_equal(
            cnn_feature.aggregate_time(0.02, 0.05),
            cnn_feature.aggregate_time(0.02, 0.05, 'mean'))

    def test_returns_1d_for_both_methods(self):
        cnn_feature = self._cnn_feature()
        self.assertEqual(cnn_feature.aggregate_time(0.02, 0.05, 'mean').shape,
            (2,))
        self.assertEqual(cnn_feature.aggregate_time(0.02, 0.05, 'middle').shape,
            (2,))

    def test_invalid_method_raises(self):
        cnn_feature = self._cnn_feature()
        with self.assertRaisesRegex(ValueError,
            "method must be 'mean' or 'middle'"):
            cnn_feature.aggregate_time(0.02, 0.05, 'bad')

    def test_percentage_overlap_forwarded(self):
        cnn_feature = self._cnn_feature()
        result = cnn_feature.aggregate_time(0.02, 0.05, 'mean',
            percentage_overlap=100)
        np.testing.assert_array_equal(result,
            cnn_feature.data[[1]].mean(axis=0))

    def test_raises_when_interval_outside_span(self):
        cnn_feature = self._cnn_feature()
        for method in ('mean', 'middle'):
            with self.assertRaisesRegex(ValueError, 'no frames overlap'):
                cnn_feature.aggregate_time(1.0, 1.1, method)

    def test_segment_delegates(self):
        cnn_feature = self._cnn_feature()
        segment = _segment(0.02, 0.05)
        for method in ('mean', 'middle'):
            np.testing.assert_array_equal(
                cnn_feature.aggregate_segment(segment, method=method),
                cnn_feature.aggregate_time(0.02, 0.05, method=method))


class TestStoreCnnFeatureLoaders(unittest.TestCase):
    def test_load_cnn_feature_returns_cnn_feature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            ensure_model(store, 'wav2vec2')
            data = np.arange(6).reshape(2, 3).astype(float)
            phraser_key = _pk('phrase-1')
            item = _put(store, phraser_key=phraser_key, collar=500,
                model_name='wav2vec2', output_type='cnn', layer=None,
                data=data)

            result = store.load_cnn_feature(item.echoframe_key)

            self.assertIsInstance(result, CNNFeature)
            self.assertEqual(result.phraser_key, phraser_key)
            self.assertEqual(result.model_name, 'wav2vec2')
            self.assertEqual(result.output_type, 'cnn')
            np.testing.assert_array_equal(result.data, data)

    def test_load_cnn_features_returns_cnn_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            ensure_model(store, 'wav2vec2')
            data_1 = np.arange(6).reshape(2, 3).astype(float)
            data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
            item_1 = _put(store, phraser_key='phrase-1', collar=500,
                model_name='wav2vec2', output_type='cnn', layer=None,
                data=data_1)
            item_2 = _put(store, phraser_key='phrase-2', collar=500,
                model_name='wav2vec2', output_type='cnn', layer=None,
                data=data_2)

            result = store.load_cnn_features(
                [item_1.echoframe_key, item_2.echoframe_key])

            self.assertIsInstance(result, CNNFeatures)
            self.assertEqual(result.count, 2)
            np.testing.assert_array_equal(result.to_numpy(), np.stack([
                data_1,
                data_2,
            ], axis=0))

    def test_phraser_key_to_cnn_feature_matches_direct_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            ensure_model(store, 'wav2vec2')
            data = np.arange(6).reshape(2, 3).astype(float)
            phraser_key = _pk('phrase-1')
            item = _put(store, phraser_key=phraser_key, collar=500,
                model_name='wav2vec2', output_type='cnn', layer=None,
                data=data)

            result = store.phraser_key_to_cnn_feature(phraser_key,
                'wav2vec2', collar=500)

            self.assertIsInstance(result, CNNFeature)
            self.assertEqual(result.echoframe_key, item.echoframe_key)
            np.testing.assert_array_equal(result.data, data)

    def test_phraser_keys_to_cnn_features_matches_direct_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            ensure_model(store, 'wav2vec2')
            data_1 = np.arange(6).reshape(2, 3).astype(float)
            data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
            phraser_key_1 = _pk('phrase-1')
            phraser_key_2 = _pk('phrase-2')
            _put(store, phraser_key=phraser_key_1, collar=500,
                model_name='wav2vec2', output_type='cnn', layer=None,
                data=data_1)
            _put(store, phraser_key=phraser_key_2, collar=500,
                model_name='wav2vec2', output_type='cnn', layer=None,
                data=data_2)

            result = store.phraser_keys_to_cnn_features(
                [phraser_key_1, phraser_key_2], 'wav2vec2', collar=500)

            self.assertIsInstance(result, CNNFeatures)
            self.assertEqual(result.phraser_keys,
                (phraser_key_1, phraser_key_2))
            np.testing.assert_array_equal(result.to_numpy(), np.stack([
                data_1,
                data_2,
            ], axis=0))


if __name__ == '__main__':
    unittest.main()
