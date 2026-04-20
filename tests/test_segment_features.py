'''Tests for segment-based feature retrieval orchestration.'''

from __future__ import annotations

from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from echoframe import Store
from echoframe.codebooks import Codebook, TokenCodebooks
from echoframe.embeddings import Embeddings, TokenEmbeddings
from echoframe.index import LmdbIndex
from echoframe.metadata import metadata_class_for_output_type
from echoframe.output_storage import Hdf5ShardStore
import echoframe.segment_features as segment_features
from echoframe.segment_features import (
    get_codebook_indices,
    get_codebook_indices_batch,
    get_embeddings,
    get_embeddings_batch,
    segment_to_echoframe_key,
)
from tests.helpers import (
    ensure_model as _ensure_model,
    FakeEnv,
    FakeH5Module,
    pk as _pk,
)


def _make_store():
    tmpdir = tempfile.TemporaryDirectory()
    index = LmdbIndex(Path(tmpdir.name) / 'index', env=FakeEnv(),
        shards_root=Path(tmpdir.name) / 'shards')
    storage = Hdf5ShardStore(Path(tmpdir.name) / 'shards',
        h5_module=FakeH5Module())
    store = Store(tmpdir.name, index=index, storage=storage)
    _ensure_model(store, 'wav2vec2')
    return tmpdir, store


def _put(store, *, phraser_key, collar, model_name, output_type, layer,
    data, tags=None):
    phraser_key = _pk(phraser_key)
    _ensure_model(store, model_name)
    key_kwargs = {
        'output_type': output_type,
        'model_name': model_name,
    }
    if output_type in {'hidden_state', 'attention'}:
        key_kwargs.update({
            'phraser_key': phraser_key,
            'layer': layer,
            'collar': collar,
        })
    elif output_type == 'codebook_indices':
        key_kwargs.update({
            'phraser_key': phraser_key,
            'collar': collar,
        })
    metadata_cls = metadata_class_for_output_type(output_type)
    metadata = metadata_cls(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=layer, tags=tags,
        echoframe_key=store.make_echoframe_key(**key_kwargs))
    return store.save(metadata.echoframe_key, metadata, data)


def _make_segment(start=1000, end=1300, key=None,
    filename=None, duration=None):
    if key is None:
        key = _pk('aabb')
    if filename is None:
        filename = str(Path(__file__).resolve())
    audio = types.SimpleNamespace(filename=filename)
    if duration is not None:
        audio.duration = duration
    return types.SimpleNamespace(key=key, start=start, end=end, audio=audio)


def _put_hidden_state(store, phraser_key, collar, model_name, layer, data):
    _put(store, phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='hidden_state', layer=layer, data=data)


def _put_codebook(store, phraser_key, collar, model_name, indices, matrix):
    _put(store, phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='codebook_indices', layer=0, data=indices)
    _put(store, phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='codebook_matrix', layer=0, data=matrix)


def _patch_runtime_dependencies(outputs=None, indices=None, matrix=None,
    frame_indices=None, batch_outputs=None):
    fake_frames = types.SimpleNamespace(
        select_frames=mock.Mock(return_value=[
            types.SimpleNamespace(index=index) for index in frame_indices or []
        ]),
    )
    fake_to_vector = types.SimpleNamespace(
        filename_to_vector=mock.Mock(return_value=outputs),
        filename_batch_to_vector=mock.Mock(return_value=batch_outputs or []),
        filename_to_codebook_artifacts=mock.Mock(return_value=types.SimpleNamespace(
            indices=indices, codebook_matrix=matrix,
        )),
    )
    fake_frame = types.SimpleNamespace(
        make_frames_from_outputs=mock.Mock(return_value=fake_frames),
    )
    return fake_to_vector, fake_frame, fake_frames


class SegmentFeatureTests(unittest.TestCase):
    def test_segment_to_echoframe_key_handles_bytes_and_strings(self):
        self.assertEqual(segment_to_echoframe_key(
            types.SimpleNamespace(key=_pk('aabb'))), _pk('aabb'))
        with self.assertRaises(TypeError):
            segment_to_echoframe_key(types.SimpleNamespace(key='my-key'))

    def test_segment_to_echoframe_key_missing_key_raises(self):
        with self.assertRaises(ValueError):
            segment_to_echoframe_key(types.SimpleNamespace())


class TestGetEmbeddings(unittest.TestCase):
    def test_cache_hit_works_without_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data = np.arange(6).reshape(2, 3).astype(float)
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 4, data)
            segment = _make_segment()

            result = get_embeddings(segment, layers=4, collar=500,
                model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, Embeddings)
            np.testing.assert_array_equal(result.to_numpy(), data)

    def test_cache_miss_requires_loaded_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(duration=2000)
            outputs = types.SimpleNamespace(hidden_states=[
                np.zeros((1, 5, 2)),
                np.ones((1, 5, 2)),
            ])
            fake_to_vector, fake_frame, _ = _patch_runtime_dependencies(
                outputs=outputs,
                frame_indices=[1, 2],
            )
            with mock.patch.object(segment_features, 'to_vector',
                fake_to_vector), mock.patch.object(segment_features, 'frame',
                fake_frame):
                with self.assertRaises(ValueError):
                    get_embeddings(segment, layers=1, collar=500,
                        model_name='wav2vec2', store=store, model=None)

                result = get_embeddings(segment, layers=1, collar=500,
                    model_name='wav2vec2', store=store, model=object())

            self.assertIsInstance(result, Embeddings)
            self.assertEqual(fake_to_vector.filename_to_vector.call_count, 1)
            fake_frame.make_frames_from_outputs.assert_called_once()
            self.assertEqual(result.dims, ('frames', 'embed_dim'))
            np.testing.assert_array_equal(result.to_numpy(),
                np.ones((2, 2)))

    def test_batch_returns_token_embeddings(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data_1 = np.arange(6).reshape(2, 3).astype(float)
            data_2 = np.arange(6, 12).reshape(2, 3).astype(float)
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 4, data_1)
            _put_hidden_state(store, 'ccdd', 500, 'wav2vec2', 4, data_2)
            segments = [
                _make_segment(key=_pk('aabb')),
                _make_segment(key=_pk('ccdd')),
            ]

            result = get_embeddings_batch(segments, layers=4, collar=500,
                model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, TokenEmbeddings)
            self.assertEqual(result.token_count, 2)
            np.testing.assert_array_equal(result.tokens[0].to_numpy(), data_1)
            np.testing.assert_array_equal(result.tokens[1].to_numpy(), data_2)

    def test_batch_uses_batch_compute_for_missing_segments(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segments = [
                _make_segment(key=_pk('aabb'), duration=2000),
                _make_segment(key=_pk('ccdd'), duration=2000),
            ]
            outputs_1 = types.SimpleNamespace(hidden_states=[
                np.zeros((1, 5, 2)),
                np.ones((1, 5, 2)),
            ])
            outputs_2 = types.SimpleNamespace(hidden_states=[
                np.zeros((1, 5, 2)),
                np.full((1, 5, 2), 2.0),
            ])
            fake_to_vector, fake_frame, _ = _patch_runtime_dependencies(
                batch_outputs=[outputs_1, outputs_2],
                frame_indices=[1, 2],
            )
            with mock.patch.object(segment_features, 'to_vector',
                fake_to_vector), mock.patch.object(segment_features, 'frame',
                fake_frame):
                result = get_embeddings_batch(segments, layers=1, collar=500,
                    model_name='wav2vec2', store=store, model=object(),
                    batch_minutes=3)

            self.assertIsInstance(result, TokenEmbeddings)
            self.assertEqual(result.token_count, 2)
            fake_to_vector.filename_batch_to_vector.assert_called_once()
            _, kwargs = fake_to_vector.filename_batch_to_vector.call_args
            self.assertEqual(kwargs['batch_minutes'], 3)
            self.assertEqual(kwargs['starts'], [0.5, 0.5])
            self.assertEqual(kwargs['ends'], [1.8, 1.8])

    def test_batch_filters_invalid_segments_and_warns(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data = np.arange(6).reshape(2, 3).astype(float)
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 4, data)
            segments = [
                _make_segment(key=_pk('aabb')),
                _make_segment(key=_pk('ccdd'), start=1000, end=1020),
                _make_segment(key=_pk('eeff'),
                    filename=str(Path(tmpdir.name) / 'missing.wav')),
            ]

            with self.assertWarnsRegex(UserWarning,
                'filtered 2 invalid segments'):
                result = get_embeddings_batch(segments, layers=4, collar=500,
                    model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, TokenEmbeddings)
            self.assertEqual(result.token_count, 1)
            np.testing.assert_array_equal(result.tokens[0].to_numpy(), data)
            self.assertEqual(len(result._failed_metadatas), 2)
            self.assertEqual(result._failed_metadatas[0]['reason'],
                'segment shorter than 25 ms')
            self.assertIn('audio file does not exist',
                result._failed_metadatas[1]['reason'])

    def test_batch_skips_cached_segments_in_compute_subset(self):
        tmpdir, store = _make_store()
        with tmpdir:
            cached = np.arange(6).reshape(2, 3).astype(float)
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 1, cached)
            segments = [
                _make_segment(key=_pk('aabb'), duration=2000),
                _make_segment(key=_pk('ccdd'), duration=2000),
            ]
            batch_output = types.SimpleNamespace(hidden_states=[
                np.zeros((1, 5, 3)),
                np.full((1, 5, 3), 4.0),
            ])
            fake_to_vector, fake_frame, _ = _patch_runtime_dependencies(
                batch_outputs=[batch_output],
                frame_indices=[1, 2],
            )

            with mock.patch.object(segment_features, 'to_vector',
                fake_to_vector), mock.patch.object(segment_features, 'frame',
                fake_frame):
                result = get_embeddings_batch(segments, layers=1, collar=500,
                    model_name='wav2vec2', store=store, model=object())

            self.assertEqual(result.token_count, 2)
            np.testing.assert_array_equal(result.tokens[0].to_numpy(), cached)
            np.testing.assert_array_equal(result.tokens[1].to_numpy(),
                np.full((2, 3), 4.0))
            _, kwargs = fake_to_vector.filename_batch_to_vector.call_args
            self.assertEqual(len(kwargs['starts']), 1)
            self.assertEqual(len(kwargs['ends']), 1)

    def test_batch_post_preflight_compute_failure_raises(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segments = [
                _make_segment(key=_pk('aabb'), duration=2000),
                _make_segment(key=_pk('ccdd'), duration=2000),
            ]
            bad_outputs = types.SimpleNamespace(hidden_states=None)
            fake_to_vector, fake_frame, _ = _patch_runtime_dependencies(
                batch_outputs=[bad_outputs, bad_outputs],
                frame_indices=[1, 2],
            )

            with mock.patch.object(segment_features, 'to_vector',
                fake_to_vector), mock.patch.object(segment_features, 'frame',
                fake_frame):
                with self.assertRaisesRegex(ValueError,
                    'did not contain hidden_states'):
                    get_embeddings_batch(segments, layers=1, collar=500,
                        model_name='wav2vec2', store=store, model=object())

    def test_batch_raises_when_no_valid_segments_remain(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segments = [
                _make_segment(key=_pk('aabb'), start=1000, end=1010),
                _make_segment(key=_pk('ccdd'),
                    filename=str(Path(tmpdir.name) / 'missing.wav')),
            ]

            with self.assertRaisesRegex(ValueError,
                'no valid segments remained'):
                get_embeddings_batch(segments, layers=4, collar=500,
                    model_name='wav2vec2', store=store, model=None)

    def test_duration_clips_collared_window(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(start=1000, end=1300, duration=1500)
            outputs = types.SimpleNamespace(hidden_states=[
                np.zeros((1, 5, 2)),
                np.ones((1, 5, 2)),
            ])
            fake_to_vector, fake_frame, _ = _patch_runtime_dependencies(
                outputs=outputs,
                frame_indices=[1],
            )
            with mock.patch.object(segment_features, 'to_vector',
                fake_to_vector), mock.patch.object(segment_features, 'frame',
                fake_frame):
                get_embeddings(segment, layers=1, collar=500,
                    model_name='wav2vec2', store=store, model=object())

            _, kwargs = fake_to_vector.filename_to_vector.call_args
            self.assertEqual(kwargs['start'], 0.5)
            self.assertEqual(kwargs['end'], 1.5)


class TestGetCodebookIndices(unittest.TestCase):
    def test_cache_hit_works_without_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            indices = np.array([[1, 0], [0, 1]])
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
            _put_codebook(store, 'aabb', 500, 'wav2vec2', indices, matrix)
            segment = _make_segment()

            result = get_codebook_indices(segment, collar=500,
                model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, Codebook)
            np.testing.assert_array_equal(result.to_numpy(), indices)

    def test_cache_miss_requires_loaded_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(duration=2000)
            outputs = types.SimpleNamespace(hidden_states=[
                np.zeros((1, 5, 2)),
                np.ones((1, 5, 2)),
            ])
            indices = np.array([
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 1],
            ])
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
            fake_to_vector, fake_frame, _ = _patch_runtime_dependencies(
                outputs=outputs,
                indices=indices,
                matrix=matrix,
                frame_indices=[1, 2],
            )
            with mock.patch.object(segment_features, 'to_vector',
                fake_to_vector), mock.patch.object(segment_features, 'frame',
                fake_frame):
                with self.assertRaises(ValueError):
                    get_codebook_indices(segment, collar=500,
                        model_name='wav2vec2', store=store, model=None)

                result = get_codebook_indices(segment, collar=500,
                    model_name='wav2vec2', store=store, model=object())

            self.assertIsInstance(result, Codebook)
            self.assertEqual(fake_to_vector.filename_to_vector.call_count, 1)
            fake_to_vector.filename_to_codebook_artifacts.assert_called_once()
            np.testing.assert_array_equal(result.to_numpy(), np.array([
                [1, 0],
                [0, 1],
            ]))

    def test_batch_returns_token_codebooks(self):
        tmpdir, store = _make_store()
        with tmpdir:
            indices_1 = np.array([[0, 1]])
            indices_2 = np.array([[1, 0]])
            matrix_1 = np.array([[1.0, 2.0], [3.0, 4.0]])
            matrix_2 = np.array([[5.0, 6.0], [7.0, 8.0]])
            _put_codebook(store, 'aabb', 500, 'wav2vec2', indices_1,
                matrix_1)
            _put_codebook(store, 'ccdd', 500, 'wav2vec2', indices_2,
                matrix_2)
            segments = [
                _make_segment(key=_pk('aabb')),
                _make_segment(key=_pk('ccdd')),
            ]

            result = get_codebook_indices_batch(segments, collar=500,
                model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, TokenCodebooks)
            self.assertEqual(result.token_count, 2)
            np.testing.assert_array_equal(result.tokens[0].to_numpy(),
                indices_1)
            np.testing.assert_array_equal(result.tokens[1].to_numpy(),
                indices_2)

    def test_batch_reuses_single_item_codebook_function(self):
        tmpdir, store = _make_store()
        with tmpdir:
            indices = np.array([[0, 1]])
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
            _put_codebook(store, 'aabb', 500, 'wav2vec2', indices, matrix)
            segments = [_make_segment(key=_pk('aabb'))]

            original = segment_features.get_codebook_indices
            with mock.patch.object(segment_features, 'get_codebook_indices',
                wraps=original) as patched:
                result = get_codebook_indices_batch(segments, collar=500,
                    model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, TokenCodebooks)
            patched.assert_called_once()

    def test_codebook_batch_skip_is_default_and_omits_failed_items(self):
        tmpdir, store = _make_store()
        with tmpdir:
            indices = np.array([[0, 1]])
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
            _put_codebook(store, 'aabb', 500, 'wav2vec2', indices, matrix)
            segments = [
                _make_segment(key=_pk('aabb')),
                _make_segment(key=_pk('ccdd')),
            ]

            result = get_codebook_indices_batch(segments, collar=500,
                model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, TokenCodebooks)
            self.assertEqual(result.token_count, 1)
            self.assertEqual(result.echoframe_keys, (
                store.make_echoframe_key(
                    'codebook_indices',
                    model_name='wav2vec2',
                    phraser_key=_pk('aabb'),
                    collar=500,
                ),
            ))
            np.testing.assert_array_equal(result.tokens[0].to_numpy(), indices)

    def test_codebook_batch_raise_propagates_first_failure(self):
        tmpdir, store = _make_store()
        with tmpdir:
            indices = np.array([[0, 1]])
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
            _put_codebook(store, 'aabb', 500, 'wav2vec2', indices, matrix)
            segments = [
                _make_segment(key=_pk('aabb')),
                _make_segment(key=_pk('ccdd')),
            ]

            with self.assertRaises(ValueError):
                get_codebook_indices_batch(segments, collar=500,
                    model_name='wav2vec2', store=store, model=None,
                    on_error='raise')

    def test_codebook_batch_skip_raises_when_all_items_fail(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segments = [
                _make_segment(key=_pk('aabb')),
                _make_segment(key=_pk('ccdd')),
            ]

            with self.assertRaisesRegex(ValueError,
                'no codebook indices succeeded'):
                get_codebook_indices_batch(segments, collar=500,
                    model_name='wav2vec2', store=store, model=None)

    def test_duration_clips_collared_window(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(start=1000, end=1300, duration=1500)
            outputs = types.SimpleNamespace(hidden_states=[
                np.zeros((1, 5, 2)),
                np.ones((1, 5, 2)),
            ])
            indices = np.array([
                [0, 1],
                [1, 0],
                [0, 0],
                [1, 1],
                [0, 1],
            ])
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
            fake_to_vector, fake_frame, _ = _patch_runtime_dependencies(
                outputs=outputs,
                indices=indices,
                matrix=matrix,
                frame_indices=[1],
            )
            with mock.patch.object(segment_features, 'to_vector',
                fake_to_vector), mock.patch.object(segment_features, 'frame',
                fake_frame):
                get_codebook_indices(segment, collar=500,
                    model_name='wav2vec2', store=store, model=object())

            _, kwargs = fake_to_vector.filename_to_vector.call_args
            self.assertEqual(kwargs['start'], 0.5)
            self.assertEqual(kwargs['end'], 1.5)


if __name__ == '__main__':
    unittest.main()
