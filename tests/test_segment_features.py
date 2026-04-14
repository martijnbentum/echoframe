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
from echoframe.output_storage import Hdf5ShardStore
import echoframe.segment_features as segment_features
from echoframe.segment_features import (
    get_codebook_indices,
    get_codebook_indices_batch,
    get_embeddings,
    get_embeddings_batch,
    segment_to_echoframe_key,
)
from tests.test_public_api import FakeEnv, FakeH5Module


def _make_store():
    tmpdir = tempfile.TemporaryDirectory()
    index = LmdbIndex(Path(tmpdir.name) / 'index', env=FakeEnv(),
        shards_root=Path(tmpdir.name) / 'shards')
    storage = Hdf5ShardStore(Path(tmpdir.name) / 'shards',
        h5_module=FakeH5Module())
    store = Store(tmpdir.name, index=index, storage=storage)
    return tmpdir, store


def _make_segment(start=1000, end=1300, key=b'\xaa\xbb',
    filename='audio.wav', duration=None):
    audio = types.SimpleNamespace(filename=filename)
    if duration is not None:
        audio.duration = duration
    return types.SimpleNamespace(key=key, start=start, end=end, audio=audio)


def _put_hidden_state(store, phraser_key, collar, model_name, layer, data):
    store.put(phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='hidden_state', layer=layer, data=data)


def _put_codebook(store, phraser_key, collar, model_name, indices, matrix):
    store.put(phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='codebook_indices', layer=0, data=indices)
    store.put(phraser_key=phraser_key, collar=collar, model_name=model_name,
        output_type='codebook_matrix', layer=0, data=matrix)


def _patch_runtime_dependencies(outputs=None, indices=None, matrix=None,
    frame_indices=None):
    fake_frames = types.SimpleNamespace(
        select_frames=mock.Mock(return_value=[
            types.SimpleNamespace(index=index) for index in frame_indices or []
        ]),
    )
    fake_to_vector = types.SimpleNamespace(
        filename_to_vector=mock.Mock(return_value=outputs),
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
            types.SimpleNamespace(key=b'\xaa\xbb')), 'aabb')
        self.assertEqual(segment_to_echoframe_key(
            types.SimpleNamespace(key='my-key')), 'my-key')

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
                _make_segment(key=b'\xaa\xbb'),
                _make_segment(key=b'\xcc\xdd'),
            ]

            result = get_embeddings_batch(segments, layers=4, collar=500,
                model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, TokenEmbeddings)
            self.assertEqual(result.token_count, 2)
            np.testing.assert_array_equal(result.tokens[0].to_numpy(), data_1)
            np.testing.assert_array_equal(result.tokens[1].to_numpy(), data_2)

    def test_batch_uses_typed_store_loader(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data = np.arange(6).reshape(2, 3).astype(float)
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 4, data)
            segments = [_make_segment(key=b'\xaa\xbb')]

            original = store.load_many_embeddings
            with mock.patch.object(store, 'load_many_embeddings',
                wraps=original) as patched:
                result = get_embeddings_batch(segments, layers=4, collar=500,
                    model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, TokenEmbeddings)
            patched.assert_called_once()

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
                _make_segment(key=b'\xaa\xbb'),
                _make_segment(key=b'\xcc\xdd'),
            ]

            result = get_codebook_indices_batch(segments, collar=500,
                model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, TokenCodebooks)
            self.assertEqual(result.token_count, 2)
            np.testing.assert_array_equal(result.tokens[0].to_numpy(),
                indices_1)
            np.testing.assert_array_equal(result.tokens[1].to_numpy(),
                indices_2)

    def test_batch_uses_typed_codebook_loader(self):
        tmpdir, store = _make_store()
        with tmpdir:
            indices = np.array([[0, 1]])
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
            _put_codebook(store, 'aabb', 500, 'wav2vec2', indices, matrix)
            segments = [_make_segment(key=b'\xaa\xbb')]

            original = store.load_many_codebooks
            with mock.patch.object(store, 'load_many_codebooks',
                wraps=original) as patched:
                result = get_codebook_indices_batch(segments, collar=500,
                    model_name='wav2vec2', store=store, model=None)

            self.assertIsInstance(result, TokenCodebooks)
            patched.assert_called_once()

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
