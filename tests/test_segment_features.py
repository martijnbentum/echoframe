'''Tests for the current single-segment feature helpers.'''

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from echoframe import Store
from echoframe.codebooks import Codebook
from echoframe.embeddings import Embeddings
from echoframe.index import LmdbIndex
from echoframe.metadata import EchoframeMetadata
from echoframe.output_storage import Hdf5ShardStore

sys.modules.setdefault('frame', types.SimpleNamespace())
sys.modules.setdefault('to_vector', types.SimpleNamespace())

import echoframe.segment_features as segment_features
from echoframe.segment_features import (
    _segment_times,
    get_codebook_indices,
    get_embeddings,
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


def _put_hidden_state(store, phraser_key, collar, model_name, layer, data):
    phraser_key = _pk(phraser_key)
    echoframe_key = store.make_echoframe_key('hidden_state',
        model_name=model_name, phraser_key=phraser_key, layer=layer,
        collar=collar)
    metadata = EchoframeMetadata(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=layer, echoframe_key=echoframe_key,
        output_type='hidden_state')
    store.save(metadata.echoframe_key, metadata, data)


def _put_codebook_indices(store, phraser_key, collar, model_name, data):
    phraser_key = _pk(phraser_key)
    echoframe_key = store.make_echoframe_key('codebook_indices',
        model_name=model_name, phraser_key=phraser_key, collar=collar)
    metadata = EchoframeMetadata(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=0, echoframe_key=echoframe_key,
        output_type='codebook_indices')
    store.save(metadata.echoframe_key, metadata, data)


def _put_codebook_matrix(store, phraser_key, collar, model_name, data):
    phraser_key = _pk(phraser_key)
    echoframe_key = store.make_echoframe_key('codebook_matrix',
        model_name=model_name)
    metadata = EchoframeMetadata(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=0, echoframe_key=echoframe_key,
        output_type='codebook_matrix')
    store.save(metadata.echoframe_key, metadata, np.asarray(data))


def _make_segment(start_seconds=1.0, end_seconds=1.3, key=None,
    filename=None, duration=2000):
    if key is None:
        key = _pk('aabb')
    if filename is None:
        filename = str(Path(__file__).resolve())
    audio = types.SimpleNamespace(filename=filename, duration=duration)
    return types.SimpleNamespace(key=key, start_seconds=start_seconds,
        end_seconds=end_seconds, audio=audio)


def _make_selected_frames(indices):
    return [types.SimpleNamespace(index=index) for index in indices]


class FakeFrames:
    def __init__(self, n_frames=None, start_time=None, selected_indices=None):
        self.n_frames = n_frames
        self.start_time = start_time
        self.selected_indices = [] if selected_indices is None else (
            list(selected_indices))
        self.select_frames = mock.Mock(
            return_value=_make_selected_frames(self.selected_indices))


class TestSegmentTimes(unittest.TestCase):
    def test_negative_collar_raises(self):
        segment = _make_segment()

        with self.assertRaisesRegex(ValueError, 'collar must be non-negative'):
            _segment_times(segment, -1)

    def test_end_must_exceed_start(self):
        segment = _make_segment(start_seconds=1.0, end_seconds=1.0)

        with self.assertRaisesRegex(ValueError,
            'segment end must be greater than start'):
            _segment_times(segment, 500)

    def test_duration_clips_collared_window(self):
        segment = _make_segment(start_seconds=1.0, end_seconds=1.3,
            duration=1500)

        start, end, collared_start, collared_end = _segment_times(segment,
            500)

        self.assertEqual(start, 1.0)
        self.assertEqual(end, 1.3)
        self.assertEqual(collared_start, 0.5)
        self.assertEqual(collared_end, 1.5)


class TestGetEmbeddings(unittest.TestCase):
    def test_cache_hit_works_without_loading_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            data = np.arange(6).reshape(2, 3).astype(float)
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 4, data)
            segment = _make_segment()
            store.load_model = mock.Mock()

            with mock.patch.object(segment_features.to_vector,
                'filename_to_vector', create=True) as compute:
                result = get_embeddings(segment, layers=[4], collar=500,
                    model_name='wav2vec2', store=store)

            self.assertIsInstance(result, Embeddings)
            np.testing.assert_array_equal(result.to_numpy(), data[None, ...])
            compute.assert_not_called()
            store.load_model.assert_not_called()

    def test_cache_miss_computes_and_stores_selected_frames(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment()
            model = object()
            outputs = types.SimpleNamespace(hidden_states=[
                np.zeros((1, 5, 2)),
                np.arange(10).reshape(1, 5, 2),
                np.arange(10, 20).reshape(1, 5, 2),
            ])
            fake_frames = FakeFrames(selected_indices=[1, 3])
            fake_frame = types.SimpleNamespace(
                make_frames_from_outputs=mock.Mock(return_value=fake_frames),
            )
            fake_to_vector = types.SimpleNamespace(
                filename_to_vector=mock.Mock(return_value=outputs),
            )
            store.load_model = mock.Mock(return_value=model)

            with mock.patch.object(segment_features, 'frame', fake_frame), (
                mock.patch.object(segment_features, 'to_vector',
                fake_to_vector)
            ):
                result = get_embeddings(segment, layers=[1, 2],
                    model_name='wav2vec2', collar=500, store=store, gpu=True,
                    tags=['fresh'])

            np.testing.assert_array_equal(result.to_numpy(), np.array([
                [[2, 3], [6, 7]],
                [[12, 13], [16, 17]],
            ]))
            store.load_model.assert_called_once_with('wav2vec2', gpu=True)
            fake_to_vector.filename_to_vector.assert_called_once_with(
                segment.audio.filename,
                start=0.5,
                end=1.8,
                model=model,
                gpu=True,
                numpify_output=True,
            )
            fake_frame.make_frames_from_outputs.assert_called_once_with(
                outputs,
                start_time=0.5,
            )


class TestGetCodebookIndices(unittest.TestCase):
    def test_cache_hit_returns_codebook_without_loading_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment()
            indices = np.array([[1, 3], [0, 2]])
            matrix = np.arange(24).reshape(6, 4)
            _put_codebook_indices(store, 'aabb', 500, 'wav2vec2', indices)
            _put_codebook_matrix(store, 'aabb', 500, 'wav2vec2', matrix)
            store.load_model = mock.Mock()

            with mock.patch.object(segment_features.to_vector,
                'filename_to_codebook_artifacts', create=True) as compute:
                result = get_codebook_indices(segment, model_name='wav2vec2',
                    collar=500, store=store)

            self.assertIsInstance(result, Codebook)
            np.testing.assert_array_equal(result.to_numpy(), indices)
            result.bind_store(store)
            np.testing.assert_array_equal(result.codebook_matrix, matrix)
            compute.assert_not_called()
            store.load_model.assert_not_called()

    def test_cache_miss_stores_indices_and_matrix(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment()
            model = object()
            artifacts = types.SimpleNamespace(
                indices=np.array([[2, 4], [5, 1], [3, 0], [4, 2]]),
                codebook_matrix=np.arange(24).reshape(6, 4),
            )
            fake_frames = FakeFrames(selected_indices=[1, 2])
            fake_frame = types.SimpleNamespace(
                Frames=mock.Mock(return_value=fake_frames),
            )
            fake_to_vector = types.SimpleNamespace(
                filename_to_codebook_artifacts=mock.Mock(
                    return_value=artifacts),
            )
            store.load_model = mock.Mock(return_value=model)

            with mock.patch.object(segment_features, 'frame', fake_frame), (
                mock.patch.object(segment_features, 'to_vector',
                fake_to_vector)
            ):
                result = get_codebook_indices(segment, model_name='wav2vec2',
                    collar=500, store=store, gpu=True, tags=['fresh'])

            np.testing.assert_array_equal(result.to_numpy(),
                np.array([[5, 1], [3, 0]]))
            result.bind_store(store)
            np.testing.assert_array_equal(result.codebook_matrix,
                artifacts.codebook_matrix)
            store.load_model.assert_called_once_with('wav2vec2', gpu=True)
            fake_to_vector.filename_to_codebook_artifacts.\
                assert_called_once_with(
                    segment.audio.filename,
                    start=0.5,
                    end=1.8,
                    model=model,
                    gpu=True,
                )
            fake_frame.Frames.assert_called_once_with(4, start_time=0.5)

    def test_cache_miss_does_not_overwrite_existing_matrix(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment()
            model = object()
            existing_matrix = np.arange(24).reshape(6, 4)
            new_matrix = np.arange(100, 124).reshape(6, 4)
            _put_codebook_matrix(store, 'older', 250, 'wav2vec2',
                existing_matrix)
            artifacts = types.SimpleNamespace(
                indices=np.array([[1, 2], [3, 4], [0, 5]]),
                codebook_matrix=new_matrix,
            )
            fake_frames = FakeFrames(selected_indices=[0, 2])
            fake_frame = types.SimpleNamespace(
                Frames=mock.Mock(return_value=fake_frames),
            )
            fake_to_vector = types.SimpleNamespace(
                filename_to_codebook_artifacts=mock.Mock(
                    return_value=artifacts),
            )
            store.load_model = mock.Mock(return_value=model)

            with mock.patch.object(segment_features, 'frame', fake_frame), (
                mock.patch.object(segment_features, 'to_vector',
                fake_to_vector)
            ):
                result = get_codebook_indices(segment, model_name='wav2vec2',
                    collar=500, store=store)

            np.testing.assert_array_equal(result.to_numpy(),
                np.array([[1, 2], [0, 5]]))
            result.bind_store(store)
            np.testing.assert_array_equal(result.codebook_matrix,
                existing_matrix)
            store.load_model.assert_called_once_with('wav2vec2', gpu=False)


if __name__ == '__main__':
    unittest.main()
