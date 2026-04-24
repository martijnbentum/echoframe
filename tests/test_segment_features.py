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
from echoframe.index import LmdbIndex
from echoframe.metadata import EchoframeMetadata
from echoframe.output_storage import Hdf5ShardStore

sys.modules.setdefault('frame', types.SimpleNamespace())
sys.modules.setdefault('to_vector', types.SimpleNamespace())

import echoframe.segment_features as segment_features
from echoframe.segment_features import (
    _segment_times,
    compute_embeddings,
    get_codebook_indices,
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
    metadata = EchoframeMetadata(echoframe_key, model_name=model_name)
    store.save(metadata.echoframe_key, metadata, data)


def _put_codebook_indices(store, phraser_key, collar, model_name, data):
    phraser_key = _pk(phraser_key)
    echoframe_key = store.make_echoframe_key('codebook_indices',
        model_name=model_name, phraser_key=phraser_key, collar=collar)
    metadata = EchoframeMetadata(echoframe_key, model_name=model_name)
    store.save(metadata.echoframe_key, metadata, data)


def _put_codebook_matrix(store, phraser_key, collar, model_name, data):
    phraser_key = _pk(phraser_key)
    echoframe_key = store.make_echoframe_key('codebook_matrix',
        model_name=model_name)
    metadata = EchoframeMetadata(echoframe_key, model_name=model_name)
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


class TestComputeEmbeddings(unittest.TestCase):
    def test_cache_hit_works_without_loading_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 3,
                np.array([[1.0, 2.0], [3.0, 4.0]]))
            segment = _make_segment(key=_pk('aabb'))
            with mock.patch.object(store, 'load_model') as load_model:
                with mock.patch.object(segment_features.to_vector,
                    'filename_to_vector', create=True) as filename_to_vector:
                    with mock.patch('builtins.print') as print_mock:
                        result = compute_embeddings(segment, 3, 'wav2vec2',
                            store=store)

            self.assertIsNone(result)
            print_mock.assert_called_once_with(
                'embeddings found in store for layers [3]')
            load_model.assert_not_called()
            filename_to_vector.assert_not_called()

    def test_cache_miss_computes_and_stores_selected_frames(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(key=_pk('aabb'))
            hidden_states = [np.zeros((1, 4, 2)) for _ in range(4)]
            hidden_states[3] = np.array([[
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]])
            outputs = types.SimpleNamespace(hidden_states=hidden_states)
            with mock.patch.object(store, 'load_model',
                return_value='model') as load_model:
                with mock.patch.object(segment_features.to_vector,
                    'filename_to_vector', create=True,
                    return_value=outputs) as filename_to_vector:
                    with mock.patch.object(segment_features.frame,
                        'make_frames_from_outputs', create=True,
                        return_value=FakeFrames(selected_indices=[1, 2])):
                        with mock.patch('builtins.print') as print_mock:
                            result = compute_embeddings(segment, 3,
                                'wav2vec2', store=store, tags=['exp-a'])

            stored_key = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=segment.key, layer=3,
                collar=500)
            stored = store.load_metadata(stored_key)
            embedding = store.load_embedding(segment.key, 'wav2vec2', 3,
                collar=500)

            self.assertIsNone(result)
            print_mock.assert_called_once_with(
                'embeddings computed for layers [3]')
            np.testing.assert_array_equal(embedding.data,
                np.array([[3.0, 4.0], [5.0, 6.0]]))
            self.assertEqual(stored.tags, ['exp-a'])
            load_model.assert_called_once_with('wav2vec2', gpu=False)
            filename_to_vector.assert_called_once()

    def test_mixed_cache_reports_found_and_computed_layers(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(key=_pk('aabb'))
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 1,
                np.array([[10.0, 11.0], [12.0, 13.0]]))
            hidden_states = [np.zeros((1, 4, 2)) for _ in range(4)]
            hidden_states[3] = np.array([[
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]])
            outputs = types.SimpleNamespace(hidden_states=hidden_states)
            with mock.patch.object(store, 'load_model',
                return_value='model') as load_model:
                with mock.patch.object(segment_features.to_vector,
                    'filename_to_vector', create=True,
                    return_value=outputs) as filename_to_vector:
                    with mock.patch.object(segment_features.frame,
                        'make_frames_from_outputs', create=True,
                        return_value=FakeFrames(selected_indices=[1, 2])):
                        with mock.patch('builtins.print') as print_mock:
                            result = compute_embeddings(segment, [1, 3],
                                'wav2vec2', store=store, tags=['exp-a'])

            cached = store.load_embedding(segment.key, 'wav2vec2', 1,
                collar=500)
            computed = store.load_embedding(segment.key, 'wav2vec2', 3,
                collar=500)

            self.assertIsNone(result)
            self.assertEqual(print_mock.call_args_list, [
                mock.call('embeddings found in store for layers [1]'),
                mock.call('embeddings computed for layers [3]'),
            ])
            np.testing.assert_array_equal(cached.data,
                np.array([[10.0, 11.0], [12.0, 13.0]]))
            np.testing.assert_array_equal(computed.data,
                np.array([[3.0, 4.0], [5.0, 6.0]]))
            load_model.assert_called_once_with('wav2vec2', gpu=False)
            filename_to_vector.assert_called_once()


class TestGetCodebookIndices(unittest.TestCase):
    def test_cache_hit_returns_codebook_without_loading_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            _put_codebook_indices(store, 'aabb', 500, 'wav2vec2',
                np.array([[0, 1], [2, 3]]))
            _put_codebook_matrix(store, 'aabb', 500, 'wav2vec2',
                np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0],
                    [40.0, 41.0]]))
            segment = _make_segment(key=_pk('aabb'))
            with mock.patch.object(store, 'load_model') as load_model:
                with mock.patch.object(segment_features.to_vector,
                    'filename_to_codebook_artifacts', create=True
                    ) as filename_to_artifacts:
                    result = get_codebook_indices(segment, 'wav2vec2',
                        store=store)
            result.bind_store(store)

            self.assertIsInstance(result, Codebook)
            np.testing.assert_array_equal(result.to_numpy(),
                np.array([[0, 1], [2, 3]]))
            np.testing.assert_array_equal(result.codebook_matrix,
                np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0],
                    [40.0, 41.0]]))
            load_model.assert_not_called()
            filename_to_artifacts.assert_not_called()

    def test_cache_miss_stores_indices_and_matrix(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(key=_pk('aabb'))
            artifacts = types.SimpleNamespace(
                indices=np.array([[0, 1], [2, 3], [1, 0]]),
                codebook_matrix=np.array([
                    [10.0, 11.0],
                    [20.0, 21.0],
                    [30.0, 31.0],
                    [40.0, 41.0],
                ]),
            )
            with mock.patch.object(store, 'load_model',
                return_value='model') as load_model:
                with mock.patch.object(segment_features.to_vector,
                    'filename_to_codebook_artifacts', create=True,
                    return_value=artifacts) as filename_to_artifacts:
                    with mock.patch.object(segment_features.frame, 'Frames',
                        create=True, side_effect=lambda n_frames, start_time:
                        FakeFrames(n_frames, start_time, [0, 2])):
                        result = get_codebook_indices(segment, 'wav2vec2',
                            store=store, tags=['exp-a'])
            result.bind_store(store)

            indices_key = store.make_echoframe_key('codebook_indices',
                model_name='wav2vec2', phraser_key=segment.key, collar=500)
            matrix_key = store.make_echoframe_key('codebook_matrix',
                model_name='wav2vec2')
            indices_md = store.load_metadata(indices_key)
            matrix_md = store.load_metadata(matrix_key)

            self.assertIsInstance(result, Codebook)
            np.testing.assert_array_equal(result.to_numpy(),
                np.array([[0, 1], [1, 0]]))
            np.testing.assert_array_equal(result.codebook_matrix,
                artifacts.codebook_matrix)
            self.assertEqual(indices_md.tags, ['exp-a'])
            self.assertEqual(matrix_md.tags, ['exp-a'])
            load_model.assert_called_once_with('wav2vec2', gpu=False)
            filename_to_artifacts.assert_called_once()

    def test_cache_miss_does_not_overwrite_existing_matrix(self):
        tmpdir, store = _make_store()
        with tmpdir:
            original_matrix = np.array([
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
            ])
            _put_codebook_matrix(store, 'aabb', 500, 'wav2vec2',
                original_matrix)
            segment = _make_segment(key=_pk('aabb'))
            artifacts = types.SimpleNamespace(
                indices=np.array([[0, 1], [2, 3], [1, 0]]),
                codebook_matrix=np.array([
                    [100.0, 101.0],
                    [200.0, 201.0],
                    [300.0, 301.0],
                    [400.0, 401.0],
                ]),
            )
            with mock.patch.object(store, 'load_model',
                return_value='model'):
                with mock.patch.object(segment_features.to_vector,
                    'filename_to_codebook_artifacts', create=True,
                    return_value=artifacts):
                    with mock.patch.object(segment_features.frame, 'Frames',
                        create=True, side_effect=lambda n_frames, start_time:
                        FakeFrames(n_frames, start_time, [0, 2])):
                        result = get_codebook_indices(segment, 'wav2vec2',
                            store=store)
            result.bind_store(store)

            matrix_key = store.make_echoframe_key('codebook_matrix',
                model_name='wav2vec2')
            stored_matrix = store.load(matrix_key)

            np.testing.assert_array_equal(result.codebook_matrix,
                original_matrix)
            np.testing.assert_array_equal(stored_matrix, original_matrix)


if __name__ == '__main__':
    unittest.main()
