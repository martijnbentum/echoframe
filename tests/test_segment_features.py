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
from echoframe.index import LmdbIndex
from echoframe.metadata import EchoframeMetadata
from echoframe.output_storage import Hdf5ShardStore

sys.modules.setdefault('frame', types.SimpleNamespace())
sys.modules.setdefault('to_vector', types.SimpleNamespace())
if not hasattr(sys.modules['to_vector'], 'wav2vec2_codebook'):
    sys.modules['to_vector'].wav2vec2_codebook = types.SimpleNamespace()

import echoframe.segment_features as segment_features
import echoframe.batch_codebook_indices as batch_codebook_indices
import echoframe.batch_segment_features as batch_segment_features
import echoframe.utils_segment_features as utils_segment_features
from echoframe.batch_codebook_indices import (
    MissingIndices,
    compute_codebook_indices_batch,
)
from echoframe.batch_segment_features import (
    MissingSegments,
    compute_embeddings_batch,
)
from echoframe.segment_features import (
    _segment_times,
    compute_codebook_indices,
    compute_embeddings,
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
    def test_segment_request_builds_expected_keys_and_times(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(key=_pk('aabb'))
            parent = types.SimpleNamespace(store=store)
            request = batch_segment_features.SegmentRequest(segment, [3], 500,
                'wav2vec2', parent)

            expected_key = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=segment.key, layer=3,
                collar=500)

            self.assertEqual(request.audio_filename, segment.audio.filename)
            self.assertEqual(request.collared_start, 0.5)
            self.assertEqual(request.collared_end, 1.8)
            self.assertEqual(request.echoframe_keys, [expected_key])

    def test_missing_segments_reports_found_and_missing_layers(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(key=_pk('aabb'))
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 1,
                np.array([[1.0, 2.0], [3.0, 4.0]]))
            missing_segments = MissingSegments([segment], [1, 3], 'wav2vec2',
                500, store)

            self.assertEqual(len(missing_segments.metadatas), 2)
            self.assertEqual(len(missing_segments.found), 1)
            self.assertEqual(len(missing_segments.missing), 1)
            self.assertEqual(missing_segments.found[0].missing_layers, [3])
            self.assertEqual(missing_segments.missing[0].missing_layers, [3])
            self.assertEqual(missing_segments.audio_filenames,
                [segment.audio.filename])
            np.testing.assert_allclose(missing_segments.starts, [0.5])
            np.testing.assert_allclose(missing_segments.ends, [1.8])

    def test_batch_cache_hit_works_without_loading_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(key=_pk('aabb'))
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 3,
                np.array([[1.0, 2.0], [3.0, 4.0]]))
            with mock.patch.object(store, 'load_model') as load_model:
                with mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector', create=True
                    ) as iter_filename_batch_to_vector:
                    with mock.patch('builtins.print') as print_mock:
                        result = compute_embeddings_batch([segment], 3,
                            'wav2vec2', store=store)

            self.assertIsNone(result)
            print_mock.assert_called_once()
            printed = str(print_mock.call_args.args[0])
            self.assertEqual(printed,
                'MissingSegments model=wav2vec2 layers=[3]\n'
                'collar: 500ms\n'
                'n segments: 1\n'
                'missing segments: 0\n'
                'found segments: 1\n'
                'missing layer items: 0\n'
                'found layer items: 1')
            load_model.assert_not_called()
            iter_filename_batch_to_vector.assert_not_called()

    def test_cache_hit_works_without_loading_model(self):
        tmpdir, store = _make_store()
        with tmpdir:
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 3,
                np.array([[1.0, 2.0], [3.0, 4.0]]))
            segment = _make_segment(key=_pk('aabb'))
            with mock.patch.object(store, 'load_model') as load_model:
                with mock.patch.object(utils_segment_features.to_vector,
                    'filename_to_vector', create=True) as filename_to_vector:
                    with mock.patch('builtins.print') as print_mock:
                        result = compute_embeddings(segment, 3, 'wav2vec2',
                            store=store, verbose=True)

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
                with mock.patch.object(utils_segment_features.to_vector,
                    'filename_to_vector', create=True,
                    return_value=outputs) as filename_to_vector:
                    with mock.patch.object(utils_segment_features.frame,
                        'make_frames_from_outputs', create=True,
                        return_value=FakeFrames(selected_indices=[1, 2])):
                        with mock.patch('builtins.print') as print_mock:
                            result = compute_embeddings(segment, 3,
                                'wav2vec2', store=store, tags=['exp-a'],
                                verbose=True)

            stored_key = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=segment.key, layer=3,
                collar=500)
            stored = store.load_metadata(stored_key)
            embedding = store.phraser_key_to_embedding(segment.key,
                'wav2vec2', 3, collar=500)

            self.assertIsNone(result)
            print_mock.assert_called_once_with(
                'embeddings computed for layers [3]')
            np.testing.assert_array_equal(embedding.data,
                np.array([[3.0, 4.0], [5.0, 6.0]]))
            self.assertEqual(stored.tags, ['exp-a'])
            load_model.assert_called_once_with('wav2vec2', gpu=False)
            filename_to_vector.assert_called_once()

    def test_batch_cache_miss_computes_and_stores_selected_frames(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment_a = _make_segment(key=_pk('aabb'))
            segment_b = _make_segment(key=_pk('ccdd'), start_seconds=1.1,
                end_seconds=1.4)
            hidden_states_a = [np.zeros((1, 4, 2)) for _ in range(4)]
            hidden_states_b = [np.zeros((1, 4, 2)) for _ in range(4)]
            hidden_states_a[3] = np.array([[
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]])
            hidden_states_b[3] = np.array([[
                [10.0, 11.0],
                [12.0, 13.0],
                [14.0, 15.0],
                [16.0, 17.0],
            ]])
            outputs = [
                types.SimpleNamespace(hidden_states=hidden_states_a),
                types.SimpleNamespace(hidden_states=hidden_states_b),
            ]
            frame_values = [
                FakeFrames(selected_indices=[1, 2]),
                FakeFrames(selected_indices=[0, 1]),
            ]
            with mock.patch.object(store, 'load_model',
                return_value='model') as load_model:
                with mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector', create=True,
                    return_value=iter(outputs)
                    ) as iter_filename_batch_to_vector:
                    with mock.patch.object(utils_segment_features.frame,
                        'make_frames_from_outputs', create=True,
                        side_effect=frame_values):
                        with mock.patch('builtins.print') as print_mock:
                            result = compute_embeddings_batch(
                                [segment_a, segment_b], 3, 'wav2vec2',
                                store=store, tags=['exp-a'], batch_size=2)

            embedding_a = store.phraser_key_to_embedding(segment_a.key,
                'wav2vec2', 3, collar=500)
            embedding_b = store.phraser_key_to_embedding(segment_b.key,
                'wav2vec2', 3, collar=500)

            self.assertIsNone(result)
            self.assertEqual(print_mock.call_args_list[-1],
                mock.call('embeddings computed for 2 segments'))
            np.testing.assert_array_equal(embedding_a.data,
                np.array([[3.0, 4.0], [5.0, 6.0]]))
            np.testing.assert_array_equal(embedding_b.data,
                np.array([[10.0, 11.0], [12.0, 13.0]]))
            load_model.assert_called_once_with('wav2vec2', gpu=False)
            iter_filename_batch_to_vector.assert_called_once()
            args, kwargs = iter_filename_batch_to_vector.call_args
            self.assertEqual(args, (
                [segment_a.audio.filename, segment_b.audio.filename],))
            np.testing.assert_allclose(kwargs.pop('starts'), [0.5, 0.6])
            np.testing.assert_allclose(kwargs.pop('ends'), [1.8, 1.9])
            self.assertEqual(kwargs, {'model': 'model', 'gpu': False,
                'numpify_output': True, 'batch_size': 2})

    def test_batch_mixed_cache_only_stores_missing_layers(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(key=_pk('aabb'))
            cached_data = np.array([[10.0, 11.0], [12.0, 13.0]])
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 1, cached_data)
            hidden_states = [np.zeros((1, 4, 2)) for _ in range(4)]
            hidden_states[1] = np.array([[
                [20.0, 21.0],
                [22.0, 23.0],
                [24.0, 25.0],
                [26.0, 27.0],
            ]])
            hidden_states[3] = np.array([[
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]])
            outputs = [types.SimpleNamespace(hidden_states=hidden_states)]
            with mock.patch.object(store, 'load_model',
                return_value='model') as load_model:
                with mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector', create=True,
                    return_value=iter(outputs)
                    ) as iter_filename_batch_to_vector:
                    with mock.patch.object(utils_segment_features.frame,
                        'make_frames_from_outputs', create=True,
                        return_value=FakeFrames(selected_indices=[1, 2])):
                        with mock.patch('builtins.print') as print_mock:
                            result = compute_embeddings_batch([segment], [1, 3],
                                'wav2vec2', store=store, tags=['exp-a'])

            cached = store.phraser_key_to_embedding(segment.key, 'wav2vec2',
                1, collar=500)
            computed = store.phraser_key_to_embedding(segment.key,
                'wav2vec2', 3, collar=500)
            cached_key = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=segment.key, layer=1,
                collar=500)
            computed_key = store.make_echoframe_key('hidden_state',
                model_name='wav2vec2', phraser_key=segment.key, layer=3,
                collar=500)

            self.assertIsNone(result)
            self.assertEqual(print_mock.call_args_list[-1],
                mock.call('embeddings computed for 1 segments'))
            np.testing.assert_array_equal(cached.data, cached_data)
            np.testing.assert_array_equal(computed.data,
                np.array([[3.0, 4.0], [5.0, 6.0]]))
            self.assertEqual(store.load_metadata(cached_key).tags, [])
            self.assertEqual(store.load_metadata(computed_key).tags, ['exp-a'])
            load_model.assert_called_once_with('wav2vec2', gpu=False)
            iter_filename_batch_to_vector.assert_called_once()
            args, kwargs = iter_filename_batch_to_vector.call_args
            self.assertEqual(args, ([segment.audio.filename],))
            np.testing.assert_allclose(kwargs.pop('starts'), [0.5])
            np.testing.assert_allclose(kwargs.pop('ends'), [1.8])
            self.assertEqual(kwargs, {'model': 'model', 'gpu': False,
                'numpify_output': True, 'batch_size': None})

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
                with mock.patch.object(utils_segment_features.to_vector,
                    'filename_to_vector', create=True,
                    return_value=outputs) as filename_to_vector:
                    with mock.patch.object(utils_segment_features.frame,
                        'make_frames_from_outputs', create=True,
                        return_value=FakeFrames(selected_indices=[1, 2])):
                        with mock.patch('builtins.print') as print_mock:
                            result = compute_embeddings(segment, [1, 3],
                                'wav2vec2', store=store, tags=['exp-a'],
                                verbose=True)

            cached = store.phraser_key_to_embedding(segment.key, 'wav2vec2',
                1, collar=500)
            computed = store.phraser_key_to_embedding(segment.key,
                'wav2vec2', 3, collar=500)

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

    def test_duplicate_requests_collapse_by_filename_start_and_end(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment_a = _make_segment(key=_pk('aabb'))
            segment_b = _make_segment(key=_pk('aabb'))

            missing_segments = MissingSegments([segment_a, segment_b], [3],
                'wav2vec2', 500, store)

            self.assertEqual(len(missing_segments.segment_requests), 2)
            self.assertEqual(len(missing_segments.missing), 1)
            self.assertEqual(missing_segments.audio_filenames,
                [segment_a.audio.filename])
            np.testing.assert_allclose(missing_segments.starts, [0.5])
            np.testing.assert_allclose(missing_segments.ends, [1.8])

    def test_different_windows_do_not_collapse_requests(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment_a = _make_segment(key=_pk('aabb'))
            segment_b = _make_segment(key=_pk('ccdd'), start_seconds=1.05)

            missing_segments = MissingSegments([segment_a, segment_b], [3],
                'wav2vec2', 500, store)

            self.assertEqual(len(missing_segments.segment_requests), 2)
            self.assertEqual(len(missing_segments.missing), 2)
            self.assertEqual(missing_segments.audio_filenames, [
                segment_a.audio.filename,
                segment_b.audio.filename,
            ])
            np.testing.assert_allclose(missing_segments.starts, [0.5, 0.55])
            np.testing.assert_allclose(missing_segments.ends, [1.8, 1.8])

    def test_batch_multi_segment_multi_layer_partial_cache(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment_a = _make_segment(key=_pk('aabb'))
            segment_b = _make_segment(key=_pk('ccdd'), start_seconds=1.1,
                end_seconds=1.4)
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 1,
                np.array([[10.0, 11.0], [12.0, 13.0]]))
            _put_hidden_state(store, 'ccdd', 500, 'wav2vec2', 3,
                np.array([[30.0, 31.0], [32.0, 33.0]]))
            hidden_states_a = [np.zeros((1, 4, 2)) for _ in range(4)]
            hidden_states_b = [np.zeros((1, 4, 2)) for _ in range(4)]
            hidden_states_a[3] = np.array([[
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]])
            hidden_states_b[1] = np.array([[
                [10.0, 11.0],
                [12.0, 13.0],
                [14.0, 15.0],
                [16.0, 17.0],
            ]])
            outputs = [
                types.SimpleNamespace(hidden_states=hidden_states_a),
                types.SimpleNamespace(hidden_states=hidden_states_b),
            ]
            frame_values = [
                FakeFrames(selected_indices=[1, 2]),
                FakeFrames(selected_indices=[0, 1]),
            ]
            with mock.patch.object(store, 'load_model',
                return_value='model') as load_model:
                with mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector', create=True,
                    return_value=iter(outputs)
                    ) as iter_filename_batch_to_vector:
                    with mock.patch.object(utils_segment_features.frame,
                        'make_frames_from_outputs', create=True,
                        side_effect=frame_values):
                        with mock.patch('builtins.print') as print_mock:
                            result = compute_embeddings_batch(
                                [segment_a, segment_b], [1, 3], 'wav2vec2',
                                store=store, tags=['exp-a'])

            self.assertIsNone(result)
            self.assertEqual(print_mock.call_args_list[-1],
                mock.call('embeddings computed for 2 segments'))
            np.testing.assert_array_equal(
                store.phraser_key_to_embedding(segment_a.key, 'wav2vec2', 1,
                    collar=500).data,
                np.array([[10.0, 11.0], [12.0, 13.0]]))
            np.testing.assert_array_equal(
                store.phraser_key_to_embedding(segment_a.key, 'wav2vec2', 3,
                    collar=500).data,
                np.array([[3.0, 4.0], [5.0, 6.0]]))
            np.testing.assert_array_equal(
                store.phraser_key_to_embedding(segment_b.key, 'wav2vec2', 1,
                    collar=500).data,
                np.array([[10.0, 11.0], [12.0, 13.0]]))
            np.testing.assert_array_equal(
                store.phraser_key_to_embedding(segment_b.key, 'wav2vec2', 3,
                    collar=500).data,
                np.array([[30.0, 31.0], [32.0, 33.0]]))
            load_model.assert_called_once_with('wav2vec2', gpu=False)
            iter_filename_batch_to_vector.assert_called_once()
            args, kwargs = iter_filename_batch_to_vector.call_args
            self.assertEqual(args, (
                [segment_a.audio.filename, segment_b.audio.filename],))
            np.testing.assert_allclose(kwargs.pop('starts'), [0.5, 0.6])
            np.testing.assert_allclose(kwargs.pop('ends'), [1.8, 1.9])
            self.assertEqual(kwargs, {'model': 'model', 'gpu': False,
                'numpify_output': True, 'batch_size': None})

    def test_missing_segments_metadata_mapping_is_aligned(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(key=_pk('aabb'))
            _put_hidden_state(store, 'aabb', 500, 'wav2vec2', 1,
                np.array([[1.0, 2.0], [3.0, 4.0]]))

            missing_segments = MissingSegments([segment], [1, 3], 'wav2vec2',
                500, store)
            key_to_metadata = missing_segments.echoframe_key_to_metadata_dict

            self.assertEqual(len(missing_segments.echoframe_keys), 2)
            self.assertEqual(len(missing_segments.metadatas), 2)
            for key, metadata in zip(missing_segments.echoframe_keys,
                missing_segments.metadatas, strict=True):
                self.assertIs(key_to_metadata[key], metadata)
            self.assertIsNotNone(missing_segments.metadatas[0])
            self.assertIsNone(missing_segments.metadatas[1])

    def test_segment_request_equality_uses_segment_key(self):
        tmpdir, store = _make_store()
        with tmpdir:
            parent = types.SimpleNamespace(store=store)
            segment_a = _make_segment()
            segment_b = _make_segment(start_seconds=1.0 + 1e-12,
                end_seconds=1.3 + 1e-12)
            segment_c = _make_segment(key=_pk('ccdd'))

            request_a = batch_segment_features.SegmentRequest(segment_a, [3],
                500, 'wav2vec2', parent)
            request_b = batch_segment_features.SegmentRequest(segment_b, [3],
                500, 'wav2vec2', parent)
            request_c = batch_segment_features.SegmentRequest(segment_c, [3],
                500, 'wav2vec2', parent)

            self.assertEqual(request_a, request_b)
            self.assertNotEqual(request_a, request_c)

    def test_missing_segments_accepts_generator_input(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segments = (
                segment for segment in [
                    _make_segment(key=_pk('aabb')),
                    _make_segment(key=_pk('ccdd'), start_seconds=1.1,
                        end_seconds=1.4),
                ]
            )

            missing_segments = MissingSegments(segments, [3], 'wav2vec2', 500,
                store)

            self.assertEqual(len(missing_segments.segments), 2)
            self.assertEqual(len(missing_segments.segment_requests), 2)
            self.assertEqual(len(missing_segments.missing), 2)


class TestComputeCodebookIndices(unittest.TestCase):
    def test_cache_hit_skips_model_and_leaves_store_unchanged(self):
        tmpdir, store = _make_store()
        with tmpdir:
            _put_codebook_indices(store, 'aabb', 500, 'wav2vec2',
                np.array([[0, 1], [2, 3]]))
            _put_codebook_matrix(store, 'aabb', 500, 'wav2vec2',
                np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0],
                    [40.0, 41.0]]))
            segment = _make_segment(key=_pk('aabb'))
            with mock.patch.object(store, 'load_model') as load_model:
                with mock.patch.object(utils_segment_features.to_vector,
                    'filename_to_codebook_artifacts', create=True
                    ) as filename_to_artifacts:
                    compute_codebook_indices(segment, 'wav2vec2', store=store)
            echoframe_key = store.make_echoframe_key('codebook_indices',
                model_name='wav2vec2', phraser_key=segment.key, collar=500)
            codevector = store.load_codevector(echoframe_key)
            np.testing.assert_array_equal(codevector.to_numpy(),
                np.array([[0, 1], [2, 3]]))
            np.testing.assert_array_equal(codevector.codebook_matrix,
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
                with mock.patch.object(utils_segment_features.to_vector,
                    'filename_to_codebook_artifacts', create=True,
                    return_value=artifacts) as filename_to_artifacts:
                    with mock.patch.object(utils_segment_features.frame,
                        'Frames',
                        create=True, side_effect=lambda n_frames, start_time:
                        FakeFrames(n_frames, start_time, [0, 2])):
                        compute_codebook_indices(segment, 'wav2vec2',
                            store=store, tags=['exp-a'])
            indices_key = store.make_echoframe_key('codebook_indices',
                model_name='wav2vec2', phraser_key=segment.key, collar=500)
            matrix_key = store.make_echoframe_key('codebook_matrix',
                model_name='wav2vec2')
            indices_md = store.load_metadata(indices_key)
            matrix_md = store.load_metadata(matrix_key)

            codevector = store.load_codevector(indices_key)
            np.testing.assert_array_equal(codevector.to_numpy(),
                np.array([[0, 1], [1, 0]]))
            np.testing.assert_array_equal(codevector.codebook_matrix,
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
                with mock.patch.object(utils_segment_features.to_vector,
                    'filename_to_codebook_artifacts', create=True,
                    return_value=artifacts):
                    with mock.patch.object(utils_segment_features.frame,
                        'Frames',
                        create=True, side_effect=lambda n_frames, start_time:
                        FakeFrames(n_frames, start_time, [0, 2])):
                        compute_codebook_indices(segment, 'wav2vec2',
                            store=store)
            indices_key = store.make_echoframe_key('codebook_indices',
                model_name='wav2vec2', phraser_key=segment.key, collar=500)
            matrix_key = store.make_echoframe_key('codebook_matrix',
                model_name='wav2vec2')
            codevector = store.load_codevector(indices_key)
            stored_matrix = store.load(matrix_key)

            np.testing.assert_array_equal(codevector.codebook_matrix,
                original_matrix)
            np.testing.assert_array_equal(stored_matrix, original_matrix)


class TestComputeCodebookIndicesBatch(unittest.TestCase):
    def test_missing_indices_reports_found_missing_and_matrix_state(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment_a = _make_segment(key=_pk('aabb'))
            segment_b = _make_segment(key=_pk('ccdd'), start_seconds=1.1,
                end_seconds=1.4)
            _put_codebook_indices(store, 'aabb', 500, 'wav2vec2',
                np.array([[0, 1], [2, 3]]))

            missing = MissingIndices([segment_a, segment_b], 'wav2vec2',
                500, store)

            self.assertEqual(len(missing.segments), 2)
            self.assertEqual(len(missing.segment_requests), 2)
            self.assertEqual(len(missing.found), 1)
            self.assertEqual(len(missing.missing), 1)
            self.assertIs(missing.found[0].segment, segment_a)
            self.assertIs(missing.missing[0].segment, segment_b)
            self.assertEqual(missing.indices_items_found, 1)
            self.assertEqual(missing.indices_items_missing, 1)
            self.assertTrue(missing.matrix_missing)
            self.assertEqual(missing.audio_filenames,
                [segment_b.audio.filename])
            np.testing.assert_allclose(missing.starts, [0.6])
            np.testing.assert_allclose(missing.ends, [1.9])

    def test_batch_cache_hit_skips_model_and_iterator(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment = _make_segment(key=_pk('aabb'))
            _put_codebook_indices(store, 'aabb', 500, 'wav2vec2',
                np.array([[0, 1], [2, 3]]))
            with mock.patch.object(store, 'load_codebook_model'
                ) as load_codebook_model:
                with mock.patch.object(
                    batch_codebook_indices.wav2vec2_codebook,
                    'iter_filename_batch_to_codebook_indices', create=True
                    ) as iter_indices:
                    with mock.patch('builtins.print') as print_mock:
                        result = compute_codebook_indices_batch([segment],
                            'wav2vec2', store=store)

            self.assertIsNone(result)
            print_mock.assert_called_once()
            self.assertIn('missing segments: 0',
                str(print_mock.call_args.args[0]))
            load_codebook_model.assert_not_called()
            iter_indices.assert_not_called()

    def test_batch_cache_miss_stores_indices_and_matrix(self):
        tmpdir, store = _make_store()
        with tmpdir:
            segment_a = _make_segment(key=_pk('aabb'))
            segment_b = _make_segment(key=_pk('ccdd'), start_seconds=1.1,
                end_seconds=1.4)
            matrix = np.array([
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
            ])
            outputs = [
                np.array([[0, 1], [2, 3], [1, 0]]),
                np.array([[3, 2], [1, 1], [0, 0]]),
            ]
            frame_values = [
                FakeFrames(selected_indices=[0, 2]),
                FakeFrames(selected_indices=[1, 2]),
            ]
            with mock.patch.object(store, 'load_codebook_model',
                return_value='model') as load_codebook_model:
                with mock.patch.object(
                    batch_codebook_indices.wav2vec2_codebook,
                    'load_codebook', return_value=matrix) as load_codebook:
                    with mock.patch.object(
                        batch_codebook_indices.wav2vec2_codebook,
                        'iter_filename_batch_to_codebook_indices',
                        return_value=iter(outputs)) as iter_indices:
                        with mock.patch.object(
                            utils_segment_features.frame, 'Frames',
                            side_effect=lambda n_frames, start_time:
                            frame_values.pop(0), create=True):
                            with mock.patch('builtins.print') as print_mock:
                                result = compute_codebook_indices_batch(
                                    [segment_a, segment_b], 'wav2vec2',
                                    store=store, tags=['exp-a'], batch_size=2)

            key_a = store.make_echoframe_key('codebook_indices',
                model_name='wav2vec2', phraser_key=segment_a.key, collar=500)
            key_b = store.make_echoframe_key('codebook_indices',
                model_name='wav2vec2', phraser_key=segment_b.key, collar=500)
            matrix_key = store.make_echoframe_key('codebook_matrix',
                model_name='wav2vec2')

            self.assertIsNone(result)
            self.assertEqual(print_mock.call_args_list[-1],
                mock.call('codebook indices computed for 2 segments'))
            np.testing.assert_array_equal(store.load(key_a),
                np.array([[0, 1], [1, 0]]))
            np.testing.assert_array_equal(store.load(key_b),
                np.array([[1, 1], [0, 0]]))
            np.testing.assert_array_equal(store.load(matrix_key), matrix)
            self.assertEqual(store.load_metadata(key_a).tags, ['exp-a'])
            self.assertEqual(store.load_metadata(matrix_key).tags, ['exp-a'])
            load_codebook_model.assert_called_once_with('wav2vec2', gpu=False)
            load_codebook.assert_called_once_with('model')
            iter_indices.assert_called_once()
            args, kwargs = iter_indices.call_args
            self.assertEqual(args, (
                [segment_a.audio.filename, segment_b.audio.filename],))
            np.testing.assert_allclose(kwargs.pop('starts'), [0.5, 0.6])
            np.testing.assert_allclose(kwargs.pop('ends'), [1.8, 1.9])
            self.assertEqual(kwargs, {'model_pt': 'model', 'gpu': False,
                'batch_size': 2})

    def test_batch_does_not_overwrite_existing_matrix(self):
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
            output = np.array([[0, 1], [2, 3], [1, 0]])
            with mock.patch.object(store, 'load_codebook_model',
                return_value='model'):
                with mock.patch.object(
                    batch_codebook_indices.wav2vec2_codebook,
                    'load_codebook') as load_codebook:
                    with mock.patch.object(
                        batch_codebook_indices.wav2vec2_codebook,
                        'iter_filename_batch_to_codebook_indices',
                        return_value=iter([output])):
                        with mock.patch.object(
                            utils_segment_features.frame, 'Frames',
                            side_effect=lambda n_frames, start_time:
                            FakeFrames(n_frames, start_time, [0, 2]),
                            create=True):
                            compute_codebook_indices_batch([segment],
                                'wav2vec2', store=store)

            matrix_key = store.make_echoframe_key('codebook_matrix',
                model_name='wav2vec2')
            np.testing.assert_array_equal(store.load(matrix_key),
                original_matrix)
            load_codebook.assert_not_called()


if __name__ == '__main__':
    unittest.main()
