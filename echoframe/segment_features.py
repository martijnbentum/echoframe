'''Segment-based feature retrieval orchestration for echoframe.'''

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import echoframe
import numpy as np
from .metadata import metadata_class_for_output_type

try:
    import frame
except ModuleNotFoundError:  # pragma: no cover - exercised via monkeypatching
    frame = None

try:
    import to_vector
except ModuleNotFoundError:  # pragma: no cover - exercised via monkeypatching
    to_vector = None


_VALID_FRAME_AGGREGATIONS = (None, 'mean', 'centroid')


def get_embeddings(segment, layers, collar=500, model_name='wav2vec2',
    frame_aggregation=None, model=None, store=None,
    store_root='echoframe', gpu=False, tags=None):
    '''Return embeddings for one segment object.'''
    single_layer = isinstance(layers, Integral)
    layers_list = _normalise_layers(layers)
    _validate_frame_aggregation(frame_aggregation)
    store = _resolve_store(store, store_root)
    phraser_key, audio_filename, col_start_ms, col_end_ms, orig_start_ms, \
        orig_end_ms = _segment_context(segment, collar)

    if _embeddings_missing(store, phraser_key, collar, model_name,
        layers_list):
        compute_model = _require_loaded_model(model, 'embeddings')
        _compute_and_store_embeddings(audio_filename, col_start_ms,
            col_end_ms, orig_start_ms, orig_end_ms, collar, layers_list,
            model_name, compute_model, phraser_key, store, gpu, tags)

    load_layers = layers if single_layer else layers_list
    return store.load_embeddings(phraser_key, collar, model_name, load_layers,
        frame_aggregation=frame_aggregation)


def get_embeddings_batch(segments, layers, collar=500, model_name='wav2vec2',
    frame_aggregation=None, model=None, store=None,
    store_root='echoframe', gpu=False, tags=None):
    '''Return embeddings for multiple segment objects.'''
    segments = _require_segments(segments)
    layers_list = _normalise_layers(layers)
    _validate_frame_aggregation(frame_aggregation)
    store = _resolve_store(store, store_root)
    requests = []
    for segment in segments:
        phraser_key, audio_filename, col_start_ms, col_end_ms, \
            orig_start_ms, orig_end_ms = _segment_context(segment, collar)
        if _embeddings_missing(store, phraser_key, collar, model_name,
            layers_list):
            compute_model = _require_loaded_model(model, 'embeddings')
            _compute_and_store_embeddings(audio_filename, col_start_ms,
                col_end_ms, orig_start_ms, orig_end_ms, collar, layers_list,
                model_name, compute_model, phraser_key, store, gpu, tags)
        requests.append({
            'phraser_key': phraser_key,
            'collar': collar,
            'model_name': model_name,
            'layers': layers,
            'frame_aggregation': frame_aggregation,
        })
    return store.load_many_embeddings(requests)


def get_codebook_indices(segment, collar=500, model_name='wav2vec2',
    model=None, store=None, store_root='echoframe', gpu=False, tags=None):
    '''Return codebook indices for one segment object.'''
    store = _resolve_store(store, store_root)
    phraser_key, audio_filename, col_start_ms, col_end_ms, orig_start_ms, \
        orig_end_ms = _segment_context(segment, collar)

    if _codebook_artifacts_missing(store, phraser_key, collar, model_name):
        compute_model = _require_loaded_model(model, 'codebook indices')
        _compute_and_store_codebook_indices(audio_filename, col_start_ms,
            col_end_ms, orig_start_ms, orig_end_ms, collar, model_name,
            compute_model, phraser_key, store, gpu, tags)

    return store.load_codebook(phraser_key, collar, model_name)


def get_codebook_indices_batch(segments, collar=500, model_name='wav2vec2',
    model=None, store=None, store_root='echoframe', gpu=False, tags=None):
    '''Return codebook indices for multiple segment objects.'''
    segments = _require_segments(segments)
    store = _resolve_store(store, store_root)
    requests = []
    for segment in segments:
        phraser_key, audio_filename, col_start_ms, col_end_ms, \
            orig_start_ms, orig_end_ms = _segment_context(segment, collar)
        if _codebook_artifacts_missing(store, phraser_key, collar,
            model_name):
            compute_model = _require_loaded_model(model, 'codebook indices')
            _compute_and_store_codebook_indices(audio_filename, col_start_ms,
                col_end_ms, orig_start_ms, orig_end_ms, collar, model_name,
                compute_model, phraser_key, store, gpu, tags)
        requests.append({
            'phraser_key': phraser_key,
            'collar': collar,
            'model_name': model_name,
        })
    return store.load_many_codebooks(requests)


def segment_to_echoframe_key(segment):
    '''Return the raw binary segment key used by echoframe.'''
    key = getattr(segment, 'key', None)
    if key is None:
        raise ValueError('segment must expose a key')
    if isinstance(key, bytes):
        return key
    raise TypeError('segment.key must be bytes')


def _resolve_store(store, store_root):
    if store is not None:
        return store
    return echoframe.Store(store_root)


def _require_segments(segments):
    segments = list(segments)
    if not segments:
        raise ValueError('segments must contain at least one segment')
    return segments


def _normalise_layers(layers):
    if isinstance(layers, Integral):
        return [int(layers)]
    layers = [int(layer) for layer in layers]
    if not layers:
        raise ValueError('layers must not be empty')
    return layers


def _validate_frame_aggregation(frame_aggregation):
    if frame_aggregation not in _VALID_FRAME_AGGREGATIONS:
        message = 'frame_aggregation must be one of '
        message += f'{_VALID_FRAME_AGGREGATIONS}, got {frame_aggregation!r}'
        raise ValueError(message)


def _segment_context(segment, collar):
    phraser_key = segment_to_echoframe_key(segment)
    audio = getattr(segment, 'audio', None)
    if audio is None:
        raise ValueError('segment must be linked to an audio object')
    audio_filename = getattr(audio, 'filename', None)
    if not audio_filename:
        raise ValueError('segment.audio must expose a filename')
    start_ms = getattr(segment, 'start', None)
    end_ms = getattr(segment, 'end', None)
    if start_ms is None or end_ms is None:
        raise ValueError('segment must expose start and end in milliseconds')
    collar = int(collar)
    if collar < 0:
        raise ValueError('collar must be non-negative')
    orig_start_ms = int(start_ms)
    orig_end_ms = int(end_ms)
    if orig_end_ms <= orig_start_ms:
        raise ValueError('segment end must be greater than start')
    col_start_ms = max(0, orig_start_ms - collar)
    col_end_ms = orig_end_ms + collar
    duration = getattr(audio, 'duration', None)
    if duration is not None:
        col_end_ms = min(col_end_ms, int(duration))
    if col_end_ms <= col_start_ms:
        raise ValueError('resolved segment window is invalid')
    return (
        phraser_key,
        str(Path(audio_filename).resolve()),
        col_start_ms,
        col_end_ms,
        orig_start_ms,
        orig_end_ms,
    )


def _embeddings_missing(store, phraser_key, collar, model_name, layers):
    for layer in layers:
        echoframe_key = store.make_echoframe_key(
            'hidden_state',
            model_name=model_name,
            phraser_key=phraser_key,
            layer=layer,
            collar=collar,
        )
        if store.load_metadata(echoframe_key) is None:
            return True
    return False


def _codebook_artifacts_missing(store, phraser_key, collar, model_name):
    indices_key = store.make_echoframe_key(
        'codebook_indices',
        model_name=model_name,
        phraser_key=phraser_key,
        collar=collar,
    )
    matrix_key = store.make_echoframe_key(
        'codebook_matrix',
        model_name=model_name,
    )
    return (store.load_metadata(indices_key) is None or
        store.load_metadata(matrix_key) is None)


def _compute_and_store_embeddings(audio_filename, col_start_ms,
    col_end_ms, orig_start_ms, orig_end_ms, collar, layers, model_name,
    compute_model, phraser_key, store, gpu, tags):
    if frame is None:
        raise ImportError('frame is required to compute embeddings')
    if to_vector is None:
        raise ImportError('to_vector is required to compute embeddings')
    outputs = to_vector.filename_to_vector(audio_filename,
        start=_ms_to_s(col_start_ms), end=_ms_to_s(col_end_ms),
        model=compute_model, gpu=gpu, numpify_output=True)

    hidden_states = getattr(outputs, 'hidden_states', None)
    if hidden_states is None:
        raise ValueError('to-vector outputs did not contain hidden_states')

    frames = frame.make_frames_from_outputs(outputs,
        start_time=_ms_to_s(col_start_ms))
    selected = frames.select_frames(_ms_to_s(orig_start_ms),
        _ms_to_s(orig_end_ms), percentage_overlap=100)
    if not selected:
        message = 'no frames fully within '
        message += f'[{orig_start_ms}, {orig_end_ms}] ms'
        raise ValueError(message)

    indices = [f.index for f in selected]

    for layer in layers:
        if layer >= len(hidden_states):
            message = f'layer {layer} out of range '
            message += f'(model has {len(hidden_states)} layers)'
            raise ValueError(message)
        hs = hidden_states[layer]
        data = hs[0, indices, :] if hs.ndim == 3 else hs[indices, :]
        metadata_cls = metadata_class_for_output_type('hidden_state')
        echoframe_key = store.make_echoframe_key(
            'hidden_state',
            model_name=model_name,
            phraser_key=phraser_key,
            layer=layer,
            collar=collar,
        )
        metadata = metadata_cls(phraser_key=phraser_key, collar=collar,
            model_name=model_name, layer=layer, tags=tags,
            echoframe_key=echoframe_key)
        store.put(metadata.echoframe_key, metadata, data)


def _compute_and_store_codebook_indices(audio_filename, col_start_ms,
    col_end_ms, orig_start_ms, orig_end_ms, collar, model_name,
    compute_model, phraser_key, store, gpu, tags):
    if frame is None:
        raise ImportError('frame is required to compute codebook indices')
    if to_vector is None:
        raise ImportError('to_vector is required to compute codebook indices')
    outputs = to_vector.filename_to_vector(audio_filename,
        start=_ms_to_s(col_start_ms), end=_ms_to_s(col_end_ms),
        model=compute_model, gpu=gpu, numpify_output=True)
    frames = frame.make_frames_from_outputs(outputs,
        start_time=_ms_to_s(col_start_ms))
    selected = frames.select_frames(_ms_to_s(orig_start_ms),
        _ms_to_s(orig_end_ms), percentage_overlap=100)
    if not selected:
        message = 'no frames fully within '
        message += f'[{orig_start_ms}, {orig_end_ms}] ms'
        raise ValueError(message)
    frame_indices = [item.index for item in selected]
    artifacts = to_vector.filename_to_codebook_artifacts(
        audio_filename, start=_ms_to_s(col_start_ms), end=_ms_to_s(col_end_ms),
        model=compute_model, gpu=gpu)
    selected_indices = np.asarray(artifacts.indices)[frame_indices]
    ci_cls = metadata_class_for_output_type('codebook_indices')
    ci_key = store.make_echoframe_key(
        'codebook_indices',
        model_name=model_name,
        phraser_key=phraser_key,
        collar=collar,
    )
    ci_metadata = ci_cls(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=0, tags=tags, echoframe_key=ci_key)
    store.put(ci_metadata.echoframe_key, ci_metadata, selected_indices)
    cm_cls = metadata_class_for_output_type('codebook_matrix')
    cm_key = store.make_echoframe_key(
        'codebook_matrix',
        model_name=model_name,
    )
    cm_metadata = cm_cls(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=0, tags=tags, echoframe_key=cm_key)
    store.put(cm_metadata.echoframe_key, cm_metadata,
        np.asarray(artifacts.codebook_matrix))


def _ms_to_s(value):
    return int(value) / 1000.0


def _require_loaded_model(model, output_label):
    if model is None:
        raise ValueError(
            f'model is required as a loaded model object when '
            f'{output_label} must be computed')
    if isinstance(model, (str, Path)):
        raise TypeError(
            'model must be a loaded model object; string and path values '
            'are not accepted for compute paths')
    return model


__all__ = [
    'get_embeddings',
    'get_embeddings_batch',
    'get_codebook_indices',
    'get_codebook_indices_batch',
    'segment_to_echoframe_key',
]
