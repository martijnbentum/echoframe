'''Segment-based feature retrieval orchestration for echoframe.'''
from pathlib import Path

import echoframe
import numpy as np
from .metadata import EchoframeMetadata 
import frame
import to_vector

_VALID_FRAME_AGGREGATIONS = (None, 'mean', 'centroid')

def compute_embeddings(segment, layers, model_name, collar=500, store=None,
    store_root='echoframe', gpu=False, tags=None):
    '''Compute and store embeddings for one segment object.
    segment:              phraser segment object with key, timing, and audio
    layers:               layer index or iterable of layer indices
    model_name:           registered model name for store storage
    collar:               context window in milliseconds
    store:                optional Store instance
    store_root:           root used when creating a Store lazily
    gpu:                  whether to run vectorization on GPU
    tags:                 optional tags stored on newly written metadata
    '''
    layers_list = _normalise_layers(layers)
    if store is None: store = echoframe.Store(store_root)
    phraser_key = segment.key
    found_layers, missing_layers = _find_embedding_layers(store, phraser_key,
        collar, model_name, layers_list)
    if found_layers:
        print(f'embeddings found in store for layers {found_layers}')
    if missing_layers:
        model = store.load_model(model_name, gpu=gpu)
        outputs = _compute_embeddings(segment, collar, model, gpu)
        _store_embeddings_from_outputs(outputs, segment, collar, missing_layers,
            model_name, store, tags)
        print(f'embeddings computed for layers {missing_layers}')

def compute_embeddings_batch(segments, layers, model_name, collar=500, 
    store=None, store_root='echoframe', gpu=False, tags=None,
    batch_size=None):
    '''Compute and store embeddings for multiple segment objects.
    segments:             iterable of phraser segment objects
    layers:               layer index or iterable of layer indices
    model_name:           registered model name for store storage
    collar:               context window in milliseconds
    store:                optional Store instance
    store_root:           root used when creating a Store lazily
    gpu:                  whether to run vectorization on GPU
    tags:                 optional tags stored on newly written metadata
    batch_size:           optional item count per batch
    '''
    layers_list = _normalise_layers(layers)
    if store is None: store = echoframe.Store(store_root)
    missing = _find_missing_segments(segments, collar, model_name, store, 
        layers_list)
    if not missing: return
    model = store.load_model(model_name, gpu=gpu)
    audio_filenames = [segment.audio.filename for segment, _, _, _ in missing]
    starts = [collared_start for _, _, collared_start, _ in missing]
    ends = [collared_end for _, _, _, collared_end in missing]
    outputs = to_vector.filename_batch_to_vector(audio_filenames,
        starts=starts, ends=ends, model=model, gpu=gpu,
        numpify_output=True, batch_size=batch_size)
    if len(outputs) != len(missing):
        m = f'number of outputs {len(outputs)} '
        m += f'does not match number of segments {len(missing)}'
        raise ValueError(m)
    for output, (segment, missing_layers, _, _) in zip(outputs, missing):
        _store_embeddings_from_outputs(output, segment, collar, missing_layers,
            model_name, store, tags)
    print(f'embeddings computed for {len(missing)} segments')

def _find_missing_segments(segments, collar, model_name, store, layers_list):
    missing, found = [], []
    for segment in segments:
        phraser_key = segment.key
        found_layers, missing_layers = _find_embedding_layers(store, 
            phraser_key, collar, model_name, layers_list)
        if found_layers: found.append(found_layers)
        if missing_layers:
            _, _, collared_start, collared_end = _segment_times(segment, collar)
            missing.append((segment, missing_layers, collared_start,
                collared_end))
    found_layer_count = sum(len(layers) for layers in found)
    missing_layer_count = sum(len(l) for _,l,_,_ in missing)
    if found and not missing: print(f'all items in store: {found_layer_count}')
    if missing and not found: print(f'items to compute: {missing_layer_count}')
    if found and missing:
        m = f'items in store: {found_layer_count}\n'
        m += f'items to compute: {missing_layer_count}'
        print(m)
    return missing

def _find_embedding_layers(store, phraser_key, collar, model_name, layers):
    '''Return found and missing hidden-state layers for one phraser key.'''
    echoframe_keys = []
    for layer in layers:
        echoframe_key = store.make_echoframe_key('hidden_state',
            model_name=model_name, phraser_key=phraser_key, layer=layer,
            collar=collar)
        echoframe_keys.append(echoframe_key)
    metadatas = store.load_many_metadata(echoframe_keys, keep_missing=True)
    found_layers, missing_layers = [], []
    for layer, metadata in zip(layers, metadatas):
        if metadata is not None: found_layers.append(layer)
        else: missing_layers.append(layer)
    return found_layers, missing_layers

def get_codebook_indices(segment, model_name,
    collar=500, store=None, store_root='echoframe', gpu=False, tags=None):
    '''Return codebook indices for one segment object.
    segment:      phraser segment object with key, timing, and audio
    model_name:   registered model name
    collar:       context window in milliseconds
    store:        optional Store instance
    store_root:   root used when creating a Store lazily
    gpu:          whether to run codebook extraction on GPU
    tags:         optional tags stored on newly written metadata
    '''
    if store is None: store = echoframe.Store(store_root)
    phraser_key = segment.key
    if codebook_indices_missing(store, phraser_key, collar, model_name):
        model = store.load_model(model_name, gpu=gpu)
        artifacts = _compute_codebook_indices(segment, collar, model, gpu)
        _store_codebook_indices_from_artifacts(artifacts, segment, collar,
            model_name, store, tags)
        if codebook_matrix_missing(store, model_name):
            _store_codebook_matrix(artifacts.codebook_matrix, phraser_key,
                collar, model_name, store, tags)
    return store.load_codebook(phraser_key, collar, model_name)

def codebook_indices_missing(store, phraser_key, collar, model_name):
    '''Return whether segment-level codebook indices are absent.'''
    indices_key = store.make_echoframe_key('codebook_indices',
        model_name=model_name, phraser_key=phraser_key, collar=collar)
    return store.load_metadata(indices_key) is None

def codebook_matrix_missing(store, model_name):
    '''Return whether the model-level codebook matrix is absent.'''
    matrix_key = store.make_echoframe_key('codebook_matrix',
        model_name=model_name)
    return store.load_metadata(matrix_key) is None


def _compute_embeddings(segment, collar,  model, gpu):
    '''Compute hidden states for one collared segment window.'''
    _, _, collared_start, collared_end = _segment_times(segment, collar)
    outputs = to_vector.filename_to_vector(segment.audio.filename,
        start=collared_start, end=collared_end, model=model, gpu=gpu,
        numpify_output=True)
    return outputs

def _store_embeddings_from_outputs(outputs, segment, collar, layers,
    model_name, store, tags):
    '''Select segment frames from model outputs and store layer specific elments 
    in the store.'''
    _validate_hidden_states(outputs, layers) 
    indices = _get_selected_frame_indices(outputs, segment, collar, layers) 
    phraser_key = segment.key
    for layer in layers:
        hs = outputs.hidden_states[layer]
        data = hs[0, indices, :] if hs.ndim == 3 else hs[indices, :]
        echoframe_key = store.make_echoframe_key('hidden_state',
            model_name=model_name,phraser_key=phraser_key,layer=layer,
            collar=collar)
        metadata = EchoframeMetadata(echoframe_key=echoframe_key,
            store=store, model_name=model_name, tags=tags)
        store.save(metadata.echoframe_key, metadata, data)

def _compute_codebook_indices(segment, collar, model, gpu):
    '''Compute storage-oriented codebook artifacts for one segment window.'''
    _, _, collared_start, collared_end = _segment_times(segment, collar)
    artifacts = to_vector.filename_to_codebook_artifacts(segment.audio.filename,
        start=collared_start, end=collared_end, model=model, gpu=gpu)
    return artifacts

def _store_codebook_indices_from_artifacts(artifacts, segment, collar,
    model_name, store, tags):
    '''Persist selected codebook indices for one segment from artifacts.'''
    _validate_codebook_artifacts(artifacts)
    indices = _get_selected_codebook_frame_indices(artifacts, segment, collar)
    selected_indices = np.asarray(artifacts.indices)[indices]
    phraser_key = segment.key
    ci_key = store.make_echoframe_key('codebook_indices',
        model_name=model_name, phraser_key=phraser_key, collar=collar)
    ci_metadata = EchoframeMetadata(echoframe_key=ci_key, store=store,
        model_name=model_name, tags=tags)
    store.save(ci_metadata.echoframe_key, ci_metadata, selected_indices)

def _store_codebook_matrix(codebook_matrix, phraser_key, collar,
    model_name, store, tags):
    '''Persist the shared codebook matrix for one model.'''
    cm_key = store.make_echoframe_key('codebook_matrix', model_name=model_name)
    codebook_matrix = np.asarray(codebook_matrix)
    cm_metadata = EchoframeMetadata(echoframe_key=cm_key, store=store,
        model_name=model_name, tags=tags)
    store.save(cm_metadata.echoframe_key, cm_metadata, codebook_matrix)

def _segment_times(segment, collar):
    '''Return original and collared segment bounds in seconds.'''
    if collar < 0:
        raise ValueError('collar must be non-negative')
    collar = _ms_to_seconds(collar)
    start_seconds = float(segment.start_seconds)
    end_seconds = float(segment.end_seconds)
    duration_seconds = float(_ms_to_seconds(segment.audio.duration))
    if end_seconds <= start_seconds:
        raise ValueError('segment end must be greater than start')
    collared_start = max(0.0, start_seconds - collar)
    collared_end = end_seconds + collar
    collared_end = min(collared_end, float(duration_seconds))
    return start_seconds, end_seconds, collared_start, collared_end

def _ms_to_seconds(value):
    '''Convert milliseconds to seconds as float.'''
    return float(value) / 1000.0

def _validate_frame_aggregation(frame_aggregation):
    '''Reject unsupported frame aggregation modes early.'''
    if frame_aggregation in _VALID_FRAME_AGGREGATIONS: return
    raise ValueError( f'{frame_aggregation} not in {_VALID_FRAME_AGGREGATIONS}')

def _normalise_layers(layers):
    '''Normalize layer input to a validated list of non-negative ints.'''
    if layers is None:
        raise ValueError('layers must be int or a list of ints {layers}')
    layers_list = [layers] if isinstance(layers, int) else list(layers)
    if len(layers_list) == 0: 
        raise ValueError('layers must be a non-empty list')
    for layer in layers_list:
        if not isinstance(layer, int):
            raise ValueError(f'layer must be int {layer}, {layers_list}') 
        if layer < 0:
            raise ValueError(f'layer must be non-negative integers {layer}')
    return layers_list

def _validate_hidden_states(outputs, layers, batch=False):
    '''Validate requested hidden-state layers before frame selection.'''
    if not hasattr(outputs, 'hidden_states'):
        m = 'to-vector outputs did not contain hidden_states attribute\n'
        m += f'outputs type: {type(outputs)}\n'
        m += f'outputs attributes: {dir(outputs)}'
        raise ValueError(m)
    hidden_states = outputs.hidden_states
    if not isinstance(hidden_states, list):
        m = 'to-vector outputs hidden_states is not a list, '
        m += f'got {type(hidden_states)}'
        raise ValueError(m)
    for layer in layers:
        if layer >= len(hidden_states):
            m = f'layer {layer} out of range '
            m += f'(model has {len(hidden_states)} layers)'
            raise ValueError(m)
        if not isinstance(hidden_states[layer], np.ndarray):
            m = f'hidden state for layer {layer} is not a numpy array, '
            m += f'got {type(hidden_states[layer])}'
            raise ValueError(m)
        shape = hidden_states[layer].shape
        if  len(shape) not in (2, 3):
            m = f'hidden state for layer {layer} has invalid shape '
            m += f'{hidden_states[layer].shape}'
            raise ValueError(m)
        if len(shape) == 3 and shape[0] != 1 and batch is False:
            m = f'batch size for hidden states {shape} is {shape[0]}, expected 1'
            raise ValueError(m)


def _validate_codebook_artifacts(artifacts):
    '''Validate artifact arrays needed for codebook storage.'''
    if not hasattr(artifacts, 'indices'):
        m = 'to-vector artifacts did not contain indices attribute\n'
        m += f'artifacts type: {type(artifacts)}\n'
        m += f'artifacts attributes: {dir(artifacts)}'
        raise ValueError(m)
    if not hasattr(artifacts, 'codebook_matrix'):
        m = 'to-vector artifacts did not contain codebook_matrix attribute\n'
        m += f'artifacts type: {type(artifacts)}\n'
        m += f'artifacts attributes: {dir(artifacts)}'
        raise ValueError(m)
    if not isinstance(artifacts.indices, np.ndarray):
        m = 'to-vector artifacts indices is not a numpy array, '
        m += f'got {type(artifacts.indices)}'
        raise ValueError(m)
    if not isinstance(artifacts.codebook_matrix, np.ndarray):
        m = 'to-vector artifacts codebook_matrix is not a numpy array, '
        m += f'got {type(artifacts.codebook_matrix)}'
        raise ValueError(m)
    if len(artifacts.indices.shape) != 2:
        m = 'to-vector artifacts indices has invalid shape '
        m += f'{artifacts.indices.shape}'
        raise ValueError(m)
    if len(artifacts.codebook_matrix.shape) != 2:
        m = 'to-vector artifacts codebook_matrix has invalid shape '
        m += f'{artifacts.codebook_matrix.shape}'
        raise ValueError(m)


def _get_selected_frame_indices(outputs, segment, collar, layers):
    '''Return frame indices fully contained in the original segment span.'''
    start, end, collared_start, _ = _segment_times(segment, collar)
    frames = frame.make_frames_from_outputs(outputs, start_time=collared_start)
    selected = frames.select_frames(start, end, percentage_overlap=100)
    if not selected:
        m= f'no frames fully within {start} - {end}s, {segment!r}'
        m+= f'for segment {segment!r} with collar {collar} seconds'
        raise ValueError(m)
    hs = outputs.hidden_states[layers[0]]
    indices = [f.index for f in selected]
    frame_count = hs.shape[1] if hs.ndim == 3 else hs.shape[0]
    if max(indices) >= frame_count:
        m = f'frame index {max(indices)} out of range for layer {layers[0]} '
        m += f'with {frame_count} frames'
        m += f'for segment {segment!r} with collar {collar} seconds'
        raise ValueError(m)
    return indices


def _get_selected_codebook_frame_indices(artifacts, segment, collar):
    '''Return selected frame indices using artifact frame count only.'''
    start, end, collared_start, _ = _segment_times(segment, collar)
    n_frames = artifacts.indices.shape[0]
    frames = frame.Frames(n_frames, start_time=collared_start)
    selected = frames.select_frames(start, end, percentage_overlap=100)
    if not selected:
        m = f'no frames fully within {start} - {end}s, {segment!r}'
        m += f'for segment {segment!r} with collar {collar} seconds'
        raise ValueError(m)
    indices = [f.index for f in selected]
    if max(indices) >= n_frames:
        m = f'frame index {max(indices)} out of range for codebook indices '
        m += f'with {n_frames} frames'
        m += f'for segment {segment!r} with collar {collar} seconds'
        raise ValueError(m)
    return indices
