'''Batch segment-based feature retrieval orchestration for echoframe.'''

from __future__ import annotations

from numbers import Integral

import echoframe
import frame
import to_vector

from echoframe.codebooks import TokenCodebooks
from echoframe.embeddings import TokenEmbeddings
from echoframe.segment_features import _get_selected_frame_indices
from echoframe.segment_features import _normalise_layers
from echoframe.segment_features import _segment_times
from echoframe.segment_features import _store_embeddings_from_outputs
from echoframe.segment_features import _validate_frame_aggregation
from echoframe.segment_features import embeddings_missing
from echoframe.segment_features import get_codebook_indices
from scripts.segment_batch_validation import validate_segment_batch_contexts
from scripts.segment_batch_validation import warn_for_failed_metadatas


def get_embeddings_batch(segments, layers, collar=500, model_name='wav2vec2',
    frame_aggregation=None, model=None, store=None,
    store_root='echoframe', gpu=False, tags=None, batch_minutes=None):
    '''Return embeddings for multiple segment objects.'''
    segments = _require_segments(segments)
    single_layer = isinstance(layers, Integral)
    layers_list = _normalise_layers(layers)
    _validate_frame_aggregation(frame_aggregation)
    if store is None:
        store = echoframe.Store(store_root)
    contexts = [_build_segment_batch_context(segment, collar)
        for segment in segments]
    valid_contexts, failed_metadatas = validate_segment_batch_contexts(
        contexts)
    warn_for_failed_metadatas(failed_metadatas, item_label='segments')
    if not valid_contexts:
        raise ValueError('no valid segments remained after batch validation')

    missing_contexts = []
    for context in valid_contexts:
        if embeddings_missing(store, context['phraser_key'], collar,
            model_name, layers_list):
            missing_contexts.append(context)

    if missing_contexts:
        if model is None:
            raise ValueError('model must be provided')
        _compute_and_store_embeddings_batch(missing_contexts, collar,
            layers_list, model_name, model, store, gpu, tags,
            batch_minutes=batch_minutes)

    tokens = []
    load_layers = layers if single_layer else layers_list
    for context in valid_contexts:
        token = store.load_embeddings(context['phraser_key'], collar,
            model_name, load_layers, frame_aggregation=frame_aggregation)
        tokens.append(token)
    return TokenEmbeddings(tokens=tokens, path=store.root,
        _failed_metadatas=tuple(failed_metadatas)).bind_store(store)


def get_codebook_indices_batch(segments, collar=500, model_name='wav2vec2',
    model=None, store=None, store_root='echoframe', gpu=False, tags=None,
    on_error='skip'):
    '''Return codebook indices for multiple segment objects.'''
    segments = _require_segments(segments)
    if store is None:
        store = echoframe.Store(store_root)
    _validate_on_error(on_error)
    tokens = []
    for segment in segments:
        try:
            token = get_codebook_indices(segment, model_name=model_name,
                model=model, collar=collar, store=store,
                store_root=store_root, gpu=gpu, tags=tags)
        except Exception:
            if on_error == 'raise':
                raise
            continue
        tokens.append(token)
    if not tokens:
        raise ValueError('no codebook indices succeeded for the requested '
            'segments')
    return TokenCodebooks(tokens=tokens, path=store.root).bind_store(store)


def _require_segments(segments):
    segments = list(segments)
    if not segments:
        raise ValueError('segments must contain at least one segment')
    return segments


def _validate_on_error(on_error):
    if on_error not in {'skip', 'raise'}:
        raise ValueError("on_error must be 'skip' or 'raise'")


def _build_segment_batch_context(segment, collar):
    start_seconds, end_seconds, collared_start, collared_end = \
        _segment_times(segment, collar)
    return {
        'segment': segment,
        'phraser_key': segment.key,
        'audio_filename': segment.audio.filename,
        'col_start_ms': int(round(collared_start * 1000.0)),
        'col_end_ms': int(round(collared_end * 1000.0)),
        'orig_start_ms': int(round(start_seconds * 1000.0)),
        'orig_end_ms': int(round(end_seconds * 1000.0)),
    }


def _compute_and_store_embeddings_batch(contexts, collar, layers, model_name,
    compute_model, store, gpu, tags, batch_minutes=None):
    audio_filenames = [context['audio_filename'] for context in contexts]
    starts = [context['col_start_ms'] / 1000.0 for context in contexts]
    ends = [context['col_end_ms'] / 1000.0 for context in contexts]
    outputs_list = to_vector.filename_batch_to_vector(audio_filenames,
        starts=starts, ends=ends, model=compute_model, gpu=gpu,
        numpify_output=True, batch_minutes=batch_minutes)
    for context, outputs in zip(contexts, outputs_list):
        _store_embeddings_from_outputs(outputs, context['segment'], collar,
            layers, model_name, store, tags)


__all__ = [
    'get_embeddings_batch',
    'get_codebook_indices_batch',
]
