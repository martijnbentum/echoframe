'''Typed Store loaders for embedding and codebook containers.'''

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .codebooks import Codebook, TokenCodebooks
from .embeddings import Embeddings, TokenEmbeddings


_VALID_FRAME_AGGREGATIONS = (None, 'mean', 'centroid')


@dataclass(frozen=True)
class EmbeddingLoadRequest:
    '''Canonical identifier for one embedding load.'''

    phraser_key: str
    collar: int
    model_name: str
    layers: tuple[int, ...]
    frame_aggregation: str | None

    @classmethod
    def from_mapping(cls, data):
        return cls(
            phraser_key=data['phraser_key'],
            collar=data['collar'],
            model_name=data['model_name'],
            layers=tuple(_normalise_layers(data['layers'])[0]),
            frame_aggregation=data.get('frame_aggregation'),
        )


@dataclass(frozen=True)
class CodebookLoadRequest:
    '''Canonical identifier for one codebook load.'''

    phraser_key: str
    collar: int
    model_name: str

    @classmethod
    def from_mapping(cls, data):
        return cls(
            phraser_key=data['phraser_key'],
            collar=data['collar'],
            model_name=data['model_name'],
        )


def load_embeddings(store, phraser_key, collar, model_name, layers,
    frame_aggregation=None):
    '''Load one embedding container from store-backed arrays.'''
    layers_list, single_layer = _normalise_layers(layers)
    _validate_frame_aggregation(frame_aggregation)
    arrays = []
    echoframe_keys = []
    for layer in layers_list:
        metadata = _find_one(store, phraser_key, collar, model_name,
            'hidden_state', layer)
        arrays.append(store.load_with_echoframe_key(metadata.entry_id))
        echoframe_keys.append(metadata.entry_id)
    return _build_embeddings(arrays, layers_list, single_layer,
        frame_aggregation, tuple(echoframe_keys))


def load_many_embeddings(store, requests):
    '''Load many embedding containers with deduped request identities.'''
    unique_requests = _dedupe_embedding_requests(requests)
    tokens = [load_embeddings(store, request.phraser_key, request.collar,
        request.model_name, request.layers,
        frame_aggregation=request.frame_aggregation)
        for request in unique_requests]
    return TokenEmbeddings(tokens=tokens)


def load_codebook(store, phraser_key, collar, model_name):
    '''Load one codebook container from store-backed artifacts.'''
    indices_metadata = _find_one(store, phraser_key, collar, model_name,
        'codebook_indices', 0)
    matrix_metadata = _find_one(store, phraser_key, collar, model_name,
        'codebook_matrix', 0)
    indices = store.load_with_echoframe_key(indices_metadata.entry_id)
    model_architecture = _infer_architecture(matrix_metadata)
    return Codebook(
        echoframe_key=indices_metadata.entry_id,
        data=indices,
        model_architecture=model_architecture,
        codebook_matrix_echoframe_key=matrix_metadata.entry_id,
    ).bind_store(store)


def load_many_codebooks(store, requests):
    '''Load many codebook containers with deduped request identities.'''
    unique_requests = _dedupe_codebook_requests(requests)
    tokens = [load_codebook(store, request.phraser_key, request.collar,
        request.model_name) for request in unique_requests]
    return TokenCodebooks(tokens=tokens)


def _dedupe_embedding_requests(requests):
    unique = []
    seen = set()
    for request in requests:
        if isinstance(request, EmbeddingLoadRequest):
            canonical = request
        else:
            canonical = EmbeddingLoadRequest.from_mapping(request)
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(canonical)
    return unique


def _dedupe_codebook_requests(requests):
    unique = []
    seen = set()
    for request in requests:
        if isinstance(request, CodebookLoadRequest):
            canonical = request
        else:
            canonical = CodebookLoadRequest.from_mapping(request)
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(canonical)
    return unique


def _find_one(store, phraser_key, collar, model_name, output_type, layer):
    metadata = store.find_one(phraser_key=phraser_key, collar=collar,
        model_name=model_name, output_type=output_type, layer=layer)
    if metadata is None:
        raise ValueError('no stored output matched the requested criteria')
    return metadata


def _normalise_layers(layers):
    if isinstance(layers, Integral):
        return [int(layers)], True
    layers = list(layers)
    if not layers:
        raise ValueError('layers must not be empty')
    return layers, False


def _validate_frame_aggregation(frame_aggregation):
    if frame_aggregation not in _VALID_FRAME_AGGREGATIONS:
        message = 'frame_aggregation must be one of '
        message += f'{_VALID_FRAME_AGGREGATIONS}, got {frame_aggregation!r}'
        raise ValueError(message)


def _build_embeddings(arrays, layers_list, single_layer, frame_aggregation,
    echoframe_keys):
    processed = [_apply_aggregation(arr, frame_aggregation) for arr in arrays]

    if single_layer:
        data = processed[0]
        dims = ('embed_dim',) if frame_aggregation else ('frames', 'embed_dim')
        return Embeddings(data=data, dims=dims, layers=None,
            echoframe_keys=echoframe_keys,
            frame_aggregation=frame_aggregation)

    data = np.stack(processed, axis=0)
    dims = (('layers', 'embed_dim') if frame_aggregation else
        ('layers', 'frames', 'embed_dim'))
    return Embeddings(data=data, dims=dims, layers=tuple(layers_list),
        echoframe_keys=echoframe_keys,
        frame_aggregation=frame_aggregation)


def _apply_aggregation(data, frame_aggregation):
    if frame_aggregation is None:
        return data
    if frame_aggregation == 'mean':
        return np.mean(data, axis=0)
    if frame_aggregation == 'centroid':
        return data[len(data) // 2]
    raise ValueError(f'unknown aggregation: {frame_aggregation!r}')


def _infer_architecture(matrix_metadata):
    shape = getattr(matrix_metadata, 'shape', None)
    if shape:
        shape = tuple(shape)
        if len(shape) == 3:
            return 'spidr'
        return 'wav2vec2'

    payload = matrix_metadata.load_payload()
    matrix = np.asarray(payload)
    if matrix.ndim == 3:
        return 'spidr'
    return 'wav2vec2'
