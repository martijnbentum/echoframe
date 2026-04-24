'''Typed Store loaders for codebook containers.'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codebooks import Codebook, TokenCodebooks


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


def load_codebook(store, phraser_key, collar, model_name):
    '''Load one codebook container from store-backed artifacts.'''
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
    indices_metadata = _require_metadata(store, indices_key)
    matrix_metadata = _require_metadata(store, matrix_key)
    indices = store.load(indices_key)
    model_architecture = _infer_architecture(matrix_metadata)
    return Codebook(
        echoframe_key=indices_metadata.echoframe_key,
        data=indices,
        model_architecture=model_architecture,
        codebook_matrix_echoframe_key=matrix_metadata.echoframe_key,
        path=store.root,
    )


def load_many_codebooks(store, requests):
    '''Load many codebook containers with deduped request identities.'''
    unique_requests = _dedupe_codebook_requests(requests)
    tokens = [load_codebook(store, request.phraser_key, request.collar,
        request.model_name) for request in unique_requests]
    return TokenCodebooks(tokens=tokens, path=store.root)


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


def _require_metadata(store, echoframe_key):
    metadata = store.load_metadata(echoframe_key)
    if metadata is None:
        raise ValueError('no stored output matched the requested echoframe key')
    return metadata


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
