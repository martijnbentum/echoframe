'''Immutable containers for stored codebook payloads.'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings
from typing import Any

import numpy as np


def _validate_key(key, field_name):
    valid = isinstance(key, str) and bool(key)
    valid = valid or (isinstance(key, bytes) and bool(key))
    if not valid:
        raise ValueError(f'{field_name} must be a non-empty string or bytes')


def _normalize_wav2vec2_indices(indices):
    array = np.asarray(indices)
    if array.ndim == 1:
        if array.shape[0] != 2:
            raise ValueError(
                'wav2vec2 codebook indices must contain two indices')
        return array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            'wav2vec2 codebook indices must have shape (frames, 2)')
    return array


def _normalize_spidr_indices(indices):
    array = np.asarray(indices)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(
            'spidr codebook indices must have shape (frames, heads)')
    return array


@dataclass(frozen=True)
class Codebook:
    '''Immutable container for one token of codebook indices.'''

    echoframe_key: str
    data: Any
    model_architecture: str
    codebook_matrix_echoframe_key: str
    path: Path | None = field(default=None, compare=False)
    _store: Any = field(default=None, init=False, repr=False, compare=False)
    _codebook_matrix: Any = field(default=None, init=False, repr=False,
        compare=False)
    _codebook_matrix_loaded: bool = field(default=False, init=False,
        repr=False, compare=False)

    def __post_init__(self):
        _validate_key(self.echoframe_key, 'echoframe_key')
        _validate_key(self.codebook_matrix_echoframe_key,
            'codebook_matrix_echoframe_key')
        if self.model_architecture not in {'spidr', 'wav2vec2'}:
            raise ValueError(
                "model_architecture must be 'spidr' or 'wav2vec2'")
        self._normalized_indices()

    def bind_store(self, store):
        '''Attach a store for lazy linked-artifact loading.'''
        object.__setattr__(self, '_store', store)
        root = getattr(store, 'root', None)
        if root is not None:
            object.__setattr__(self, 'path', Path(root))
        return self

    @property
    def store(self):
        '''Return the bound store or reopen one from path metadata.'''
        store = self._store
        if store is not None:
            return store
        if self.path is None:
            raise ValueError('codebook is not bound to a store and has no path')
        from .store import Store
        store = Store(self.path)
        object.__setattr__(self, '_store', store)
        return store

    def _normalized_indices(self):
        if self.model_architecture == 'wav2vec2':
            return _normalize_wav2vec2_indices(self.data)
        return _normalize_spidr_indices(self.data)

    def _load_codebook_matrix(self):
        if self._codebook_matrix_loaded:
            return self._codebook_matrix
        payload = self.store.load(self.codebook_matrix_echoframe_key)
        object.__setattr__(self, '_codebook_matrix', payload)
        object.__setattr__(self, '_codebook_matrix_loaded', True)
        return payload

    @property
    def codebook_matrix(self):
        '''Load and cache the linked codebook matrix artifact.'''
        return self._load_codebook_matrix()

    @property
    def metadata(self):
        '''Load metadata for the stored indices artifact.'''
        return self.store.load_metadata(self.echoframe_key)

    def to_numpy(self):
        '''Return normalized codebook indices as a numpy array.'''
        return self._normalized_indices()

    @property
    def codevectors(self):
        '''Reconstruct codevectors from the stored indices.'''
        indices = self._normalized_indices()
        codebook_matrix = np.asarray(self.codebook_matrix)
        if self.model_architecture == 'wav2vec2':
            codevectors = []
            for index1, index2 in indices:
                codevectors.append(np.hstack((
                    codebook_matrix[index1],
                    codebook_matrix[index2],
                )))
            return np.asarray(codevectors)

        if codebook_matrix.ndim != 3:
            raise ValueError(
                'spidr codebook matrix must have shape '
                '(heads, codebook_size, codevector_dim)')
        if indices.shape[1] != codebook_matrix.shape[0]:
            raise ValueError(
                'spidr head count does not match the linked codebook matrix')
        codevectors = []
        for frame_indices in indices:
            codevectors.append(np.stack([
                codebook_matrix[head_index, code_index]
                for head_index, code_index in enumerate(frame_indices)
            ], axis=0))
        return np.asarray(codevectors)

    def __repr__(self):
        shape = self.to_numpy().shape
        text = 'Codebook('
        text += f'shape={shape}, '
        text += f'model_architecture={self.model_architecture!r})'
        return text


@dataclass(frozen=True)
class TokenCodebooks:
    '''Immutable ordered collection of token codebook objects.'''

    tokens: list[Codebook]
    path: Path | None = field(default=None, compare=False)
    _store: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.tokens, list):
            raise ValueError('tokens must be a list of Codebook')
        if not self.tokens:
            raise ValueError('tokens must contain at least one Codebook')

        deduped = []
        seen = set()
        duplicate_keys = []
        for token in self.tokens:
            if not isinstance(token, Codebook):
                raise ValueError('tokens must contain only Codebook')
            if token.echoframe_key in seen:
                if token.echoframe_key not in duplicate_keys:
                    duplicate_keys.append(token.echoframe_key)
                continue
            seen.add(token.echoframe_key)
            deduped.append(token)
        reference = deduped[0]
        token_path = None
        for token in deduped[1:]:
            if token.model_architecture != reference.model_architecture:
                raise ValueError('token model_architecture mismatch')
        for token in deduped:
            if token.path is not None:
                if token_path is None:
                    token_path = token.path
                elif token.path != token_path:
                    raise ValueError('token path mismatch')
        if self.path is None and token_path is not None:
            object.__setattr__(self, 'path', token_path)
        elif self.path is not None and token_path is not None and (
            token_path != self.path
        ):
            raise ValueError('token path mismatch')
        if self._store is None:
            token_store = deduped[0].__dict__.get('_store')
            if token_store is not None and all(
                token.__dict__.get('_store') is token_store
                for token in deduped
            ):
                object.__setattr__(self, '_store', token_store)
        if duplicate_keys:
            message = 'duplicate echoframe_key values were removed'
            for key in duplicate_keys:
                message += f'\n{key}'
            warnings.warn(message, stacklevel=2)
        object.__setattr__(self, 'tokens', deduped)

    def bind_store(self, store):
        '''Attach a store for lazy linked-artifact loading.'''
        object.__setattr__(self, '_store', store)
        root = getattr(store, 'root', None)
        if root is not None:
            object.__setattr__(self, 'path', Path(root))
        for token in self.tokens:
            token.bind_store(store)
        return self

    @property
    def token_count(self):
        return len(self.tokens)

    @property
    def echoframe_keys(self):
        keys = []
        for token in self.tokens:
            keys.append(token.echoframe_key)
        return tuple(keys)

    @property
    def metadatas(self):
        return tuple(token.metadata for token in self.tokens)

    @property
    def model_architecture(self):
        return self.tokens[0].model_architecture

    @property
    def store(self):
        '''Return the bound store or reopen one from path metadata.'''
        store = self._store
        if store is not None:
            return store
        token_store = self.tokens[0].__dict__.get('_store')
        if token_store is not None and all(
            token.__dict__.get('_store') is token_store for token in self.tokens
        ):
            object.__setattr__(self, '_store', token_store)
            return token_store
        if self.path is not None:
            from .store import Store
            store = Store(self.path)
            object.__setattr__(self, '_store', store)
            return store
        token_paths = {token.path for token in self.tokens if token.path is not None}
        if len(token_paths) == 1:
            path = token_paths.pop()
            object.__setattr__(self, 'path', path)
            from .store import Store
            store = Store(path)
            object.__setattr__(self, '_store', store)
            return store
        raise ValueError(
            'token codebooks are not bound to a store and have no path')

    def to_numpy(self):
        '''Return a stacked numpy array when token shapes are uniform.'''
        arrays = [token.to_numpy() for token in self.tokens]
        reference = arrays[0].shape
        if any(array.shape != reference for array in arrays[1:]):
            message = 'TokenCodebooks.to_numpy() requires identical token '
            message += 'shapes'
            raise NotImplementedError(message)
        return np.stack(arrays, axis=0)

    @property
    def codevectors(self):
        '''Return stacked codevectors when token shapes are uniform.'''
        arrays = [token.codevectors for token in self.tokens]
        reference = arrays[0].shape
        if any(array.shape != reference for array in arrays[1:]):
            message = 'TokenCodebooks.codevectors requires identical token '
            message += 'shapes'
            raise NotImplementedError(message)
        return np.stack(arrays, axis=0)

    def __repr__(self):
        text = 'TokenCodebooks('
        text += f'token_count={self.token_count}, '
        text += f'model_architecture={self.model_architecture!r})'
        return text
