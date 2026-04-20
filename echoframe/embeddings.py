'''Named array container for model embedding outputs.'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class Embeddings:
    '''Immutable named array container for embedding outputs.
    echoframe_keys:  echoframe metadata identifiers for this token
    data:    raw embedding array
    dims:    axis labels; length must equal data.ndim
    layers:  layer indices when 'layers' is in dims; None otherwise
    frame_aggregation:  aggregation over frames, or None if frames remain
    '''

    echoframe_keys: tuple[str, ...]
    data: np.ndarray
    dims: tuple
    layers: Optional[tuple] = None
    frame_aggregation: Optional[str] = None
    path: Path | None = field(default=None, compare=False)
    _store: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.echoframe_keys, tuple) or not self.echoframe_keys:
            raise ValueError('echoframe_keys must be a non-empty tuple')
        for k in self.echoframe_keys:
            valid = isinstance(k, str) and bool(k)
            valid = valid or (isinstance(k, bytes) and bool(k))
            if not valid:
                raise ValueError(
                    'every element of echoframe_keys must be a non-empty '
                    'string or bytes')
        if len(self.dims) != self.data.ndim:
            message = f'dims length mismatch: len(dims)={len(self.dims)} '
            message += f'data.ndim={self.data.ndim}'
            raise ValueError(message)
        if 'layers' in self.dims and self.layers is None:
            message = "'layers' in dims requires layers metadata"
            raise ValueError(message)
        if self.layers is not None and 'layers' not in self.dims:
            message = "layers metadata requires 'layers' in dims"
            raise ValueError(message)
        if self.layers is not None:
            layers_axis = self.dims.index('layers')
            if len(self.layers) != self.data.shape[layers_axis]:
                message = f'layers length mismatch: len(layers)='
                message += f'{len(self.layers)} '
                message += f'layers axis size={self.data.shape[layers_axis]}'
                raise ValueError(message)
        if 'layers' not in self.dims:
            if len(self.echoframe_keys) != 1:
                raise ValueError(
                    'echoframe_keys must have length 1 when '
                    "'layers' is not in dims")
        else:
            if len(self.echoframe_keys) != len(self.layers):
                raise ValueError(
                    'echoframe_keys length must equal number of layers')
        if self.frame_aggregation is not None:
            invalid = not isinstance(self.frame_aggregation, str)
            invalid = invalid or not self.frame_aggregation
            if invalid:
                raise ValueError('frame_aggregation must be a non-empty '
                    'string')
        if 'frames' in self.dims and self.frame_aggregation is not None:
            raise ValueError(
                "frame_aggregation must be None when 'frames' is in dims")

    def bind_store(self, store):
        '''Attach a store for lazy linked-artifact loading.'''
        object.__setattr__(self, '_store', store)
        root = getattr(store, 'root', None)
        if root is not None:
            object.__setattr__(self, 'path', Path(root))
        return self

    @property
    def echoframe_key(self):
        return self.echoframe_keys[0]

    @property
    def echoframe_key_hex(self) -> str:
        key = self.echoframe_key
        if isinstance(key, bytes):
            return key.hex()
        return key

    @property
    def store(self):
        '''Return the bound store or reopen one from path metadata.'''
        store = self._store
        if store is not None:
            return store
        if self.path is None:
            raise ValueError(
                'embeddings are not bound to a store and have no path')
        from .store import Store
        store = Store(self.path)
        object.__setattr__(self, '_store', store)
        return store

    @property
    def shape(self):
        '''Delegate shape to underlying array.'''
        return self.data.shape

    def __repr__(self):
        text = 'Embeddings('
        text += f'echoframe_keys={self.echoframe_keys}, '
        text += f'shape={self.shape}, dims={self.dims}, '
        text += f'layers={self.layers}, '
        text += f'frame_aggregation={self.frame_aggregation!r})'
        return text

    def key_for_layer(self, n: int) -> str:
        if 'layers' not in self.dims:
            raise ValueError("'layers' not in dims")
        if n not in self.layers:
            raise ValueError(f'layer {n} not in self.layers={self.layers}')
        pos = self.layers.index(n)
        return self.echoframe_keys[pos]

    def layer(self, n):
        '''Return a new Embeddings with only layer n (looked up by value).
        n:    layer index value (not position)
        '''
        if self.layers is None or 'layers' not in self.dims:
            raise ValueError("'layers' not in dims")
        if n not in self.layers:
            raise ValueError(f'layer {n} not in self.layers={self.layers}')
        pos = self.layers.index(n)
        layers_axis = self.dims.index('layers')
        new_data = np.take(self.data, pos, axis=layers_axis)
        new_dims = tuple(d for d in self.dims if d != 'layers')
        result = Embeddings(echoframe_keys=(self.key_for_layer(n),),
            data=new_data, dims=new_dims, layers=None,
            frame_aggregation=self.frame_aggregation, path=self.path)
        if self._store is not None:
            result.bind_store(self._store)
        return result

    def to_numpy(self):
        '''Return the underlying numpy array.'''
        return self.data


@dataclass(frozen=True)
class TokenEmbeddings:
    '''Immutable ordered collection of token embeddings.
    tokens:  ordered list of single-token Embeddings objects
    '''

    tokens: list[Embeddings]
    path: Path | None = field(default=None, compare=False)
    _failed_metadatas: tuple[dict, ...] = field(default_factory=tuple,
        compare=False)
    _store: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.tokens, list):
            raise ValueError('tokens must be a list of Embeddings')
        if not self.tokens:
            raise ValueError('tokens must contain at least one Embeddings')
        if not isinstance(self._failed_metadatas, tuple):
            raise ValueError('_failed_metadatas must be a tuple')

        deduped = []
        seen = set()
        duplicate_keys = []
        for token in self.tokens:
            if not isinstance(token, Embeddings):
                raise ValueError('tokens must contain only Embeddings')
            if token.echoframe_key in seen:
                if token.echoframe_key not in duplicate_keys:
                    duplicate_keys.append(token.echoframe_key)
                continue
            seen.add(token.echoframe_key)
            deduped.append(token)
        reference = deduped[0]
        token_path = None
        for token in deduped[1:]:
            if token.dims != reference.dims:
                raise ValueError('token dims mismatch')
            if token.layers != reference.layers:
                raise ValueError('token layers mismatch')
            if token.frame_aggregation != reference.frame_aggregation:
                raise ValueError('token frame_aggregation mismatch')
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
        '''Return the number of tokens.'''
        return len(self.tokens)

    @property
    def echoframe_keys(self):
        '''Return ordered echoframe keys.'''
        return tuple(token.echoframe_key for token in self.tokens)

    @property
    def dims(self):
        '''Return shared dims.'''
        return self.tokens[0].dims

    @property
    def layers(self):
        '''Return shared layers.'''
        return self.tokens[0].layers

    @property
    def frame_aggregation(self):
        '''Return shared frame aggregation.'''
        return self.tokens[0].frame_aggregation

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
            'token embeddings are not bound to a store and have no path')

    def __repr__(self):
        text = 'TokenEmbeddings('
        text += f'token_count={self.token_count}, '
        text += f'echoframe_keys={self.echoframe_keys}, '
        text += f'failed_count={len(self._failed_metadatas)}, '
        text += f'dims={self.dims}, layers={self.layers}, '
        text += f'frame_aggregation={self.frame_aggregation!r})'
        return text

    def layer(self, n):
        '''Return a new TokenEmbeddings with only layer n.'''
        tokens = [token.layer(n) for token in self.tokens]
        result = TokenEmbeddings(tokens=tokens, path=self.path,
            _failed_metadatas=self._failed_metadatas)
        if self._store is not None:
            result.bind_store(self._store)
        return result

    def to_numpy(self):
        '''Return a stacked numpy array when token shapes are uniform.'''
        shapes = [token.data.shape for token in self.tokens]
        reference = shapes[0]
        if any(shape != reference for shape in shapes[1:]):
            message = 'TokenEmbeddings.to_numpy() requires identical token '
            message += 'shapes'
            raise NotImplementedError(message)
        return np.stack([token.to_numpy() for token in self.tokens], axis=0)

    def aggregate(self, method='mean'):
        '''Aggregate token arrays over the token axis.
        method:    aggregation method; only 'mean' is supported
        '''
        if method != 'mean':
            message = "TokenEmbeddings.aggregate() only supports method='mean'"
            raise NotImplementedError(message)
        if 'frames' in self.dims:
            message = "TokenEmbeddings.aggregate() does not support 'frames'"
            raise NotImplementedError(message)
        return np.stack([token.to_numpy() for token in self.tokens],
            axis=0).mean(axis=0)
