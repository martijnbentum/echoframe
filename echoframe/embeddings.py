'''Named array container for model embedding outputs.'''

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Embeddings:
    '''Immutable named array container for embedding outputs.
    echoframe_key:  echoframe metadata identifier for this token
    data:    raw embedding array
    dims:    axis labels; length must equal data.ndim
    layers:  layer indices when 'layers' is in dims; None otherwise
    frame_aggregation:  aggregation over frames, or None if frames remain
    '''

    echoframe_key: str
    data: np.ndarray
    dims: tuple
    layers: Optional[tuple] = None
    frame_aggregation: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.echoframe_key, str) or not self.echoframe_key:
            raise ValueError('echoframe_key must be a non-empty string')
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
        if self.frame_aggregation is not None:
            invalid = not isinstance(self.frame_aggregation, str)
            invalid = invalid or not self.frame_aggregation
            if invalid:
                raise ValueError('frame_aggregation must be a non-empty '
                    'string')
        if 'frames' in self.dims and self.frame_aggregation is not None:
            raise ValueError(
                "frame_aggregation must be None when 'frames' is in dims")

    @property
    def shape(self):
        '''Delegate shape to underlying array.'''
        return self.data.shape

    def __repr__(self):
        text = 'Embeddings('
        text += f'echoframe_key={self.echoframe_key!r}, '
        text += f'shape={self.shape}, dims={self.dims}, '
        text += f'layers={self.layers}, '
        text += f'frame_aggregation={self.frame_aggregation!r})'
        return text

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
        return Embeddings(echoframe_key=self.echoframe_key, data=new_data,
            dims=new_dims, layers=None,
            frame_aggregation=self.frame_aggregation)

    def to_numpy(self):
        '''Return the underlying numpy array.'''
        return self.data


@dataclass(frozen=True)
class TokenEmbeddings:
    '''Immutable ordered collection of token embeddings.
    tokens:  ordered list of single-token Embeddings objects
    '''

    tokens: list[Embeddings]

    def __post_init__(self):
        if not isinstance(self.tokens, list):
            raise ValueError('tokens must be a list of Embeddings')
        if not self.tokens:
            raise ValueError('tokens must contain at least one Embeddings')

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
        for token in deduped[1:]:
            if token.dims != reference.dims:
                raise ValueError('token dims mismatch')
            if token.layers != reference.layers:
                raise ValueError('token layers mismatch')
            if token.frame_aggregation != reference.frame_aggregation:
                raise ValueError('token frame_aggregation mismatch')
        if duplicate_keys:
            message = 'duplicate echoframe_key values were removed'
            for key in duplicate_keys:
                message += f'\n{key}'
            warnings.warn(message, stacklevel=2)
        object.__setattr__(self, 'tokens', deduped)

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

    def __repr__(self):
        text = 'TokenEmbeddings('
        text += f'token_count={self.token_count}, '
        text += f'echoframe_keys={self.echoframe_keys}, '
        text += f'dims={self.dims}, layers={self.layers}, '
        text += f'frame_aggregation={self.frame_aggregation!r})'
        return text

    def layer(self, n):
        '''Return a new TokenEmbeddings with only layer n.'''
        tokens = [token.layer(n) for token in self.tokens]
        return TokenEmbeddings(tokens=tokens)

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
