'''Named array container for model embedding outputs.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Embeddings:
    '''Immutable named array container for embedding outputs.
    data:    raw embedding array
    dims:    axis labels; length must equal data.ndim
    layers:  layer indices when 'layers' is in dims; None otherwise
    '''

    data: np.ndarray
    dims: tuple
    layers: Optional[tuple] = None

    def __post_init__(self):
        if len(self.dims) != self.data.ndim:
            message = (
                f'len(dims)={len(self.dims)} must equal '
                f'data.ndim={self.data.ndim}'
            )
            raise ValueError(message)
        if 'layers' in self.dims and self.layers is None:
            message = "'layers' in dims but layers is None"
            raise ValueError(message)
        if self.layers is not None and 'layers' not in self.dims:
            message = "layers is set but 'layers' not in dims"
            raise ValueError(message)
        if self.layers is not None:
            layers_axis = self.dims.index('layers')
            if len(self.layers) != self.data.shape[layers_axis]:
                message = (
                    f'len(layers)={len(self.layers)} does not match '
                    f'layers axis size={self.data.shape[layers_axis]}'
                )
                raise ValueError(message)

    @property
    def shape(self):
        '''Delegate shape to underlying array.'''
        return self.data.shape

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
        return Embeddings(data=new_data, dims=new_dims,
            layers=None)

    @staticmethod
    def concat(items, axis):
        '''Concatenate a list of Embeddings along a named axis.
        items:   list of Embeddings with identical dims
        axis:    named axis to concatenate along
        '''
        if not items:
            raise ValueError('items must not be empty')
        dims = items[0].dims
        for item in items[1:]:
            if item.dims != dims:
                message = f'dims mismatch: {item.dims!r} != {dims!r}'
                raise ValueError(message)
        if axis not in dims:
            message = f"axis '{axis}' not in dims={dims!r}"
            raise ValueError(message)
        axis_index = dims.index(axis)
        new_data = np.concatenate([item.data for item in items],
            axis=axis_index)
        if axis == 'layers':
            merged_layers = sum((item.layers for item in items), ())
            return Embeddings(data=new_data, dims=dims,
                layers=merged_layers)
        layers = items[0].layers
        return Embeddings(data=new_data, dims=dims, layers=layers)

    def __add__(self, other):
        '''Concatenate along the frames axis.'''
        if 'frames' not in self.dims:
            raise ValueError("'frames' not in dims")
        return Embeddings.concat([self, other], axis='frames')
