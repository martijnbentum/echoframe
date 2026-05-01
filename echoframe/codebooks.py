'''Stored codevector containers.'''

from __future__ import annotations

import numpy as np


class Codevector:
    '''One stored codebook-indices payload with its metadata.'''
    def __init__(self, echoframe_key, store, metadata=None, data=None):
        self.echoframe_key = echoframe_key
        self.store = store
        self.metadata = metadata
        self.data = data
        self.matrix_key = None
        self.matrix_metadata = None
        self.model_architecture = None
        self._codebook_matrix = None
        self._load_missing()
        self._validate()
        self.phraser_key = self.metadata.phraser_key
        self.model_name = self.metadata.model_name
        self.output_type = self.metadata.output_type
        self.collar = self.metadata.collar

    def __repr__(self):
        text = f'Codevector(shape={self.shape}, '
        text += f'model_architecture={self.model_architecture!r})'
        return text

    def to_numpy(self):
        '''Return normalized codebook indices as a numpy array.'''
        return self.indices

    @property
    def indices(self):
        return self._normalized_indices()

    @property
    def vectors(self):
        '''Reconstruct codevectors from the stored indices.'''
        return self.to_vectors()

    @property
    def codebook_matrix(self):
        '''Load and cache the linked codebook matrix artifact.'''
        if self._codebook_matrix is None:
            self._codebook_matrix = self.store.load(self.matrix_key)
        return self._codebook_matrix

    def to_vectors(self, codebook_matrix=None):
        '''Reconstruct codevectors, optionally using a shared matrix.'''
        if codebook_matrix is None:
            codebook_matrix = self.codebook_matrix
        return _indices_to_vectors(self.indices, codebook_matrix,
            self.model_architecture)

    @property
    def shape(self):
        return self.indices.shape

    def _validate(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError('data must be a numpy array')
        if self.model_architecture not in {'spidr', 'wav2vec2'}:
            raise ValueError(
                "model_architecture must be 'spidr' or 'wav2vec2'")
        if self.metadata.output_type != 'codebook_indices':
            raise ValueError('metadata.output_type must be codebook_indices')
        if self.metadata.echoframe_key != self.echoframe_key:
            message = 'metadata.echoframe_key did not match echoframe_key'
            raise ValueError(message)
        self._normalized_indices()

    def _normalized_indices(self):
        if self.model_architecture == 'wav2vec2':
            return _normalize_wav2vec2_indices(self.data)
        return _normalize_spidr_indices(self.data)

    def _load_missing(self):
        if self.metadata is None:
            self.metadata = self.store.load_metadata(self.echoframe_key)
        if self.metadata is None:
            m = f'no metadata found for echoframe_key {self.echoframe_key!r}'
            raise ValueError(m)
        if self.data is None:
            self.data = self.store.metadata_to_payload(self.metadata)
        if self.data is None:
            m = f'no codevector data found for echoframe_key {self.echoframe_key!r}'
            raise ValueError(m)
        self.matrix_key = self.store.make_echoframe_key('codebook_matrix',
            model_name=self.metadata.model_name)
        self.matrix_metadata = self.store.load_metadata(self.matrix_key)
        if self.matrix_metadata is None:
            raise ValueError('no stored codebook matrix matched the codevector')
        self.model_architecture = infer_codebook_architecture(
            self.matrix_metadata)


class Codevectors:
    '''A validated collection of stored Codevector objects.'''
    def __init__(self, codevectors, store):
        self._check_codevectors(codevectors)
        self.store = store
        self.codevectors = tuple(codevectors)
        self.count = len(self.codevectors)
        self.phraser_keys = tuple(x.phraser_key for x in self.codevectors)
        self.metadatas = tuple(x.metadata for x in self.codevectors)
        self.model_name = self.codevectors[0].model_name
        self.output_type = self.codevectors[0].output_type
        self.model_architecture = self.codevectors[0].model_architecture
        self._validate()

    @classmethod
    def from_echoframe_keys(cls, store, keys):
        codevectors = []
        skipped_count = 0
        for key in keys:
            try: codevector = Codevector(key, store)
            except ValueError as e:
                skipped_count += 1
                print(f'skipping echoframe_key {key!r}: {e}')
                continue
            codevectors.append(codevector)
        if not codevectors:
            message = f'no codevectors were loaded skipped keys {skipped_count}'
            raise ValueError(message)
        return cls(codevectors, store)

    @property
    def data(self):
        return self.to_numpy()

    @property
    def indices(self):
        return self.to_numpy()

    def to_numpy(self):
        arrays = [codevector.indices for codevector in self.codevectors]
        reference = arrays[0].shape
        if any(array.shape != reference for array in arrays[1:]):
            message = 'Codevectors.to_numpy() requires identical codevector '
            message += 'shapes'
            raise NotImplementedError(message)
        return np.stack(arrays, axis=0)

    @property
    def vectors(self):
        matrix = self.codevectors[0].codebook_matrix
        arrays = [codevector.to_vectors(matrix)
            for codevector in self.codevectors]
        reference = arrays[0].shape
        if any(array.shape != reference for array in arrays[1:]):
            message = 'Codevectors.vectors requires identical codevector shapes'
            raise NotImplementedError(message)
        return np.stack(arrays, axis=0)

    def __repr__(self):
        text = f'Codevectors(# {self.count}, '
        text += f'model_architecture={self.model_architecture!r})'
        return text

    def _validate(self):
        seen = set()
        for phraser_key in self.phraser_keys:
            if phraser_key in seen:
                raise ValueError(f'duplicate phraser_key: {phraser_key}')
            seen.add(phraser_key)
        for codevector in self.codevectors[1:]:
            if codevector.model_name != self.model_name:
                raise ValueError('codevector model_name mismatch')
            if codevector.output_type != self.output_type:
                raise ValueError('codevector output_type mismatch')
            if codevector.model_architecture != self.model_architecture:
                raise ValueError('codevector model_architecture mismatch')

    def _check_codevectors(self, codevectors):
        if not isinstance(codevectors, (list, tuple)):
            raise ValueError('codevectors must be a list or tuple of Codevector')
        if not codevectors:
            raise ValueError('codevectors must contain at least one Codevector')
        for codevector in codevectors:
            if not isinstance(codevector, Codevector):
                raise ValueError('codevectors must contain only Codevector')


def infer_codebook_architecture(matrix_metadata):
    '''Infer codebook architecture from matrix metadata or payload.'''
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


def _indices_to_vectors(indices, codebook_matrix, model_architecture):
    codebook_matrix = np.asarray(codebook_matrix)
    if model_architecture == 'wav2vec2':
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
