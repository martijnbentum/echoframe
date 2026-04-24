'''Stored embedding containers.'''

from __future__ import annotations

import numpy as np


class Embedding:
    '''One stored hidden-state payload with its metadata.'''

    def __init__(self, echoframe_key, store, metadata=None, data=None):
        self.echoframe_key = echoframe_key
        self.store = store
        self.metadata = metadata
        self.data = data
        self._load_missing()
        self._validate()
        self.phraser_key = self.metadata.phraser_key
        self.model_name = self.metadata.model_name
        self.output_type = self.metadata.output_type
        self.layer = self.metadata.layer

    def _load_missing(self):
        if self.metadata is None:
            self.metadata = self.store.load_metadata(self.echoframe_key)
        if self.metadata is None:
            m = f'no metadata found for echoframe_key {self.echoframe_key!r}'
            raise ValueError(m)
        if self.data is None:
            self.data = self.store.metadata_to_payload(self.metadata)
        if self.data is None:
            m = f'no embedding data found for echoframe_key {self.echoframe_key!r}'
            raise ValueError(m)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        text = f'Embedding(shape={self.shape}, layer={self.layer})'
        return text

    def _validate(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError('data must be a numpy array')
        if self.data.ndim not in (1, 2):
            raise ValueError(f'data must be ndim 1 or 2, got {self.data.ndim}')
        if self.metadata.output_type != 'hidden_state':
            raise ValueError('metadata.output_type must be hidden_state')
        if self.metadata.layer is None:
            raise ValueError('embedding metadata.layer must not be None')
        if not hasattr(self.metadata, 'model_name'):
            raise ValueError('embedding metadata must have model_name')
        if self.metadata.echoframe_key != self.echoframe_key:
            message = 'metadata.echoframe_key did not match echoframe_key'
            raise ValueError(message)


class Embeddings:
    '''A validated collection of stored Embedding objects.'''

    def __init__(self, embeddings, store):
        self._check_embeddings(embeddings) 
        self.store = store
        self.embeddings = tuple(embeddings)
        self.count = len(self.embeddings)
        self.phraser_keys = tuple(x.phraser_key for x in self.embeddings)
        self.metadatas = tuple(x.metadata for x in self.embeddings)
        self.model_name = self.embeddings[0].model_name
        self.output_type = self.embeddings[0].output_type
        self.layer = self.embeddings[0].layer
        self._validate()

    @classmethod
    def from_echoframe_keys(cls, store, keys):
        embeddings = []
        skipped_count = 0
        for key in keys:
            try: embedding = Embedding(key, store)
            except ValueError as e:
                skipped_count += 1
                print(f'skipping echoframe_key {key!r}: {e}')
                continue
            embeddings.append(embedding)
        if not embeddings:
            message = f'no embeddings were loaded skipped keys {skipped_count}'
            raise ValueError(message)
        return cls(embeddings, store)

    @property
    def data(self):
        return self.to_numpy()

    def __repr__(self):
        text = f'Embeddings(# {self.count}, layer={self.layer})'
        return text

    def to_numpy(self):
        shapes = [embedding.data.shape for embedding in self.embeddings]
        reference = shapes[0]
        if any(shape != reference for shape in shapes[1:]):
            message = 'Embeddings.to_numpy() requires identical embedding shapes'
            raise NotImplementedError(message)
        return np.stack([embedding.data for embedding in self.embeddings], axis=0)

    def _validate(self):
        seen = set()
        for phraser_key in self.phraser_keys:
            if phraser_key in seen:
                raise ValueError(f'duplicate phraser_key: {phraser_key}')
            seen.add(phraser_key)
        for embedding in self.embeddings[1:]:
            if embedding.model_name != self.model_name:
                raise ValueError('embedding model_name mismatch')
            if embedding.output_type != self.output_type:
                raise ValueError('embedding output_type mismatch')
            if embedding.layer != self.layer:
                raise ValueError('embedding layer mismatch')

    def _check_embeddings(self, embeddings):
        if not isinstance(embeddings, (list, tuple)):
            raise ValueError('embeddings must be a list or tuple of Embedding')
        if not embeddings:
            raise ValueError('embeddings must contain at least one Embedding')
        for embedding in embeddings:
            if not isinstance(embedding, Embedding):
                raise ValueError('embeddings must contain only Embedding')
