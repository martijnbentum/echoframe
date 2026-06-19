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

    @property
    def phraser_object(self):
        return self.metadata.phraser_object

    @property
    def object_class(self):
        return self.phraser_object.object_type

    def __repr__(self):
        text = (f'Embedding(shape={self.shape}, layer={self.layer}, '
            f'class={self.object_class})')
        return text

    def to_frames(self, stride=0.02, field=0.025):
        '''Wrap the payload as a frame.Frames grid anchored at the segment start.
        Frame index equals matrix row index. The grid is anchored at the
        segment start; this is exact when the collar is a multiple of stride
        (true for the default 500ms collar), otherwise row 0 may be offset
        from the segment start by up to one stride.
        '''
        from frame.frames import make_frames_from_numpy_matrix
        if self.data.ndim != 2:
            raise ValueError('slicing requires a 2D (frames, dim) payload')
        start_time = self.metadata.phraser_object.start_seconds
        return make_frames_from_numpy_matrix(self.data, stride, field,
            start_time=start_time)

    def slice_time(self, start, end, percentage_overlap=None, stride=0.02,
        field=0.025):
        '''Return payload rows overlapping the absolute [start, end] seconds.'''
        frames = self.to_frames(stride, field)
        selected = frames.select_frames(start, end,
            percentage_overlap=percentage_overlap)
        rows = [frame.index for frame in selected]
        if not rows:
            raise ValueError(f'no frames overlap {start:.3f}-{end:.3f}s')
        return self.data[rows]

    def slice_segment(self, segment, percentage_overlap=None, stride=0.02,
        field=0.025):
        '''Return payload rows for a descendant phraser segment.
        segment:  phraser segment (word, syllable, or phone) with seconds API
        '''
        rows = self.slice_time(segment.start_seconds, segment.end_seconds,
            percentage_overlap=percentage_overlap, stride=stride, field=field)
        return rows

    def middle_frame_time(self, start, end, percentage_overlap=None,
        stride=0.02, field=0.025):
        '''Return the single payload row for the middle frame overlapping
        [start, end] seconds.'''
        from frame.frames import select_middle_frame
        frames = self.to_frames(stride, field)
        selected = frames.select_frames(start, end,
            percentage_overlap=percentage_overlap)
        if not selected:
            raise ValueError(f'no frames overlap {start:.3f}-{end:.3f}s')
        frame = select_middle_frame(selected)
        return self.data[frame.index]

    def middle_frame_segment(self, segment, percentage_overlap=None,
        stride=0.02, field=0.025):
        '''Middle-frame row for a descendant phraser segment.'''
        row = self.middle_frame_time(segment.start_seconds,
            segment.end_seconds, percentage_overlap=percentage_overlap,
            stride=stride, field=field)
        return row

    def aggregate_time(self, start, end, method='mean', percentage_overlap=None,
        stride=0.02, field=0.025):
        '''Aggregate payload rows overlapping [start, end] to one vector.
        method:  'mean'   -> average of all overlapping rows
                 'middle' -> the middle frame's row
        Returns a 1D (dim,) array.
        '''
        if method == 'middle':
            vector = self.middle_frame_time(start, end,
                percentage_overlap=percentage_overlap, stride=stride,
                field=field)
            return vector
        if method == 'mean':
            rows = self.slice_time(start, end,
                percentage_overlap=percentage_overlap, stride=stride,
                field=field)
            return rows.mean(axis=0)
        raise ValueError(f"method must be 'mean' or 'middle', got {method!r}")

    def aggregate_segment(self, segment, method='mean', percentage_overlap=None,
        stride=0.02, field=0.025):
        '''Aggregate rows for a descendant phraser segment (see
        aggregate_time).'''
        vector = self.aggregate_time(segment.start_seconds, segment.end_seconds,
            method=method, percentage_overlap=percentage_overlap,
            stride=stride, field=field)
        return vector

    def sub_embedding(self, phraser_object, aggregate=None,
        percentage_overlap=None, stride=0.02, field=0.025):
        '''Return a SlicedEmbedding for a descendant phraser object.
        phraser_object:  descendant phraser segment (word, syllable, or phone)
        aggregate:       None -> 2D rows; 'mean' or 'middle' -> 1D vector
        '''
        frames = self.to_frames(stride, field)
        selected = frames.select_frames(phraser_object.start_seconds,
            phraser_object.end_seconds, percentage_overlap=percentage_overlap)
        if not selected:
            raise ValueError(f'no frames overlap {phraser_object.object_type} '
                f'{phraser_object.label!r}')
        if aggregate == 'middle':
            from frame.frames import select_middle_frame
            frame = select_middle_frame(selected)
            rows = [frame.index]
            data = self.data[frame.index]
        else:
            rows = [frame.index for frame in selected]
            if aggregate is None: data = self.data[rows]
            elif aggregate == 'mean': data = self.data[rows].mean(axis=0)
            else: raise ValueError(
                "aggregate must be None, 'mean', or 'middle'")
        return SlicedEmbedding(self, phraser_object, data, rows)

    def sub_embeddings(self, object_class, aggregate=None,
        percentage_overlap=None, stride=0.02, field=0.025):
        '''Return a list of SlicedEmbeddings for all descendant objects of
        object_class.
        object_class:  'word', 'syllable', or 'phone' (singular or plural,
                       case-insensitive); must be a descendant segment type of
                       this embedding's phraser_object.
        aggregate:     None -> 2D rows; 'mean' or 'middle' -> 1D vector each
        '''
        accessor = object_class.lower()
        if not accessor.endswith('s'): accessor += 's'
        segments = getattr(self.phraser_object, accessor, None)
        if segments is None:
            m = f'{self.object_class} has no descendant {object_class!r}'
            raise ValueError(m)
        sub_embeddings = []
        for segment in segments:
            sub = self.sub_embedding(segment, aggregate=aggregate,
                percentage_overlap=percentage_overlap, stride=stride,
                field=field)
            sub_embeddings.append(sub)
        return sub_embeddings

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


class SlicedEmbedding:
    '''A view of a stored Embedding sliced to a descendant phraser object.

    Produced by Embedding.sub_embedding(...). Holds the rows of the parent
    (stored) embedding that overlap a descendant phraser object, and keeps a
    reference to the parent embedding it was derived from.
    '''

    def __init__(self, parent_embedding, phraser_object, data, rows):
        self.parent_embedding = parent_embedding
        self.parent_phraser_key = parent_embedding.phraser_key
        self.parent_collar = parent_embedding.metadata.collar
        self.parent_class = parent_embedding.metadata.phraser_object.object_type
        self.phraser_object = phraser_object
        self.object_class = phraser_object.object_type
        self.data = data
        self.rows = rows
        self.model_name = parent_embedding.model_name
        self.output_type = parent_embedding.output_type
        self.layer = parent_embedding.layer

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return (f'SlicedEmbedding(shape={self.shape}, layer={self.layer}, '
            f'class={self.object_class}, parent_class={self.parent_class})')


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
