'''Stored CNN feature-extractor containers.'''

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CNNFeature:
    '''One stored CNN feature-extractor payload with its metadata.'''

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

    def _load_missing(self):
        if self.metadata is None:
            self.metadata = self.store.load_metadata(self.echoframe_key)
        if self.metadata is None:
            m = f'no metadata found for echoframe_key {self.echoframe_key!r}'
            raise ValueError(m)
        if self.data is None:
            self.data = self.store.metadata_to_payload(self.metadata)
        if self.data is None:
            m = f'no cnn feature data found for echoframe_key '
            m += f'{self.echoframe_key!r}'
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

    def _object_class_repr(self):
        '''Return object_class for display; '?' when phraser is unreachable.'''
        try: return self.object_class
        except Exception: return '?'

    def __repr__(self):
        text = (f'CNNFeature(shape={self.shape}, model={self.model_name}, '
            f'class={self._object_class_repr()})')
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
        '''Return payload rows overlapping the absolute [start, end] seconds.

        The frame grid is anchored at the collared segment start. For
        segments within one collar of the audio file start the collar is
        clamped to 0, so selected rows can be off by up to one frame.
        '''
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

    def sub_feature(self, phraser_object, aggregate=None,
        percentage_overlap=None, stride=0.02, field=0.025):
        '''Return a SlicedCNNFeature for a descendant phraser object.
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
        return SlicedCNNFeature(self, phraser_object, data, rows)

    def sub_features(self, object_class, aggregate=None,
        percentage_overlap=None, stride=0.02, field=0.025):
        '''Return a list of SlicedCNNFeatures for all descendant objects of
        object_class.
        object_class:  'word', 'syllable', or 'phone' (singular or plural,
                       case-insensitive); must be a descendant segment type of
                       this feature's phraser_object.
        aggregate:     None -> 2D rows; 'mean' or 'middle' -> 1D vector each
        '''
        accessor = object_class.lower()
        if not accessor.endswith('s'): accessor += 's'
        segments = getattr(self.phraser_object, accessor, None)
        if segments is None:
            m = f'{self.object_class} has no descendant {object_class!r}'
            raise ValueError(m)
        sub_features = []
        for segment in segments:
            sub = self.sub_feature(segment, aggregate=aggregate,
                percentage_overlap=percentage_overlap, stride=stride,
                field=field)
            sub_features.append(sub)
        return sub_features

    def _validate(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError('data must be a numpy array')
        if self.data.ndim not in (1, 2):
            raise ValueError(f'data must be ndim 1 or 2, got {self.data.ndim}')
        if self.metadata.output_type != 'cnn':
            raise ValueError('metadata.output_type must be cnn')
        if not hasattr(self.metadata, 'model_name'):
            raise ValueError('cnn feature metadata must have model_name')
        if self.metadata.echoframe_key != self.echoframe_key:
            message = 'metadata.echoframe_key did not match echoframe_key'
            raise ValueError(message)


class SlicedCNNFeature:
    '''A view of a stored CNNFeature sliced to a descendant phraser object.

    Produced by CNNFeature.sub_feature(...). Holds the rows of the parent
    (stored) CNN feature that overlap a descendant phraser object, and keeps
    a reference to the parent CNNFeature it was derived from.
    '''

    def __init__(self, parent_feature, phraser_object, data, rows):
        self.parent_feature = parent_feature
        self.parent_phraser_key = parent_feature.phraser_key
        self.parent_collar = parent_feature.metadata.collar
        self.parent_class = parent_feature.metadata.phraser_object.object_type
        self.phraser_object = phraser_object
        self.object_class = phraser_object.object_type
        self.data = data
        self.rows = rows
        self.model_name = parent_feature.model_name
        self.output_type = parent_feature.output_type

    @property
    def shape(self):
        return self.data.shape

    def _rows_repr(self):
        if not self.rows:
            return '[]'
        if len(self.rows) == 1:
            return f'[{self.rows[0]}]'
        lo, hi = self.rows[0], self.rows[-1]
        contiguous = self.rows == list(range(lo, hi + 1))
        if contiguous:
            return f'[{lo}..{hi}]'
        return f'[{lo}..{hi}] ({len(self.rows)} rows)'

    def __repr__(self):
        return (f'SlicedCNNFeature(shape={self.shape}, '
            f'class={self.object_class}, parent_class={self.parent_class}, '
            f'rows={self._rows_repr()})')


class CNNFeatures:
    '''A validated collection of stored CNNFeature objects.

    Validation enforces unique phraser keys and matching model_name and
    output_type. Mixed collars are allowed on purpose.
    '''

    def __init__(self, cnn_features, store):
        self._check_cnn_features(cnn_features)
        self.store = store
        self.cnn_features = tuple(cnn_features)
        self.count = len(self.cnn_features)
        self.phraser_keys = tuple(x.phraser_key for x in self.cnn_features)
        self.metadatas = tuple(x.metadata for x in self.cnn_features)
        self.model_name = self.cnn_features[0].model_name
        self.output_type = self.cnn_features[0].output_type
        self._validate()

    @classmethod
    def from_echoframe_keys(cls, store, keys):
        '''Build CNNFeatures with a single batched payload read.

        Groups payloads by shard (one HDF5 open per shard) instead of opening
        a shard per feature.
        '''
        keys = list(keys)
        metadatas = store.load_many_metadata(keys, keep_missing=True)
        payloads = store.metadatas_to_payloads(metadatas)

        cnn_features = []
        skipped_count = 0
        items = zip(keys, metadatas, payloads, strict=True)
        for key, metadata, data in items:
            if metadata is None or data is None:
                skipped_count += 1
                logger.warning(
                    f'skipping echoframe_key {key!r}: no metadata or payload')
                continue
            try: cnn_feature = CNNFeature(key, store, metadata=metadata,
                data=data)
            except ValueError as e:
                skipped_count += 1
                logger.warning(f'skipping echoframe_key {key!r}: {e}')
                continue
            cnn_features.append(cnn_feature)
        if not cnn_features:
            message = f'no cnn features were loaded skipped keys {skipped_count}'
            raise ValueError(message)
        return cls(cnn_features, store)

    @property
    def data(self):
        return self.to_numpy()

    def __repr__(self):
        text = f'CNNFeatures(# {self.count})'
        return text

    def to_numpy(self):
        shapes = [cnn_feature.data.shape for cnn_feature in self.cnn_features]
        reference = shapes[0]
        if any(shape != reference for shape in shapes[1:]):
            message = 'CNNFeatures.to_numpy() requires identical feature shapes'
            raise NotImplementedError(message)
        return np.stack(
            [cnn_feature.data for cnn_feature in self.cnn_features], axis=0)

    def _validate(self):
        seen = set()
        for phraser_key in self.phraser_keys:
            if phraser_key in seen:
                raise ValueError(f'duplicate phraser_key: {phraser_key}')
            seen.add(phraser_key)
        for cnn_feature in self.cnn_features[1:]:
            if cnn_feature.model_name != self.model_name:
                raise ValueError('cnn_feature model_name mismatch')
            if cnn_feature.output_type != self.output_type:
                raise ValueError('cnn_feature output_type mismatch')

    def _check_cnn_features(self, cnn_features):
        if not isinstance(cnn_features, (list, tuple)):
            raise ValueError('cnn_features must be a list or tuple of '
                'CNNFeature')
        if not cnn_features:
            raise ValueError('cnn_features must contain at least one '
                'CNNFeature')
        for cnn_feature in cnn_features:
            if not isinstance(cnn_feature, CNNFeature):
                raise ValueError('cnn_features must contain only CNNFeature')
