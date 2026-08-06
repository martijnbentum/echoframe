'''Batch CNN feature-extractor retrieval for echoframe.

SpidR CNN batching is intentionally not implemented here yet.
'''

from pathlib import Path

import to_vector

from .utils_segment_features import (
    StoreWriter,
    make_cnn_feature_item,
    segment_times,
)


def compute_cnn_features_batch(segments, model_name, store,
    collar=500, gpu=False, tags=None,
    batch_size=None, store_queue_size=4, verbose=True):
    '''Compute and store CNN feature-extractor output for segment objects.
    segments:             iterable of phraser segment objects
    model_name:           registered model name
    store:                echoframe Store used for model outputs
    collar:               context window in milliseconds
    gpu:                  whether to run CNN extraction on GPU
    tags:                 optional tags stored on newly written metadata
    batch_size:           optional item count per batch
    store_queue_size:     queued save chunks before compute waits

    SpidR CNN batching is not implemented in this batch path.
    '''
    if store is None: raise ValueError('store must be an echoframe Store')
    missing = MissingCnnFeatures(segments, model_name, collar, store)
    if verbose: print(missing)
    if not missing.missing:
        return
    missing_segments = []
    for item in missing.missing:
        missing_segments.append(item.segment)
    source_id = store.phraser_registry.segments_to_source_id(missing_segments)
    model = store.load_model(model_name, gpu=gpu)
    outputs = to_vector.iter_filename_batch_to_cnn(missing.audio_filenames,
        starts=missing.starts, ends=missing.ends, model=model, gpu=gpu,
        batch_size=batch_size)
    stored_count = 0
    with StoreWriter(store, max_queue_size=store_queue_size) as writer:
        for output, item in zip(outputs, missing.missing, strict=True):
            save_item = make_cnn_feature_item(output, item.segment, collar,
                model_name, store, tags, phraser_source_id=source_id)
            writer.submit([save_item])
            stored_count += 1
    if verbose: print(f'cnn features computed for {stored_count} segments')


class SegmentRequest:
    '''One segment CNN-feature request.'''
    def __init__(self, segment, collar, model_name, parent):
        self.segment = segment
        self.collar = collar
        self.model_name = model_name
        self.parent = parent
        self.audio_filename = self.segment.audio.filename
        self._set_times()

    def __repr__(self):
        f = Path(self.audio_filename).name
        m = f'SegmentRequest(filename={f}, collar={self.collar}, '
        m += f'model_name={self.model_name})'
        return m

    def __eq__(self, other):
        if not isinstance(other, SegmentRequest): return False
        if self.segment.key != other.segment.key: return False
        if self.collar != other.collar: return False
        if self.model_name != other.model_name: return False
        return True

    def _set_times(self):
        s, e, cs, ce = segment_times(self.segment, self.collar)
        self.start = s
        self.end = e
        self.collared_start = cs
        self.collared_end = ce

    @property
    def echoframe_key(self):
        if hasattr(self, '_echoframe_key'): return self._echoframe_key
        self._echoframe_key = self.parent.store.make_echoframe_key(
            'cnn', model_name=self.model_name,
            phraser_key=self.segment.key, collar=self.collar)
        return self._echoframe_key


class MissingCnnFeatures:
    '''Batch wrapper for segment-level CNN-feature requests.'''
    def __init__(self, segments, model_name, collar, store):
        self.segments = list(segments)
        self.model_name = model_name
        self.collar = collar
        self.store = store
        self._make_segment_requests()
        self._find_missing()

    def __repr__(self):
        m = f'MissingCnnFeatures({len(self.missing)}, '
        m += f'model={self.model_name}, collar={self.collar}ms)'
        m += f' with {len(self.found)} found in store'
        return m

    def __str__(self):
        m = f'MissingCnnFeatures model={self.model_name}\n'
        m += f'collar: {self.collar}ms\n'
        m += f'n segments: {len(self.segments)}\n'
        m += f'missing segments: {len(self.missing)}\n'
        m += f'found segments: {len(self.found)}\n'
        m += f'missing cnn items: {self.cnn_items_missing}\n'
        m += f'found cnn items: {self.cnn_items_found}'
        return m

    def _make_segment_requests(self):
        requests = []
        for segment in self.segments:
            request = SegmentRequest(segment, self.collar, self.model_name,
                self)
            requests.append(request)
        self.segment_requests = requests

    def _find_missing(self):
        missing, found = [], []
        for request, metadata in zip(self.segment_requests, self.metadatas,
            strict=True):
            if metadata is None: missing.append(request)
            else: found.append(request)
        self.missing = missing
        self.found = found

    @property
    def echoframe_keys(self):
        if hasattr(self, '_echoframe_keys'): return self._echoframe_keys
        self._echoframe_keys = [
            request.echoframe_key for request in self.segment_requests]
        return self._echoframe_keys

    @property
    def metadatas(self):
        if hasattr(self, '_metadatas'): return self._metadatas
        self._metadatas = self.store.load_many_metadata(self.echoframe_keys,
            keep_missing=True)
        return self._metadatas

    @property
    def audio_filenames(self):
        return [item.audio_filename for item in self.missing]

    @property
    def starts(self):
        return [item.collared_start for item in self.missing]

    @property
    def ends(self):
        return [item.collared_end for item in self.missing]

    @property
    def cnn_items_missing(self):
        if hasattr(self, '_items_missing'): return self._items_missing
        self._items_missing = len([x for x in self.metadatas if x is None])
        return self._items_missing

    @property
    def cnn_items_found(self):
        if hasattr(self, '_items_found'): return self._items_found
        self._items_found = len([x for x in self.metadatas if x is not None])
        return self._items_found
