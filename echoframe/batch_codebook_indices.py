'''Batch Wav2Vec2 codebook-index retrieval for echoframe.

SpidR codebook batching is intentionally not implemented here yet.
'''

from pathlib import Path

import echoframe
from to_vector import wav2vec2_codebook

from .utils_segment_features import (
    codebook_matrix_missing,
    segment_times,
    store_codebook_indices,
    store_codebook_matrix,
)


def compute_codebook_indices_batch(segments, model_name, collar=500,
    store=None, store_root='echoframe', gpu=False, tags=None,
    batch_size=None):
    '''Compute and store Wav2Vec2 codebook indices for segment objects.
    segments:             iterable of phraser segment objects
    model_name:           registered Wav2Vec2 model name
    collar:               context window in milliseconds
    store:                optional Store instance
    store_root:           root used when creating a Store lazily
    gpu:                  whether to run codebook extraction on GPU
    tags:                 optional tags stored on newly written metadata
    batch_size:           optional item count per batch

    SpidR codebook batching is not implemented in this Wav2Vec2-specific
    batch path.
    '''
    if store is None: store = echoframe.Store(store_root)
    missing = MissingIndices(segments, model_name, collar, store)
    print(missing)
    if not missing.missing:
        return
    model = store.load_codebook_model(model_name, gpu=gpu)
    if missing.matrix_missing:
        codebook_matrix = wav2vec2_codebook.load_codebook(model)
        first = missing.missing[0]
        store_codebook_matrix(codebook_matrix, first.segment.key, collar,
            model_name, store, tags)
    outputs = wav2vec2_codebook.iter_filename_batch_to_codebook_indices(
        missing.audio_filenames, starts=missing.starts, ends=missing.ends,
        model_pt=model, gpu=gpu, batch_size=batch_size)
    output_count = 0
    for indices, item in zip(outputs, missing.missing, strict=True):
        store_codebook_indices(indices, item.segment, collar, model_name,
            store, tags)
        output_count += 1
    print(f'codebook indices computed for {output_count} segments')


class SegmentRequest:
    '''One segment codebook-index request.'''
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
            'codebook_indices', model_name=self.model_name,
            phraser_key=self.segment.key, collar=self.collar)
        return self._echoframe_key


class MissingIndices:
    '''Batch wrapper for segment-level codebook-index requests.'''
    def __init__(self, segments, model_name, collar, store):
        self.segments = list(segments)
        self.model_name = model_name
        self.collar = collar
        self.store = store
        self._make_segment_requests()
        self._find_missing()

    def __repr__(self):
        m = f'MissingIndices({len(self.missing)}, '
        m += f'model={self.model_name}, collar={self.collar}ms)'
        m += f' with {len(self.found)} found in store'
        return m

    def __str__(self):
        m = f'MissingIndices model={self.model_name}\n'
        m += f'collar: {self.collar}ms\n'
        m += f'n segments: {len(self.segments)}\n'
        m += f'missing segments: {len(self.missing)}\n'
        m += f'found segments: {len(self.found)}\n'
        m += f'missing indices items: {self.indices_items_missing}\n'
        m += f'found indices items: {self.indices_items_found}\n'
        m += f'missing codebook matrix: {self.matrix_missing}'
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
    def matrix_missing(self):
        if hasattr(self, '_matrix_missing'): return self._matrix_missing
        self._matrix_missing = codebook_matrix_missing(self.store,
            self.model_name)
        return self._matrix_missing

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
    def indices_items_missing(self):
        if hasattr(self, '_items_missing'): return self._items_missing
        self._items_missing = len([x for x in self.metadatas if x is None])
        return self._items_missing

    @property
    def indices_items_found(self):
        if hasattr(self, '_items_found'): return self._items_found
        self._items_found = len([x for x in self.metadatas if x is not None])
        return self._items_found
