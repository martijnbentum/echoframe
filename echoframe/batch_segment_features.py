'''Batch feature retrieval orchestration for echoframe.'''
from pathlib import Path

import echoframe
import to_vector

from .utils_segment_features import (
    normalise_layers, segment_times,
    store_embeddings_from_outputs,
)


def compute_embeddings_batch(segments, layers, model_name, collar=500,
    store=None, store_root='echoframe', gpu=False, tags=None,
    batch_size=None):
    '''Compute and store embeddings for multiple segment objects.
    segments:             iterable of phraser segment objects
    layers:               layer index or iterable of layer indices
    model_name:           registered model name for store storage
    collar:               context window in milliseconds
    store:                optional Store instance
    store_root:           root used when creating a Store lazily
    gpu:                  whether to run vectorization on GPU
    tags:                 optional tags stored on newly written metadata
    batch_size:           optional item count per batch
    '''
    layers_list = normalise_layers(layers)
    if store is None: store = echoframe.Store(store_root)
    missing = MissingSegments(segments, layers_list, model_name, collar, store)
    if not missing.missing:
        print(missing)
        return
    model = store.load_model(model_name, gpu=gpu)
    outputs = to_vector.iter_filename_batch_to_vector(missing.audio_filenames,
        starts=missing.starts, ends=missing.ends, model=model, gpu=gpu,
        numpify_output=True, batch_size=batch_size)
    output_count = 0
    for output, item in zip(outputs, missing.missing, strict=True):
        store_embeddings_from_outputs(output, item.segment, collar,
            item.missing_layers, model_name, store, tags)
        output_count += 1
    print(f'embeddings computed for {output_count} segments')


class SegmentRequest:
    '''One segment-layer embedding request.'''
    def __init__(self, segment, layers, collar, model_name, parent):
        self.segment = segment
        self.layers = list(layers)
        self.collar = collar
        self.model_name = model_name
        self.parent = parent
        self.audio_filename = self.segment.audio.filename
        self._set_times()

    def __repr__(self):
        f = Path(self.audio_filename).name
        m = f'SegmentRequest(filename={f} layers={self.layers}, '
        m += f'collar={self.collar}, model_name={self.model_name})'
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
    def echoframe_keys(self):
        if hasattr(self, '_echoframe_keys'): return self._echoframe_keys
        keys = []
        for layer in self.layers:
            echoframe_key = self.parent.store.make_echoframe_key('hidden_state',
                model_name=self.model_name, phraser_key=self.segment.key,
                layer=layer, collar=self.collar)
            keys.append(echoframe_key)
        self._echoframe_keys = keys
        return self._echoframe_keys

    @property
    def echoframe_key_to_self_dict(self):
        return {key: self for key in self.echoframe_keys}

    @property
    def metadatas(self):
        if hasattr(self, '_metadatas'): return self._metadatas
        d = self.parent.echoframe_key_to_metadata_dict
        md = [d[key] for key in self.echoframe_keys]
        self._metadatas = md
        return self._metadatas

    @property
    def missing_layers(self):
        if hasattr(self, '_missing_layers'): return self._missing_layers
        missing_layers = []
        for md, layer in zip(self.metadatas, self.layers, strict=True):
            if md is None: missing_layers.append(layer)
        self._missing_layers = missing_layers
        return self._missing_layers

class MissingSegments:
    '''Batch wrapper for segment-layer embedding requests.'''
    def __init__(self, segments, layers, model_name, collar, store): 
        self.segments = list(segments)
        self.layers = list(layers)
        self.model_name = model_name
        self.collar = collar
        self.store = store
        self._make_segment_requests(self.segments, self.layers, model_name, 
            collar, self)
        self._find_missing()

    def __repr__(self):
        m = f'MissingSegments(model={self.model_name} layers={self.layers}, '
        m += f'collar={self.collar}ms)'
        return m

    def __str__(self):
        m = f'MissingSegments model={self.model_name} layers={self.layers}\n'
        m += f'collar: {self.collar}ms\n'
        m += f'n segments: {len(self.segments)}\n'
        m += f'missing layer items: {self.layer_items_missing}\n'
        m += f'found layer items: {self.layer_items_found}'
        return m

    def _make_segment_requests(self, segments, layers, model_name, collar, 
        store):
        requests = []
        for segment in segments:
            sr = SegmentRequest(segment, layers, collar, model_name,store)
            requests.append(sr)
        self.segment_requests = requests

    def _find_missing(self):
        missing, found = [], []
        seen_missing, seen_found = set(), set()
        key_to_request = self.echoframe_key_to_segment_request_dict
        for echoframe_key, metadata in zip(self.echoframe_keys, self.metadatas,
            strict=True):
            item = key_to_request[echoframe_key]
            if metadata is None:
                if item.segment.key not in seen_missing:
                    seen_missing.add(item.segment.key)
                    missing.append(item)
            else:
                if item.segment.key not in seen_found:
                    seen_found.add(item.segment.key)
                    found.append(item)
        self.missing = missing
        self.found = found

    @property
    def echoframe_keys(self):
        keys = []
        for item in self.segment_requests:
            keys.extend(item.echoframe_keys)
        return keys

    @property
    def echoframe_key_to_segment_request_dict(self):
        if hasattr(self, '_echoframe_key_to_segment_request_dict'):
            return self._echoframe_key_to_segment_request_dict
        d = {}
        for item in self.segment_requests:
            d.update(item.echoframe_key_to_self_dict)
        self._echoframe_key_to_segment_request_dict = d
        return self._echoframe_key_to_segment_request_dict

    @property
    def echoframe_key_to_metadata_dict(self):
        if hasattr(self, '_echoframe_key_to_metadata_dict'):
            return self._echoframe_key_to_metadata_dict
        d = {}
        for md, ef in zip(self.metadatas, self.echoframe_keys, strict=True): 
            d[ef] = md
        self._echoframe_key_to_metadata_dict = d
        return self._echoframe_key_to_metadata_dict

    @property
    def metadatas(self):
        if hasattr(self, '_metadatas'): return self._metadatas
        x = self.store.load_many_metadata(self.echoframe_keys, keep_missing=True)
        self._metadatas = x
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
    def layer_items_missing(self):
        if hasattr(self, '_items_missing'): return self._items_missing
        self._items_missing = len([x for x in self.metadatas if x is None])
        return self._items_missing

    @property
    def layer_items_found(self):
        if hasattr(self, '_items_found'): return self._items_found
        self._items_found = len([x for x in self.metadatas if x is not None])
        return self._items_found
