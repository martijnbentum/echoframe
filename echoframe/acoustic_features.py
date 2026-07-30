'''Store precomputed frame-aligned acoustic features (mfcc, pitch, ...).'''

import numpy as np
from progressbar import progressbar

from .metadata import EchoframeMetadata


def store_mfcc(segment, store, tags=None, verbose=False):
    '''Store the MFCC matrix for one segment, computing it if needed.
    segment:  phraser segment object with key, audio, and an mfcc property
    store:    echoframe Store used for model outputs
    tags:     optional tags stored on newly written metadata
    '''
    if store is None: raise ValueError('store must be an echoframe Store')
    echoframe_key = _mfcc_echoframe_key(segment, store)
    if store.load_metadata(echoframe_key) is not None:
        if verbose: print('mfcc found in store')
        return
    _store_acoustic_feature(segment, 'mfcc', segment.mfcc, store,
        tags=tags, verbose=verbose)


def store_mfcc_batch(segments, store, tags=None, verbose=True):
    '''Store MFCC matrices for multiple segments.

    Computes each missing segment's mfcc through its mfcc property, then
    writes all of them in one batched store call.

    segments:  iterable of phraser segment objects with an mfcc property;
               call phraser.audio.batch.mfcc_batch(segments) first for
               efficient shared computation across a large batch
    store:     echoframe Store used for model outputs
    tags:      optional tags stored on newly written metadata
    '''
    if store is None: raise ValueError('store must be an echoframe Store')
    segments = list(segments)
    keys = []
    for segment in segments:
        key = _mfcc_echoframe_key(segment, store)
        keys.append(key)
    metadatas = store.load_many_metadata(keys, keep_missing=True)
    missing = []
    for segment, key, metadata in zip(segments, keys, metadatas, strict=True):
        if metadata is None: missing.append((segment, key))
    if not missing:
        if verbose: print(f'mfcc stored for 0 of {len(segments)} segments')
        return
    missing_segments = []
    for segment, key in missing:
        missing_segments.append(segment)
    source_id = store.phraser_registry.segments_to_source_id(missing_segments)
    items = []
    for segment, key in progressbar(missing):
        item = make_acoustic_feature_item(key, 'mfcc', segment.mfcc, store,
            tags=tags, phraser_source_id=source_id)
        items.append(item)
    store.save_many(items)
    if verbose:
        m = f'mfcc stored for {len(items)} of {len(segments)} segments'
        print(m)


def make_acoustic_feature_item(echoframe_key, feature_name, feature_matrix,
    store, tags=None, phraser_source_id=None):
    '''Build one save item for a precomputed acoustic feature matrix.
    echoframe_key:      canonical acoustic feature key
    feature_name:       acoustic feature identifier
    feature_matrix:     two-dimensional frame-aligned payload
    store:              echoframe Store used for the output
    tags:               optional tags for the metadata
    phraser_source_id:  source identifier for the linked phraser segment
    '''
    feature_matrix = np.asarray(feature_matrix)
    if feature_matrix.ndim != 2:
        m = f'feature_matrix must be 2D, got {feature_matrix.ndim}'
        raise ValueError(m)
    metadata = EchoframeMetadata(echoframe_key=echoframe_key, store=store,
        feature_name=feature_name, tags=tags,
        phraser_source_id=phraser_source_id)
    item = {'echoframe_key': metadata.echoframe_key, 'metadata': metadata}
    item['data'] = feature_matrix
    return item


def _mfcc_echoframe_key(segment, store):
    return store.make_echoframe_key('acoustic_feature',
        feature_name='mfcc', phraser_key=segment.key)


def _store_acoustic_feature(segment, feature_name, feature_matrix, store,
    tags=None, verbose=False):
    '''Store one precomputed acoustic feature matrix for one segment.
    segment:         phraser segment object with key and audio
    feature_name:    acoustic feature name ('mfcc', 'pitch', ...)
    feature_matrix:  (frames, dim) array already matched to the segment
    store:           echoframe Store used for model outputs
    tags:            optional tags stored on newly written metadata
    '''
    if store is None: raise ValueError('store must be an echoframe Store')
    phraser_key = segment.key
    echoframe_key = store.make_echoframe_key('acoustic_feature',
        feature_name=feature_name, phraser_key=phraser_key)
    if store.load_metadata(echoframe_key) is not None:
        if verbose: print(f'{feature_name} found in store')
        return
    source_id = store.phraser_registry.segment_to_source_id(segment)
    item = make_acoustic_feature_item(echoframe_key, feature_name,
        feature_matrix, store, tags=tags, phraser_source_id=source_id)
    store.save(item['echoframe_key'], item['metadata'], item['data'])
    if verbose: print(f'{feature_name} stored')
