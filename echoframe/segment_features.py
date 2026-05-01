'''Single-segment feature retrieval orchestration for echoframe.'''

import echoframe

from .utils_segment_features import (
    codebook_indices_missing, codebook_matrix_missing,
    compute_codebook_indices as compute_codebook_indices_for_segment,
    compute_embeddings_for_segment,
    find_embedding_layers, normalise_layers, segment_times,
    store_codebook_indices_from_artifacts, store_codebook_matrix,
    store_embeddings_from_outputs,
)

def compute_embeddings(segment, layers, model_name, collar=500, store=None,
    store_root='echoframe', gpu=False, tags=None, verbose=False):
    '''Compute and store embeddings for one segment object.
    segment:              phraser segment object with key, timing, and audio
    layers:               layer index or iterable of layer indices
    model_name:           registered model name for store storage
    collar:               context window in milliseconds
    store:                optional Store instance
    store_root:           root used when creating a Store lazily
    gpu:                  whether to run vectorization on GPU
    tags:                 optional tags stored on newly written metadata
    '''
    layers_list = normalise_layers(layers)
    if store is None: store = echoframe.Store(store_root)
    phraser_key = segment.key
    found_layers, missing_layers = find_embedding_layers(store, phraser_key,
        collar, model_name, layers_list)
    if found_layers and verbose:
        print(f'embeddings found in store for layers {found_layers}')
    if missing_layers:
        model = store.load_model(model_name, gpu=gpu)
        outputs = compute_embeddings_for_segment(segment, collar, model, gpu)
        store_embeddings_from_outputs(outputs, segment, collar, missing_layers,
            model_name, store, tags)
        if verbose: print(f'embeddings computed for layers {missing_layers}')

def compute_codebook_indices(segment, model_name, collar=500, store=None,
    store_root='echoframe', gpu=False, tags=None, verbose=False):
    '''Compute and store codebook indices for one segment object.
    segment:      phraser segment object with key, timing, and audio
    model_name:   registered model name
    collar:       context window in milliseconds
    store:        optional Store instance
    store_root:   root used when creating a Store lazily
    gpu:          whether to run codebook extraction on GPU
    tags:         optional tags stored on newly written metadata
    '''
    if store is None: store = echoframe.Store(store_root)
    phraser_key = segment.key
    if not codebook_indices_missing(store, phraser_key, collar, model_name):
        if verbose: print('codebook indices found in store')
        return
    model = store.load_model(model_name, gpu=gpu)
    artifacts = compute_codebook_indices_for_segment(segment, collar, model,
        gpu)
    store_codebook_indices_from_artifacts(artifacts, segment, collar,
        model_name, store, tags)
    if codebook_matrix_missing(store, model_name):
        store_codebook_matrix(artifacts.codebook_matrix, phraser_key, collar,
            model_name, store, tags)
    if verbose: print('codebook indices computed and stored')


_segment_times = segment_times
