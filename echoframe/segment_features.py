'''Single-segment feature retrieval orchestration for echoframe.'''

from .utils_segment_features import (
    cnn_missing,
    codebook_indices_missing, codebook_matrix_missing,
    compute_codebook_indices as compute_codebook_indices_for_segment,
    compute_embeddings_for_segment,
    find_embedding_layers, make_cnn_feature_item, make_embedding_items,
    normalise_layers, reject_spidr_cnn_request, segment_times,
    split_requested_layers,
    store_codebook_indices_from_artifacts, store_codebook_matrix,
)

def compute_embeddings(segment, layers, model_name, store, collar=500,
    gpu=False, tags=None, verbose=False):
    '''Compute and store embeddings for one segment object.
    segment:              phraser segment object with key, timing, and audio
    layers:               layer index or iterable containing layer indices
                          and optionally 'cnn'
    model_name:           registered model name for store storage
    store:                echoframe Store used for model outputs
    collar:               context window in milliseconds
    gpu:                  whether to run vectorization on GPU
    tags:                 optional tags stored on newly written metadata
    '''
    if store is None: raise ValueError('store must be an echoframe Store')
    layers_list = normalise_layers(layers)
    hidden_state_layers, cnn_requested = split_requested_layers(layers_list)
    if cnn_requested:
        reject_spidr_cnn_request(store, model_name)
    phraser_key = segment.key
    found_layers, missing_layers = find_embedding_layers(store, phraser_key,
        collar, model_name, hidden_state_layers)
    missing_cnn = False
    if cnn_requested:
        missing_cnn = cnn_missing(store, phraser_key, collar, model_name)
    if found_layers and verbose:
        print(f'embeddings found in store for layers {found_layers}')
    if cnn_requested and not missing_cnn and verbose:
        print('cnn features found in store')
    if not missing_layers and not missing_cnn: return
    source_id = store.phraser_registry.segment_to_source_id(segment)
    model = store.load_model(model_name, gpu=gpu)
    if cnn_requested:
        reject_spidr_cnn_request(store, model_name, model=model)
    outputs = compute_embeddings_for_segment(segment, collar, model, gpu)
    items = []
    if missing_layers:
        items.extend(make_embedding_items(outputs, segment, collar,
            missing_layers, model_name, store, tags,
            phraser_source_id=source_id))
    if missing_cnn:
        items.append(make_cnn_feature_item(outputs, segment, collar,
            model_name, store, tags, phraser_source_id=source_id))
    store.save_many(items)
    if missing_layers and verbose:
        print(f'embeddings computed for layers {missing_layers}')
    if missing_cnn and verbose: print('cnn features computed and stored')

def compute_codebook_indices(segment, model_name, store, collar=500,
    gpu=False, tags=None, verbose=False):
    '''Compute and store codebook indices for one segment object.
    segment:      phraser segment object with key, timing, and audio
    model_name:   registered model name
    store:        echoframe Store used for model outputs
    collar:       context window in milliseconds
    gpu:          whether to run codebook extraction on GPU
    tags:         optional tags stored on newly written metadata
    '''
    if store is None: raise ValueError('store must be an echoframe Store')
    phraser_key = segment.key
    if not codebook_indices_missing(store, phraser_key, collar, model_name):
        if verbose: print('codebook indices found in store')
        return
    source_id = store.phraser_registry.segment_to_source_id(segment)
    model = store.load_model(model_name, gpu=gpu)
    artifacts = compute_codebook_indices_for_segment(segment, collar, model,
        gpu)
    store_codebook_indices_from_artifacts(artifacts, segment, collar,
        model_name, store, tags, phraser_source_id=source_id)
    if codebook_matrix_missing(store, model_name):
        store_codebook_matrix(artifacts.codebook_matrix, model_name, store,
            tags)
    if verbose: print('codebook indices computed and stored')

_segment_times = segment_times
