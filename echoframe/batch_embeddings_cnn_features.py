'''Batch retrieval of hidden_state and cnn output together for echoframe.

Segments missing hidden_state always go through the full forward pass, which
produces cnn's extract_features as a byproduct; segments missing only cnn go
through the cheap cnn-only path. This avoids the redundant cnn compute that
calling compute_embeddings_batch and compute_cnn_features_batch separately
would incur for segments needing both.
'''
import to_vector

from .batch_cnn_features import MissingCnnFeatures
from .batch_segment_features import MissingSegments
from .utils_segment_features import (
    StoreWriter,
    make_cnn_feature_item, make_embedding_items,
    normalise_layers,
)

_DEFAULT_WRITE_SEGMENT_COUNT = 32


def compute_embeddings_and_cnn_features_batch(segments, layers, model_name,
    store, collar=500, gpu=False, tags=None,
    batch_size=None, store_queue_size=4, verbose=True):
    '''Compute and store hidden_state and cnn output for multiple segments.
    segments:             iterable of phraser segment objects
    layers:               layer index or iterable of layer indices
    model_name:           registered model name for store storage
    store:                echoframe Store used for model outputs
    collar:               context window in milliseconds
    gpu:                  whether to run vectorization on GPU
    tags:                 optional tags stored on newly written metadata
    batch_size:           optional segment count per inference batch; storage
                          writes use the same count
    store_queue_size:     queued save chunks before compute waits
    verbose:              whether to print batch progress

    Segments missing hidden_state (with or without cnn) go through one full
    forward pass; segments missing only cnn go through the cheap cnn-only
    path. Storage writes group 32 segment results when batch_size is None.
    '''
    if store is None: raise ValueError('store must be an echoframe Store')
    layers_list = normalise_layers(layers)
    hidden_missing = MissingSegments(segments, layers_list, model_name,
        collar, store)
    cnn_missing = MissingCnnFeatures(segments, model_name, collar, store)
    if verbose:
        print(hidden_missing)
        print(cnn_missing)
    if not hidden_missing.missing and not cnn_missing.missing: return
    hidden_keys = {item.segment.key for item in hidden_missing.missing}
    cnn_lookup = {item.segment.key: item for item in cnn_missing.missing}
    cnn_only = []
    for key, item in cnn_lookup.items():
        if key not in hidden_keys: cnn_only.append(item)

    touched_segments = [item.segment for item in hidden_missing.missing]
    touched_segments.extend(item.segment for item in cnn_only)
    source_id = store.phraser_registry.segments_to_source_id(touched_segments)
    model = store.load_model(model_name, gpu=gpu)

    write_segment_count = _DEFAULT_WRITE_SEGMENT_COUNT
    if batch_size is not None: write_segment_count = int(batch_size)
    if write_segment_count <= 0:
        raise ValueError('batch_size must be greater than zero')

    hidden_count = 0
    cnn_count = 0
    with StoreWriter(store, max_queue_size=store_queue_size) as writer:
        if hidden_missing.missing:
            hidden_count, cnn_count = _run_full_forward(hidden_missing,
                cnn_lookup, model, model_name, store, collar, tags,
                source_id, gpu, batch_size, write_segment_count, writer)
        if cnn_only:
            cnn_count += _run_cnn_only(cnn_only, model, model_name, store,
                collar, tags, source_id, gpu, batch_size, writer)
    if verbose:
        print(f'embeddings computed for {hidden_count} segments, '
            f'cnn features computed for {cnn_count} segments')


def _run_full_forward(hidden_missing, cnn_lookup, model, model_name, store,
    collar, tags, source_id, gpu, batch_size, write_segment_count, writer):
    outputs = to_vector.iter_filename_batch_to_vector(
        hidden_missing.audio_filenames, starts=hidden_missing.starts,
        ends=hidden_missing.ends, model=model, gpu=gpu,
        numpify_output=True, batch_size=batch_size)
    hidden_count, cnn_count = 0, 0
    pending_items, pending_segments = [], 0
    try:
        for output, item in zip(outputs, hidden_missing.missing, strict=True):
            save_items = make_embedding_items(output, item.segment, collar,
                item.missing_layers, model_name, store, tags,
                phraser_source_id=source_id)
            if item.segment.key in cnn_lookup:
                cnn_item = make_cnn_feature_item(output, item.segment,
                    collar, model_name, store, tags,
                    phraser_source_id=source_id)
                save_items.append(cnn_item)
                cnn_count += 1
            pending_items.extend(save_items)
            pending_segments += 1
            hidden_count += 1
            if pending_segments == write_segment_count:
                chunk = pending_items
                pending_items = []
                pending_segments = 0
                writer.submit(chunk)
    finally:
        if pending_items: writer.submit(pending_items)
    return hidden_count, cnn_count


def _run_cnn_only(cnn_only, model, model_name, store, collar, tags,
    source_id, gpu, batch_size, writer):
    audio_filenames = [item.audio_filename for item in cnn_only]
    starts = [item.collared_start for item in cnn_only]
    ends = [item.collared_end for item in cnn_only]
    outputs = to_vector.iter_filename_batch_to_cnn(audio_filenames,
        starts=starts, ends=ends, model=model, gpu=gpu,
        batch_size=batch_size)
    cnn_count = 0
    for output, item in zip(outputs, cnn_only, strict=True):
        save_item = make_cnn_feature_item(output, item.segment, collar,
            model_name, store, tags, phraser_source_id=source_id)
        writer.submit([save_item])
        cnn_count += 1
    return cnn_count
