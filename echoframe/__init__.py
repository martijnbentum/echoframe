'''Disk-backed storage for model outputs linked to phraser keys.'''

from .acoustic_features import store_mfcc, store_mfcc_batch
from .cnn_features import CNNFeature, CNNFeatures, SlicedCNNFeature
from .codebooks import Codevector, Codevectors
from .embeddings import Embedding, Embeddings, SlicedEmbedding
from .metadata import EchoframeMetadata, OUTPUT_TYPES
from .metadata import STABLE_METADATA_FIELDS, filter_metadata
from .model_registry import ModelMetadata
from .store import Store


def compute_cnn(segment, model_name, store, collar=500, gpu=False, tags=None,
    overwrite=False, verbose=False):
    '''Compute and store CNN output for one phraser segment.'''
    from .segment_features import compute_cnn as _compute_cnn
    return _compute_cnn(segment, model_name, store, collar=collar, gpu=gpu,
        tags=tags, overwrite=overwrite, verbose=verbose)


__all__ = [
    'Embedding',
    'Embeddings',
    'SlicedEmbedding',
    'CNNFeature',
    'CNNFeatures',
    'SlicedCNNFeature',
    'Codevector',
    'Codevectors',
    'compute_cnn',
    'store_mfcc',
    'store_mfcc_batch',
    'Store',
    'OUTPUT_TYPES',
    'EchoframeMetadata',
    'ModelMetadata',
    'STABLE_METADATA_FIELDS',
    'filter_metadata',
]
