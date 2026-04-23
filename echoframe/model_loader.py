'''Load registered models from echoframe model metadata.'''

from to_vector import load as to_vector_load

from .model_registry import ModelMetadata


def load_model(model_metadata, gpu=False):
    '''Load one model described by model metadata.
    model_metadata:  registered model metadata record
    gpu:             whether to move the model to GPU
    '''
    _validate_model_metadata(model_metadata)
    source = model_name_or_path(model_metadata)
    model = to_vector_load.load_model(source, gpu=gpu)
    return model


def load_model_for_attention_extraction(model_metadata, gpu=False):
    '''Load one model configured for attention extraction.
    model_metadata:  registered model metadata record
    gpu:             whether to move the model to GPU
    '''
    _validate_model_metadata(model_metadata)
    source = model_name_or_path(model_metadata)
    model = to_vector_load.load_model_for_attention_extraction(
        source, gpu=gpu)
    return model


def move_model_to_gpu(model):
    return to_vector_load.move_model_to_gpu(model)

def move_model_to_cpu(model):
    return to_vector_load.move_model_to_cpu(model)

def remove_model_from_gpu(model):
    return to_vector_load.move_model_to_cpu(model)

def model_is_on_gpu(model):
    '''Return whether a loaded model is on GPU.
    model:  loaded model instance
    '''
    return to_vector_load.model_is_on_gpu(model)


def model_name_or_path(model_metadata):
    '''Return the preferred loader source from model metadata.'''
    _validate_model_metadata(model_metadata)
    if model_metadata.local_path: return model_metadata.local_path
    if model_metadata.huggingface_id: return model_metadata.huggingface_id
    m = f'model metadata has no loadable source: {model_metadata.model_name!r}'
    raise ValueError(m)


def _validate_model_metadata(model_metadata):
    '''Reject unsupported metadata input early.'''
    if isinstance(model_metadata, ModelMetadata): return
    m = 'model_metadata must be a ModelMetadata, '
    m += f'got {type(model_metadata)}'
    raise ValueError(m)
