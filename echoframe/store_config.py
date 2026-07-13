'''Shared config.json owner for the echoframe store.'''

import json
import os
from pathlib import Path

from .model_registry import ModelMetadata


class StoreConfig:
    '''Read and write the shared store config.json.

    Holds two sections: registered models and phraser store paths
    (source_id -> path). Each read returns a fresh object graph from disk and
    each write serializes the whole graph, so independent registries can
    update their own section without clobbering the other.
    '''

    def __init__(self, config_path):
        self.config_path = Path(config_path)

    def __repr__(self):
        return f'StoreConfig(config_path={self.config_path})'

    def read(self):
        '''Return the validated in-memory config object graph.'''
        if not self.config_path.exists():
            return default_config()
        data = json.loads(self.config_path.read_text())
        return config_from_dict(data)

    def write(self, config):
        '''Write one validated in-memory config object graph.'''
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config_to_dict(config), indent=2, sort_keys=True)
        payload += '\n'
        tmp_name = f'{self.config_path.name}.{os.getpid()}.tmp'
        tmp_path = self.config_path.with_name(tmp_name)
        tmp_path.write_text(payload)
        tmp_path.replace(self.config_path)

    def read_dict(self):
        '''Return the raw serialized config dictionary.'''
        return config_to_dict(self.read())


def default_config():
    return {'models': {}, 'phraser_sources': {}}


def config_from_dict(data):
    '''Build one validated config mapping from serialized JSON data.'''
    if not isinstance(data, dict):
        raise ValueError('config.json must contain a JSON object')
    config = default_config()
    config['models'] = _models_from_dict(data.get('models', {}))
    config['phraser_sources'] = _phraser_sources_from_dict(
        data.get('phraser_sources', {}))
    return config


def config_to_dict(config):
    '''Serialize one validated config mapping for JSON output.'''
    models = {}
    for model_name, metadata in config['models'].items():
        models[model_name] = metadata.to_dict()
    phraser_sources = dict(config.get('phraser_sources', {}))
    return {'models': models, 'phraser_sources': phraser_sources}


def _models_from_dict(raw_models):
    if not isinstance(raw_models, dict):
        raise ValueError('config.json models must be a JSON object')
    models = {}
    for model_name, record in raw_models.items():
        if not isinstance(record, dict):
            raise ValueError('config.json model records must be JSON objects')
        record = dict(record)
        record['model_name'] = model_name
        models[model_name] = ModelMetadata.from_dict(record)
    return models


def _phraser_sources_from_dict(raw_sources):
    if not isinstance(raw_sources, dict):
        raise ValueError('config.json phraser_sources must be a JSON object')
    sources = {}
    for source_id, path in raw_sources.items():
        if not isinstance(path, str) or not path.strip():
            message = 'phraser_sources values must be non-empty path strings'
            raise ValueError(message)
        sources[source_id] = path
    return sources
