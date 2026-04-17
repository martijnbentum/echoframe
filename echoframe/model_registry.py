'''Store-owned model registry backed by config.json.'''

import json
from pathlib import Path

from . import util_formatting


class ModelRegistry:
    '''Persist and query model metadata in a store config file.'''

    def __init__(self, config_path):
        self.config_path = Path(config_path)

    def __repr__(self):
        m = f'ModelRegistry(config_path={self.config_path})'
        return m

    def __str__(self):
        try:
            summary = self.registry_summary()
            return util_formatting.format_model_registry_str(summary)
        except Exception:
            return self.__repr__()

    def register_model(self, model_name, local_path=None, huggingface_id=None,
        language=None, size=None, architecture=None):
        '''Register one model and persist it in config.json.'''
        metadata = ModelMetadata(model_name, model_id=None,
            local_path=local_path, huggingface_id=huggingface_id,
            language=language, size=size, architecture=architecture)
        config = self.read_config()
        if check_model_name_conflict(config, metadata):
            message = f'model_name already registered: {model_name!r}'
            raise ValueError(message)
        metadata.model_id = _next_model_id(config['models'].values())
        config['models'][model_name] = metadata
        self.write_config(config)
        return metadata

    def registry_summary(self):
        '''Return compact summary stats for this model registry.'''
        return util_formatting.build_model_registry_summary(self)

    @property
    def model_metadatas(self):
        '''Return all registered model metadata objects.'''
        config = self.read_config()
        return list(config['models'].values())

    def load_model_metadata(self, model_name):
        '''Return the model metadata object for model_name, or None.'''
        config = self.read_config()
        return config['models'].get(model_name)

    def register_models_from_file(self, path):
        '''Import model definitions from a JSON file into config.json.'''
        metadata_list = load_model_seed_file(path)
        config = self.read_config()
        conflicts = check_model_names_conflict(config, metadata_list)
        if conflicts:
            names = ', '.join(repr(metadata.model_name)
                for metadata in conflicts)
            message = f'model names already registered in store: {names}'
            raise ValueError(message)

        next_id = _next_model_id(config['models'].values())
        stored = []
        for metadata in metadata_list:
            metadata.model_id = next_id
            config['models'][metadata.model_name] = metadata
            stored.append(metadata)
            next_id += 1
        self.write_config(config)
        return stored

    def read_config_dict(self):
        '''Return the raw serialized config dictionary.'''
        if not self.config_path.exists():
            return config_to_dict(_default_config())
        data = json.loads(self.config_path.read_text())
        config = config_from_dict(data)
        return config_to_dict(config)

    def read_config(self):
        '''Return the validated in-memory config object graph.'''
        if not self.config_path.exists():
            return _default_config()
        data = json.loads(self.config_path.read_text())
        return config_from_dict(data)

    def write_config(self, config):
        '''Write one validated in-memory config object graph.'''
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config_to_dict(config), indent=2, sort_keys=True)
        payload += '\n'
        tmp_path = self.config_path.with_suffix('.tmp')
        tmp_path.write_text(payload)
        tmp_path.replace(self.config_path)


class ModelMetadata:
    '''Small validated model metadata record.'''

    def __init__(self, model_name, model_id=None, local_path=None,
        huggingface_id=None, language=None, size=None, architecture=None):
        self.model_name = model_name
        self.model_id = model_id
        self.local_path = local_path
        self.huggingface_id = huggingface_id
        self.language = language
        self.size = size
        self.architecture = architecture
        self._validate()

    def __repr__(self):
        m = f'ModelMetadata(model_name={self.model_name}, '
        m += f'model_id={self.model_id}, '
        if self.local_path is not None:
            p = Path(self.local_path)
            m += f'local_path={p.name}, '
        if self.huggingface_id is not None:
            m += f'hf_id={self.huggingface_id}, '
        m += f'language={self.language}, size={self.size}, '
        m += f'architecture={self.architecture})'
        return m

    def _validate(self):
        _validate_model_name(self.model_name)
        _validate_model_id(self.model_id)
        _validate_optional_string(self.local_path, 'local_path')
        _validate_optional_string(self.huggingface_id, 'huggingface_id')
        _validate_optional_string(self.language, 'language')
        _validate_optional_string(self.architecture, 'architecture')
        _validate_optional_size(self.size)

    def to_dict(self):
        return {
            'model_id': self.model_id,
            'local_path': self.local_path,
            'huggingface_id': self.huggingface_id,
            'language': self.language,
            'size': self.size,
            'architecture': self.architecture,
            'model_name': self.model_name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(model_name=data.get('model_name'),
            model_id=data.get('model_id'),
            local_path=data.get('local_path'),
            huggingface_id=data.get('huggingface_id'),
            language=data.get('language'), size=data.get('size'),
            architecture=data.get('architecture'))


def _default_config():
    return {
        'models': {},
    }


def config_from_dict(data):
    if not isinstance(data, dict):
        raise ValueError('config.json must contain a JSON object')
    config = _default_config()
    raw_models = data.get('models', {})
    if not isinstance(raw_models, dict):
        raise ValueError('config.json models must be a JSON object')
    models = {}
    for model_name, record in raw_models.items():
        if not isinstance(record, dict):
            raise ValueError('config.json model records must be JSON objects')
        record = dict(record)
        record['model_name'] = model_name
        models[model_name] = ModelMetadata.from_dict(record)
    config['models'] = models
    return config


def config_to_dict(config):
    return {
        'models': {
            model_name: metadata.to_dict()
            for model_name, metadata in config['models'].items()
        },
    }


def _next_model_id(model_metadatas):
    max_id = -1
    for metadata in model_metadatas:
        candidate = metadata.model_id if metadata.model_id is not None else -1
        if candidate > max_id:
            max_id = candidate
    return max_id + 1


def check_model_name_conflict(config, metadata):
    return metadata.model_name in config['models']


def check_model_names_conflict(config, metadata_list):
    conflicts = [
        metadata for metadata in metadata_list
        if check_model_name_conflict(config, metadata)
    ]
    if not conflicts:
        return None
    return conflicts


def _validate_model_name(model_name):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError('model_name must be a non-empty string')


def _validate_model_id(model_id):
    if model_id is None:
        return
    if not isinstance(model_id, int) or model_id < 0:
        raise ValueError('model_id must be a non-negative integer')


def _validate_optional_string(value, field_name):
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty string')


def _validate_optional_size(size):
    if size is None:
        return
    if isinstance(size, bool):
        raise ValueError('size must not be a boolean')
    if not isinstance(size, (int, float, str)):
        raise ValueError('size must be a string or number')


def load_model_seed_file(path):
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f'model file must contain a JSON list, got '
            f'{type(data).__name__!r}')

    seen_names = set()
    records = []
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(
                f'record {i} must be a dict, got '
                f'{type(record).__name__!r}')
        model_name = record.get('model_name')
        if model_name is None:
            raise ValueError(
                f'record {i} is missing required field: \'model_name\'')
        metadata = ModelMetadata(
            model_name=model_name, model_id=None,
            local_path=record.get('local_path'),
            huggingface_id=record.get('huggingface_id'),
            language=record.get('language'),
            size=record.get('size'),
            architecture=record.get('architecture'))
        if model_name in seen_names:
            raise ValueError(
                f'duplicate model_name in model file: {model_name!r}')
        seen_names.add(model_name)
        records.append(metadata)
    return records
