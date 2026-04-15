'''Metadata records for stored model outputs.'''

from datetime import datetime, timezone
from hashlib import sha1
from pprint import pformat


class EchoframeMetadata:
    '''Shared metadata base for echoframe output records.'''

    OUTPUT_TYPE = None

    def __new__(cls, *args, **kwargs):
        if cls is EchoframeMetadata:
            output_type = _extract_output_type(args, kwargs)
            metadata_cls = METADATA_CLASS_BY_OUTPUT_TYPE.get(output_type)
            if metadata_cls is not None:
                return super().__new__(metadata_cls)
        return super().__new__(cls)

    def __init__(self, phraser_key, collar, model_name, output_type, layer,
        storage_status='live', shard_id=None, dataset_path=None, shape=None,
        dtype=None, tags=None, created_at=None, deleted_at=None,
        accessed_at=None, model_id=None, local_path=None,
        huggingface_id=None, language=None, echoframe_key=None):
        self.output_type = self._resolve_output_type(output_type)
        self.phraser_key = phraser_key
        self.collar = collar
        self.model_name = model_name
        self.layer = layer
        self.model_id = model_id
        self.local_path = local_path
        self.huggingface_id = huggingface_id
        self.language = language
        self.storage_status = storage_status
        self.shard_id = shard_id
        self.dataset_path = dataset_path
        self.shape = shape
        self.dtype = dtype
        self.tags = tags
        self.created_at = created_at
        self.deleted_at = deleted_at
        self.accessed_at = accessed_at
        self._echoframe_key = echoframe_key
        self._store = None
        self._phraser_object = None
        self._phraser_object_loaded = False
        self._validate()

    def _resolve_output_type(self, output_type):
        if self.OUTPUT_TYPE is None:
            return output_type
        if output_type is None:
            return self.OUTPUT_TYPE
        if output_type != self.OUTPUT_TYPE:
            raise ValueError(
                f'output_type must be {self.OUTPUT_TYPE!r} for '
                f'{self.__class__.__name__}')
        return output_type

    def _validate(self):
        if self.output_type not in OUTPUT_TYPES:
            message = f'output_type must be one of {sorted(OUTPUT_TYPES)}'
            raise ValueError(message)
        if not self.model_name:
            raise ValueError('model_name must not be empty')
        if self.storage_status not in {'live', 'deleted'}:
            message = "storage_status must be 'live' or 'deleted'"
            raise ValueError(message)
        if self.created_at is None:
            self.created_at = utc_now()
        if self.storage_status == 'deleted' and self.deleted_at is None:
            self.deleted_at = utc_now()
        if self.shape is not None:
            self.shape = tuple(self.shape)
        self.tags = normalize_tags(self.tags)
        self._validate_specific()

    def _validate_specific(self):
        '''Validate output-type-specific fields.'''

    def __repr__(self):
        limit = 80
        tags = ','.join(self.tags) if self.tags else '-'
        body = self._repr_body() + f'status={self.storage_status}, tags={tags}'
        return 'MD(' + _truncate_text(body, limit) + ')'

    def _repr_body(self):
        return f'model={self.model_name}, layer={self.layer}, '

    def __str__(self):
        return pformat(self._display_dict(), sort_dicts=False, width=80)

    def bind_store(self, store):
        '''Attach a store for convenience payload loading.'''
        self._store = store
        return self

    def load_payload(self):
        '''Load the stored output payload via an attached store.'''
        if self._store is None:
            raise ValueError('metadata is not bound to a store')
        return self._store.metadata_to_payload(self)

    @property
    def echoframe_key(self):
        '''Canonical raw binary echoframe key when provided.

        Detached/manual metadata instances fall back to a deterministic
        printable-only key shape until a real echoframe_key is assigned.
        '''
        if self._echoframe_key is None:
            self._echoframe_key = self._fallback_echoframe_key()
        return self._echoframe_key

    def format_echoframe_key(self):
        '''Return a printable hex string for the binary echoframe key.'''
        return self.echoframe_key.hex()

    def _fallback_echoframe_key(self):
        text = ':'.join(self._fallback_key_components())
        return sha1(text.encode('utf-8')).digest()

    def _fallback_key_components(self):
        raise NotImplementedError

    def mark_deleted(self):
        '''Return a tombstoned copy.'''
        data = self.to_dict()
        data['storage_status'] = 'deleted'
        data['deleted_at'] = utc_now()
        metadata = self.__class__.from_dict(data)
        return metadata.bind_store(self._store)

    def with_tags(self, tags):
        '''Return a copy with updated tags.'''
        data = self.to_dict()
        data['tags'] = tags
        metadata = self.__class__.from_dict(data)
        return metadata.bind_store(self._store)

    def with_accessed_at(self, timestamp):
        '''Return a copy with updated accessed_at.'''
        data = self.to_dict()
        data['accessed_at'] = timestamp
        metadata = self.__class__.from_dict(data)
        return metadata.bind_store(self._store)

    def _display_dict(self):
        data = {'echoframe_key_hex': self.format_echoframe_key()}
        data.update(self._display_identity_dict())
        if self.storage_status != 'live':
            data['storage_status'] = self.storage_status
        if self.shard_id is not None:
            data['shard_id'] = self.shard_id
        if self.dataset_path is not None:
            data['dataset_path'] = self.dataset_path
        if self.shape is not None:
            data['shape'] = self.shape
        if self.dtype is not None:
            data['dtype'] = self.dtype
        if self.tags:
            data['tags'] = self.tags
        if self.phraser_object is not None:
            data['phraser_object'] = repr(self.phraser_object)
        if self.created_at is not None:
            data['created_at'] = self.created_at
        if self.deleted_at is not None:
            data['deleted_at'] = self.deleted_at
        if self.accessed_at is not None:
            data['accessed_at'] = self.accessed_at
        return data

    def _display_identity_dict(self):
        raise NotImplementedError

    def to_dict(self):
        '''Serialize to a JSON-friendly dictionary.'''
        data = {'phraser_key': self.phraser_key, 'collar': self.collar,
            'model_name': self.model_name, 'model_id': self.model_id,
            'output_type': self.output_type, 'layer': self.layer,
            'local_path': self.local_path,
            'huggingface_id': self.huggingface_id,
            'language': self.language,
            'echoframe_key_hex': self.echoframe_key.hex(),
            'storage_status': self.storage_status,
            'shard_id': self.shard_id, 'dataset_path': self.dataset_path,
            'shape': self.shape, 'dtype': self.dtype, 'tags': self.tags,
            'created_at': self.created_at,
            'deleted_at': self.deleted_at,
            'accessed_at': self.accessed_at}
        if isinstance(self.phraser_key, bytes):
            data['phraser_key'] = None
            data['phraser_key_hex'] = self.phraser_key.hex()
        return data

    @property
    def phraser_object(self):
        if self._phraser_object_loaded:
            return self._phraser_object
        try:
            from phraser import models
        except ImportError:
            return None
        try:
            self._phraser_object = models.cache.load(self.phraser_key)
        except Exception:
            self._phraser_object = None
        self._phraser_object_loaded = True
        return self._phraser_object

    @property
    def label(self):
        phraser_object = self.phraser_object
        if phraser_object is None:
            return None
        return getattr(phraser_object, 'label', None)

    @classmethod
    def from_dict(cls, data):
        '''Create an instance from serialized data.'''
        data = dict(data)
        key_hex = data.pop('echoframe_key_hex', None)
        if key_hex is not None:
            data['echoframe_key'] = bytes.fromhex(key_hex)
        phraser_key_hex = data.pop('phraser_key_hex', None)
        if phraser_key_hex is not None:
            data['phraser_key'] = bytes.fromhex(phraser_key_hex)
        metadata_cls = cls
        if cls is EchoframeMetadata:
            metadata_cls = metadata_class_for_output_type(data['output_type'])
        return metadata_cls(**data)


class _PhraserKeyMetadata(EchoframeMetadata):
    '''Shared metadata behavior for phraser-key-based outputs.'''

    def _validate_specific(self):
        if not self.phraser_key:
            raise ValueError('phraser_key must not be empty')
        if self.collar < 0:
            raise ValueError('collar must be >= 0')
        if self.layer < 0:
            raise ValueError('layer must be >= 0')

    def _fallback_key_components(self):
        return [
            text_key_component(self.phraser_key),
            self.model_name,
            self.output_type,
            f'{self.layer:04d}',
            f'{self.collar:09d}',
        ]

    def _display_identity_dict(self):
        data = {
            'phraser_key': self.phraser_key,
            'collar': self.collar,
            'model_name': self.model_name,
            'output_type': self.output_type,
            'layer': self.layer,
        }
        if self.model_id is not None:
            data['model_id'] = self.model_id
        return data


class HiddenStateMetadata(_PhraserKeyMetadata):
    OUTPUT_TYPE = 'hidden_state'

    def __init__(self, phraser_key, collar, model_name, output_type=None,
        layer=0, **kwargs):
        super().__init__(phraser_key=phraser_key, collar=collar,
            model_name=model_name, output_type=output_type, layer=layer,
            **kwargs)


class AttentionMetadata(_PhraserKeyMetadata):
    OUTPUT_TYPE = 'attention'

    def __init__(self, phraser_key, collar, model_name, output_type=None,
        layer=0, **kwargs):
        super().__init__(phraser_key=phraser_key, collar=collar,
            model_name=model_name, output_type=output_type, layer=layer,
            **kwargs)


class CodebookIndicesMetadata(_PhraserKeyMetadata):
    OUTPUT_TYPE = 'codebook_indices'

    def __init__(self, phraser_key, collar, model_name, output_type=None,
        layer=0, **kwargs):
        super().__init__(phraser_key=phraser_key, collar=collar,
            model_name=model_name, output_type=output_type, layer=layer,
            **kwargs)

    def _validate_specific(self):
        super()._validate_specific()
        if self.layer != 0:
            raise ValueError(
                'codebook output types require layer to be exactly 0')


class CodebookMatrixMetadata(_PhraserKeyMetadata):
    OUTPUT_TYPE = 'codebook_matrix'

    def __init__(self, phraser_key, collar, model_name, output_type=None,
        layer=0, **kwargs):
        super().__init__(phraser_key=phraser_key, collar=collar,
            model_name=model_name, output_type=output_type, layer=layer,
            **kwargs)

    def _validate_specific(self):
        super()._validate_specific()
        if self.layer != 0:
            raise ValueError(
                'codebook output types require layer to be exactly 0')


class ModelMetadata(EchoframeMetadata):
    OUTPUT_TYPE = 'model_metadata'

    def __init__(self, model_name, output_type=None, model_id=None,
        local_path=None, huggingface_id=None, language=None,
        storage_status='live', shard_id=None, dataset_path=None, shape=None,
        dtype=None, tags=None, created_at=None, deleted_at=None,
        accessed_at=None, phraser_key=None, collar=None, layer=None,
        echoframe_key=None):
        super().__init__(phraser_key=phraser_key, collar=collar,
            model_name=model_name, output_type=output_type, layer=layer,
            storage_status=storage_status, shard_id=shard_id,
            dataset_path=dataset_path, shape=shape, dtype=dtype, tags=tags,
            created_at=created_at, deleted_at=deleted_at,
            accessed_at=accessed_at, model_id=model_id,
            local_path=local_path, huggingface_id=huggingface_id,
            language=language, echoframe_key=echoframe_key)

    def _validate_specific(self):
        if self.phraser_key not in (None, ''):
            raise ValueError('model_metadata does not use phraser_key')
        if self.collar not in (None, 0):
            raise ValueError('model_metadata does not use collar')
        if self.layer not in (None, 0):
            raise ValueError('model_metadata does not use layer')

    def _repr_body(self):
        return f'type={self.output_type}, model={self.model_name}, '

    def _fallback_key_components(self):
        return [self.model_name, self.output_type]

    def _display_identity_dict(self):
        data = {
            'model_name': self.model_name,
            'output_type': self.output_type,
        }
        if self.model_id is not None:
            data['model_id'] = self.model_id
        if self.local_path is not None:
            data['local_path'] = self.local_path
        if self.huggingface_id is not None:
            data['huggingface_id'] = self.huggingface_id
        if self.language is not None:
            data['language'] = self.language
        return data


METADATA_CLASS_BY_OUTPUT_TYPE = {
    'hidden_state': HiddenStateMetadata,
    'attention': AttentionMetadata,
    'codebook_indices': CodebookIndicesMetadata,
    'codebook_matrix': CodebookMatrixMetadata,
    'model_metadata': ModelMetadata,
}


OUTPUT_TYPES = set(METADATA_CLASS_BY_OUTPUT_TYPE)


STABLE_METADATA_FIELDS = (
    'phraser_key',
    'collar',
    'model_name',
    'model_id',
    'output_type',
    'layer',
    'local_path',
    'huggingface_id',
    'language',
    'storage_status',
    'shard_id',
    'dataset_path',
    'shape',
    'dtype',
    'tags',
    'created_at',
    'deleted_at',
    'accessed_at',
)


def metadata_class_for_output_type(output_type):
    '''Return the metadata class for one output type.'''
    if output_type not in METADATA_CLASS_BY_OUTPUT_TYPE:
        message = f'output_type must be one of {sorted(OUTPUT_TYPES)}'
        raise ValueError(message)
    return METADATA_CLASS_BY_OUTPUT_TYPE[output_type]


def filter_metadata(records, model_name=None, output_type=None, layer=None,
    collar=None, match='exact'):
    '''Filter metadata records by common artifact fields.'''
    items = [record for record in records if record is not None]
    if model_name is not None:
        items = [record for record in items if record.model_name == model_name]
    if output_type is not None:
        items = [record for record in items
            if record.output_type == output_type]
    if layer is not None:
        items = [record for record in items if record.layer == layer]
    items.sort(key=lambda record: (
        -1 if record.collar is None else record.collar,
        record.output_type,
        record.layer if record.layer is not None else -1,
        record.model_name or '',
    ))

    if collar is None:
        return items
    if match not in _VALID_MATCHES:
        message = "match must be one of 'exact', 'min', 'max', 'nearest'"
        raise ValueError(message)
    if not items:
        return []
    if match == 'exact':
        return [record for record in items if record.collar == collar]
    if match == 'min':
        for record in items:
            if record.collar >= collar:
                return [record]
        return []
    if match == 'max':
        for record in reversed(items):
            if record.collar <= collar:
                return [record]
        return []
    return [min(items, key=lambda record: abs(record.collar - collar))]


def utc_now():
    '''Return an ISO-8601 UTC timestamp.'''
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_tags(tags):
    '''Validate and normalize tag values.'''
    if tags is None:
        return []

    values = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError('tags must contain only strings')
        value = tag.strip()
        if not value:
            raise ValueError('tags must not be empty')
        if ':' in value:
            raise ValueError("tags must not contain ':'")
        values.append(value)
    return sorted(set(values))


def text_key_component(value):
    '''Return a stable text form for key components used in fallback keys.'''
    if isinstance(value, bytes):
        return value.hex()
    return value


def _truncate_text(text, max_length):
    '''Return text clipped to max_length with a trailing ellipsis.'''
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[:max_length - 3] + '...'


def _extract_output_type(args, kwargs):
    if 'output_type' in kwargs:
        return kwargs['output_type']
    if len(args) >= 4:
        return args[3]
    return None


_VALID_MATCHES = {'exact', 'min', 'max', 'nearest'}
