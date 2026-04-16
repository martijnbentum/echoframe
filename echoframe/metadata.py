'''Metadata records for stored model outputs.'''

from datetime import datetime, timezone
from functools import partial

from .util_formatting import format_pretty_dict, truncate_text


SEGMENT_OUTPUT_TYPES = {
    'hidden_state',
    'attention',
    'codebook_indices',
    'codebook_matrix',
}


OUTPUT_TYPES = SEGMENT_OUTPUT_TYPES


STABLE_METADATA_FIELDS = (
    'model_name',
    'output_type',
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


class _BaseMetadata:
    '''Shared metadata behavior for echoframe records.'''

    def __init__(self, model_name, output_type, storage_status='live',
        shard_id=None, dataset_path=None, shape=None, dtype=None, tags=None,
        created_at=None, deleted_at=None, accessed_at=None,
        echoframe_key=None):
        self.model_name = model_name
        self.output_type = output_type
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
        self._validate()

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
        '''Validate record-specific fields.'''

    def __repr__(self):
        limit = 80
        tags = ','.join(self.tags) if self.tags else '-'
        body = self._repr_body() + f'status={self.storage_status}, tags={tags}'
        return 'MD(' + truncate_text(body, limit) + ')'

    def _repr_body(self):
        return f'type={self.output_type}, model={self.model_name}, '

    def __str__(self):
        return format_pretty_dict(self._display_dict())

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
        '''Canonical raw binary echoframe key for stored metadata.'''
        if self._echoframe_key is None:
            raise ValueError('metadata does not have an echoframe_key')
        return self._echoframe_key

    @property
    def has_echoframe_key(self):
        '''Return whether one canonical echoframe key is available.'''
        return self._echoframe_key is not None

    def format_echoframe_key(self):
        '''Return a printable hex string for the binary echoframe key.'''
        return self.echoframe_key.hex()

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
        data = self._display_header_dict()
        data.update(self._display_specific_dict())
        data.update(self._display_storage_dict())
        return data

    def _display_header_dict(self):
        data = {
            'model_name': self.model_name,
            'output_type': self.output_type,
        }
        if self.has_echoframe_key:
            data['echoframe_key_hex'] = self.format_echoframe_key()
        return data

    def _display_specific_dict(self):
        return {}

    def _display_storage_dict(self):
        data = {}
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
        if self.created_at is not None:
            data['created_at'] = self.created_at
        if self.deleted_at is not None:
            data['deleted_at'] = self.deleted_at
        if self.accessed_at is not None:
            data['accessed_at'] = self.accessed_at
        return data

    def to_dict(self):
        '''Serialize to a JSON-friendly dictionary.'''
        data = {
            'model_name': self.model_name,
            'output_type': self.output_type,
            'storage_status': self.storage_status,
            'shard_id': self.shard_id,
            'dataset_path': self.dataset_path,
            'shape': self.shape,
            'dtype': self.dtype,
            'tags': self.tags,
            'created_at': self.created_at,
            'deleted_at': self.deleted_at,
            'accessed_at': self.accessed_at,
        }
        if self.has_echoframe_key:
            data['echoframe_key_hex'] = self.echoframe_key.hex()
        data.update(self._to_dict_specific())
        return data

    def _to_dict_specific(self):
        return {}


class EchoframeMetadata(_BaseMetadata):
    '''Metadata for one stored segment-scoped output.'''

    def __init__(self, phraser_key, collar, model_name, output_type, layer=0,
        model_id=None, storage_status='live', shard_id=None,
        dataset_path=None, shape=None, dtype=None, tags=None,
        created_at=None, deleted_at=None, accessed_at=None,
        echoframe_key=None):
        self.phraser_key = phraser_key
        self.collar = collar
        self.layer = layer
        self.model_id = model_id
        self._phraser_object = None
        self._phraser_object_loaded = False
        super().__init__(model_name=model_name, output_type=output_type,
            storage_status=storage_status, shard_id=shard_id,
            dataset_path=dataset_path, shape=shape, dtype=dtype, tags=tags,
            created_at=created_at, deleted_at=deleted_at,
            accessed_at=accessed_at, echoframe_key=echoframe_key)

    def _validate_specific(self):
        if self.output_type not in SEGMENT_OUTPUT_TYPES:
            message = 'segment metadata output_type must be one of '
            message += f'{sorted(SEGMENT_OUTPUT_TYPES)}'
            raise ValueError(message)
        if not self.phraser_key:
            raise ValueError('phraser_key must not be empty')
        if self.collar < 0:
            raise ValueError('collar must be >= 0')
        if self.layer < 0:
            raise ValueError('layer must be >= 0')
        if self.output_type in {'codebook_indices', 'codebook_matrix'}:
            if self.layer != 0:
                raise ValueError(
                    'codebook output types require layer to be exactly 0')

    def _repr_body(self):
        return f'model={self.model_name}, layer={self.layer}, '

    def _display_header_dict(self):
        data = {
            'phraser_key': self.phraser_key,
            'collar': self.collar,
            'model_name': self.model_name,
            'output_type': self.output_type,
            'layer': self.layer,
        }
        if self.has_echoframe_key:
            data['echoframe_key_hex'] = self.format_echoframe_key()
        if self.model_id is not None:
            data['model_id'] = self.model_id
        return data

    def _display_dict(self):
        data = self._display_header_dict()
        data.update(self._display_specific_dict())
        storage = super()._display_storage_dict()
        created_at = storage.pop('created_at', None)
        deleted_at = storage.pop('deleted_at', None)
        accessed_at = storage.pop('accessed_at', None)
        data.update(storage)
        if self.phraser_object is not None:
            data['phraser_object'] = repr(self.phraser_object)
        if created_at is not None:
            data['created_at'] = created_at
        if deleted_at is not None:
            data['deleted_at'] = deleted_at
        if accessed_at is not None:
            data['accessed_at'] = accessed_at
        return data

    def _to_dict_specific(self):
        data = {
            'phraser_key': self.phraser_key,
            'collar': self.collar,
            'layer': self.layer,
            'model_id': self.model_id,
        }
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
        '''Create one metadata record from serialized data.'''
        data = dict(data)
        key_hex = data.pop('echoframe_key_hex', None)
        if key_hex is not None:
            data['echoframe_key'] = bytes.fromhex(key_hex)
        phraser_key_hex = data.pop('phraser_key_hex', None)
        if phraser_key_hex is not None:
            data['phraser_key'] = bytes.fromhex(phraser_key_hex)
        return cls(**data)


def metadata_class_for_output_type(output_type):
    '''Return the metadata class for one output type.'''
    if output_type not in OUTPUT_TYPES:
        message = f'output_type must be one of {sorted(OUTPUT_TYPES)}'
        raise ValueError(message)
    return partial(EchoframeMetadata, output_type=output_type)


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
        items = [record for record in items if getattr(record, 'layer', None)
            == layer]
    items.sort(key=lambda record: (
        -1 if getattr(record, 'collar', None) is None else record.collar,
        record.output_type,
        getattr(record, 'layer', None)
            if getattr(record, 'layer', None) is not None else -1,
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
        return [record for record in items
            if getattr(record, 'collar', None) == collar]
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


_VALID_MATCHES = {'exact', 'min', 'max', 'nearest'}
