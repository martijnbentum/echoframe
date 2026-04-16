'''Metadata records for stored model outputs.'''

from datetime import datetime, timezone
from functools import partial

from .util_formatting import format_pretty_dict, truncate_text


class EchoframeMetadata:
    '''Metadata for one stored echoframe output.
    phraser_key:    phraser object key linked to this output
    collar:         collar in milliseconds
    model_name:     registered model name
    output_type:    stored output type
    layer:          model layer index
    '''

    def __init__(self, phraser_key, collar, model_name, output_type, layer=0,
        model_id=None, storage_status='live', shard_id=None,
        dataset_path=None, shape=None, dtype=None, tags=None,
        created_at=None, deleted_at=None, accessed_at=None,
        echoframe_key=None):
        self.phraser_key = phraser_key
        self.collar = collar
        self.model_name = model_name
        self.output_type = output_type
        self.layer = layer
        self.model_id = model_id
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

    def bind_store(self, store):
        '''Attach one store so the metadata can load its payload.'''
        self._store = store
        return self

    def load_payload(self):
        '''Load the stored output payload through the attached store.'''
        if self._store is None:
            raise ValueError('metadata is not bound to a store')
        return self._store.metadata_to_payload(self)

    @property
    def echoframe_key(self):
        '''Return the canonical binary echoframe key.

        Raises ValueError when the metadata is detached and has not been
        assigned a real store key yet.
        '''
        if self._echoframe_key is None:
            raise ValueError('metadata does not have an echoframe_key')
        return self._echoframe_key

    @property
    def has_echoframe_key(self):
        '''Return whether this metadata has a real echoframe key.'''
        return self._echoframe_key is not None

    def format_echoframe_key(self):
        '''Return the echoframe key as a printable hex string.'''
        return self.echoframe_key.hex()

    @property
    def phraser_object(self):
        '''Load the linked phraser object when phraser is available.'''
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
        '''Return the linked phraser object label when available.'''
        phraser_object = self.phraser_object
        if phraser_object is None:
            return None
        return getattr(phraser_object, 'label', None)

    def to_dict(self):
        '''Serialize one metadata record to a JSON-friendly dictionary.'''
        data = {
            'phraser_key': self.phraser_key,
            'collar': self.collar,
            'model_name': self.model_name,
            'output_type': self.output_type,
            'layer': self.layer,
            'model_id': self.model_id,
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
        if isinstance(self.phraser_key, bytes):
            data['phraser_key'] = None
            data['phraser_key_hex'] = self.phraser_key.hex()
        if self.has_echoframe_key:
            data['echoframe_key_hex'] = self.echoframe_key.hex()
        return data

    @classmethod
    def from_dict(cls, data):
        '''Create one metadata record from serialized metadata data.'''
        data = dict(data)
        key_hex = data.pop('echoframe_key_hex', None)
        if key_hex is not None:
            data['echoframe_key'] = bytes.fromhex(key_hex)
        phraser_key_hex = data.pop('phraser_key_hex', None)
        if phraser_key_hex is not None:
            data['phraser_key'] = bytes.fromhex(phraser_key_hex)
        return cls(**data)

    def mark_deleted(self):
        '''Return a copy marked as deleted with a fresh deleted_at value.'''
        data = self.to_dict()
        data['storage_status'] = 'deleted'
        data['deleted_at'] = utc_now()
        metadata = self.__class__.from_dict(data)
        return metadata.bind_store(self._store)

    def with_tags(self, tags):
        '''Return a copy with a replaced normalized tag set.'''
        data = self.to_dict()
        data['tags'] = tags
        metadata = self.__class__.from_dict(data)
        return metadata.bind_store(self._store)

    def with_accessed_at(self, timestamp):
        '''Return a copy with an updated accessed_at timestamp.'''
        data = self.to_dict()
        data['accessed_at'] = timestamp
        metadata = self.__class__.from_dict(data)
        return metadata.bind_store(self._store)

    def __repr__(self):
        limit = 80
        tags = ','.join(self.tags) if self.tags else '-'
        body = f'model={self.model_name}, layer={self.layer}, '
        body += f'status={self.storage_status}, tags={tags}'
        return 'MD(' + truncate_text(body, limit) + ')'

    def __str__(self):
        return format_pretty_dict(self._display_dict())

    def _validate(self):
        _validate_output_type(self.output_type)
        _validate_model_name(self.model_name)
        _validate_storage_status(self.storage_status)
        if self.created_at is None:
            self.created_at = utc_now()
        if self.storage_status == 'deleted' and self.deleted_at is None:
            self.deleted_at = utc_now()
        if self.shape is not None:
            self.shape = tuple(self.shape)
        self.tags = normalize_tags(self.tags)
        self._validate_segment_fields()

    def _validate_segment_fields(self):
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

    def _display_dict(self):
        data = self._display_header_dict()
        storage = _display_storage_dict(self)
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


def metadata_class_for_output_type(output_type):
    '''Return the metadata constructor for one output type.'''
    _validate_output_type(output_type)
    return partial(EchoframeMetadata, output_type=output_type)


def filter_metadata(records, model_name=None, output_type=None, layer=None,
    collar=None, match='exact'):
    '''Filter metadata records by common echoframe fields.
    records:       metadata iterable
    model_name:    optional model filter
    output_type:   optional output type filter
    layer:         optional layer filter
    collar:        optional collar filter
    match:         exact, min, max, or nearest
    '''
    items = [record for record in records if record is not None]
    if model_name is not None:
        items = [record for record in items if record.model_name == model_name]
    if output_type is not None:
        items = [record for record in items
            if record.output_type == output_type]
    if layer is not None:
        items = [record for record in items if record.layer == layer]
    items.sort(key=lambda record: (
        record.collar,
        record.output_type,
        record.layer,
        record.model_name,
    ))

    if collar is None:
        return items
    if match not in VALID_MATCHES:
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


def _validate_output_type(output_type):
    if output_type not in OUTPUT_TYPES:
        message = f'output_type must be one of {sorted(OUTPUT_TYPES)}'
        raise ValueError(message)


def _validate_model_name(model_name):
    if not model_name:
        raise ValueError('model_name must not be empty')


def _validate_storage_status(storage_status):
    if storage_status not in {'live', 'deleted'}:
        raise ValueError("storage_status must be 'live' or 'deleted'")


def _display_storage_dict(metadata):
    data = {}
    if metadata.storage_status != 'live':
        data['storage_status'] = metadata.storage_status
    if metadata.shard_id is not None:
        data['shard_id'] = metadata.shard_id
    if metadata.dataset_path is not None:
        data['dataset_path'] = metadata.dataset_path
    if metadata.shape is not None:
        data['shape'] = metadata.shape
    if metadata.dtype is not None:
        data['dtype'] = metadata.dtype
    if metadata.tags:
        data['tags'] = metadata.tags
    if metadata.created_at is not None:
        data['created_at'] = metadata.created_at
    if metadata.deleted_at is not None:
        data['deleted_at'] = metadata.deleted_at
    if metadata.accessed_at is not None:
        data['accessed_at'] = metadata.accessed_at
    return data


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

VALID_MATCHES = {'exact', 'min', 'max', 'nearest'}
