'''Metadata records for stored model outputs.'''

from datetime import datetime, timezone

from . import key_helper
from .util_formatting import format_pretty_dict, truncate_text


class EchoframeMetadata:
    '''Metadata for one stored echoframe output.'''

    def __init__(self, echoframe_key, store = None, tags=None, model_name=None):
        '''Initialize one metadata record.
        echoframe_key:  canonical binary echoframe key
        store:          store used for model-name resolution
        tags:           optional grouping labels
        '''
        self.echoframe_key = echoframe_key
        self.store = store
        self._model_name = model_name
        self._set_echoframe_key_attributes()
        self.shard_id = None
        self.dataset_path = None
        self.shape = None
        self.tags = tags
        self.created_at = None
        self.accessed_at = None
        self._validate()

    def __repr__(self):
        limit = 80
        tags = ','.join(self.tags) if self.tags else '-'
        body = f'model={self.model_name}, layer={self.layer}, '
        body += f'tags={tags}'
        return 'MD(' + truncate_text(body, limit) + ')'

    def __str__(self):
        return format_pretty_dict(self._display_dict())

    def _set_echoframe_key_attributes(self):
        self._key_fields = key_helper.unpack_echoframe_key(self.echoframe_key)
        self.echoframe_key_attributes = tuple(sorted(self._key_fields))
        self.model_id = self._key_fields.get('model_id', None)
        self.output_type = self._key_fields.get('output_type', None)
        self.phraser_key = self._key_fields.get('phraser_key', None)
        self.collar = self._key_fields.get('collar', None)
        self.layer = self._key_fields.get('layer', None)

    def load_payload(self):
        '''Load the stored output payload through the attached store.'''
        if self.store is None:
            raise ValueError('store is not attached to metadata')
        return self.store.metadata_to_payload(self)

    @property
    def model_name(self):
        '''Resolve model_name through the bound store registry if not present.'''
        if self._model_name is not None:
            return self._model_name
        if self.store is None:
            raise ValueError('store is not attached to metadata')
        if self.store.registry is None:
            raise ValueError('store does not have a registry')
        return _model_name_from_registry(self.store.registry, self.model_id)

    @property
    def phraser_object(self):
        '''Load the linked phraser object when phraser is available.'''
        if hasattr(self, '_phraser_object'): return self._phraser_object
        from phraser import phraser_models
        try: self._phraser_object = phraser_models.cache.load(self.phraser_key)
        except Exception as e:
            print(f"Error loading phraser object: {self.phraser_key}: {e}")
            self._phraser_object = None
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
        return {
            'model_name': self.model_name,
            'shard_id': self.shard_id,
            'dataset_path': self.dataset_path,
            'shape': self.shape,
            'tags': self.tags,
            'created_at': self.created_at,
            'accessed_at': self.accessed_at,
        }

    @classmethod
    def from_dict(cls, metadata, echoframe_key, store=None):
        '''Create one metadata record from serialized metadata data.'''
        data = dict(metadata)
        return cls._from_state(
            echoframe_key=echoframe_key,
            store=store,
            model_name=data.pop('model_name', None),
            tags=data.pop('tags', None),
            shard_id=data.pop('shard_id', None),
            dataset_path=data.pop('dataset_path', None),
            shape=data.pop('shape', None),
            created_at=data.pop('created_at', None),
            accessed_at=data.pop('accessed_at', None))

    def with_tags(self, tags):
        '''Return a copy with a replaced normalized tag set.'''
        return self.copy(tags=tags)

    def with_accessed_at(self, timestamp):
        '''Return a copy with an updated accessed_at timestamp.'''
        return self.copy(accessed_at=timestamp)


    def _validate(self):
        if self.created_at is None:
            self.created_at = utc_now()
        if self.shape is not None:
            self.shape = tuple(self.shape)
        self.tags = normalize_tags(self.tags)
        self._validate_key_fields()

    def copy(self, **updates):
        metadata = self.__class__._from_state(
            echoframe_key=self.echoframe_key,
            store=self.store,
            model_name=updates.pop('model_name', self._model_name),
            tags=updates.pop('tags', self.tags),
            shard_id=updates.pop('shard_id', self.shard_id),
            dataset_path=updates.pop('dataset_path', self.dataset_path),
            shape=updates.pop('shape', self.shape),
            created_at=updates.pop('created_at', self.created_at),
            accessed_at=updates.pop('accessed_at', self.accessed_at))
        return metadata

    @classmethod
    def _from_state(cls, echoframe_key, store, model_name=None, tags=None,
        shard_id=None, dataset_path=None, shape=None, created_at=None,
        accessed_at=None):
        metadata = cls(echoframe_key=echoframe_key, store=store, tags=tags,
            model_name=model_name)
        metadata.shard_id = shard_id
        metadata.dataset_path = dataset_path
        metadata.shape = shape
        metadata.created_at = created_at
        metadata.accessed_at = accessed_at
        metadata._validate()
        return metadata

    def _validate_key_fields(self):
        _validate_output_type(self.output_type)
        if self.collar is not None and self.collar < 0:
            raise ValueError('collar must be >= 0')
        if self.layer is not None and self.layer < 0:
            raise ValueError('layer must be >= 0')

    def _display_dict(self):
        data = self._display_header_dict()
        storage = _display_storage_dict(self)
        created_at = storage.pop('created_at', None)
        accessed_at = storage.pop('accessed_at', None)
        data.update(storage)
        if self.phraser_object is not None:
            data['phraser_object'] = repr(self.phraser_object)
        data['created_at'] = created_at
        if accessed_at is not None:
            data['accessed_at'] = accessed_at
        return data

    def _display_header_dict(self):
        data = {'output_type': self.output_type}
        data['model_name'] = self.model_name
        data['phraser_key'] = self.phraser_key
        data['collar'] = self.collar
        data['layer'] = self.layer
        data['echoframe_key_hex'] = self.echoframe_key.hex()
        data['model_id'] = self.model_id
        return data


def filter_metadata(records, model_name=None, output_type=None, layer=None,
    collar=None, collar_match='exact'):
    '''Filter metadata records by common echoframe fields.
    records:       metadata iterable
    model_name:    optional model filter
    output_type:   optional output type filter
    layer:         optional layer filter
    collar:        optional collar filter
    collar_match:         exact, min, max, or nearest
    '''
    items = [x for x in records if x is not None]
    if model_name is not None:
        items = [record for record in items if record.model_name == model_name]
    if output_type is not None:
        items = [record for record in items if record.output_type == output_type]
    if layer is not None:
        items = [record for record in items if record.layer == layer]
    if not items: return []
        
    items.sort(key=lambda record: (_sort_collar(record), record.output_type,
        _sort_layer(record), _sort_model_name(record)))
    if collar is None: return items
        
    if collar_match not in VALID_MATCHES:
        message = "collar_match must be one of 'exact', 'min', 'max', "
        message += "'nearest'"
        raise ValueError(message)
    if collar_match == 'exact':
        items = [record for record in items if record.collar == collar]
        return items
    if collar_match == 'min':
        items = [record for record in items if record.collar >= collar]
        return items
    if collar_match == 'max':
        items = [record for record in items if record.collar <= collar]
        return items
    # nearest
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
        message = 'tags must contain only strings'
        if not isinstance(tag, str):
            raise ValueError(message)
        value = tag.strip()
        if not value:
            raise ValueError('tags must not be empty')
        if ':' in value:
            raise ValueError("tags must not contain ':'")
        values.append(value)
    return sorted(set(values))


def _validate_output_type(output_type):
    if output_type not in OUTPUT_TYPES:
        message = f'output_type must be one of {OUTPUT_TYPES}'
        raise ValueError(message)


def _display_storage_dict(metadata):
    data = {}
    if metadata.shard_id is not None:
        data['shard_id'] = metadata.shard_id
    if metadata.dataset_path is not None:
        data['dataset_path'] = metadata.dataset_path
    if metadata.shape is not None:
        data['shape'] = metadata.shape
    if metadata.tags:
        data['tags'] = metadata.tags
    if metadata.created_at is not None:
        data['created_at'] = metadata.created_at
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
    'shard_id',
    'dataset_path',
    'shape',
    'tags',
    'created_at',
    'accessed_at',
)

VALID_MATCHES = {'exact', 'min', 'max', 'nearest'}



def _model_name_from_registry(registry, model_id):
    for metadata in registry.model_metadatas:
        if metadata.model_id == model_id:
            return metadata.model_name
    raise ValueError(f'model_id {model_id} not found in registry')


def _sort_collar(record):
    if record.collar is None:
        return -1
    return record.collar


def _sort_layer(record):
    if record.layer is None:
        return -1
    return record.layer


def _sort_model_name(record):
    if record.model_name is None:
        return ''
    return record.model_name
