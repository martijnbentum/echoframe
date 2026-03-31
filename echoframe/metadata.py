'''Metadata records for stored model outputs.'''

from datetime import datetime, timezone
from hashlib import sha1

OUTPUT_TYPES = {
    'attention',
    'codebook_indices',
    'hidden_state',
}


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


class Metadata:
    '''EchoFrame metadata.
    phraser_key:          unique phraser object key
    collar:               collar in milliseconds
    model_name:           model identifier
    output_type:          hidden_state, attention, or codebook_indices
    layer:                model layer index
    tags:                 optional grouping labels
    '''

    def __init__(self, phraser_key, collar, model_name, output_type, layer,
        storage_status='live', shard_id=None, dataset_path=None, shape=None,
        dtype=None, tags=None, created_at=None, deleted_at=None,
        to_vector_version=None):
        self.phraser_key = phraser_key
        self.collar = collar
        self.model_name = model_name
        self.output_type = output_type
        self.layer = layer
        self.storage_status = storage_status
        self.shard_id = shard_id
        self.dataset_path = dataset_path
        self.shape = shape
        self.dtype = dtype
        self.tags = tags
        self.created_at = created_at
        self.deleted_at = deleted_at
        self.to_vector_version = to_vector_version
        self._validate()

    def _validate(self):
        if not self.phraser_key:
            raise ValueError('phraser_key must not be empty')
        if self.collar < 0:
            raise ValueError('collar must be >= 0')
        if not self.model_name:
            raise ValueError('model_name must not be empty')
        if self.output_type not in OUTPUT_TYPES:
            message = f'output_type must be one of {sorted(OUTPUT_TYPES)}'
            raise ValueError(message)
        if self.layer < 0:
            raise ValueError('layer must be >= 0')
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

    @property
    def entry_id(self):
        '''Stable identifier for one canonical output unit.'''
        digest = sha1(self.identity_key.encode('utf-8')).hexdigest()
        return digest

    @property
    def identity_key(self):
        '''Canonical identity for a stored output.'''
        return ':'.join([self.phraser_key, self.model_name,
            self.output_type, f'{self.layer:04d}',
            f'{self.collar:09d}'])

    @property
    def object_key(self):
        '''Sortable object index key.'''
        return ':'.join(['obj', self.phraser_key, self.model_name,
            self.output_type, f'{self.layer:04d}',
            f'{self.collar:09d}'])

    def mark_deleted(self):
        '''Return a tombstoned copy.'''
        return Metadata(phraser_key=self.phraser_key,
            collar=self.collar, model_name=self.model_name,
            output_type=self.output_type, layer=self.layer,
            storage_status='deleted', shard_id=self.shard_id,
            dataset_path=self.dataset_path, shape=self.shape,
            dtype=self.dtype, tags=self.tags, created_at=self.created_at,
            deleted_at=utc_now(),
            to_vector_version=self.to_vector_version)

    def with_tags(self, tags):
        '''Return a copy with updated tags.'''
        return Metadata(phraser_key=self.phraser_key,
            collar=self.collar, model_name=self.model_name,
            output_type=self.output_type, layer=self.layer,
            storage_status=self.storage_status, shard_id=self.shard_id,
            dataset_path=self.dataset_path, shape=self.shape,
            dtype=self.dtype, tags=tags, created_at=self.created_at,
            deleted_at=self.deleted_at,
            to_vector_version=self.to_vector_version)

    def to_dict(self):
        '''Serialize to a JSON-friendly dictionary.'''
        return {
            'phraser_key': self.phraser_key,
            'collar': self.collar,
            'model_name': self.model_name,
            'output_type': self.output_type,
            'layer': self.layer,
            'storage_status': self.storage_status,
            'shard_id': self.shard_id,
            'dataset_path': self.dataset_path,
            'shape': self.shape,
            'dtype': self.dtype,
            'tags': self.tags,
            'created_at': self.created_at,
            'deleted_at': self.deleted_at,
            'to_vector_version': self.to_vector_version,
        }

    @classmethod
    def from_dict(cls, data):
        '''Create an instance from serialized data.
        data:    serialized metadata mapping
        '''
        return cls(**data)
