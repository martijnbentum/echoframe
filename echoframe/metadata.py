'''Metadata records for stored model outputs.'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha1

OUTPUT_TYPES = {
    'attention',
    'codebook_indices',
    'hidden_state',
}


def utc_now() -> str:
    '''Return an ISO-8601 UTC timestamp.'''
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Metadata:
    '''EchoFrame metadata.

    This intentionally stores only metadata about model outputs. Phraser
    metadata stays in phraser and is joined through `phraser_key`.
    '''

    phraser_key: str
    collar_ms: int
    model_name: str
    output_type: str
    layer: int
    storage_status: str = 'live'
    shard_id: str | None = None
    dataset_path: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    created_at: str | None = None
    deleted_at: str | None = None
    to_vector_version: str | None = None

    def __post_init__(self) -> None:
        if not self.phraser_key:
            raise ValueError('phraser_key must not be empty')
        if self.collar_ms < 0:
            raise ValueError('collar_ms must be >= 0')
        if not self.model_name:
            raise ValueError('model_name must not be empty')
        if self.output_type not in OUTPUT_TYPES:
            raise ValueError(
                'output_type must be one of '
                f'{sorted(OUTPUT_TYPES)}'
            )
        if self.layer < 0:
            raise ValueError('layer must be >= 0')
        if self.storage_status not in {'live', 'deleted'}:
            raise ValueError("storage_status must be 'live' or 'deleted'")
        if self.created_at is None:
            object.__setattr__(self, 'created_at', utc_now())
        if self.storage_status == 'deleted' and self.deleted_at is None:
            object.__setattr__(self, 'deleted_at', utc_now())
        if self.shape is not None:
            object.__setattr__(self, 'shape', tuple(self.shape))

    @property
    def entry_id(self) -> str:
        '''Stable identifier for one canonical output unit.'''
        digest = sha1(self.identity_key.encode('utf-8')).hexdigest()
        return digest

    @property
    def identity_key(self) -> str:
        '''Canonical identity for a stored output.'''
        return ':'.join([
            self.phraser_key,
            self.model_name,
            self.output_type,
            f'{self.layer:04d}',
            f'{self.collar_ms:09d}',
        ])

    @property
    def object_key(self) -> str:
        '''Sortable object index key.'''
        return ':'.join([
            'obj',
            self.phraser_key,
            self.model_name,
            self.output_type,
            f'{self.layer:04d}',
            f'{self.collar_ms:09d}',
        ])

    def mark_deleted(self) -> 'Metadata':
        '''Return a tombstoned copy.'''
        return Metadata(
            phraser_key=self.phraser_key,
            collar_ms=self.collar_ms,
            model_name=self.model_name,
            output_type=self.output_type,
            layer=self.layer,
            storage_status='deleted',
            shard_id=self.shard_id,
            dataset_path=self.dataset_path,
            shape=self.shape,
            dtype=self.dtype,
            created_at=self.created_at,
            deleted_at=utc_now(),
            to_vector_version=self.to_vector_version,
        )

    def to_dict(self) -> dict[str, object]:
        '''Serialize to a JSON-friendly dictionary.'''
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> 'Metadata':
        '''Create an instance from serialized data.'''
        return cls(**data)
