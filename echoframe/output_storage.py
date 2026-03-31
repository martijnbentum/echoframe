'''HDF5 shard storage for model output payloads.'''

import re
from pathlib import Path

from .metadata import Metadata


def sanitize_name(value):
    '''Convert a user-facing value into a shard-safe name.'''
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', value).strip('_') or 'unknown'


class Hdf5ShardStore:
    '''Store payloads in rolling HDF5 shard files.'''

    def __init__(self, root, max_shard_size_bytes=1_000_000_000,
        h5_module=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_shard_size_bytes = max_shard_size_bytes
        self.h5 = h5_module or self._import_h5()

    def _import_h5(self):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError('h5py is required to use Store') from exc
        return h5py

    def store(self, metadata, data):
        '''Store payload data and return updated metadata.'''
        shard_id = self._active_shard_id(model_name=metadata.model_name,
            output_type=metadata.output_type)
        dataset_path = self._dataset_path(metadata)
        file_path = self.root / f'{shard_id}.h5'

        with self.h5.File(file_path, 'a') as handle:
            group = handle.require_group(f'/layer_{metadata.layer:04d}')
            if metadata.entry_id in group:
                del group[metadata.entry_id]
            dataset = group.create_dataset(metadata.entry_id, data=data)
            shape = tuple(getattr(dataset, 'shape', ()) or ())
            dtype = getattr(dataset, 'dtype', None)
            if dtype is None: dtype = getattr(data, 'dtype', 'unknown')
            dtype = str(dtype)

        return Metadata(phraser_key=metadata.phraser_key,
            collar=metadata.collar, model_name=metadata.model_name,
            output_type=metadata.output_type, layer=metadata.layer,
            storage_status=metadata.storage_status, shard_id=shard_id,
            dataset_path=dataset_path, shape=shape, dtype=dtype,
            tags=metadata.tags, created_at=metadata.created_at,
            deleted_at=metadata.deleted_at,
            to_vector_version=metadata.to_vector_version)

    def load(self, metadata):
        '''Load stored payload data.'''
        if metadata.shard_id is None or metadata.dataset_path is None:
            raise ValueError('metadata does not point to a stored payload')
        file_path = self.root / f'{metadata.shard_id}.h5'
        with self.h5.File(file_path, 'r') as handle:
            return handle[metadata.dataset_path][()]

    def delete(self, metadata):
        '''Best-effort payload deletion.

        HDF5 files are not compacted here. Compaction should be a later,
        offline maintenance step.
        '''
        if metadata.shard_id is None or metadata.dataset_path is None:
            return
        file_path = self.root / f'{metadata.shard_id}.h5'
        if not file_path.exists():
            return
        with self.h5.File(file_path, 'a') as handle:
            if metadata.dataset_path in handle:
                del handle[metadata.dataset_path]

    def compact_shard(self, shard_id, entries):
        '''Rewrite live datasets from one shard into a new shard.'''
        if not entries:
            self._delete_file(shard_id)
            return []

        next_shard_id = self._replacement_shard_id(shard_id)
        old_file_path = self.root / f'{shard_id}.h5'
        new_file_path = self.root / f'{next_shard_id}.h5'
        updated = []

        with self.h5.File(old_file_path, 'r') as source:
            with self.h5.File(new_file_path, 'a') as target:
                for metadata in entries:
                    dataset = source[metadata.dataset_path][()]
                    group = target.require_group(f'/layer_{metadata.layer:04d}')
                    if metadata.entry_id in group:
                        del group[metadata.entry_id]
                    created = group.create_dataset(metadata.entry_id,
                        data=dataset)
                    updated.append(Metadata(
                        phraser_key=metadata.phraser_key,
                        collar=metadata.collar,
                        model_name=metadata.model_name,
                        output_type=metadata.output_type,
                        layer=metadata.layer,
                        storage_status=metadata.storage_status,
                        shard_id=next_shard_id,
                        dataset_path=self._dataset_path(metadata),
                        shape=tuple(getattr(created, 'shape', ()) or ()),
                        dtype=str(getattr(created, 'dtype', 'unknown')),
                        tags=metadata.tags,
                        created_at=metadata.created_at,
                        deleted_at=metadata.deleted_at,
                        to_vector_version=metadata.to_vector_version,
                    ))

        self._delete_file(shard_id)
        return updated

    def _active_shard_id(self, model_name, output_type):
        stem = f'{sanitize_name(model_name)}_{sanitize_name(output_type)}'
        index = 1
        while True:
            shard_id = f'{stem}_{index:04d}'
            file_path = self.root / f'{shard_id}.h5'
            if not file_path.exists():
                return shard_id
            if self._file_size(file_path) < self.max_shard_size_bytes:
                return shard_id
            index += 1

    def _dataset_path(self, metadata):
        return f'/layer_{metadata.layer:04d}/{metadata.entry_id}'

    def _file_size(self, file_path):
        return file_path.stat().st_size

    def _replacement_shard_id(self, shard_id):
        stem, _, suffix = shard_id.rpartition('_')
        if not stem or not suffix.isdigit():
            raise ValueError('invalid shard_id')
        index = int(suffix) + 1
        while True:
            candidate = f'{stem}_{index:04d}'
            if not (self.root / f'{candidate}.h5').exists():
                return candidate
            index += 1

    def _delete_file(self, shard_id):
        file_path = self.root / f'{shard_id}.h5'
        if hasattr(self.h5, 'files'):
            self.h5.files.pop(str(file_path), None)
        if file_path.exists():
            file_path.unlink()
