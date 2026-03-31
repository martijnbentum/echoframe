'''Public store facade for echoframe.'''

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .index import LmdbIndex
from .metadata import Metadata
from .output_storage import Hdf5ShardStore


class Store:
    '''Link phraser keys to stored model outputs.'''

    def __init__(self, root: str | Path,
        max_shard_size_bytes: int=1_000_000_000,
        index: LmdbIndex | None=None,
        storage: Hdf5ShardStore | None=None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = index or LmdbIndex(self.root / 'index.lmdb')
        self.storage = storage or Hdf5ShardStore(self.root / 'shards',
            max_shard_size_bytes=max_shard_size_bytes)

    def put(self, phraser_key: str, collar_ms: int, model_name: str,
        output_type: str, layer: int, data: object,
        to_vector_version: str | None=None) -> Metadata:
        '''Store one output payload and index its metadata.'''
        metadata = Metadata(phraser_key=phraser_key,
            collar_ms=collar_ms, model_name=model_name,
            output_type=output_type, layer=layer,
            to_vector_version=to_vector_version)
        stored = self.storage.store(metadata, data=data)
        return self.index.upsert(stored)

    def find(self, phraser_key: str, model_name: str | None=None,
        output_type: str | None=None, layer: int | None=None,
        include_deleted: bool=False) -> list[Metadata]:
        '''List matching metadata records for one phraser key.'''
        return self.index.find(phraser_key=phraser_key,
            model_name=model_name, output_type=output_type, layer=layer,
            include_deleted=include_deleted)

    def find_one(self, phraser_key: str, collar_ms: int, model_name: str,
        output_type: str, layer: int,
        match: str='exact') -> Metadata | None:
        '''Find one matching output metadata record.'''
        return self.index.find_one(phraser_key=phraser_key,
            model_name=model_name, output_type=output_type, layer=layer,
            collar_ms=collar_ms, match=match)

    def exists(self, phraser_key: str, collar_ms: int, model_name: str,
        output_type: str, layer: int, match: str='exact') -> bool:
        '''Return whether a matching output is stored.'''
        return self.find_one(phraser_key=phraser_key,
            collar_ms=collar_ms, model_name=model_name,
            output_type=output_type, layer=layer,
            match=match) is not None

    def load(self, phraser_key: str, collar_ms: int, model_name: str,
        output_type: str, layer: int, match: str='exact') -> object:
        '''Load one stored output payload.'''
        metadata = self.find_one(phraser_key=phraser_key,
            collar_ms=collar_ms, model_name=model_name,
            output_type=output_type, layer=layer, match=match)
        if metadata is None:
            raise ValueError('no stored output matched the requested criteria')
        return self.storage.load(metadata)

    def delete(self, phraser_key: str, collar_ms: int, model_name: str,
        output_type: str, layer: int,
        match: str='exact') -> Metadata | None:
        '''Delete one stored output from live indexes.'''
        metadata = self.find_one(phraser_key=phraser_key,
            collar_ms=collar_ms, model_name=model_name,
            output_type=output_type, layer=layer, match=match)
        if metadata is None:
            return None
        self.storage.delete(metadata)
        return self.index.delete(metadata)

    def find_or_compute(self, phraser_key: str, collar_ms: int,
        model_name: str, output_type: str, layer: int,
        compute: Callable[[], object], match: str='exact',
        to_vector_version: str | None=None) -> tuple[Metadata, bool]:
        '''Load metadata if present, otherwise compute and store a payload.'''
        metadata = self.find_one(phraser_key=phraser_key,
            collar_ms=collar_ms, model_name=model_name,
            output_type=output_type, layer=layer, match=match)
        if metadata is not None:
            return metadata, False
        data = compute()
        metadata = self.put(phraser_key=phraser_key,
            collar_ms=collar_ms, model_name=model_name,
            output_type=output_type, layer=layer, data=data,
            to_vector_version=to_vector_version)
        return metadata, True
