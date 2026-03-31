'''Public store facade for echoframe.'''

from pathlib import Path

from .index import LmdbIndex
from .metadata import Metadata
from .output_storage import Hdf5ShardStore


class Store:
    '''Link phraser keys to stored model outputs.'''

    def __init__(self, root, max_shard_size_bytes=1_000_000_000,
        index=None, storage=None):
        '''Initialize a store.
        root:                  store root directory
        max_shard_size_bytes:  HDF5 shard size cap
        index:                 optional LMDB index instance
        storage:               optional payload storage instance
        '''
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = index or LmdbIndex(self.root / 'index.lmdb')
        self.storage = storage or Hdf5ShardStore(self.root / 'shards',
            max_shard_size_bytes=max_shard_size_bytes)

    def put(self, phraser_key, collar, model_name, output_type, layer,
        data, tags=None, to_vector_version=None):
        '''Store one output payload and index its metadata.
        phraser_key:          unique phraser object key
        collar:               collar in milliseconds
        model_name:           model identifier
        output_type:          hidden_state, attention, or codebook_indices
        layer:                model layer index
        data:                 payload to store
        tags:                 optional grouping labels
        to_vector_version:    optional debug-only version marker
        '''
        metadata = Metadata(phraser_key=phraser_key,
            collar=collar, model_name=model_name,
            output_type=output_type, layer=layer,
            tags=tags, to_vector_version=to_vector_version)
        stored = self.storage.store(metadata, data=data)
        return self.index.upsert(stored)

    def find(self, phraser_key, model_name=None, output_type=None,
        layer=None, include_deleted=False):
        '''List metadata records for one phraser key.
        phraser_key:       unique phraser object key
        model_name:        optional model filter
        output_type:       optional output type filter
        layer:             optional layer filter
        include_deleted:   include tombstoned entries
        '''
        return self.index.find(phraser_key=phraser_key,
            model_name=model_name, output_type=output_type, layer=layer,
            include_deleted=include_deleted)

    def find_one(self, phraser_key, collar, model_name, output_type,
        layer, match='exact'):
        '''Find one matching metadata record.
        phraser_key:    unique phraser object key
        collar:         requested collar in milliseconds
        model_name:     model identifier
        output_type:    output type to match
        layer:          layer to match
        match:          exact, min, max, or nearest
        '''
        return self.index.find_one(phraser_key=phraser_key,
            model_name=model_name, output_type=output_type, layer=layer,
            collar=collar, match=match)

    def exists(self, phraser_key, collar, model_name, output_type,
        layer, match='exact'):
        '''Return whether a matching output is stored.
        phraser_key:    unique phraser object key
        collar:         requested collar in milliseconds
        model_name:     model identifier
        output_type:    output type to match
        layer:          layer to match
        match:          exact, min, max, or nearest
        '''
        return self.find_one(phraser_key=phraser_key,
            collar=collar, model_name=model_name,
            output_type=output_type, layer=layer,
            match=match) is not None

    def load(self, phraser_key, collar, model_name, output_type, layer,
        match='exact'):
        '''Load one stored output payload.
        phraser_key:    unique phraser object key
        collar:         requested collar in milliseconds
        model_name:     model identifier
        output_type:    output type to match
        layer:          layer to match
        match:          exact, min, max, or nearest
        '''
        metadata = self.find_one(phraser_key=phraser_key,
            collar=collar, model_name=model_name,
            output_type=output_type, layer=layer, match=match)
        if metadata is None:
            raise ValueError('no stored output matched the requested criteria')
        return self.storage.load(metadata)

    def delete(self, phraser_key, collar, model_name, output_type, layer,
        match='exact'):
        '''Delete one stored output from live indexes.
        phraser_key:    unique phraser object key
        collar:         requested collar in milliseconds
        model_name:     model identifier
        output_type:    output type to match
        layer:          layer to match
        match:          exact, min, max, or nearest
        '''
        metadata = self.find_one(phraser_key=phraser_key,
            collar=collar, model_name=model_name,
            output_type=output_type, layer=layer, match=match)
        if metadata is None:
            return None
        self.storage.delete(metadata)
        return self.index.delete(metadata)

    def find_by_tag(self, tag, include_deleted=False):
        '''List metadata records for one tag.
        tag:               grouping label
        include_deleted:   include tombstoned entries
        '''
        return self.index.find_by_tag(tag,
            include_deleted=include_deleted)

    def add_tags(self, entry_id, tags):
        '''Add tags to one metadata record.
        entry_id:    metadata identifier
        tags:        grouping labels to add
        '''
        return self.index.add_tags(entry_id, tags)

    def remove_tags(self, entry_id, tags):
        '''Remove tags from one metadata record.
        entry_id:    metadata identifier
        tags:        grouping labels to remove
        '''
        return self.index.remove_tags(entry_id, tags)

    def find_or_compute(self, phraser_key, collar, model_name,
        output_type, layer, compute, match='exact',
        tags=None, to_vector_version=None):
        '''Load metadata if present, otherwise compute and store a payload.
        phraser_key:          unique phraser object key
        collar:               requested collar in milliseconds
        model_name:           model identifier
        output_type:          output type to match
        layer:                layer to match
        compute:              callback that returns the payload
        match:                exact, min, max, or nearest
        tags:                 optional grouping labels
        to_vector_version:    optional debug-only version marker
        '''
        metadata = self.find_one(phraser_key=phraser_key,
            collar=collar, model_name=model_name,
            output_type=output_type, layer=layer, match=match)
        if metadata is not None:
            return metadata, False
        data = compute()
        metadata = self.put(phraser_key=phraser_key,
            collar=collar, model_name=model_name,
            output_type=output_type, layer=layer, data=data, tags=tags,
            to_vector_version=to_vector_version)
        return metadata, True
