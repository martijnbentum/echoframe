'''Public store facade for echoframe.'''

from pathlib import Path
from progressbar import progressbar

from . import compaction
from .codebooks import Codevector, Codevectors
from .embeddings import Embedding, Embeddings
from . import model_loader
from . import util_formatting
from .index import LmdbIndex
from .key_helper import pack_echoframe_key
from .metadata import EchoframeMetadata, filter_metadata
from .metadata import validate_metadata_for_save
from .model_registry import ModelMetadata, ModelRegistry
from .output_storage import Hdf5ShardStore
from .phraser_registry import phraser_source_id_to_phraser_source


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
        shards_root = self.root / 'shards'
        self.config_path = self.root / 'config.json'
        if index is None: 
            index = LmdbIndex(self.root / 'index.lmdb', shards_root=shards_root)
        self.index = index
        if storage is None: 
            storage = Hdf5ShardStore(shards_root, max_shard_size_bytes)
        self.storage = storage 
        self.registry = ModelRegistry(self.config_path)
        self._model = None
        self._model_name = None
        self._attached_phraser_sources = {}

    def __repr__(self):
        m = f'Store(root={str(self.root)})'
        return m

    def __str__(self):
        summary = self.store_summary()
        return util_formatting.format_store_str(summary)

    def register_model(self, model_name, local_path=None, huggingface_id=None,
        language=None, size=None, architecture=None):
        '''Register one model in the store registry.
        model_name:      unique registered model name
        local_path:      optional local model path
        huggingface_id:  optional Hugging Face model identifier
        language:        optional language label
        size:            optional model size label
        architecture:    optional model architecture label
        '''
        return self.registry.register_model(model_name,
            local_path=local_path, huggingface_id=huggingface_id,
            language=language, size=size, architecture=architecture)

    def import_models(self, path):
        '''Import model definitions from a JSON file.

        Validates the entire file before writing anything. Raises ValueError
        if any model name already exists in the store.
        Returns a list of ModelMetadata objects.
        '''
        return self.registry.register_models_from_file(path)

    def load_model_metadata(self, model_name):
        '''Look up a model_metadata record by model name.
        Returns a ModelMetadata object, or None.
        '''
        return self.registry.load_model_metadata(model_name)

    def register_phraser_source(self, source_id, root):
        '''Register one phraser source in this store config.'''
        return self.registry.register_phraser_source(source_id, root)

    def attach_phraser_store(self, source_id, phraser_store, root=None):
        '''Attach an already-open phraser Store for this process.
        Registers the source when it is not already present.
        '''
        source = self.load_phraser_source(source_id)
        if source is None:
            if root is None:
                root = getattr(phraser_store, 'path', None)
            if root is None:
                raise ValueError('root is required for an unregistered '
                    'phraser store')
            source = self.register_phraser_source(source_id, root)
        source._phraser_store = phraser_store
        self._attached_phraser_sources[source_id] = source
        return source

    def load_phraser_source(self, source_id):
        '''Return one registered phraser source, or None.'''
        if source_id in self._attached_phraser_sources:
            return self._attached_phraser_sources[source_id]
        return self.registry.load_phraser_source(source_id)

    def list_phraser_sources(self):
        '''Return all registered phraser source records.'''
        sources = self.registry.list_phraser_sources()
        attached = self._attached_phraser_sources
        by_id = {source.source_id: source for source in sources}
        by_id.update(attached)
        return [by_id[source_id] for source_id in sorted(by_id)]

    def list_phraser_source_ids(self):
        '''Return registered phraser source identifiers.'''
        return [source.source_id for source in self.list_phraser_sources()]

    def load_phraser_object(self, phraser_key, phraser_source_id):
        '''Load one phraser object through a registered phraser source.'''
        source = phraser_source_id_to_phraser_source(self, phraser_source_id)
        return source.open().load(phraser_key)

    def group_metadatas_by_source(self, metadata_list):
        '''Return {source_id: [metadata, ...]} grouped by phraser_source_id.
        Skips metadata records without a phraser_key.
        '''
        groups = {}
        for md in metadata_list:
            if md.phraser_key is None: continue
            if md.phraser_source_id not in groups:
                groups[md.phraser_source_id] = []
            groups[md.phraser_source_id].append(md)
        return groups

    def collect_labels(self, metadata_list):
        '''Return {phraser_key: label} for a list of metadata records.
        Opens each phraser source once.
        '''
        labels = {}
        seen = set()
        source_id_to_metadata = self.group_metadatas_by_source(metadata_list)
        for source_id, metadatas in source_id_to_metadata.items():
            source_store = phraser_source_id_to_phraser_source(self, source_id).open()
            for md in metadatas:
                if md.phraser_key in seen: continue
                seen.add(md.phraser_key)
                obj = source_store.load(md.phraser_key)
                label = getattr(obj, 'label', None) if obj else None
                if label is not None: labels[md.phraser_key] = label
        return labels

    def backfill_phraser_source_id(self, phraser_source_id):
        '''Set phraser_source_id on metadata records that do not have one.
        '''
        source = phraser_source_id_to_phraser_source(self, phraser_source_id)
        metadatas = self.index.all_metadatas(store=self)
        updates = []
        for metadata in progressbar(metadatas):
            if metadata.phraser_key is None: continue
            if metadata.phraser_source_id is not None: continue
            updates.append(metadata.copy(phraser_source_id=source.source_id))
        self.index.save_many(updates)
        return len(updates)

    def load_model(self, model_name, gpu=False, flush_model_cache=False):
        '''Load one registered model and cache only one model.
        model_name:   registered model name
        gpu:          whether the returned model should be on GPU
        '''
        return self._load_cached_model(model_name, gpu, flush_model_cache,
            model_loader.load_model)

    def load_codebook_model(self, model_name, gpu=False,
        flush_model_cache=False):
        '''Load one registered Wav2Vec2 pretraining model for codebooks.
        model_name:   registered model name
        gpu:          whether the returned model should be on GPU
        '''
        return self._load_cached_model(model_name, gpu, flush_model_cache,
            model_loader.load_codebook_model)

    def _load_cached_model(self, model_name, gpu, flush_model_cache,
        load_func):
        if flush_model_cache: self.remove_cached_model()
        model_metadata = self.load_model_metadata(model_name)
        if model_metadata is None:
            raise ValueError(f'model_name is not registered: {model_name!r}')
        if self._model_name == model_name and self._model is None:
            return self._load_cached_model(model_name, gpu, True, load_func)
        if self._model_name != model_name:
            if self._model is not None: self.remove_cached_model()
            model = load_func(model_metadata, gpu=gpu)
            self._model = model
            self._model_name = model_name
            return model
        if gpu and not model_loader.model_is_on_gpu(self._model):
            self._model = model_loader.move_model_to_gpu(self._model) 
        if not gpu and model_loader.model_is_on_gpu(self._model):
            self._model = model_loader.move_model_to_cpu(self._model)
        return self._model

    def remove_cached_model(self):
        if self._model is not None:
            if model_loader.model_is_on_gpu(self._model):
                self._model = model_loader.move_model_to_cpu(self._model)
        self._model = None
        self._model_name = None

    def make_echoframe_key(self, output_type, *, model_name, phraser_key=None,
        layer=None, collar=None):
        '''Build one canonical echoframe key.
        output_type:    output type to encode in the key
        model_name:     registered model name
        phraser_key:    optional phraser object key
        layer:          optional layer index
        collar:         optional collar in milliseconds
        '''
        model_id = self.registry.load_model_id(model_name)
        return pack_echoframe_key(output_type, model_id,
            phraser_key=phraser_key, layer=layer, collar=collar)

    def save(self, echoframe_key, metadata, data):
        '''Store one output payload.
        echoframe_key:  canonical echoframe identifier
        metadata:       metadata record for this payload
        data:           output payload to store
        '''
        validate_metadata_for_save(metadata)
        if data is None:
            raise ValueError('data must not be None for this output type')
        if bytes(echoframe_key) != metadata.echoframe_key:
            message = 'metadata.echoframe_key must match the first argument'
            raise ValueError(message)
        stored_metadata = self.storage.store(metadata, data=data)
        self.index.save(stored_metadata)
        return stored_metadata

    def save_many(self, items):
        '''Store multiple output payloads.
        items:    iterable of save-like keyword mappings
        '''
        prepared = []
        for item in items:
            metadata = item['metadata']
            echoframe_key = item['echoframe_key']
            validate_metadata_for_save(metadata)
            if bytes(echoframe_key) != metadata.echoframe_key:
                message = 'metadata.echoframe_key must match the item key'
                raise ValueError(message)
            if item['data'] is None:
                raise ValueError('data must not be None for this output type')
            prepared_item = {'metadata': metadata, 'data': item['data']}
            prepared.append(prepared_item)
        stored_metadatas = self.storage.store_many(prepared)
        self.index.save_many(stored_metadatas)
        return len(stored_metadatas)

    def load_metadata(self, echoframe_key) :
        '''Load one metadata record by echoframe key.
        echoframe_key:  canonical metadata identifier; bytes or hex string
        '''
        metadata = self.index.load(echoframe_key, store=self)
        return metadata

    def load_many_metadata(self, echoframe_keys, keep_missing=False):
        '''Load multiple metadata records by echoframe key.
        echoframe_keys:  iterable of canonical metadata identifiers;
                         each can be bytes or hex string
        keep_missing:    whether to keep None for missing keys or skip them
        '''
        metadata_list = self.index.load_many(echoframe_keys, store=self,
            keep_missing=keep_missing)
        return metadata_list

    def load(self, echoframe_key):
        '''Load one stored payload by echoframe key.
        echoframe_key:  canonical metadata identifier; bytes or hex string
        '''
        metadata = self.load_metadata(echoframe_key)
        if metadata is None: return None
        return self.storage.load(metadata)

    def load_many(self, echoframe_keys, keep_missing=False):
        '''Load multiple stored output payloads.
        echoframe_keys:  iterable of canonical metadata identifiers
        keep_missing:    whether to keep None for missing keys or skip them

        By default, missing keys are skipped. With keep_missing=True, the
        returned payload list preserves input alignment and contains None for
        missing keys.
        '''
        echoframe_keys = list(echoframe_keys)
        metadata_list = self.load_many_metadata(echoframe_keys, keep_missing)
        if not keep_missing and len(metadata_list) != len(echoframe_keys):
            print('WARNING: some echoframe keys were not found in the index')
        payloads = self.metadatas_to_payloads(metadata_list)
        return payloads

    def load_frame(self, echoframe_key, frame='center'):
        '''Load one frame vector or reduction from a matrix payload.
        echoframe_key:  canonical metadata identifier
        frame:          center, mean, first, or last
        '''
        metadata = self.load_metadata(echoframe_key)
        if metadata is None: return None
        return self.storage.load_frame(metadata, frame=frame)

    def load_many_frames(self, echoframe_keys, frame='center',
        keep_missing=False):
        '''Load frame vectors for multiple matrix payloads.
        echoframe_keys:  iterable of canonical metadata identifiers
        frame:           center, mean, first, or last
        keep_missing:    whether to keep None for missing keys or skip them
        '''
        echoframe_keys = list(echoframe_keys)
        metadata_list = self.load_many_metadata(echoframe_keys, keep_missing)
        if not keep_missing and len(metadata_list) != len(echoframe_keys):
            print('WARNING: some echoframe keys were not found in the index')
        return self.storage.load_many_frames(metadata_list, frame=frame)

    def delete(self, echoframe_key):
        '''Delete one stored payload by echoframe key.
        echoframe_key:  canonical metadata identifier; bytes or hex string
        '''
        metadata = self.load_metadata(echoframe_key)
        if metadata is None: return None
        self._delete_payload(metadata)
        self.index.delete(metadata)

    def _delete_payload(self, metadata):
        try:self.storage.delete(metadata)
        except Exception as e: 
            key = metadata.echoframe_key
            print(f'failed to delete payload for {key}: {e}')

    def delete_many(self, echoframe_keys):
        '''delete multiple stored payloads by echoframe key.'''
        metadata_list = self.load_many_metadata(echoframe_keys)
        for metadata in metadata_list:
            self._delete_payload(metadata)
        self.index.delete_many(metadata_list)

    def metadata_to_payload(self, metadata):
        '''Load one stored output payload from echoframe metadata.
        metadata:    metadata record that points to a stored payload
        '''
        return self.storage.load(metadata)

    def metadatas_to_payloads(self, metadata_list):
        '''Load payloads for multiple echoframe metadata records.
        metadata_list:   iterable of metadata records or None values
        '''
        metadata_list = list(metadata_list)
        return self.storage.load_many(metadata_list)

    def phraser_key_to_echoframe_key(self, phraser_key, model_name, 
        layer = None, collar=500):
        echoframe_key = self.make_echoframe_key('codebook_indices', 
            model_name=model_name, phraser_key=phraser_key, layer=layer, 
            collar=collar)
        return echoframe_key

    def phraser_key_to_embedding(self, phraser_key, model_name, layer,
        collar=500):
        '''Load one hidden-state embedding from a phraser key.'''
        echoframe_key = self.make_echoframe_key('hidden_state',
            model_name=model_name, phraser_key=phraser_key, layer=layer,
            collar=collar)
        return self.load_embedding(echoframe_key)

    def phraser_keys_to_embeddings(self, phraser_keys, model_name, layer,
        collar=500):
        '''Load hidden-state embeddings from multiple phraser keys.'''
        echoframe_keys = [
            self.make_echoframe_key('hidden_state', model_name=model_name,
                phraser_key=phraser_key, layer=layer, collar=collar)
            for phraser_key in phraser_keys
        ]
        return self.load_embeddings(echoframe_keys)

    def load_embedding(self, echoframe_key):
        '''Load one typed Embedding object.
        echoframe_key:  canonical echoframe identifier
        '''
        return Embedding(echoframe_key, self)

    def load_embeddings(self, echoframe_keys):
        '''Load typed Embeddings for multiple echoframe keys.
        echoframe_keys:  canonical echoframe identifiers
        '''
        if not isinstance(echoframe_keys, (list, tuple)):
            raise ValueError('echoframe_keys must be a list or tuple')
        return Embeddings.from_echoframe_keys(self, echoframe_keys)


    def load_codevector(self, echoframe_key):
        '''Load one typed Codevector object.
        echoframe_key:  canonical echoframe identifier
        '''
        return Codevector(echoframe_key, self)

    def load_codevectors(self, echoframe_keys):
        '''Load typed Codevectors for multiple echoframe keys.
        echoframe_keys:  canonical echoframe identifiers
        '''
        if not isinstance(echoframe_keys, (list, tuple)):
            raise ValueError('echoframe_keys must be a list or tuple')
        return Codevectors.from_echoframe_keys(self, echoframe_keys)

    def delete_phraser_key(self, phraser_key, model_name, output_type,
        layer = None, collar = None, collar_match='exact'):
        '''Delete stored outputs linked to phraser_key that match the criteria.
        phraser_key:    unique phraser object key
        collar:         requested collar in milliseconds
        model_name:     model identifier
        output_type:    output type to match
        layer:          layer to match
        collar_match:   how to match collar time: exact, min, max, or nearest
        '''
        metadatas = self.find_phraser(phraser_key)
        matches = filter_metadata(metadatas,model_name=model_name,
             output_type=output_type, layer=layer, collar=collar,
             collar_match=collar_match)
        if not matches: return 
        echoframe_keys = [metadata.echoframe_key for metadata in matches]
        self.delete_many(echoframe_keys)
        print(f'deleted {len(matches)} metadata and payloads key:{phraser_key}')

    def store_summary(self):
        '''Return compact summary stats for this store.'''
        return util_formatting.build_store_summary(self)

    def store_state(self):
        '''Return detailed summary stats for this store.'''
        return util_formatting.build_store_state(self)

    def show_store_state(self):
        '''Return a formatted store state summary.'''
        state = self.store_state()
        return util_formatting.format_store_state(state)

    def find_phraser(self, phraser_key):
        '''List metadata records for one phraser key.
        phraser_key:       unique phraser object key
        '''
        metadatas = self.index.find_phraser(phraser_key=phraser_key, store=self)
        return metadatas

    def find_by_tag(self, tag):
        '''List metadata records for one tag.
        tag:               grouping label
        '''
        metadatas = self.index.find_by_tag(tag, store=self)
        return metadatas

    def find_by_tags(self, tags, match='all'):
        '''List metadata records that match a tag set.
        tags:              grouping labels to match
        match:             require all or any tags
        '''
        metadatas = self.index.find_by_tags(tags, match=match, store=self)
        return metadatas

    def find_by_label(self, label, model_name=None, output_type=None,
        layer=None, collar=None, collar_match='exact'):
        '''List metadata records for one phraser object label.
        label:             phraser object label to match
        model_name:        optional model filter
        output_type:       optional output type filter
        layer:             optional layer filter
        '''
        if not isinstance(label, str) or not label.strip():
            raise ValueError('label must be a non-empty string')

        metadatas = self.index.all_metadatas(store=self)
        metadatas = filter_metadata(metadatas, model_name=model_name,
            output_type=output_type, layer=layer, collar=collar, 
            collar_match=collar_match)
        if len(metadatas) == 0: return None
        labels_by_key = self.collect_labels(metadatas)
        return [md for md in metadatas if labels_by_key.get(md.phraser_key) == label]

    def list_tags(self):
        '''List all known tags.
        '''
        return self.index.list_tags()

    def tag_counts(self):
        '''Count metadata records per tag.
        '''
        return self.index.tag_counts()

    def add_tags(self, echoframe_key, tags):
        '''Add tags to one metadata record.
        echoframe_key:  canonical metadata identifier
        tags:        grouping labels to add
        '''
        metadata = self.index.add_tags(echoframe_key, tags, self)
        return metadata

    def add_tags_many(self, echoframe_keys, tags):
        '''Add tags to multiple metadata records.
        echoframe_keys:  canonical metadata identifiers
        tags:        grouping labels to add
        '''
        metadata_list = self.index.add_tags_many(echoframe_keys, tags, self)
        return metadata_list

    def remove_tags(self, echoframe_key, tags):
        '''Remove tags from one metadata record.
        echoframe_key:  canonical metadata identifier
        tags:        grouping labels to remove
        '''
        metadata = self.index.remove_tags(echoframe_key, tags, self)
        return metadata

    def remove_tags_many(self, echoframe_keys, tags):
        '''Remove tags from multiple metadata records.
        echoframe_keys:  canonical metadata identifiers
        tags:        grouping labels to remove
        '''
        metadata_list = self.index.remove_tags_many(echoframe_keys, tags, self)
        return metadata_list

    def shard_stats(self):
        '''Return shard-level stats.'''
        return self.index.list_shard_metadata()

    @property
    def metadatas(self):
        '''Return stored metadata records.'''
        if not hasattr(self, '_metadatas'):
            self._metadatas = self.index.all_metadatas(store=self)
        return self._metadatas

    def overview(self, health_event_limit=20,include_integrity=False):
        '''Return a compact overview of store contents.
        health_event_limit:   recent shard health events to include
        include_integrity:    run a full integrity scan
        '''
        metadatas = self.metadatas
        data = {}
        data['metadata_count'] = len(metadatas)
        data['metadatas'] = [metadata.to_dict() for metadata in metadatas]
        data['shard_count'] = len(self.index.list_shards())
        data['shards'] = self.shard_stats()
        data['tags'] = self.list_tags()
        if include_integrity:
            data['integrity'] = self.verify_integrity()
        else:
            data['integrity'] = None
        data['recent_shard_health_events'] = self.get_shard_health_events(
            limit=health_event_limit)
        return data

    def get_shard_health_events(self, limit=None):
        '''Return recent shard health events from storage.'''
        if not hasattr(self.storage, 'get_shard_health_events'): return []
        return self.storage.get_shard_health_events(limit=limit)

    def verify_integrity(self):
        '''Verify that metadata records point to existing datasets.'''
        return compaction.verify_integrity(self)

    def compact_shards(self, shard_ids=None, dry_run=False,
        resume_pending=False):
        '''Compact shard files by rewriting existing shard contents.
        shard_ids:          optional shard identifiers to compact
        dry_run:            report what would be compacted
        resume_pending:     resume running journal entries first
        '''
        return compaction.compact_shards(self, shard_ids=shard_ids,
            dry_run=dry_run, resume_pending=resume_pending)

    def resume_compactions(self):
        '''Resume interrupted shard compactions.'''
        return compaction.resume_compactions(self)

    def compaction_journal(self, status=None):
        '''Return compaction journal records.'''
        return compaction.compaction_journal(self, status=status)

    def _storage_bytes(self):
        '''Return total bytes used by shard files.'''
        shards_root = self.root / 'shards'
        if not shards_root.exists(): return 0
        file_sizes = []
        for filename in shards_root.rglob('*'):
            if filename.is_file():
                file_sizes.append(filename.stat().st_size)
        return sum(file_sizes)

    def _make_metadata(self, echoframe_key, tags=None,
        phraser_source_id=None):
        metadata = EchoframeMetadata(echoframe_key=echoframe_key, tags=tags,
            store=self, phraser_source_id=phraser_source_id)
        return metadata

def _load_phraser_models_module():
    try:
        from phraser import models
    except ImportError as exc:
        message = 'phraser is required to find metadata by label'
        raise ImportError(message) from exc
    return models
