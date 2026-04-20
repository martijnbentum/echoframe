'''Public store facade for echoframe.'''

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import compaction
from . import util_formatting
from .index import LmdbIndex
from .key_helper import pack_echoframe_key
from .metadata import EchoframeMetadata, filter_metadata
from .metadata import metadata_class_for_output_type, utc_now
from .model_registry import ModelMetadata, ModelRegistry
from .output_storage import Hdf5ShardStore
from .typed_loaders import load_codebook, load_embeddings
from .typed_loaders import load_many_codebooks, load_many_embeddings


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

    def get_model_metadata(self, model_name):
        '''Look up a model_metadata record by model name.
        Returns a ModelMetadata object, or None.
        '''
        return self.registry.load_model_metadata(model_name)

    def make_echoframe_key(self, output_type, *, model_name, phraser_key=None,
        layer=None, collar=None):
        '''Build one canonical echoframe key.
        output_type:    output type to encode in the key
        model_name:     registered model name
        phraser_key:    optional phraser object key
        layer:          optional layer index
        collar:         optional collar in milliseconds
        '''
        record = self.registry.load_model_metadata(model_name)
        if record is None:
            raise ValueError(f'model_name is not registered: {model_name!r}')
        return pack_echoframe_key(output_type, record.model_id,
            phraser_key=phraser_key, layer=layer, collar=collar)

    def put(self, echoframe_key, metadata_obj, data):
        '''Store one output payload.
        echoframe_key:  canonical echoframe identifier
        metadata_obj:   metadata record for this payload
        data:           output payload to store
        '''
        if not isinstance(metadata_obj, EchoframeMetadata):
            raise ValueError('metadata must be an EchoframeMetadata')
        if data is None:
            raise ValueError('data must not be None for this output type')
        if bytes(echoframe_key) != metadata_obj.echoframe_key:
            message = 'metadata.echoframe_key must match the first argument'
            raise ValueError(message)
        stored = self.storage.store(metadata_obj, data=data)
        indexed = self.index.upsert(stored)
        return self._bind_metadata(indexed)

    def put_many(self, items):
        '''Store multiple output payloads.
        items:    iterable of put-like keyword mappings
        '''
        prepared = []
        for item in items:
            metadata_obj = item['metadata']
            echoframe_key = item['echoframe_key']
            if not isinstance(metadata_obj, EchoframeMetadata):
                raise ValueError('metadata must be an EchoframeMetadata')
            if bytes(echoframe_key) != metadata_obj.echoframe_key:
                message = 'metadata.echoframe_key must match the item key'
                raise ValueError(message)
            if item['data'] is None:
                raise ValueError('data must not be None for this output type')
            prepared_item = {'metadata': metadata_obj, 'data': item['data']}
            prepared.append(prepared_item)
        stored = self.storage.store_many(prepared)
        indexed = self.index.upsert_many(stored)
        return self._bind_metadatas(indexed)

    def find_phraser(self, phraser_key, include_deleted=False):
        '''List metadata records for one phraser key.
        phraser_key:       unique phraser object key
        include_deleted:   include tombstoned entries
        '''
        records = self.index.find_phraser(phraser_key=phraser_key,
            include_deleted=include_deleted)
        return self._bind_metadatas(records)

    def load_metadata(self, echoframe_key):
        '''Load one metadata record by echoframe key.'''
        metadata_obj = self.index.get(echoframe_key)
        return self._bind_metadata(metadata_obj)

    def load(self, echoframe_key):
        '''Load one stored payload by echoframe key.'''
        metadata_obj = self.load_metadata(echoframe_key)
        if metadata_obj is None:
            message = 'no stored output matched the requested echoframe key'
            raise ValueError(message)
        self._touch_accessed_at(metadata_obj)
        return self.storage.load(metadata_obj)

    def metadata_to_payload(self, metadata_obj):
        '''Load one stored output payload from echoframe metadata.
        metadata:    metadata record that points to a stored payload
        '''
        return self.storage.load(metadata_obj)

    def load_embeddings(self, phraser_key, collar, model_name, layers,
        frame_aggregation=None):
        '''Load one typed Embeddings object.
        phraser_key:         unique phraser object key
        collar:              collar in milliseconds
        model_name:          registered model name
        layers:              layer indexes to load
        frame_aggregation:   optional frame aggregation mode
        '''
        embeddings = load_embeddings(self, phraser_key, collar, model_name,
            layers, frame_aggregation=frame_aggregation)
        return embeddings

    def load_many_embeddings(self, requests):
        '''Load many typed Embeddings objects.'''
        return load_many_embeddings(self, requests)

    def load_codebook(self, phraser_key, collar, model_name):
        '''Load one typed Codebook object.
        phraser_key:  unique phraser object key
        collar:       collar in milliseconds
        model_name:   registered model name
        '''
        codebook = load_codebook(self, phraser_key, collar, model_name)
        return codebook

    def load_many_codebooks(self, requests):
        '''Load many typed Codebook objects.'''
        return load_many_codebooks(self, requests)

    def load_many(self, echoframe_keys, strict=False):
        '''Load multiple stored output payloads.
        echoframe_keys:  iterable of canonical metadata identifiers
        strict:          raise when any key does not match
        '''
        metadata_list = self.get_many_metadata(echoframe_keys)
        payloads = self.metadatas_to_payloads(metadata_list, strict=strict)
        return payloads

    def get_many_metadata(self, echoframe_keys):
        '''Load multiple metadata records by echoframe key.'''
        metadata_list = self.index.get_many(echoframe_keys)
        return self._bind_metadatas(metadata_list)

    def metadatas_to_payloads(self, metadata_list, strict=False):
        '''Load payloads for multiple echoframe metadata records.
        metadata_list:   iterable of metadata records or None values
        strict:          raise when any metadata item is None
        '''
        payloads = []
        for index, metadata_obj in enumerate(metadata_list):
            if metadata_obj is None and strict:
                message = 'no stored output matched one of the requested '
                message += 'echoframe key at index {index}, {echoframe_key}'
                raise ValueError(message)
            if metadata_obj is None: 
                print(f'WARNING: no output for echoframe key at index {index}')
                continue
            self._touch_accessed_at(metadata_obj)
            payload = self.storage.load(metadata_obj)
            payloads.append(payload)

        return payloads

    def phraser_key_to_output(self, phraser_key, model_name, layer, collar=500,
        output_type='hidden_state', match='exact'):
        '''Load frame outputs for one object.
        phraser_key:    unique phraser object key
        model_name:     model identifier
        layer:          layer to match
        collar:         exact collar or None for all collars
        output_type:    output type to match
        match:          exact, min, max, or nearest
        '''
        phraser_records = self.find_phraser(phraser_key)
        matches = filter_metadata(phraser_records,
            model_name=model_name, output_type=output_type, layer=layer,
            collar=collar, match=match)
        if not matches:
            raise ValueError('no stored output matched the requested criteria')
        if len(matches) > 1:
            print(f'{len(matches)} stored outputs matched the criteria')
            print(f'returning echoframe_key {matches[0].echoframe_key}')
        return self.load(matches[0].echoframe_key), matches[0]

    def phraser_key_to_outputs(self, phraser_key, model_name, layer = None,
        collar=None, output_type='hidden_state', match='exact'):
        '''Iterate lazily over frame outputs for one object.
        phraser_key:    unique phraser object key
        model_name:     model identifier
        layer:          layer to match
        collar:         exact collar, matched collar, or None for all
        output_type:    output type to match
        match:          exact, min, max, or nearest
        '''
        phraser_records = self.find_phraser(phraser_key)
        matches = filter_metadata(phraser_records,
            model_name=model_name, output_type=output_type, layer=layer,
            collar=collar, match=match)
        if not matches:
            raise ValueError('no stored output matched the requested criteria')
        keys = [metadata_obj.echoframe_key for metadata_obj in matches]
        outputs = self.load_many(keys)
        return outputs, matches

    def delete_phraser_key(self, phraser_key, model_name, output_type,
        layer = None, collar = None, match='exact'):
        '''Delete stored outputs linked to phraser_key that match the criteria.
        phraser_key:    unique phraser object key
        collar:         requested collar in milliseconds
        model_name:     model identifier
        output_type:    output type to match
        layer:          layer to match
        match:          exact, min, max, or nearest
        '''
        phraser_records = self.find_phraser(phraser_key)
        matches = filter_metadata(phraser_records,
            model_name=model_name, output_type=output_type, layer=layer,
            collar=collar, match=match)
        if not matches: return None
        for metadata_obj in matches:
            self.storage.delete(metadata_obj)
            self.index.delete(metadata_obj)
        print(f'deleted {len(matches)} entries for phraser_key {phraser_key}')

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

    def find_by_tag(self, tag, include_deleted=False):
        '''List metadata records for one tag.
        tag:               grouping label
        include_deleted:   include tombstoned entries
        '''
        records = self.index.find_by_tag(tag,
            include_deleted=include_deleted)
        return self._bind_metadatas(records)

    def find_by_tags(self, tags, match='all', include_deleted=False):
        '''List metadata records that match a tag set.
        tags:              grouping labels to match
        match:             require all or any tags
        include_deleted:   include tombstoned entries
        '''
        records = self.index.find_by_tags(tags, match=match,
            include_deleted=include_deleted)
        return self._bind_metadatas(records)

    def find_by_label(self, label, model_name=None, output_type=None,
        layer=None, include_deleted=False, collar=None, match='exact'):
        '''List metadata records for one phraser object label.
        label:             phraser object label to match
        model_name:        optional model filter
        output_type:       optional output type filter
        layer:             optional layer filter
        include_deleted:   include tombstoned entries
        '''
        if not isinstance(label, str) or not label.strip():
            raise ValueError('label must be a non-empty string')

        records = self.index.list_entries(include_deleted=include_deleted)
        records = filter_metadata(records, model_name=model_name,
            output_type=output_type, layer=layer, collar=collar, match=match)
        if len(records) == 0: return None
        phraser_models= _load_phraser_models_module()
            
        labels_by_key = {}
        matches = []
        for record in records:
            phraser_key = record.phraser_key
            if phraser_key not in labels_by_key:
                labels_by_key[phraser_key] = record.label
            if labels_by_key[phraser_key] == label:
                matches.append(record)
        return self._bind_metadatas(matches)

    def list_tags(self, include_deleted=False):
        '''List all known tags.
        include_deleted:   include tombstoned entries
        '''
        return self.index.list_tags(include_deleted=include_deleted)

    def tag_counts(self, include_deleted=False):
        '''Count metadata records per tag.
        include_deleted:   include tombstoned entries
        '''
        return self.index.tag_counts(include_deleted=include_deleted)

    def add_tags(self, echoframe_key, tags):
        '''Add tags to one metadata record.
        echoframe_key:  canonical metadata identifier
        tags:        grouping labels to add
        '''
        metadata_obj = self.index.add_tags(echoframe_key, tags)
        return self._bind_metadata(metadata_obj)

    def add_tags_many(self, echoframe_keys, tags):
        '''Add tags to multiple metadata records.
        echoframe_keys:  canonical metadata identifiers
        tags:        grouping labels to add
        '''
        metadata_list = self.index.add_tags_many(echoframe_keys, tags)
        return self._bind_metadatas(metadata_list)

    def remove_tags(self, echoframe_key, tags):
        '''Remove tags from one metadata record.
        echoframe_key:  canonical metadata identifier
        tags:        grouping labels to remove
        '''
        metadata_obj = self.index.remove_tags(echoframe_key, tags)
        return self._bind_metadata(metadata_obj)

    def remove_tags_many(self, echoframe_keys, tags):
        '''Remove tags from multiple metadata records.
        echoframe_keys:  canonical metadata identifiers
        tags:        grouping labels to remove
        '''
        metadata_list = self.index.remove_tags_many(echoframe_keys, tags)
        return self._bind_metadatas(metadata_list)

    def shard_stats(self):
        '''Return shard-level stats.'''
        return self.index.list_shard_metadata()

    def list_entries(self, include_deleted=False):
        '''List stored metadata records across all shards.
        include_deleted:   include tombstoned entries
        '''
        entries = self.index.list_entries(include_deleted=include_deleted)
        entries.sort(key=lambda metadata: (metadata.phraser_key,
            metadata.model_name, metadata.output_type, metadata.layer,
            metadata.collar))
        bound_entries = self._bind_metadatas(entries)
        return bound_entries

    @property
    def metadata(self):
        '''Return stored metadata records.'''
        if not hasattr(self, '_metadata'):
            self._metadata = self.list_entries()
        return self._metadata

    def overview(self, include_deleted=False, health_event_limit=20,
        include_integrity=False):
        '''Return a compact overview of store contents.
        include_deleted:      include tombstoned entries
        health_event_limit:   recent shard health events to include
        include_integrity:    run a full integrity scan
        '''
        entries = self.list_entries(include_deleted=include_deleted)
        data = {}
        data['entry_count'] = len(entries)
        data['entries'] = [metadata.to_dict() for metadata in entries]
        data['shard_count'] = len(self.index.list_shards())
        data['shards'] = self.shard_stats()
        data['tags'] = self.list_tags(include_deleted=include_deleted)
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

    def find_or_compute(self, phraser_key, collar, model_name,
        output_type, layer, compute, match='exact',
        tags=None, add_tags_on_hit=False):
        '''Load metadata if present, otherwise compute and store a payload.
        phraser_key:          unique phraser object key
        collar:               requested collar in milliseconds
        model_name:           model identifier
        output_type:          output type to match
        layer:                layer to match
        compute:              callback that returns the payload
        match:                exact, min, max, or nearest
        tags:                 optional grouping labels
        add_tags_on_hit:      add tags to an existing matching entry
        '''
        phraser_records = self.find_phraser(phraser_key)
        matches = filter_metadata(phraser_records,
            model_name=model_name, output_type=output_type, layer=layer,
            collar=collar, match=match)
        if matches:
            metadata_obj = matches[0]
            if tags and add_tags_on_hit:
                metadata_obj = self.add_tags(metadata_obj.echoframe_key, tags)
            return metadata_obj, False
        data = compute()
        metadata_obj = self._make_metadata(phraser_key, collar, model_name,
            output_type, layer, tags=tags)
        metadata_obj = self.put(metadata_obj.echoframe_key, metadata_obj, data)
        return metadata_obj, True

    def evict_by_recency(self):
        '''Soft-delete entries older than the recency window until under budget.

        Reads ECHOFRAME_RECENCY_WINDOW_DAYS (default 30) and
        ECHOFRAME_STORAGE_BUDGET_GB (default no limit) from the environment.
        Entries without accessed_at are skipped. Oldest entries are deleted
        first until storage is within the budget.
        '''
        window_days_text = os.environ.get('ECHOFRAME_RECENCY_WINDOW_DAYS', 30)
        window_days = int(window_days_text)
        budget_gb = os.environ.get('ECHOFRAME_STORAGE_BUDGET_GB')
        budget_bytes = float(budget_gb) * 1_000_000_000 if budget_gb else None

        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

        if budget_bytes is not None and self._storage_bytes() <= budget_bytes: 
            return []

        entries = self.list_entries()
        stale = []
        for metadata_obj in entries:
            if metadata_obj.accessed_at is None:
                continue
            accessed = datetime.fromisoformat(metadata_obj.accessed_at)
            if accessed < cutoff:
                stale.append(metadata_obj)

        stale.sort(key=lambda m: m.accessed_at)

        evicted = []
        for metadata_obj in stale:
            if budget_bytes is not None:
                if self._storage_bytes() <= budget_bytes:
                    break
            self.storage.delete(metadata_obj)
            self.index.delete(metadata_obj)
            evicted.append(metadata_obj)
        return evicted

    def verify_integrity(self):
        '''Verify that live metadata records point to existing datasets.'''
        broken = []
        checked = 0
        for shard_id in self.index.list_shards():
            for metadata_obj in self.index.entries_for_shard(shard_id):
                checked += 1
                if metadata_obj.shard_id is None:
                    broken_item = self._broken_reference(metadata_obj,
                        reason='missing shard pointer')
                    broken.append(broken_item)
                    continue
                if metadata_obj.dataset_path is None:
                    broken_item = self._broken_reference(metadata_obj,
                        reason='missing shard pointer')
                    broken.append(broken_item)
                    continue
                if not self.storage.dataset_exists(metadata_obj.shard_id,
                    metadata_obj.dataset_path):
                    broken_item = self._broken_reference(metadata_obj,
                        reason='missing dataset')
                    broken.append(broken_item)
        data = {}
        data['ok'] = not broken
        data['checked_entries'] = checked
        data['broken_references'] = broken
        return data

    def compact_shards(self, shard_ids=None, dry_run=False,
        resume_pending=False):
        '''Compact shard files by removing deleted payloads.
        shard_ids:          optional shard identifiers to compact
        dry_run:            report what would be compacted
        resume_pending:     resume running journal entries first
        '''
        if shard_ids is None: shard_ids = self.index.list_shards()
        if resume_pending: self.resume_compactions()

        compacted = []
        plans = []
        for shard_id in shard_ids:
            plan = self._compaction_plan(shard_id)
            if not plan['needs_compaction']: continue
            if dry_run:
                plans.append(plan)
                continue
            compacted_plan = self._run_compaction_plan(plan)
            compacted.append(compacted_plan)
        return plans if dry_run else compacted

    def resume_compactions(self):
        '''Resume interrupted shard compactions.'''
        completed = []
        for record in self.index.list_compaction_journal(status='running'):
            completed_plan = self._run_compaction_plan(record,
                from_journal=True)
            completed.append(completed_plan)
        return completed

    def compaction_journal(self, status=None):
        '''Return compaction journal records.'''
        return self.index.list_compaction_journal(status=status)

    def _storage_bytes(self):
        '''Return total bytes used by shard files.'''
        shards_root = self.root / 'shards'
        if not shards_root.exists(): return 0
        file_sizes = []
        for filename in shards_root.rglob('*'):
            if filename.is_file():
                file_sizes.append(filename.stat().st_size)
        return sum(file_sizes)

    def _bind_metadata(self, metadata_obj):
        if metadata_obj is None: return None
        return metadata_obj.bind_store(self)

    def _bind_metadatas(self, metadata_list):
        bound = []
        for metadata_obj in metadata_list:
            bound_metadata = self._bind_metadata(metadata_obj)
            bound.append(bound_metadata)
        return bound

    def _make_metadata(self, phraser_key, collar, model_name, output_type,
        layer, tags=None):
        metadata_cls = metadata_class_for_output_type(output_type)
        echoframe_key = self.make_echoframe_key(output_type,
            model_name=model_name, phraser_key=phraser_key, layer=layer,
            collar=collar)
        return metadata_cls(phraser_key=phraser_key, collar=collar,
            model_name=model_name, layer=layer, tags=tags,
            echoframe_key=echoframe_key)

    def _touch_accessed_at(self, metadata_obj):
        '''Update accessed_at on a metadata record and persist to index.'''
        updated = metadata_obj.with_accessed_at(utc_now())
        self.index.upsert(updated)
        return updated

    def _broken_reference(self, metadata, reason):
        return compaction.broken_reference(metadata, reason)

    def _build_shard_health_report(self, shard_id, error):
        return compaction.build_shard_health_report(self.index,
            self.storage, shard_id, error)

    def _compaction_plan(self, shard_id):
        plan = compaction.build_compaction_plan(self.index, self.storage,
            shard_id)
        return plan

    def _run_compaction_plan(self, plan, from_journal=False):
        return compaction.run_compaction_plan(self.index, self.storage,
            plan, from_journal=from_journal)


def _load_phraser_models_module():
    try:
        from phraser import models
    except ImportError as exc:
        message = 'phraser is required to find entries by label'
        raise ImportError(message) from exc
    return models
