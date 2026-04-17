'''Public store facade for echoframe.'''

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import compaction
from .index import LmdbIndex
from .key_helper import pack_echoframe_key
from .metadata import (
    EchoframeMetadata,
    filter_metadata,
    metadata_class_for_output_type,
    utc_now,
)
from .model_registry import ModelMetadata, ModelRegistry
from .output_storage import Hdf5ShardStore
from .typed_loaders import (
    load_codebook as _load_codebook,
    load_embeddings as _load_embeddings,
    load_many_codebooks as _load_many_codebooks,
    load_many_embeddings as _load_many_embeddings,
)


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
        self.index = index or LmdbIndex(self.root / 'index.lmdb',
            shards_root=shards_root)
        self.storage = storage or Hdf5ShardStore(shards_root,
            max_shard_size_bytes=max_shard_size_bytes)
        self.registry = ModelRegistry(self.config_path)

    def __repr__(self):
        m = f'Store(root={str(self.root)})'
        return m

    def _bind_metadata(self, metadata):
        if metadata is None:
            return None
        return metadata.bind_store(self)

    def _bind_metadatas(self, metadata_list):
        return [self._bind_metadata(metadata) for metadata in metadata_list]

    def _make_metadata(self, phraser_key, collar, model_name, output_type,
        layer, tags=None):
        metadata_cls = metadata_class_for_output_type(output_type)
        echoframe_key = self.make_echoframe_key(output_type,
            model_name=model_name, phraser_key=phraser_key, layer=layer,
            collar=collar)
        return metadata_cls(phraser_key=phraser_key, collar=collar,
            model_name=model_name, layer=layer, tags=tags,
            echoframe_key=echoframe_key)

    def register_model(self, model_name, local_path=None,
        huggingface_id=None, language=None, size=None):
        '''Register one model in the store registry.

        model_name:   str model identifier
        Returns a ModelMetadata object.
        Raises ValueError if model_name is already registered.
        '''
        return self.registry.register_model(model_name,
            local_path=local_path, huggingface_id=huggingface_id,
            language=language, size=size)

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
        '''Build a canonical echoframe_key from user-facing parameters.'''
        record = self.registry.load_model_metadata(model_name)
        if record is None:
            raise ValueError(f'model_name is not registered: {model_name!r}')
        kwargs = {
            'output_type': output_type,
            'model_id': record.model_id,
        }
        if output_type in {'hidden_state', 'attention'}:
            kwargs.update({
                'layer': layer,
                'phraser_key': phraser_key,
                'collar': collar,
            })
        elif output_type == 'codebook_indices':
            kwargs.update({
                'phraser_key': phraser_key,
                'collar': collar,
            })
        elif output_type == 'codebook_matrix':
            pass
        else:
            raise ValueError(f'unknown output type: {output_type!r}')
        return pack_echoframe_key(**kwargs)

    def put(self, echoframe_key, metadata, data):
        '''Store one output payload.

        Canonical form:
            store.put(echoframe_key, metadata, data)
        '''
        if not isinstance(metadata, EchoframeMetadata):
            raise ValueError('metadata must be an EchoframeMetadata')
        if data is None:
            raise ValueError('data must not be None for this output type')
        if bytes(echoframe_key) != metadata.echoframe_key:
            raise ValueError(
                'metadata.echoframe_key must match the first argument')
        stored = self.storage.store(metadata, data=data)
        return self._bind_metadata(self.index.upsert(stored))

    def put_many(self, items):
        '''Store multiple output payloads.
        items:    iterable of put-like keyword mappings
        '''
        prepared = []
        for item in items:
            metadata = item['metadata']
            echoframe_key = item['echoframe_key']
            if not isinstance(metadata, EchoframeMetadata):
                raise ValueError('metadata must be an EchoframeMetadata')
            if bytes(echoframe_key) != metadata.echoframe_key:
                raise ValueError(
                    'metadata.echoframe_key must match the item key')
            if item['data'] is None:
                raise ValueError('data must not be None for this output type')
            prepared.append({
                'metadata': metadata,
                'data': item['data'],
            })
        stored = self.storage.store_many(prepared)
        return self._bind_metadatas(self.index.upsert_many(stored))

    def find_phraser(self, phraser_key, include_deleted=False):
        '''List metadata records for one phraser key.'''
        return self._bind_metadatas(self.index.find_phraser(
            phraser_key=phraser_key, include_deleted=include_deleted))

    def _touch_accessed_at(self, metadata):
        '''Update accessed_at on a metadata record and persist to index.'''
        updated = metadata.with_accessed_at(utc_now())
        self.index.upsert(updated)
        return updated

    def load_metadata(self, echoframe_key):
        '''Load one metadata record by echoframe key.'''
        return self._bind_metadata(self.index.get(echoframe_key))

    def load(self, echoframe_key):
        '''Load one stored payload by echoframe key.'''
        metadata = self.load_metadata(echoframe_key)
        if metadata is None:
            raise ValueError(
                'no stored output matched the requested echoframe key')
        self._touch_accessed_at(metadata)
        return self.storage.load(metadata)

    def metadata_to_payload(self, metadata):
        '''Load one stored output payload from echoframe metadata.
        metadata:    metadata record that points to a stored payload
        '''
        return self.storage.load(metadata)

    def load_embeddings(self, phraser_key, collar, model_name, layers,
        frame_aggregation=None):
        '''Load one typed Embeddings object.'''
        return _load_embeddings(self, phraser_key, collar, model_name,
            layers, frame_aggregation=frame_aggregation)

    def load_many_embeddings(self, requests):
        '''Load many typed Embeddings objects.'''
        return _load_many_embeddings(self, requests)

    def load_codebook(self, phraser_key, collar, model_name):
        '''Load one typed Codebook object.'''
        return _load_codebook(self, phraser_key, collar, model_name)

    def load_many_codebooks(self, requests):
        '''Load many typed Codebook objects.'''
        return _load_many_codebooks(self, requests)

    def load_many(self, echoframe_keys, strict=False):
        '''Load multiple stored output payloads.
        echoframe_keys: iterable of canonical metadata identifiers
        strict:     raise when any key does not match
        '''
        metadata_list = self.get_many_metadata(echoframe_keys)
        if not strict:
            payloads = []
            for metadata in metadata_list:
                if metadata is None:
                    payloads.append(None)
                else:
                    self._touch_accessed_at(metadata)
                    payloads.append(self.storage.load(metadata))
            return payloads

        payloads = []
        for metadata in metadata_list:
            if metadata is None:
                raise ValueError(
                    'no stored output matched one of the requested '
                    'echoframe keys'
                )
            self._touch_accessed_at(metadata)
            payloads.append(self.storage.load(metadata))
        return payloads

    def get_many_metadata(self, echoframe_keys):
        '''Load multiple metadata records by echoframe key.'''
        return self._bind_metadatas(self.index.get_many(echoframe_keys))

    def metadatas_to_payloads(self, metadata_list, strict=False):
        '''Load payloads for multiple echoframe metadata records.
        metadata_list:   iterable of metadata records or None values
        strict:          raise when any metadata item is None
        '''
        if not strict:
            return [None if metadata is None else self.storage.load(metadata)
                for metadata in metadata_list]

        payloads = []
        for metadata in metadata_list:
            if metadata is None:
                raise ValueError(
                    'no stored output matched one of the requested metadata '
                    'records'
                )
            payloads.append(self.storage.load(metadata))
        return payloads

    def load_object_frames(self, phraser_key, model_name, layer, collar=500,
        output_type='hidden_state', match='exact'):
        '''Load frame outputs for one object.
        phraser_key:    unique phraser object key
        model_name:     model identifier
        layer:          layer to match
        collar:         exact collar or None for all collars
        output_type:    output type to match
        match:          exact, min, max, or nearest
        '''
        if collar is None:
            entries = filter_metadata(self.find_phraser(phraser_key),
                model_name=model_name, output_type=output_type, layer=layer)
            return {metadata.collar: self.storage.load(metadata)
                for metadata in entries}
        matches = filter_metadata(self.find_phraser(phraser_key),
            model_name=model_name, output_type=output_type, layer=layer,
            collar=collar, match=match)
        if not matches:
            raise ValueError('no stored output matched the requested criteria')
        return self.load(matches[0].echoframe_key)

    def iter_object_frames(self, phraser_key, model_name, layer,
        collar=None, output_type='hidden_state', match='exact'):
        '''Iterate lazily over frame outputs for one object.
        phraser_key:    unique phraser object key
        model_name:     model identifier
        layer:          layer to match
        collar:         exact collar, matched collar, or None for all
        output_type:    output type to match
        match:          exact, min, max, or nearest
        '''
        if collar is None:
            entries = filter_metadata(self.find_phraser(phraser_key),
                model_name=model_name, output_type=output_type, layer=layer)
            for metadata in entries:
                yield metadata, self.storage.load(metadata)
            return

        matches = filter_metadata(self.find_phraser(phraser_key),
            model_name=model_name, output_type=output_type, layer=layer,
            collar=collar, match=match)
        if not matches:
            raise ValueError('no stored output matched the requested criteria')
        metadata = matches[0]
        yield metadata, self.storage.load(metadata)

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
        matches = filter_metadata(self.find_phraser(phraser_key),
            model_name=model_name, output_type=output_type, layer=layer,
            collar=collar, match=match)
        if not matches:
            return None
        metadata = matches[0]
        self.storage.delete(metadata)
        return self.index.delete(metadata)

    def find_by_tag(self, tag, include_deleted=False):
        '''List metadata records for one tag.
        tag:               grouping label
        include_deleted:   include tombstoned entries
        '''
        return self._bind_metadatas(self.index.find_by_tag(tag,
            include_deleted=include_deleted))

    def find_by_tags(self, tags, match='all', include_deleted=False):
        '''List metadata records that match a tag set.
        tags:              grouping labels to match
        match:             require all or any tags
        include_deleted:   include tombstoned entries
        '''
        return self._bind_metadatas(self.index.find_by_tags(tags,
            match=match, include_deleted=include_deleted))

    def find_by_label(self, label, model_name=None, output_type=None,
        layer=None, include_deleted=False):
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
        if model_name is not None:
            records = [record for record in records
                if record.model_name == model_name]
        if output_type is not None:
            records = [record for record in records
                if record.output_type == output_type]
        if layer is not None:
            records = [record for record in records if record.layer == layer]

        models = _load_phraser_models()
        labels_by_key = {}
        matches = []
        for record in records:
            phraser_key = record.phraser_key
            if phraser_key not in labels_by_key:
                phraser_object = models.cache.load(phraser_key)
                labels_by_key[phraser_key] = getattr(phraser_object,
                    'label', None)
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
        return self._bind_metadata(self.index.add_tags(echoframe_key, tags))

    def add_tags_many(self, echoframe_keys, tags):
        '''Add tags to multiple metadata records.
        echoframe_keys:  canonical metadata identifiers
        tags:        grouping labels to add
        '''
        return self._bind_metadatas(self.index.add_tags_many(echoframe_keys,
            tags))

    def remove_tags(self, echoframe_key, tags):
        '''Remove tags from one metadata record.
        echoframe_key:  canonical metadata identifier
        tags:        grouping labels to remove
        '''
        return self._bind_metadata(self.index.remove_tags(echoframe_key, tags))

    def remove_tags_many(self, echoframe_keys, tags):
        '''Remove tags from multiple metadata records.
        echoframe_keys:  canonical metadata identifiers
        tags:        grouping labels to remove
        '''
        return self._bind_metadatas(self.index.remove_tags_many(
            echoframe_keys, tags))

    def shard_stats(self):
        '''Return shard-level stats.'''
        return self.index.list_shard_metadata()

    def list_entries(self, include_deleted=False):
        '''List stored metadata records across all shards.
        include_deleted:   include tombstoned entries
        '''
        entries = self.index.list_entries(
            include_deleted=include_deleted)
        entries.sort(key=lambda metadata: (
            metadata.phraser_key,
            metadata.model_name,
            metadata.output_type,
            metadata.layer,
            metadata.collar,
        ))
        return self._bind_metadatas(entries)

    @property
    def metadata(self):
        '''Return stored metadata records.'''
        if not hasattr(self, '_metadata'):
            self._metadata= self.list_entries()
        return self._metadata

    def overview(self, include_deleted=False, health_event_limit=20,
        include_integrity=False):
        '''Return a compact overview of store contents.
        include_deleted:      include tombstoned entries
        health_event_limit:   recent shard health events to include
        include_integrity:    run a full integrity scan
        '''
        entries = self.list_entries(include_deleted=include_deleted)
        return {
            'entry_count': len(entries),
            'entries': [metadata.to_dict() for metadata in entries],
            'shard_count': len(self.index.list_shards()),
            'shards': self.shard_stats(),
            'tags': self.list_tags(include_deleted=include_deleted),
            'integrity': (self.verify_integrity()
                if include_integrity else None),
            'recent_shard_health_events': self.get_shard_health_events(
                limit=health_event_limit),
        }

    def get_shard_health_events(self, limit=None):
        '''Return recent shard health events from storage.'''
        if not hasattr(self.storage, 'get_shard_health_events'):
            return []
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
        matches = filter_metadata(self.find_phraser(phraser_key),
            model_name=model_name, output_type=output_type, layer=layer,
            collar=collar, match=match)
        if matches:
            metadata = matches[0]
            if tags and add_tags_on_hit:
                metadata = self.add_tags(metadata.echoframe_key, tags)
            return metadata, False
        data = compute()
        metadata = self._make_metadata(phraser_key, collar, model_name,
            output_type, layer, tags=tags)
        metadata = self.put(metadata.echoframe_key, metadata, data)
        return metadata, True

    def _storage_bytes(self):
        '''Return total bytes used by shard files.'''
        shards_root = self.root / 'shards'
        if not shards_root.exists():
            return 0
        return sum(f.stat().st_size for f in shards_root.rglob('*')
            if f.is_file())

    def evict_by_recency(self):
        '''Soft-delete entries older than the recency window until under budget.

        Reads ECHOFRAME_RECENCY_WINDOW_DAYS (default 30) and
        ECHOFRAME_STORAGE_BUDGET_GB (default no limit) from the environment.
        Entries without accessed_at are skipped. Oldest entries are deleted
        first until storage is within the budget.
        '''
        window_days = int(os.environ.get('ECHOFRAME_RECENCY_WINDOW_DAYS', 30))
        budget_gb = os.environ.get('ECHOFRAME_STORAGE_BUDGET_GB')
        budget_bytes = float(budget_gb) * 1_000_000_000 if budget_gb else None

        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

        if budget_bytes is not None and self._storage_bytes() <= budget_bytes:
            return []

        entries = self.list_entries()
        stale = []
        for metadata in entries:
            if metadata.accessed_at is None:
                continue
            accessed = datetime.fromisoformat(metadata.accessed_at)
            if accessed < cutoff:
                stale.append(metadata)

        stale.sort(key=lambda m: m.accessed_at)

        evicted = []
        for metadata in stale:
            if budget_bytes is not None and self._storage_bytes() <= budget_bytes:
                break
            self.storage.delete(metadata)
            self.index.delete(metadata)
            evicted.append(metadata)
        return evicted

    def verify_integrity(self):
        '''Verify that live metadata records point to existing datasets.'''
        broken = []
        checked = 0
        for shard_id in self.index.list_shards():
            for metadata in self.index.entries_for_shard(shard_id):
                checked += 1
                if metadata.shard_id is None or metadata.dataset_path is None:
                    broken.append(self._broken_reference(metadata,
                        reason='missing shard pointer'))
                    continue
                if not self.storage.dataset_exists(metadata.shard_id,
                    metadata.dataset_path):
                    broken.append(self._broken_reference(metadata,
                        reason='missing dataset'))
        return {
            'ok': not broken,
            'checked_entries': checked,
            'broken_references': broken,
        }

    def compact_shards(self, shard_ids=None, dry_run=False,
        resume_pending=False):
        '''Compact shard files by removing deleted payloads.
        shard_ids:          optional shard identifiers to compact
        dry_run:            report what would be compacted
        resume_pending:     resume running journal entries first
        '''
        if shard_ids is None:
            shard_ids = self.index.list_shards()
        if resume_pending:
            self.resume_compactions()

        compacted = []
        plans = []
        for shard_id in shard_ids:
            plan = self._compaction_plan(shard_id)
            if not plan['needs_compaction']:
                continue
            if dry_run:
                plans.append(plan)
                continue
            compacted.append(self._run_compaction_plan(plan))
        return plans if dry_run else compacted

    def resume_compactions(self):
        '''Resume interrupted shard compactions.'''
        completed = []
        for record in self.index.list_compaction_journal(status='running'):
            completed.append(self._run_compaction_plan(record,
                from_journal=True))
        return completed

    def compaction_journal(self, status=None):
        '''Return compaction journal records.'''
        return self.index.list_compaction_journal(status=status)

    def _broken_reference(self, metadata, reason):
        return compaction.broken_reference(metadata, reason)

    def _build_shard_health_report(self, shard_id, error):
        return compaction.build_shard_health_report(self.index,
            self.storage, shard_id, error)

    def _compaction_plan(self, shard_id):
        return compaction.build_compaction_plan(self.index, self.storage,
            shard_id)

    def _run_compaction_plan(self, plan, from_journal=False):
        return compaction.run_compaction_plan(self.index, self.storage,
            plan, from_journal=from_journal)


def _load_phraser_models():
    try:
        from phraser import models
    except ImportError as exc:
        raise ImportError(
            'phraser is required to find entries by label'
        ) from exc
    return models
