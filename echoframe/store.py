'''Public store facade for echoframe.'''

from pathlib import Path

from .index import LmdbIndex
from .metadata import Metadata, utc_now
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
        shards_root = self.root / 'shards'
        self.index = index or LmdbIndex(self.root / 'index.lmdb',
            shards_root=shards_root)
        self.storage = storage or Hdf5ShardStore(shards_root,
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
            output_type=output_type, layer=layer, tags=tags,
            to_vector_version=to_vector_version)
        stored = self.storage.store(metadata, data=data)
        return self.index.upsert(stored)

    def put_many(self, items):
        '''Store multiple output payloads.
        items:    iterable of put-like keyword mappings
        '''
        prepared = []
        for item in items:
            metadata = Metadata(phraser_key=item['phraser_key'],
                collar=item['collar'], model_name=item['model_name'],
                output_type=item['output_type'], layer=item['layer'],
                tags=item.get('tags'),
                to_vector_version=item.get('to_vector_version'))
            prepared.append({
                'metadata': metadata,
                'data': item['data'],
            })
        stored = self.storage.store_many(prepared)
        return self.index.upsert_many(stored)

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

    def find_many(self, queries):
        '''Find multiple records using collar matching rules.
        queries:    iterable of find_one-like keyword mappings
        '''
        return self.index.find_many(queries)

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

    def load_many(self, queries, strict=False):
        '''Load multiple stored output payloads.
        queries:    iterable of find_one-like keyword mappings
        strict:     raise when any query does not match
        '''
        metadata_list = self.find_many(queries)
        if not strict:
            return [None if metadata is None else self.storage.load(metadata)
                for metadata in metadata_list]

        payloads = []
        for metadata in metadata_list:
            if metadata is None:
                raise ValueError(
                    'no stored output matched one of the requested queries'
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
            entries = self.find(phraser_key=phraser_key,
                model_name=model_name, output_type=output_type, layer=layer)
            return {metadata.collar: self.storage.load(metadata)
                for metadata in entries}
        return self.load(phraser_key=phraser_key, collar=collar,
            model_name=model_name, output_type=output_type, layer=layer,
            match=match)

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
            entries = self.find(phraser_key=phraser_key,
                model_name=model_name, output_type=output_type, layer=layer)
            for metadata in entries:
                yield metadata, self.storage.load(metadata)
            return

        metadata = self.find_one(phraser_key=phraser_key, collar=collar,
            model_name=model_name, output_type=output_type, layer=layer,
            match=match)
        if metadata is None:
            raise ValueError('no stored output matched the requested criteria')
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

    def find_by_tags(self, tags, match='all', include_deleted=False):
        '''List metadata records that match a tag set.
        tags:              grouping labels to match
        match:             require all or any tags
        include_deleted:   include tombstoned entries
        '''
        return self.index.find_by_tags(tags, match=match,
            include_deleted=include_deleted)

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

    def add_tags(self, entry_id, tags):
        '''Add tags to one metadata record.
        entry_id:    metadata identifier
        tags:        grouping labels to add
        '''
        return self.index.add_tags(entry_id, tags)

    def add_tags_many(self, entry_ids, tags):
        '''Add tags to multiple metadata records.
        entry_ids:   metadata identifiers
        tags:        grouping labels to add
        '''
        return self.index.add_tags_many(entry_ids, tags)

    def remove_tags(self, entry_id, tags):
        '''Remove tags from one metadata record.
        entry_id:    metadata identifier
        tags:        grouping labels to remove
        '''
        return self.index.remove_tags(entry_id, tags)

    def remove_tags_many(self, entry_ids, tags):
        '''Remove tags from multiple metadata records.
        entry_ids:   metadata identifiers
        tags:        grouping labels to remove
        '''
        return self.index.remove_tags_many(entry_ids, tags)

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
        return entries

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
            'entries': [{
                'entry_id': metadata.entry_id,
                **metadata.to_dict(),
            } for metadata in entries],
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
        tags=None, add_tags_on_hit=False, to_vector_version=None):
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
        to_vector_version:    optional debug-only version marker
        '''
        metadata = self.find_one(phraser_key=phraser_key,
            collar=collar, model_name=model_name,
            output_type=output_type, layer=layer, match=match)
        if metadata is not None:
            if tags and add_tags_on_hit:
                metadata = self.add_tags(metadata.entry_id, tags)
            return metadata, False
        data = compute()
        metadata = self.put(phraser_key=phraser_key,
            collar=collar, model_name=model_name,
            output_type=output_type, layer=layer, data=data, tags=tags,
            to_vector_version=to_vector_version)
        return metadata, True

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
        return {
            'entry_id': metadata.entry_id,
            'phraser_key': metadata.phraser_key,
            'model_name': metadata.model_name,
            'output_type': metadata.output_type,
            'layer': metadata.layer,
            'collar': metadata.collar,
            'shard_id': metadata.shard_id,
            'dataset_path': metadata.dataset_path,
            'reason': reason,
        }

    def _build_shard_health_report(self, shard_id, error):
        report = {
            'created_at': utc_now(),
            'failing_shard_id': shard_id,
            'error': error,
            'checked_entries': 0,
            'lost_items': [],
            'shards': [],
        }
        for current_shard_id in self.index.list_shards():
            entries = self.index.entries_for_shard(current_shard_id)
            shard_report = self.storage.validate_shard(current_shard_id,
                entries=entries, read_data=True)
            report['shards'].append(shard_report)
            report['checked_entries'] += len(entries)
            missing = set(shard_report['missing_entries'])
            unreadable = {
                item['entry_id'] for item in shard_report['unreadable_entries']
            }
            if shard_report['open_error'] is not None:
                for metadata in entries:
                    report['lost_items'].append(self._broken_reference(
                        metadata, reason=shard_report['open_error']))
                continue
            for metadata in entries:
                if metadata.entry_id in missing:
                    report['lost_items'].append(self._broken_reference(
                        metadata, reason='missing dataset'))
                elif metadata.entry_id in unreadable:
                    report['lost_items'].append(self._broken_reference(
                        metadata, reason='unreadable dataset'))
        return report

    def _compaction_plan(self, shard_id):
        all_entries = self.index.entries_for_shard(shard_id,
            include_deleted=True)
        live_entries = [entry for entry in all_entries
            if entry.storage_status == 'live']
        stats = self.index.get_shard_metadata(shard_id) or {
            'live_entry_count': len(live_entries),
            'deleted_entry_count': len(all_entries) - len(live_entries),
            'byte_size': self.storage.shard_size(shard_id),
        }
        target_shard_id = self.storage._replacement_shard_id(shard_id)
        return {
            'shard_id': shard_id,
            'target_shard_id': target_shard_id,
            'source_entry_ids': [entry.entry_id for entry in all_entries],
            'live_entry_ids': [entry.entry_id for entry in live_entries],
            'live_entry_count': len(live_entries),
            'deleted_entry_count': len(all_entries) - len(live_entries),
            'byte_size': stats.get('byte_size', 0),
            'needs_compaction': len(live_entries) != len(all_entries),
        }

    def _run_compaction_plan(self, plan, from_journal=False):
        shard_id = plan['shard_id']
        live_entries = []
        for entry_id in plan['live_entry_ids']:
            metadata = self.index.get(entry_id)
            if metadata is None:
                continue
            if metadata.storage_status != 'live':
                continue
            if metadata.shard_id != shard_id:
                continue
            live_entries.append(metadata)

        if from_journal:
            journal = plan
        else:
            journal = self.index.create_compaction_journal(shard_id,
                source_entry_ids=plan['source_entry_ids'],
                live_entry_ids=plan['live_entry_ids'],
                target_shard_id=plan['target_shard_id'])

        try:
            updated = self.storage.compact_shard_to(shard_id, live_entries,
                target_shard_id=journal['target_shard_id'],
                delete_source=False)
            if updated:
                self.index.upsert_many(updated)
            deleted_ids = [entry_id for entry_id in journal['source_entry_ids']
                if entry_id not in journal['live_entry_ids']]
            if deleted_ids:
                self.index.remove_shard_entries(shard_id, deleted_ids)
            old_live_ids = list(journal['live_entry_ids'])
            if old_live_ids:
                self.index.remove_shard_entries(shard_id, old_live_ids)
            self.storage._delete_file(shard_id)
            self.index.update_compaction_journal(journal['journal_id'],
                status='completed')
            return shard_id
        except Exception as exc:
            self.index.update_compaction_journal(journal['journal_id'],
                status='failed', error=str(exc))
            raise
