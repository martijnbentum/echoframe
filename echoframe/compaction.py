'''Compaction and shard-health helpers for Store.'''

from .metadata import utc_now


def broken_reference(metadata, reason):
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


def build_shard_health_report(index, storage, shard_id, error):
    report = {
        'created_at': utc_now(),
        'failing_shard_id': shard_id,
        'error': error,
        'checked_entries': 0,
        'lost_items': [],
        'shards': [],
    }
    for current_shard_id in index.list_shards():
        entries = index.entries_for_shard(current_shard_id)
        shard_report = storage.validate_shard(current_shard_id,
            entries=entries, read_data=True)
        report['shards'].append(shard_report)
        report['checked_entries'] += len(entries)
        missing = set(shard_report['missing_entries'])
        unreadable = {
            item['entry_id'] for item in shard_report['unreadable_entries']
        }
        if shard_report['open_error'] is not None:
            for metadata in entries:
                report['lost_items'].append(broken_reference(metadata,
                    reason=shard_report['open_error']))
            continue
        for metadata in entries:
            if metadata.entry_id in missing:
                report['lost_items'].append(broken_reference(metadata,
                    reason='missing dataset'))
            elif metadata.entry_id in unreadable:
                report['lost_items'].append(broken_reference(metadata,
                    reason='unreadable dataset'))
    return report


def build_compaction_plan(index, storage, shard_id):
    all_entries = index.entries_for_shard(shard_id, include_deleted=True)
    live_entries = [entry for entry in all_entries
        if entry.storage_status == 'live']
    stats = index.get_shard_metadata(shard_id) or {
        'live_entry_count': len(live_entries),
        'deleted_entry_count': len(all_entries) - len(live_entries),
        'byte_size': storage.shard_size(shard_id),
    }
    target_shard_id = storage._replacement_shard_id(shard_id)
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


def run_compaction_plan(index, storage, plan, from_journal=False):
    shard_id = plan['shard_id']
    live_entries = []
    for entry_id in plan['live_entry_ids']:
        metadata = index.get(entry_id)
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
        journal = index.create_compaction_journal(shard_id,
            source_entry_ids=plan['source_entry_ids'],
            live_entry_ids=plan['live_entry_ids'],
            target_shard_id=plan['target_shard_id'])

    try:
        updated = storage.compact_shard_to(shard_id, live_entries,
            target_shard_id=journal['target_shard_id'],
            delete_source=False)
        if updated:
            index.upsert_many(updated)
        deleted_ids = [entry_id for entry_id in journal['source_entry_ids']
            if entry_id not in journal['live_entry_ids']]
        if deleted_ids:
            index.remove_shard_entries(shard_id, deleted_ids)
        old_live_ids = list(journal['live_entry_ids'])
        if old_live_ids:
            index.remove_shard_entries(shard_id, old_live_ids)
        storage._delete_file(shard_id)
        index.update_compaction_journal(journal['journal_id'],
            status='completed')
        return shard_id
    except Exception as exc:
        index.update_compaction_journal(journal['journal_id'],
            status='failed', error=str(exc))
        raise
