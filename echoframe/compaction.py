'''Compaction and shard-health helpers for Store.'''

import json
from uuid import uuid4

from . import lmdb_helper

from .metadata import utc_now


def broken_reference(metadata, reason):
    return {
        'echoframe_key_hex': metadata.echoframe_key.hex(),
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
        missing = set(shard_report['missing_echoframe_keys'])
        unreadable = {item['echoframe_key_hex']
            for item in shard_report['unreadable_echoframe_keys']}
        if shard_report['open_error'] is not None:
            for metadata in entries:
                report['lost_items'].append(broken_reference(metadata,
                    reason=shard_report['open_error']))
            continue
        for metadata in entries:
            if metadata.format_echoframe_key() in missing:
                report['lost_items'].append(broken_reference(metadata,
                    reason='missing dataset'))
            elif metadata.format_echoframe_key() in unreadable:
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
        'source_echoframe_keys': [entry.echoframe_key.hex()
            for entry in all_entries],
        'live_echoframe_keys': [entry.echoframe_key.hex()
            for entry in live_entries],
        'live_entry_count': len(live_entries),
        'deleted_entry_count': len(all_entries) - len(live_entries),
        'byte_size': stats.get('byte_size', 0),
        'needs_compaction': len(live_entries) != len(all_entries),
    }


def create_compaction_journal(index, shard_id, source_echoframe_keys,
    live_echoframe_keys, target_shard_id):
    journal_id = (
        f'{utc_now()}:{shard_id}:{target_shard_id}:{uuid4().hex}'
    )
    record = {
        'journal_id': journal_id,
        'shard_id': shard_id,
        'target_shard_id': target_shard_id,
        'source_echoframe_keys': list(source_echoframe_keys),
        'live_echoframe_keys': list(live_echoframe_keys),
        'status': 'running',
        'started_at': utc_now(),
        'finished_at': None,
        'error': None,
    }
    with lmdb_helper.write_txn(index.env) as txn:
        txn.put(journal_id.encode('utf-8'), _dump_json(record),
            db=index.compaction_db)
    return record


def update_compaction_journal(index, journal_id, status, error=None):
    with lmdb_helper.write_txn(index.env) as txn:
        key = journal_id.encode('utf-8')
        payload = txn.get(key, db=index.compaction_db)
        if payload is None:
            return None
        record = json.loads(payload.decode('utf-8'))
        record['status'] = status
        record['error'] = error
        record['finished_at'] = utc_now()
        txn.put(key, _dump_json(record), db=index.compaction_db)
    return record


def list_compaction_journal(index, status=None):
    rows = []
    with lmdb_helper.read_txn(index.env) as txn:
        cursor = txn.cursor(db=index.compaction_db)
        if not cursor.set_range(b''):
            return rows
        for _, value in cursor:
            record = json.loads(value.decode('utf-8'))
            if status is not None and record['status'] != status:
                continue
            rows.append(record)
    rows.sort(key=lambda row: row['journal_id'])
    return rows


def run_compaction_plan(index, storage, plan, from_journal=False):
    shard_id = plan['shard_id']
    live_entries = []
    for echoframe_key in plan['live_echoframe_keys']:
        metadata = index.get(echoframe_key)
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
            source_echoframe_keys=plan['source_echoframe_keys'],
            live_echoframe_keys=plan['live_echoframe_keys'],
            target_shard_id=plan['target_shard_id'])

    try:
        updated = storage.compact_shard_to(shard_id, live_entries,
            target_shard_id=journal['target_shard_id'],
            delete_source=False)
        if updated:
            index.upsert_many(updated)
        deleted_keys = [echoframe_key
            for echoframe_key in journal['source_echoframe_keys']
            if echoframe_key not in journal['live_echoframe_keys']]
        if deleted_keys:
            index.remove_shard_entries(shard_id, deleted_keys)
        old_live_keys = list(journal['live_echoframe_keys'])
        if old_live_keys:
            index.remove_shard_entries(shard_id, old_live_keys)
        storage._delete_file(shard_id)
        index.update_compaction_journal(journal['journal_id'],
            status='completed')
        return shard_id
    except Exception as exc:
        index.update_compaction_journal(journal['journal_id'],
            status='failed', error=str(exc))
        raise


def _dump_json(value):
    return json.dumps(value, sort_keys=True).encode('utf-8')
