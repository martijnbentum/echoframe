'''Compaction and shard-health helpers for Store.'''

import json
from uuid import uuid4

from . import lmdb_helper

from .metadata import utc_now

MIN_GARBAGE_BYTES = 10_000_000
MIN_GARBAGE_RATIO = 0.1

# Measured HDF5 structural overhead is ~4KB per file plus ~350 bytes per
# dataset; both are padded so pristine shards report zero garbage.
BASE_FILE_OVERHEAD_BYTES = 8192
PER_DATASET_OVERHEAD_BYTES = 512


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


def build_shard_health_report(store, shard_id, error):
    index = store.index
    storage = store.storage
    report = {
        'created_at': utc_now(),
        'failing_shard_id': shard_id,
        'error': error,
        'checked_entries': 0,
        'lost_items': [],
        'shards': [],
    }
    for current_shard_id in index.list_shards():
        entries = _entries_for_shard(store, current_shard_id)
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
            echoframe_key_hex = metadata.echoframe_key.hex()
            if echoframe_key_hex in missing:
                report['lost_items'].append(broken_reference(metadata,
                    reason='missing dataset'))
            elif echoframe_key_hex in unreadable:
                report['lost_items'].append(broken_reference(metadata,
                    reason='unreadable dataset'))
    return report


def build_compaction_plan(store, shard_id, min_garbage_bytes=MIN_GARBAGE_BYTES,
    min_garbage_ratio=MIN_GARBAGE_RATIO):
    index = store.index
    storage = store.storage
    all_entries = _entries_for_shard(store, shard_id)
    stats = index.load_shard_metadata(shard_id) or {
        'entry_count': len(all_entries),
        'byte_size': storage.shard_size(shard_id),
    }
    byte_size = stats.get('byte_size', 0)
    dataset_paths = []
    for entry in all_entries:
        if entry.dataset_path is None: continue
        dataset_paths.append(entry.dataset_path)
    live_bytes = storage.live_dataset_bytes(shard_id, dataset_paths)
    dataset_count = len(dataset_paths)
    garbage_bytes = _estimated_garbage_bytes(byte_size, live_bytes,
        dataset_count)
    needs_compaction = _needs_compaction(byte_size, garbage_bytes,
        min_garbage_bytes, min_garbage_ratio)
    target_shard_id = storage._replacement_shard_id(shard_id)
    return {
        'shard_id': shard_id,
        'target_shard_id': target_shard_id,
        'echoframe_keys': [entry.echoframe_key.hex() for entry in all_entries],
        'entry_count': len(all_entries),
        'byte_size': byte_size,
        'live_bytes': live_bytes,
        'garbage_bytes': garbage_bytes,
        'needs_compaction': needs_compaction,
    }


def create_compaction_journal(index, shard_id, echoframe_keys,
    target_shard_id):
    journal_id = (
        f'{utc_now()}:{shard_id}:{target_shard_id}:{uuid4().hex}'
    )
    record = {
        'journal_id': journal_id,
        'shard_id': shard_id,
        'target_shard_id': target_shard_id,
        'echoframe_keys': list(echoframe_keys),
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


def run_compaction_plan(store, plan, from_journal=False):
    index = store.index
    storage = store.storage
    shard_id = plan['shard_id']
    entries = []
    for echoframe_key_hex in plan['echoframe_keys']:
        echoframe_key = bytes.fromhex(echoframe_key_hex)
        metadata = index.load(echoframe_key, store=store)
        if metadata is None:
            continue
        if metadata.shard_id != shard_id:
            continue
        entries.append(metadata)

    if from_journal:
        journal = plan
    else:
        journal = index.create_compaction_journal(shard_id,
            echoframe_keys=plan['echoframe_keys'],
            target_shard_id=plan['target_shard_id'])

    try:
        updated = storage.compact_shard_to(shard_id, entries,
            target_shard_id=journal['target_shard_id'],
            delete_source=False)
        if updated:
            index.save_many(updated)
        echoframe_keys = [bytes.fromhex(key)
            for key in journal['echoframe_keys']]
        if echoframe_keys:
            index.remove_shard_entries(shard_id, echoframe_keys)
        storage._delete_file(shard_id)
        index.update_compaction_journal(journal['journal_id'],
            status='completed')
        return shard_id
    except Exception as exc:
        index.update_compaction_journal(journal['journal_id'],
            status='failed', error=str(exc))
        raise


def verify_integrity(store):
    '''Verify that metadata records point to existing datasets.'''
    broken = []
    checked = 0
    for shard_id in store.index.list_shards():
        pointed = []
        for metadata in _entries_for_shard(store, shard_id):
            checked += 1
            if metadata.shard_id is None or metadata.dataset_path is None:
                broken.append(broken_reference(metadata,
                    reason='missing shard pointer'))
                continue
            pointed.append(metadata)
        dataset_paths = [metadata.dataset_path for metadata in pointed]
        missing = store.storage.missing_datasets(shard_id, dataset_paths)
        for metadata in pointed:
            if metadata.dataset_path not in missing: continue
            broken.append(broken_reference(metadata,
                reason='missing dataset'))
    data = {}
    data['ok'] = not broken
    data['checked_metadata_count'] = checked
    data['broken_metadata_references'] = broken
    data['unreferenced_shard_files'] = find_unreferenced_shard_files(store)
    return data


def find_unreferenced_shard_files(store):
    '''Return shard files on disk that no index entry references.
    These files are invisible to list_shards() and compact_shards(). They
    can only appear when every record in a shard was overwritten or when a
    batch wrote payloads but crashed before the index commit. They count as
    waste, not damage, and do not affect the integrity verdict.
    '''
    if not hasattr(store.storage, 'root'): return []
    known_shard_ids = set(store.index.list_shards())
    rows = []
    for file_path in sorted(store.storage.root.glob('*.h5')):
        if file_path.stem in known_shard_ids: continue
        byte_size = file_path.stat().st_size
        rows.append({'shard_id': file_path.stem, 'byte_size': byte_size})
    return rows


def compact_shards(store, shard_ids=None, dry_run=False,
    resume_pending=False, min_garbage_bytes=MIN_GARBAGE_BYTES,
    min_garbage_ratio=MIN_GARBAGE_RATIO):
    '''Compact shard files that hold enough reclaimable garbage.'''
    if shard_ids is None:
        shard_ids = store.index.list_shards()
    if resume_pending:
        resume_compactions(store)

    compacted = []
    plans = []
    for shard_id in shard_ids:
        plan = build_compaction_plan(store, shard_id,
            min_garbage_bytes=min_garbage_bytes,
            min_garbage_ratio=min_garbage_ratio)
        if not plan['needs_compaction']:
            continue
        if dry_run:
            plans.append(plan)
            continue
        compacted.append(run_compaction_plan(store, plan))
    return plans if dry_run else compacted


def resume_compactions(store):
    '''Resume interrupted shard compactions.'''
    completed = []
    for record in store.index.list_compaction_journal(status='running'):
        completed.append(run_compaction_plan(store, record,
            from_journal=True))
    return completed


def compaction_journal(store, status=None):
    '''Return compaction journal records.'''
    return store.index.list_compaction_journal(status=status)


def _estimated_garbage_bytes(byte_size, live_bytes, dataset_count):
    overhead = BASE_FILE_OVERHEAD_BYTES
    overhead += PER_DATASET_OVERHEAD_BYTES * dataset_count
    return max(byte_size - live_bytes - overhead, 0)


def _needs_compaction(byte_size, garbage_bytes, min_garbage_bytes,
    min_garbage_ratio):
    if byte_size <= 0: return False
    if garbage_bytes >= min_garbage_bytes: return True
    return garbage_bytes / byte_size >= min_garbage_ratio


def _entries_for_shard(store, shard_id):
    entries = store.index.find_by_shard(shard_id, store=store)
    return [metadata for metadata in entries if metadata.shard_id == shard_id]


def _dump_json(value):
    return json.dumps(value, sort_keys=True).encode('utf-8')
