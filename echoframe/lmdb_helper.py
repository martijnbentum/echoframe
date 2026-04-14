'''Helpers for LMDB environment and transaction handling.'''

from contextlib import contextmanager
import json
from pathlib import Path


DB_NAMES = {
    'entries_db': b'entries',
    'by_object_db': b'by_object',
    'live_by_object_db': b'live_by_object',
    'by_shard_db': b'by_shard',
    'by_tag_db': b'by_tag',
    'shard_meta_db': b'shard_meta',
    'compaction_db': b'compaction',
}

_ENV_CACHE = {}


def open_env(path, map_size):
    import lmdb

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    cached = _ENV_CACHE.get(resolved_root)
    if cached is not None:
        if cached['map_size'] != map_size:
            message = 'LMDB environment is already open for '
            message += f'{resolved_root} with map_size='
            message += f"{cached['map_size']}; requested map_size={map_size}"
            raise ValueError(message)
        return cached['env']
    env = lmdb.open(str(resolved_root), create=True, max_dbs=12,
        map_size=map_size, subdir=True)
    _ENV_CACHE[resolved_root] = {'env': env, 'map_size': map_size}
    return env


def open_databases(env):
    handles = {}
    for field_name, db_name in DB_NAMES.items():
        handles[field_name] = env.open_db(db_name)
    return handles


@contextmanager
def read_txn(env):
    with env.begin() as txn:
        yield txn


@contextmanager
def write_txn(env):
    with env.begin(write=True) as txn:
        yield txn


def scan_prefix_in_txn(txn, db, prefix):
    entry_ids = []
    cursor = txn.cursor(db=db)
    if not cursor.set_range(prefix):
        return entry_ids
    for key, value in cursor:
        if not key.startswith(prefix):
            break
        entry_ids.append(value.decode('utf-8'))
    return entry_ids


def get_metadata(env, entries_db, metadata_cls, entry_id):
    with read_txn(env) as txn:
        return get_metadata_in_txn(txn, entries_db, metadata_cls, entry_id)


def get_many_metadata(env, entries_db, metadata_cls, entry_ids):
    if not entry_ids:
        return []
    with read_txn(env) as txn:
        return [get_metadata_in_txn(txn, entries_db, metadata_cls, entry_id)
            for entry_id in entry_ids]


def list_metadata(env, entries_db, metadata_cls, include_deleted=False):
    with read_txn(env) as txn:
        return list_metadata_in_txn(txn, entries_db, metadata_cls,
            include_deleted=include_deleted)


def get_metadata_in_txn(txn, entries_db, metadata_cls, entry_id):
    payload = txn.get(entry_id.encode('utf-8'), db=entries_db)
    if payload is None:
        return None
    return metadata_cls.from_dict(json.loads(payload.decode('utf-8')))


def list_metadata_in_txn(txn, entries_db, metadata_cls,
    include_deleted=False):
    entries = []
    cursor = txn.cursor(db=entries_db)
    if not cursor.set_range(b''):
        return entries
    for _, value in cursor:
        metadata = metadata_cls.from_dict(json.loads(value.decode('utf-8')))
        if not include_deleted and metadata.storage_status != 'live':
            continue
        entries.append(metadata)
    return entries


def write_metadata(txn, db_handles, metadata, payload):
    entry_id = metadata.entry_id.encode('utf-8')
    object_key = metadata.object_key.encode('utf-8')
    txn.put(entry_id, payload, db=db_handles['entries_db'])
    txn.put(object_key, entry_id, db=db_handles['by_object_db'])
    if metadata.storage_status == 'live':
        txn.put(object_key, entry_id, db=db_handles['live_by_object_db'])
    else:
        txn.delete(object_key, db=db_handles['live_by_object_db'])
    touched = set()
    if metadata.shard_id:
        txn.put(shard_key(metadata.shard_id, metadata.entry_id), entry_id,
            db=db_handles['by_shard_db'])
        touched.add(metadata.shard_id)
    for tag in metadata.tags:
        txn.put(tag_key(tag, metadata.entry_id), entry_id,
            db=db_handles['by_tag_db'])
    return touched


def delete_tag_keys(txn, by_tag_db, metadata):
    for tag in metadata.tags:
        txn.delete(tag_key(tag, metadata.entry_id), db=by_tag_db)


def delete_shard_keys(txn, by_shard_db, previous, metadata):
    if previous.shard_id is None:
        return
    if previous.shard_id == metadata.shard_id:
        return
    txn.delete(shard_key(previous.shard_id, previous.entry_id),
        db=by_shard_db)


def shard_key(shard_id, entry_id):
    return f'shard:{shard_id}:{entry_id}'.encode('utf-8')


def tag_key(tag, entry_id):
    return f'tag:{tag}:{entry_id}'.encode('utf-8')
