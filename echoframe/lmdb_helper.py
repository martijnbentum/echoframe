'''Helpers for LMDB environment and transaction handling.'''

from contextlib import contextmanager
import json
from pathlib import Path


DB_NAMES = {
    'entries_db': b'entries',
    'by_phraser_db': b'by_phraser',
    'live_by_phraser_db': b'live_by_phraser',
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
        entry_ids.append(bytes(value))
    return entry_ids


def _normalize_echoframe_key(echoframe_key):
    if isinstance(echoframe_key, bytes):
        return echoframe_key
    if isinstance(echoframe_key, bytearray):
        return bytes(echoframe_key)
    if isinstance(echoframe_key, str):
        return bytes.fromhex(echoframe_key)
    raise ValueError('echoframe_key must be bytes or hex string')


def encode_phraser_key(phraser_key):
    if isinstance(phraser_key, bytes):
        return b'b:' + phraser_key
    if isinstance(phraser_key, bytearray):
        return b'b:' + bytes(phraser_key)
    if isinstance(phraser_key, str):
        return b's:' + phraser_key.encode('utf-8')
    raise ValueError('phraser_key must be bytes or string')


def get_metadata(env, entries_db, metadata_cls, echoframe_key):
    with read_txn(env) as txn:
        return get_metadata_in_txn(txn, entries_db, metadata_cls,
            echoframe_key)


def get_many_metadata(env, entries_db, metadata_cls, echoframe_keys):
    if not echoframe_keys:
        return []
    with read_txn(env) as txn:
        return [get_metadata_in_txn(txn, entries_db, metadata_cls,
            echoframe_key) for echoframe_key in echoframe_keys]


def list_metadata(env, entries_db, metadata_cls, include_deleted=False):
    with read_txn(env) as txn:
        return list_metadata_in_txn(txn, entries_db, metadata_cls,
            include_deleted=include_deleted)


def get_metadata_in_txn(txn, entries_db, metadata_cls, echoframe_key):
    echoframe_key = _normalize_echoframe_key(echoframe_key)
    payload = txn.get(echoframe_key, db=entries_db)
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
    echoframe_key = metadata.echoframe_key
    phraser_key = phraser_scan_key(metadata.phraser_key,
        metadata.echoframe_key)
    txn.put(echoframe_key, payload, db=db_handles['entries_db'])
    if metadata.phraser_key not in (None, ''):
        txn.put(phraser_key, echoframe_key, db=db_handles['by_phraser_db'])
        if metadata.storage_status == 'live':
            txn.put(phraser_key, echoframe_key,
                db=db_handles['live_by_phraser_db'])
        else:
            txn.delete(phraser_key, db=db_handles['live_by_phraser_db'])
    touched = set()
    if metadata.shard_id:
        txn.put(shard_key(metadata.shard_id, metadata.echoframe_key),
            echoframe_key, db=db_handles['by_shard_db'])
        touched.add(metadata.shard_id)
    for tag in metadata.tags:
        txn.put(tag_key(tag, metadata.echoframe_key), echoframe_key,
            db=db_handles['by_tag_db'])
    return touched


def delete_tag_keys(txn, by_tag_db, metadata):
    for tag in metadata.tags:
        txn.delete(tag_key(tag, metadata.echoframe_key), db=by_tag_db)


def delete_shard_keys(txn, by_shard_db, previous, metadata):
    if previous.shard_id is None:
        return
    if previous.shard_id == metadata.shard_id:
        return
    txn.delete(shard_key(previous.shard_id, previous.echoframe_key),
        db=by_shard_db)


def delete_phraser_keys(txn, db_handles, metadata):
    if metadata.phraser_key in (None, ''):
        return
    key = phraser_scan_key(metadata.phraser_key, metadata.echoframe_key)
    txn.delete(key, db=db_handles['by_phraser_db'])
    txn.delete(key, db=db_handles['live_by_phraser_db'])


def shard_key(shard_id, echoframe_key):
    echoframe_key = _normalize_echoframe_key(echoframe_key)
    return f'shard:{shard_id}:'.encode('utf-8') + echoframe_key


def tag_key(tag, echoframe_key):
    echoframe_key = _normalize_echoframe_key(echoframe_key)
    return f'tag:{tag}:'.encode('utf-8') + echoframe_key


def phraser_scan_prefix(phraser_key):
    return b'phraser:' + encode_phraser_key(phraser_key) + b':'


def phraser_scan_key(phraser_key, echoframe_key):
    echoframe_key = _normalize_echoframe_key(echoframe_key)
    return phraser_scan_prefix(phraser_key) + echoframe_key
