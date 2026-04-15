'''Store-owned model registry for echoframe.

Persists model_metadata records using the canonical binary key from the
schema plan. Model ids are store-internal and assigned by the registry.

Key layout: 8-byte model_name_hash || uint8 output_type_id (9 bytes total).
Value: JSON payload with model_name, model_id, and created_at.
'''

import json
from datetime import datetime, timezone

from . import lmdb_helper
from .key_helper import pack_model_metadata_key

_REGISTRY_DB = b'model_registry'


class ModelRegistry:
    '''Persist and query model_metadata records in LMDB.'''

    def __init__(self, env):
        self.env = env
        self.db = env.open_db(_REGISTRY_DB)

    def register(self, model_name):
        '''Register one model and persist a model_metadata record.

        model_name:   str — must be a non-empty string
        Returns the stored record dict with model_name, model_id, created_at.
        Raises ValueError if model_name is invalid or already registered.
        '''
        _validate_model_name(model_name)
        key = pack_model_metadata_key(model_name)
        with lmdb_helper.write_txn(self.env) as txn:
            if txn.get(key, db=self.db) is not None:
                raise ValueError(
                    f'model_name already registered: {model_name!r}')
            model_id = _next_model_id(txn, self.db)
            record = _make_record(model_name, model_id)
            txn.put(key, _dump(record), db=self.db)
        return record

    def get(self, model_name):
        '''Return the model_metadata record for model_name, or None.'''
        key = pack_model_metadata_key(model_name)
        with lmdb_helper.read_txn(self.env) as txn:
            value = txn.get(key, db=self.db)
        if value is None:
            return None
        return json.loads(value.decode('utf-8'))

    def import_seeds(self, path, _loader=None):
        '''Import model definitions from a seed file.

        Validates the entire seed file and checks for name conflicts with the
        store before writing anything. All records are written in one
        transaction — if any model name already exists the import fails and
        nothing is written.

        path:      path-like pointing to a JSON seed file
        _loader:   optional callable(path) → list of dicts (for testing)
        Returns list of stored record dicts.
        Raises ValueError for invalid seed files or any name conflict.
        '''
        if _loader is None:
            from .model_registry_loader import load_seed_file as _loader
        seeds = _loader(path)

        with lmdb_helper.write_txn(self.env) as txn:
            conflicts = [
                seed['model_name']
                for seed in seeds
                if txn.get(
                    pack_model_metadata_key(seed['model_name']),
                    db=self.db) is not None
            ]
            if conflicts:
                names = ', '.join(repr(n) for n in conflicts)
                raise ValueError(
                    f'model names already registered in store: {names}')

            next_id = _next_model_id(txn, self.db)
            stored = []
            for seed in seeds:
                model_name = seed['model_name']
                record = _make_record(model_name, next_id)
                txn.put(pack_model_metadata_key(model_name),
                        _dump(record), db=self.db)
                stored.append(record)
                next_id += 1
        return stored


# -------- helpers --------

def _validate_model_name(model_name):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError('model_name must be a non-empty string')


def _next_model_id(txn, db):
    '''Return the next available model_id (max existing + 1, starting at 0).'''
    max_id = -1
    cursor = txn.cursor(db=db)
    if not cursor.set_range(b''):
        return 0
    for _, value in cursor:
        record = json.loads(value.decode('utf-8'))
        candidate = record.get('model_id', -1)
        if candidate > max_id:
            max_id = candidate
    return max_id + 1


def _make_record(model_name, model_id):
    return {
        'model_name': model_name,
        'model_id': model_id,
        'created_at': _utc_now(),
    }


def _dump(record):
    return json.dumps(record, sort_keys=True).encode('utf-8')


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
