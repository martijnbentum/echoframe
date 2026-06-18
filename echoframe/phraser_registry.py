'''Phraser store registry: source_id <-> path records and loaded stores.'''

import os
from pathlib import Path


def normalise_phraser_store_path(path):
    '''Return an absolute, symlink-resolved phraser store path.'''
    return str(Path(path).expanduser().resolve())


class PhraserStoreRegistry:
    '''Persist source_id -> path records and hold loaded phraser stores.

    Records persist in config.json (shared with the model registry); loaded
    phraser Store objects live in memory for this process only.
    '''

    def __init__(self, config):
        self._config = config        # shared StoreConfig
        self._phraser_stores = {}    # source_id -> live phraser Store

    def register(self, source_id, path):
        '''Persist source_id -> path. Idempotent on the same path; raises on a
        conflicting path for an existing source_id.
        '''
        _validate_source_id(source_id)
        _validate_path(path)
        path = normalise_phraser_store_path(path)
        config = self._config.read()
        paths = config['phraser_sources']
        existing = paths.get(source_id)
        if existing is not None:
            if normalise_phraser_store_path(existing) == path:
                return path
            message = 'phraser_source_id already registered with a '
            message += f'different path: {source_id!r} -> {existing!r}'
            raise ValueError(message)
        paths[source_id] = path
        self._config.write(config)
        return path

    def load_path(self, source_id):
        '''Return the persisted path for source_id, or None.'''
        return self._config.read()['phraser_sources'].get(source_id)

    def source_ids(self):
        '''Return all registered phraser source identifiers.'''
        return sorted(self._config.read()['phraser_sources'])

    def attach(self, source_id, phraser_store):
        '''Register source_id -> phraser_store.path and hold the live store.'''
        self.register(source_id, phraser_store.path)
        self._phraser_stores[source_id] = phraser_store
        return phraser_store

    def load_store(self, source_id):
        '''Return the live phraser Store for source_id.

        Opens the registered path on first use and caches it. A cached store
        that has been closed is not silently reopened: this raises with a hint
        to call open_phraser_stores().
        '''
        cached = self._phraser_stores.get(source_id)
        if cached is not None:
            if _is_open(cached):
                return cached
            message = f'phraser store {source_id!r} is closed; call '
            message += 'store.open_phraser_stores() to reopen'
            raise RuntimeError(message)
        path = self.load_path(source_id)
        if path is None:
            raise ValueError(f'unknown phraser_source_id: {source_id!r}')
        if not Path(path).exists():
            print(f'WARNING: phraser store path does not exist, opening may '
                f'create a new empty store: {path}')
        import phraser
        phraser_store = phraser.Store(path=path)
        self._phraser_stores[source_id] = phraser_store
        return phraser_store

    def open_phraser_stores(self):
        '''Reopen cached phraser stores that have been closed.
        Returns the number of stores reopened. Leaves already-open stores and
        registered-but-never-opened sources untouched.
        '''
        reopened = 0
        for store in self._phraser_stores.values():
            if _is_open(store): continue
            store.open()
            reopened += 1
        return reopened

    def close_phraser_stores(self):
        '''Close all cached phraser stores, attached ones included.
        Closed stores stay cached so load_store can fail loud instead of
        silently reopening. Returns the number of stores closed.
        '''
        closed = 0
        for store in self._phraser_stores.values():
            if not _is_open(store): continue
            store.close()
            closed += 1
        return closed

    def segment_to_source_id(self, segment):
        '''Return the registered source_id for a bound phraser segment.'''
        path = normalise_phraser_store_path(segment.store.path)
        paths = self._config.read()['phraser_sources']
        matches = [source_id for source_id, p in paths.items()
            if normalise_phraser_store_path(p) == path]
        if len(matches) == 1: return matches[0]
        if not matches:
            raise ValueError('phraser segment store is not registered')
        raise ValueError('multiple phraser sources match segment store')

    def segments_to_source_id(self, segments):
        '''Return the single source_id shared by bound segments.'''
        segments = list(segments)
        if not segments:
            raise ValueError('segments must not be empty')
        source_ids = []
        for segment in segments:
            source_id = self.segment_to_source_id(segment)
            if source_id not in source_ids: source_ids.append(source_id)
        if len(source_ids) == 1: return source_ids[0]
        raise ValueError('batch segments must come from one phraser source')


def _is_open(store):
    '''Return whether a phraser store is open.
    A store that does not expose is_open() (e.g. a test double) is treated as
    open.
    '''
    is_open = getattr(store, 'is_open', None)
    if is_open is None: return True
    return is_open()


def _validate_source_id(source_id):
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError('phraser_source_id must be a non-empty string')


def _validate_path(path):
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError('phraser store path must be a non-empty string or path')
