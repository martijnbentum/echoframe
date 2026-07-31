'''Transfer model outputs or relocate complete stores.'''

from pathlib import Path

from . import lmdb_helper
from .metadata import EchoframeMetadata
from .store import Store


__all__ = [
    'copy_hidden_states_for_model',
    'move_hidden_states_for_model',
    'move_store',
]


def move_store(source_path, destination_path):
    '''Move a closed store directory to a new location.
    source_path:       existing Echoframe store directory
    destination_path:  exact non-existing destination directory

    The move is one atomic directory rename and therefore requires source and
    destination to be on the same filesystem. Callers must ensure no other
    process has the source store open.
    '''
    source, destination = _validate_store_move_paths(source_path,
        destination_path)
    index_path = source / 'index.lmdb'
    if lmdb_helper.env_is_open(index_path):
        raise ValueError('source store is open in the current process')

    _verify_store_for_move(source, 'source')
    file_count, byte_count = _store_file_totals(source)
    source.rename(destination)

    try:
        integrity = _verify_store_for_move(destination, 'destination')
    except Exception as exc:
        _rollback_store_move(source, destination, exc)
        raise

    return {
        'source_path': str(source),
        'destination_path': str(destination),
        'file_count': file_count,
        'byte_count': byte_count,
        'integrity': integrity,
    }


def copy_hidden_states_for_model(source_store, destination_store, model_name,
    batch_size=100):
    '''Copy one model's hidden states to an empty store.
    source_store:       store containing the hidden states
    destination_store:  distinct, empty store receiving the hidden states
    model_name:         registered model used to select hidden states
    batch_size:         maximum payloads loaded and saved per batch

    The source is not changed. Referenced phraser registrations are copied by
    persisted path; live phraser store objects are not copied.
    '''
    transfer = _prepare_transfer(source_store, destination_store, model_name,
        batch_size)
    return _copy_transfer(transfer)


def move_hidden_states_for_model(source_store, destination_store, model_name,
    batch_size=100):
    '''Move one model's hidden states to an empty store.
    source_store:       store containing the hidden states
    destination_store:  distinct, empty store receiving the hidden states
    model_name:         registered model used to select hidden states
    batch_size:         maximum payloads loaded and saved per batch

    Complete source shards are deleted only after the destination copy and its
    integrity have been verified. Source model and phraser registrations are
    retained.
    '''
    transfer = _prepare_transfer(source_store, destination_store, model_name,
        batch_size)
    _assert_complete_source_shards(transfer)
    result = _copy_transfer(transfer)
    _assert_complete_source_shards(transfer)

    source_store.index.delete_many(transfer['metadatas'])
    shard_ids = sorted(transfer['metadatas_by_shard'])
    try:
        for shard_id in shard_ids:
            source_store.storage._delete_file(shard_id)
    finally:
        _clear_active_shard_cache(source_store.storage)

    _verify_source_deletion(source_store, transfer['metadatas'], shard_ids)
    result['deleted_count'] = len(transfer['metadatas'])
    result['deleted_shard_count'] = len(shard_ids)
    return result


def _prepare_transfer(source_store, destination_store, model_name,
    batch_size):
    _validate_distinct_stores(source_store, destination_store)
    _validate_batch_size(batch_size)
    _assert_destination_empty(destination_store)

    source_model = source_store.load_model_metadata(model_name)
    if source_model is None:
        raise ValueError(f'model_name is not registered: {model_name!r}')

    metadatas = _select_hidden_states(source_store, source_model.model_id)
    if not metadatas:
        message = f'no hidden_state records found for model {model_name!r}'
        raise ValueError(message)
    _validate_selected_payloads(source_store, metadatas)

    phraser_paths = _load_referenced_phraser_paths(source_store, metadatas)
    metadatas_by_shard = _group_by_shard(metadatas)
    return {
        'source_store': source_store,
        'destination_store': destination_store,
        'model_name': model_name,
        'source_model': source_model,
        'metadatas': metadatas,
        'metadatas_by_shard': metadatas_by_shard,
        'phraser_paths': phraser_paths,
        'batch_size': batch_size,
    }


def _copy_transfer(transfer):
    source_store = transfer['source_store']
    destination_store = transfer['destination_store']
    source_model = transfer['source_model']
    model_name = transfer['model_name']

    destination_model = destination_store.register_model(model_name,
        local_path=source_model.local_path,
        huggingface_id=source_model.huggingface_id,
        language=source_model.language, size=source_model.size,
        architecture=source_model.architecture)
    for source_id, path in transfer['phraser_paths'].items():
        destination_store.register_phraser_store(source_id, path)

    destination_keys = {}
    metadatas = transfer['metadatas']
    batch_size = transfer['batch_size']
    for start in range(0, len(metadatas), batch_size):
        batch = metadatas[start:start + batch_size]
        payloads = source_store.metadatas_to_payloads(batch)
        items = []
        for source_metadata, payload in zip(batch, payloads):
            destination_key = _destination_key(destination_store, model_name,
                source_metadata)
            metadata = _destination_metadata(destination_store, model_name,
                source_metadata, destination_key)
            item = {'echoframe_key': destination_key}
            item['metadata'] = metadata
            item['data'] = payload
            items.append(item)
            destination_keys[source_metadata.echoframe_key] = destination_key
        destination_store.save_many(items)

    _verify_destination(transfer, destination_keys)
    copied_count = len(metadatas)
    destination_shards = destination_store.index.list_shards()
    return {
        'model_name': model_name,
        'source_model_id': source_model.model_id,
        'destination_model_id': destination_model.model_id,
        'copied_count': copied_count,
        'copied_payload_count': copied_count,
        'deleted_count': 0,
        'deleted_shard_count': 0,
        'phraser_source_count': len(transfer['phraser_paths']),
        'destination_shard_count': len(destination_shards),
    }


def _destination_key(destination_store, model_name, source_metadata):
    return destination_store.make_echoframe_key('hidden_state',
        model_name=model_name, phraser_key=source_metadata.phraser_key,
        layer=source_metadata.layer, collar=source_metadata.collar)


def _destination_metadata(destination_store, model_name, source_metadata,
    destination_key):
    metadata = EchoframeMetadata(destination_key, store=destination_store,
        tags=source_metadata.tags, model_name=model_name,
        phraser_source_id=source_metadata.phraser_source_id)
    metadata.created_at = source_metadata.created_at
    return metadata


def _verify_destination(transfer, destination_keys):
    destination_store = transfer['destination_store']
    expected_keys = set(destination_keys.values())
    actual_keys = set(destination_store.index.all_echoframe_keys)
    if actual_keys != expected_keys:
        message = 'destination key set does not match copied hidden states'
        raise RuntimeError(message)

    source_by_destination = {}
    for metadata in transfer['metadatas']:
        key = destination_keys[metadata.echoframe_key]
        source_by_destination[key] = metadata
    sorted_keys = sorted(expected_keys)
    copied = destination_store.load_many_metadata(sorted_keys,
        keep_missing=True)
    if any(metadata is None for metadata in copied):
        raise RuntimeError('destination index verification found missing keys')
    for metadata in copied:
        source_metadata = source_by_destination[metadata.echoframe_key]
        _verify_copied_metadata(source_metadata, metadata,
            transfer['model_name'])

    integrity = destination_store.verify_integrity()
    if not integrity.get('ok', False):
        broken = integrity.get('broken_metadata_references', [])
        message = 'destination integrity verification failed'
        if broken: message += f': {broken!r}'
        raise RuntimeError(message)


def _verify_copied_metadata(source_metadata, copied_metadata, model_name):
    expected = {
        'model_name': model_name,
        'output_type': source_metadata.output_type,
        'phraser_key': source_metadata.phraser_key,
        'phraser_source_id': source_metadata.phraser_source_id,
        'layer': source_metadata.layer,
        'collar': source_metadata.collar,
        'tags': source_metadata.tags,
        'created_at': source_metadata.created_at,
        'shape': source_metadata.shape,
    }
    actual = {
        'model_name': copied_metadata.model_name,
        'output_type': copied_metadata.output_type,
        'phraser_key': copied_metadata.phraser_key,
        'phraser_source_id': copied_metadata.phraser_source_id,
        'layer': copied_metadata.layer,
        'collar': copied_metadata.collar,
        'tags': copied_metadata.tags,
        'created_at': copied_metadata.created_at,
        'shape': copied_metadata.shape,
    }
    if actual != expected:
        key = copied_metadata.echoframe_key.hex()
        raise RuntimeError(f'destination metadata verification failed: {key}')


def _validate_distinct_stores(source_store, destination_store):
    if source_store is destination_store:
        raise ValueError('source and destination must be different stores')
    source_root = _resolved_path(source_store.root)
    destination_root = _resolved_path(destination_store.root)
    if source_root == destination_root:
        raise ValueError('source and destination must be different stores')
    if source_store.index is destination_store.index:
        raise ValueError('source and destination must be different stores')
    if source_store.storage is destination_store.storage:
        raise ValueError('source and destination must be different stores')
    source_index_path = getattr(source_store.index, 'path', None)
    destination_index_path = getattr(destination_store.index, 'path', None)
    if _paths_match(source_index_path, destination_index_path):
        raise ValueError('source and destination must be different stores')
    source_storage_root = getattr(source_store.storage, 'root', None)
    destination_storage_root = getattr(destination_store.storage, 'root', None)
    if _paths_match(source_storage_root, destination_storage_root):
        raise ValueError('source and destination must be different stores')


def _resolved_path(path):
    return Path(path).expanduser().resolve()


def _validate_store_move_paths(source_path, destination_path):
    source_input = Path(source_path).expanduser()
    destination_input = Path(destination_path).expanduser()
    if not source_input.exists():
        raise ValueError(f'source store does not exist: {source_input}')
    if not source_input.is_dir():
        raise ValueError(f'source store is not a directory: {source_input}')
    if source_input.is_symlink():
        raise ValueError('source store path must not be a symbolic link')
    if destination_input.exists() or destination_input.is_symlink():
        message = f'destination path already exists: {destination_input}'
        raise ValueError(message)

    source = source_input.resolve()
    destination = destination_input.resolve()
    if source == destination or source in destination.parents:
        raise ValueError('destination path must not be inside source store')
    if not destination.parent.is_dir():
        message = f'destination parent does not exist: {destination.parent}'
        raise ValueError(message)
    _validate_store_directory(source)
    return source, destination


def _validate_store_directory(root):
    index_path = root / 'index.lmdb'
    data_path = index_path / 'data.mdb'
    shards_path = root / 'shards'
    if not index_path.is_dir() or not data_path.is_file():
        raise ValueError(f'source is not an Echoframe store: {root}')
    if not shards_path.is_dir():
        raise ValueError(f'source is not an Echoframe store: {root}')


def _verify_store_for_move(root, label):
    store = Store(root)
    try:
        store.config.read()
        integrity = store.verify_integrity()
    finally:
        store.close()
    if not integrity.get('ok', False):
        broken = integrity.get('broken_metadata_references', [])
        message = f'{label} store integrity verification failed'
        if broken: message += f': {broken!r}'
        raise RuntimeError(message)
    return integrity


def _store_file_totals(root):
    file_paths = [path for path in root.rglob('*') if path.is_file()]
    byte_count = sum(path.stat().st_size for path in file_paths)
    return len(file_paths), byte_count


def _rollback_store_move(source, destination, verification_error):
    try:
        destination.rename(source)
    except Exception as rollback_error:
        message = 'destination verification failed and the store could not '
        message += f'be restored to {source}: {rollback_error}'
        raise RuntimeError(message) from verification_error


def _paths_match(first, second):
    if first is None or second is None: return False
    return _resolved_path(first) == _resolved_path(second)


def _validate_batch_size(batch_size):
    message = 'batch_size must be a positive integer'
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError(message)
    if batch_size <= 0: raise ValueError(message)


def _assert_destination_empty(destination_store):
    has_entries = bool(destination_store.index.all_echoframe_keys)
    has_models = bool(destination_store.model_registry.model_metadatas)
    source_ids = destination_store.phraser_registry.source_ids()
    has_phraser_sources = bool(source_ids)
    indexed_shards = destination_store.index.list_shards()
    has_indexed_shards = bool(indexed_shards)
    shard_metadata = destination_store.index.list_shard_metadata()
    has_shard_metadata = bool(shard_metadata)
    has_shard_files = _has_shard_files(destination_store.storage)
    journal = destination_store.index.list_compaction_journal()
    has_journal = bool(journal)
    if any((has_entries, has_models, has_phraser_sources,
        has_indexed_shards, has_shard_metadata, has_shard_files,
        has_journal)):
        raise ValueError('destination store must be empty')


def _has_shard_files(storage):
    root = getattr(storage, 'root', None)
    if root is None: return False
    storage_root = Path(root)
    return any(storage_root.glob('*.h5'))


def _select_hidden_states(source_store, model_id):
    selected = []
    for metadata in source_store.index.all_metadatas(store=source_store):
        if metadata.output_type != 'hidden_state': continue
        if metadata.model_id != model_id: continue
        selected.append(metadata)
    return sorted(selected, key=lambda metadata: metadata.echoframe_key)


def _validate_selected_payloads(source_store, metadatas):
    by_shard = _group_by_shard(metadatas)
    for shard_id, shard_metadatas in by_shard.items():
        dataset_paths = []
        for metadata in shard_metadatas:
            if metadata.dataset_path is None:
                message = 'selected hidden_state has no dataset path: '
                message += metadata.echoframe_key.hex()
                raise ValueError(message)
            dataset_paths.append(metadata.dataset_path)
        missing = source_store.storage.missing_datasets(shard_id,
            dataset_paths)
        if missing:
            message = f'selected hidden_state payloads are missing in shard '
            message += f'{shard_id!r}: {sorted(missing)!r}'
            raise ValueError(message)


def _load_referenced_phraser_paths(source_store, metadatas):
    source_ids = set()
    for metadata in metadatas:
        if metadata.phraser_source_id is None: continue
        source_ids.add(metadata.phraser_source_id)
    paths = {}
    for source_id in sorted(source_ids):
        path = source_store.phraser_registry.load_path(source_id)
        if path is None:
            message = 'selected hidden_state references an unregistered '
            message += f'phraser source: {source_id!r}'
            raise ValueError(message)
        paths[source_id] = path
    return paths


def _group_by_shard(metadatas):
    grouped = {}
    for metadata in metadatas:
        if metadata.shard_id is None:
            message = 'selected hidden_state has no shard: '
            message += metadata.echoframe_key.hex()
            raise ValueError(message)
        grouped.setdefault(metadata.shard_id, []).append(metadata)
    return grouped


def _assert_complete_source_shards(transfer):
    source_store = transfer['source_store']
    for shard_id, selected in transfer['metadatas_by_shard'].items():
        shard_metadatas = source_store.index.find_by_shard(shard_id,
            store=source_store)
        selected_keys = {metadata.echoframe_key for metadata in selected}
        shard_keys = {metadata.echoframe_key for metadata in shard_metadatas}
        if shard_keys != selected_keys:
            message = f'source shard {shard_id!r} contains records outside '
            message += 'the selected model hidden states'
            raise ValueError(message)


def _clear_active_shard_cache(storage):
    active_shard_ids = getattr(storage, 'active_shard_ids', None)
    if active_shard_ids is not None:
        active_shard_ids.clear()


def _verify_source_deletion(source_store, metadatas, shard_ids):
    keys = [metadata.echoframe_key for metadata in metadatas]
    remaining = source_store.load_many_metadata(keys, keep_missing=True)
    if any(metadata is not None for metadata in remaining):
        raise RuntimeError('source index deletion verification failed')
    storage_root = getattr(source_store.storage, 'root', None)
    if storage_root is None: return
    remaining_files = []
    for shard_id in shard_ids:
        shard_file = Path(storage_root) / f'{shard_id}.h5'
        if shard_file.exists(): remaining_files.append(shard_id)
    if remaining_files:
        message = f'source shard deletion verification failed: '
        message += f'{remaining_files!r}'
        raise RuntimeError(message)
