'''Formatting helpers for display-oriented repr and str output.'''

from pprint import pformat
import sys

from . import lmdb_helper


RESET = '\033[0m'
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'


def format_pretty_dict(data):
    '''Return a stable pretty-printed dict string.'''
    return pformat(data, sort_dicts=False, width=80)


def truncate_text(text, max_length):
    '''Return text clipped to max_length with a trailing ellipsis.'''
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[:max_length - 3] + '...'


def format_byte_size(byte_size):
    '''Format a byte count with binary units.'''
    if byte_size is None:
        return 'unknown'
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(byte_size)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == 'B':
                return f'{int(size)} {unit}'
            return f'{size:.1f} {unit}'
        size /= 1024.0
    return f'{int(byte_size)} B'


def format_store_str(state):
    '''Return a compact store string.'''
    rows = [
        ('root', state['root']),
        ('records', state['record_count']),
        ('  live', state['live_record_count']),
        ('  deleted', state['deleted_record_count']),
        ('model count', state['model_count']),
        ('shard count', state['shard_count']),
        ('shard storage', format_byte_size(state['shard_byte_size'])),
        ('tags', _join_values(state['tags'])),
    ]
    return _format_rows('Store', rows)


def format_store_state(state):
    '''Return a detailed store string.'''
    rows = [
        ('root', state['root']),
        ('config path', state['config_path']),
        ('lmdb path', state['lmdb_path']),
        ('records', state['record_count']),
        ('  live records', state['live_record_count']),
        ('  deleted records', state['deleted_record_count']),
        ('model count', state['model_count']),
        ('shard count', state['shard_count']),
        ('shard storage', format_byte_size(state['shard_byte_size'])),
        ('lmdb', format_byte_size(state['lmdb_byte_size'])),
        ('largest shard', format_byte_size(state['largest_shard_byte_size'])),
        ('estimated', str(state['shard_storage_is_estimated']).lower()),
        ('tag count', state['tag_count']),
        ('tags', _join_values(state['tags'])),
        ('health events', state['health_event_count']),
    ]
    return _format_rows('Store', rows)


def format_model_registry_str(state):
    '''Return a compact model registry string.'''
    rows = [
        ('config path', state['config_path']),
        ('models', state['model_count']),
        ('languages', _join_values(state['languages'])),
        ('sizes', _join_values(state['sizes'])),
        ('architectures', _join_values(state['architectures'])),
        ('local paths', state['with_local_path_count']),
        ('huggingface ids', state['with_huggingface_id_count']),
    ]
    return _format_rows('ModelRegistry', rows)


def build_store_summary(store):
    '''Return compact stats for one store.'''
    shard_rows = store.index.list_shard_metadata()
    tags = store.list_tags(include_deleted=False)
    record_count = _db_entry_count(store.index, store.index.entries_db)
    live_count = sum(row.get('live_entry_count', 0) for row in shard_rows)
    deleted_count = sum(row.get('deleted_entry_count', 0)
        for row in shard_rows)
    if record_count is None:
        record_count = live_count + deleted_count
    shard_byte_size = sum(max(row.get('byte_size') or 0, 0)
        for row in shard_rows)
    return {
        'root': str(store.root),
        'record_count': record_count,
        'live_record_count': live_count,
        'deleted_record_count': deleted_count,
        'model_count': len(store.registry.model_metadatas),
        'shard_count': len(shard_rows),
        'shard_byte_size': shard_byte_size,
        'tag_count': len(tags),
        'tags': tags,
    }


def build_store_state(store):
    '''Return detailed stats for one store.'''
    state = dict(build_store_summary(store))
    shard_rows = store.index.list_shard_metadata()
    state.update({
        'config_path': str(store.config_path),
        'lmdb_path': str(store.index.path),
        'lmdb_byte_size': _path_byte_size(store.index.path),
        'largest_shard_byte_size': max(
            [max(row.get('byte_size') or 0, 0) for row in shard_rows] or [0]),
        'shard_storage_is_estimated': any(
            row.get('byte_size_is_estimated') for row in shard_rows),
        'health_event_count': len(store.get_shard_health_events()),
    })
    return state


def build_model_registry_summary(registry):
    '''Return compact stats for one model registry.'''
    metadatas = registry.model_metadatas
    return {
        'config_path': str(registry.config_path),
        'model_count': len(metadatas),
        'languages': _sorted_values(metadata.language for metadata in metadatas),
        'sizes': _sorted_values(metadata.size for metadata in metadatas),
        'architectures': _sorted_values(
            metadata.architecture for metadata in metadatas),
        'with_local_path_count': sum(
            metadata.local_path is not None for metadata in metadatas),
        'with_huggingface_id_count': sum(
            metadata.huggingface_id is not None for metadata in metadatas),
    }


def _db_entry_count(index, db_handle):
    try:
        with lmdb_helper.read_txn(index.env) as txn:
            stats = txn.stat(db=db_handle)
    except Exception:
        return None
    return stats.get('entries')


def _join_values(values):
    if not values:
        return '-'
    return ', '.join(str(value) for value in values)


def _path_byte_size(path):
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob('*')
        if entry.is_file())


def _sorted_values(values):
    return sorted({str(value) for value in values
        if value not in (None, '')})


def _format_rows(title, rows):
    width = max(len(label) for label, _ in rows)
    lines = [title]
    for label, value in rows:
        lines.append(f'{label.ljust(width)}  {_colorize_value(value)}')
    return '\n'.join(lines)


def _colorize_value(value):
    text = str(value)
    if not _use_ansi_color():
        return text
    if _looks_like_size(text):
        return YELLOW + text + RESET
    if _looks_like_number(text):
        return CYAN + text + RESET
    return GREEN + text + RESET


def _looks_like_number(text):
    if not text:
        return False
    if text.isdigit():
        return True
    if text.startswith('-') and text[1:].isdigit():
        return True
    return False


def _looks_like_size(text):
    if not text:
        return False
    parts = text.split()
    if len(parts) != 2:
        return False
    number, unit = parts
    if not unit.endswith('B'):
        return False
    try:
        float(number)
    except ValueError:
        return False
    return True


def _use_ansi_color():
    stream = sys.stdout
    return hasattr(stream, 'isatty') and stream.isatty()
