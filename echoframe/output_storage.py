'''HDF5 shard storage for model output payloads.'''

import re
import time
from pathlib import Path

from .metadata import utc_now
from .metadata import EchoframeMetadata


def sanitize_name(value):
    '''Convert a user-facing value into a shard-safe name.'''
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', value).strip('_') or 'unknown'


class Hdf5ShardStore:
    '''Store payloads in rolling HDF5 shard files.'''

    STAT_RETRY_DELAYS = (0.1, 0.3, 0.6, 0.9, 3.0)
    MAX_SCAN_SUFFIXES = None

    def __init__(self, root, max_shard_size_bytes=1_000_000_000,
        h5_module=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_shard_size_bytes = max_shard_size_bytes
        self.h5 = h5_module or self._import_h5()
        self.health_events = []
        self.max_health_events = 500

    def _import_h5(self):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError('h5py is required to use Store') from exc
        return h5py

    def store(self, metadata, data):
        '''Store payload data and return updated metadata.'''
        if metadata.output_type == 'model_metadata':
            if data is not None:
                raise ValueError('model_metadata does not use HDF5 payload data')
            return metadata
        shard_id = self._active_shard_id(model_name=metadata.model_name,
            output_type=metadata.output_type)
        return self.store_with_shard(metadata, data=data, shard_id=shard_id)

    def store_many(self, items):
        '''Store multiple payloads and return updated metadata records.'''
        stored = []
        for item in items:
            stored.append(self.store(item['metadata'], data=item['data']))
        return stored

    def store_with_shard(self, metadata, data, shard_id):
        '''Store payload data in a specific shard.'''
        dataset_path = self._dataset_path(metadata)
        dataset_name = metadata.format_echoframe_key()
        file_path = self.root / f'{shard_id}.h5'

        with self.h5.File(file_path, 'a') as handle:
            group = handle.require_group(f'/layer_{metadata.layer:04d}')
            if dataset_name in group:
                del group[dataset_name]
            dataset = group.create_dataset(dataset_name, data=data)
            shape = tuple(getattr(dataset, 'shape', ()) or ())
            dtype = getattr(dataset, 'dtype', None)
            if dtype is None: dtype = getattr(data, 'dtype', 'unknown')
            dtype = str(dtype)
        return metadata.__class__(phraser_key=metadata.phraser_key,
            collar=metadata.collar, model_name=metadata.model_name,
            output_type=metadata.output_type, layer=metadata.layer,
            echoframe_key=metadata.echoframe_key,
            storage_status=metadata.storage_status, shard_id=shard_id,
            dataset_path=dataset_path, shape=shape, dtype=dtype,
            tags=metadata.tags, created_at=metadata.created_at,
            deleted_at=metadata.deleted_at, accessed_at=metadata.accessed_at,
            model_id=metadata.model_id, local_path=metadata.local_path,
            huggingface_id=metadata.huggingface_id,
            language=metadata.language)

    def load(self, metadata):
        '''Load stored payload data.'''
        if metadata.shard_id is None or metadata.dataset_path is None:
            raise ValueError('metadata does not point to a stored payload')
        file_path = self.root / f'{metadata.shard_id}.h5'
        with self.h5.File(file_path, 'r') as handle:
            return handle[metadata.dataset_path][()]

    def delete(self, metadata):
        '''Best-effort payload deletion.

        HDF5 files are not compacted here. Compaction should be a later,
        offline maintenance step.
        '''
        if metadata.shard_id is None or metadata.dataset_path is None:
            return
        file_path = self.root / f'{metadata.shard_id}.h5'
        if not file_path.exists():
            return
        with self.h5.File(file_path, 'a') as handle:
            if metadata.dataset_path in handle:
                del handle[metadata.dataset_path]

    def compact_shard(self, shard_id, entries):
        '''Rewrite live datasets from one shard into a new shard.'''
        next_shard_id = self._replacement_shard_id(shard_id)
        return self.compact_shard_to(shard_id, entries,
            target_shard_id=next_shard_id)

    def compact_shard_to(self, shard_id, entries, target_shard_id,
        delete_source=True):
        '''Rewrite live datasets from one shard into a target shard.'''
        if not entries:
            if delete_source:
                self._delete_file(shard_id)
            return []

        old_file_path = self.root / f'{shard_id}.h5'
        new_file_path = self.root / f'{target_shard_id}.h5'
        updated = []

        with self.h5.File(old_file_path, 'r') as source:
            with self.h5.File(new_file_path, 'a') as target:
                for metadata in entries:
                    dataset = source[metadata.dataset_path][()]
                    group = target.require_group(f'/layer_{metadata.layer:04d}')
                    dataset_name = metadata.format_echoframe_key()
                    if dataset_name in group:
                        del group[dataset_name]
                    created = group.create_dataset(dataset_name,
                        data=dataset)
                    updated.append(metadata.__class__(
                        phraser_key=metadata.phraser_key,
                        collar=metadata.collar,
                        model_name=metadata.model_name,
                        output_type=metadata.output_type,
                        layer=metadata.layer,
                        echoframe_key=metadata.echoframe_key,
                        storage_status=metadata.storage_status,
                        shard_id=target_shard_id,
                        dataset_path=self._dataset_path(metadata),
                        shape=tuple(getattr(created, 'shape', ()) or ()),
                        dtype=str(getattr(created, 'dtype', 'unknown')),
                        tags=metadata.tags,
                        created_at=metadata.created_at,
                        deleted_at=metadata.deleted_at,
                        accessed_at=metadata.accessed_at,
                        model_id=metadata.model_id,
                        local_path=metadata.local_path,
                        huggingface_id=metadata.huggingface_id,
                        language=metadata.language,
                    ))

        if delete_source:
            self._delete_file(shard_id)
        return updated

    def dataset_exists(self, shard_id, dataset_path):
        '''Return whether a dataset exists in a shard.'''
        file_path = self.root / f'{shard_id}.h5'
        if not self._path_exists(file_path):
            return False
        with self.h5.File(file_path, 'r') as handle:
            return dataset_path in handle

    def shard_size(self, shard_id):
        '''Return shard file size in bytes.'''
        file_path = self.root / f'{shard_id}.h5'
        if not self._path_exists(file_path):
            return 0
        return self._file_size(file_path)

    def _active_shard_id(self, model_name, output_type):
        stem = f'{sanitize_name(model_name)}_{sanitize_name(output_type)}'
        index = 1
        scanned = 0
        while True:
            scanned += 1
            self._raise_if_scan_limit_reached(scanned, purpose='rollover',
                shard_id=f'{stem}_{index:04d}')
            shard_id = f'{stem}_{index:04d}'
            file_path = self.root / f'{shard_id}.h5'
            state = self._shard_path_state(shard_id, file_path,
                purpose='rollover')
            if state['state'] == 'missing':
                return shard_id
            if state['state'] == 'unreadable':
                self._record_health_event('skip_unreadable_shard',
                    shard_id=shard_id, purpose='rollover',
                    error=state['error'])
                self._sleep_on_unreadable_probe(scanned,
                    purpose='rollover', shard_id=shard_id,
                    error=state['error'])
                index += 1
                continue
            if state['byte_size'] < self.max_shard_size_bytes:
                return shard_id
            index += 1

    def _dataset_path(self, metadata):
        return f'/layer_{metadata.layer:04d}/{metadata.format_echoframe_key()}'

    def _file_size(self, file_path):
        return file_path.stat().st_size

    def _replacement_shard_id(self, shard_id):
        stem, _, suffix = shard_id.rpartition('_')
        if not stem or not suffix.isdigit():
            raise ValueError('invalid shard_id')
        index = int(suffix) + 1
        unreadable_candidates = []
        scanned = 0
        while True:
            scanned += 1
            self._raise_if_scan_limit_reached(scanned, purpose='replacement',
                shard_id=f'{stem}_{index:04d}')
            candidate = f'{stem}_{index:04d}'
            file_path = self.root / f'{candidate}.h5'
            state = self._shard_path_state(candidate, file_path,
                purpose='replacement')
            if state['state'] == 'missing':
                if unreadable_candidates:
                    self._record_health_event(
                        'replacement_skip_unreadable_candidates',
                        selected_shard_id=candidate,
                        skipped_shard_ids=','.join(unreadable_candidates),
                    )
                return candidate
            if state['state'] == 'unreadable':
                unreadable_candidates.append(candidate)
                self._sleep_on_unreadable_probe(scanned,
                    purpose='replacement',
                    shard_id=candidate, error=state['error'])
            index += 1

    def _delete_file(self, shard_id):
        file_path = self.root / f'{shard_id}.h5'
        if hasattr(self.h5, 'files'):
            self.h5.files.pop(str(file_path), None)
        if self._path_exists(file_path):
            file_path.unlink()

    def get_shard_health_events(self, limit=None):
        '''Return recent shard health events.'''
        if limit is None:
            return list(self.health_events)
        return list(self.health_events[-limit:])

    def validate_shard(self, shard_id, entries=None, read_data=False):
        '''Validate that a shard can be opened and its datasets can be read.
        shard_id:    shard identifier
        entries:     optional metadata records expected in the shard
        read_data:   read payloads instead of only checking presence
        '''
        file_path = self.root / f'{shard_id}.h5'
        report = {
            'shard_id': shard_id,
            'file_path': str(file_path),
            'ok': True,
            'exists': True,
            'open_error': None,
            'missing_echoframe_keys': [],
            'unreadable_echoframe_keys': [],
        }
        try:
            file_path.stat()
        except FileNotFoundError:
            report['ok'] = False
            report['exists'] = False
            report['open_error'] = 'missing shard file'
            return report
        except OSError as exc:
            self._record_health_event('validate_stat_failure',
                shard_id=shard_id, error=str(exc))

        try:
            with self.h5.File(file_path, 'r') as handle:
                for metadata in entries or []:
                    if metadata.dataset_path not in handle:
                        report['ok'] = False
                        report['missing_echoframe_keys'].append(
                            metadata.format_echoframe_key())
                        continue
                    if not read_data:
                        continue
                    try:
                        handle[metadata.dataset_path][()]
                    except Exception as exc:
                        report['ok'] = False
                        report['unreadable_echoframe_keys'].append({
                            'echoframe_key_hex': metadata.format_echoframe_key(),
                            'dataset_path': metadata.dataset_path,
                            'error': str(exc),
                        })
        except Exception as exc:
            report['ok'] = False
            report['open_error'] = str(exc)
        return report

    def _shard_path_state(self, shard_id, file_path, purpose):
        size_info = self._size_info_with_retries(shard_id, file_path,
            purpose=purpose)
        if not size_info['exists']:
            return {'state': 'missing', 'byte_size': None, 'error': None}
        if size_info['byte_size'] is None:
            return {
                'state': 'unreadable',
                'byte_size': None,
                'error': size_info['error'],
            }
        return {
            'state': 'exists',
            'byte_size': size_info['byte_size'],
            'error': None,
        }

    def _size_info_with_retries(self, shard_id, file_path, purpose):
        attempts = len(self.STAT_RETRY_DELAYS) + 1
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return {
                    'exists': True,
                    'byte_size': file_path.stat().st_size,
                    'is_estimated': False,
                    'error': None,
                }
            except FileNotFoundError:
                return {
                    'exists': False,
                    'byte_size': None,
                    'is_estimated': False,
                    'error': None,
                }
            except OSError as exc:
                last_error = str(exc)
                self._record_health_event('stat_failure',
                    shard_id=shard_id, purpose=purpose,
                    attempt=attempt, error=last_error)
                if attempt > len(self.STAT_RETRY_DELAYS):
                    break
                delay = self.STAT_RETRY_DELAYS[attempt - 1]
                self._record_health_event('stat_retry',
                    shard_id=shard_id, purpose=purpose,
                    attempt=attempt, delay_seconds=delay)
                time.sleep(delay)
        return {
            'exists': True,
            'byte_size': None,
            'is_estimated': False,
            'error': last_error or 'stat retries exhausted',
        }

    def _record_health_event(self, event_type, **fields):
        event = {'timestamp': utc_now(), 'event_type': event_type}
        event.update(fields)
        self.health_events.append(event)
        if len(self.health_events) > self.max_health_events:
            self.health_events = self.health_events[-self.max_health_events:]

    def _sleep_on_unreadable_probe(self, count, purpose, shard_id, error):
        if count > len(self.STAT_RETRY_DELAYS):
            return
        delay = self.STAT_RETRY_DELAYS[count - 1]
        self._record_health_event('probe_retry',
            purpose=purpose, shard_id=shard_id, error=error,
            delay_seconds=delay)
        time.sleep(delay)

    def _raise_if_scan_limit_reached(self, count, purpose, shard_id):
        if self.MAX_SCAN_SUFFIXES is None:
            return
        if count <= self.MAX_SCAN_SUFFIXES:
            return
        message = (
            f'unable to probe shard path after scanning '
            f'{self.MAX_SCAN_SUFFIXES} {purpose} candidates; '
            f'latest shard={shard_id}'
        )
        self._record_health_event('probe_hard_fail',
            purpose=purpose, shard_id=shard_id,
            scanned_candidates=self.MAX_SCAN_SUFFIXES)
        raise RuntimeError(message)

    def _path_exists(self, file_path):
        try:
            file_path.stat()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return True
