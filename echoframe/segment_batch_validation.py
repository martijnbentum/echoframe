'''Generic helpers for validating segment batch inputs.'''

from __future__ import annotations

from pathlib import Path
import warnings


DEFAULT_MIN_SEGMENT_DURATION_MS = 25


def validate_segment_batch_contexts(contexts, min_duration_ms=None):
    '''Split batch contexts into valid and failed items.
    contexts:         sequence of context dicts from batch orchestration
    min_duration_ms:  minimum allowed segment duration in milliseconds
    '''
    if min_duration_ms is None:
        min_duration_ms = DEFAULT_MIN_SEGMENT_DURATION_MS
    min_duration_ms = int(min_duration_ms)
    valid_contexts = []
    failed_metadatas = []
    for context in contexts:
        failure = _validate_segment_batch_context(context, min_duration_ms)
        if failure is None:
            valid_contexts.append(context)
            continue
        failed_metadatas.append(failure)
    return valid_contexts, failed_metadatas


def warn_for_failed_metadatas(failed_metadatas, item_label='segments'):
    '''Warn about filtered batch items.'''
    failed_metadatas = tuple(failed_metadatas)
    if not failed_metadatas:
        return
    message = f'filtered {len(failed_metadatas)} invalid {item_label}'
    for metadata in failed_metadatas:
        message += '\n'
        message += _format_failed_metadata(metadata)
    warnings.warn(message, stacklevel=2)


def _validate_segment_batch_context(context, min_duration_ms):
    audio_filename = context['audio_filename']
    start_ms = context['orig_start_ms']
    end_ms = context['orig_end_ms']
    duration_ms = end_ms - start_ms
    if not Path(audio_filename).exists():
        return _failed_metadata(context, duration_ms,
            f'audio file does not exist: {audio_filename}')
    if duration_ms < min_duration_ms:
        reason = f'segment shorter than {min_duration_ms} ms'
        return _failed_metadata(context, duration_ms, reason)
    return None


def _failed_metadata(context, duration_ms, reason):
    return {
        'segment': context['segment'],
        'phraser_key': context['phraser_key'],
        'audio_filename': context['audio_filename'],
        'start_ms': context['orig_start_ms'],
        'end_ms': context['orig_end_ms'],
        'duration_ms': duration_ms,
        'reason': reason,
    }


def _format_failed_metadata(metadata):
    phraser_key = metadata['phraser_key']
    if isinstance(phraser_key, bytes):
        phraser_key = phraser_key.hex()
    text = f'phraser_key={phraser_key}'
    text += f', start_ms={metadata["start_ms"]}'
    text += f', end_ms={metadata["end_ms"]}'
    text += f', duration_ms={metadata["duration_ms"]}'
    text += f', reason={metadata["reason"]}'
    return text
