'''Formatting helpers for display-oriented repr and str output.'''

from pprint import pformat


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
