'''Load and validate model seed definitions from data/ files.

Seed definitions are human-readable JSON files that describe models to be
imported into a store. This module validates structure and content only —
it does not persist anything to the store.
'''

import json
from pathlib import Path

REQUIRED_FIELDS = ('model_name',)


def load_seed_file(path):
    '''Load and validate model seed definitions from a JSON file.

    Validates structure, required fields, and duplicate model names.
    Does not persist anything to the store.

    path:   path-like pointing to a JSON seed file
    Returns a list of model definition dicts.
    Raises ValueError for structural errors or duplicate model names.
    '''
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f'seed file must contain a JSON list, got {type(data).__name__!r}')

    seen_names = set()
    records = []
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(
                f'record {i} must be a dict, got {type(record).__name__!r}')
        for field in REQUIRED_FIELDS:
            if field not in record:
                raise ValueError(
                    f'record {i} is missing required field: {field!r}')
        model_name = record['model_name']
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                f'record {i}: model_name must be a non-empty string')
        if model_name in seen_names:
            raise ValueError(
                f'duplicate model_name in seed file: {model_name!r}')
        seen_names.add(model_name)
        records.append(dict(record))

    return records
