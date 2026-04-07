# echoframe

`echoframe` is a small Python package for storing intermediate model artifacts
on disk. The intended scope is hidden states, attention outputs, and derived
artifacts such as codebooks, with support for both temporary caches and
long-lived experiment stores.

It treats `phraser` as the source of truth for object metadata and stores only
metadata about model outputs plus pointers to payloads.

## Install

```bash
uv pip install git+https://git@github.com/martijnbentum/echoframe.git
```

After installation, import it as:

```python
import echoframe
```

## API

The public package exports `echoframe.Store`, `echoframe.Metadata`, and
`echoframe.STABLE_METADATA_FIELDS`.

```python
from echoframe import Store

store = Store('cache')
```

## Examples

Store hidden states for a `phraser` object key:

```python
from echoframe import Store

store = Store('cache')

metadata = store.put(
    phraser_key='phrase-123',
    collar=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
    data=[[0.1, 0.2], [0.3, 0.4]],
    tags=['exp-a', 'speaker-01'],
    to_vector_version='debug-only-version',
)

print(metadata.entry_id)
print(metadata.dataset_path)
```

Check whether output is already available:

```python
exists = store.exists(
    phraser_key='phrase-123',
    collar=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
)

print(exists)
```

Load a stored output:

```python
payload = store.load(
    phraser_key='phrase-123',
    collar=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
)
```

Load hidden-state frames for one object and layer:

```python
payload = store.load_object_frames(
    phraser_key='phrase-123',
    model_name='wav2vec2',
    layer=7,
    collar=500,
)

all_payloads = store.load_object_frames(
    phraser_key='phrase-123',
    model_name='wav2vec2',
    layer=7,
    collar=None,
)

for metadata, payload in store.iter_object_frames(
    phraser_key='phrase-123',
    model_name='wav2vec2',
    layer=7,
):
    print(metadata.collar, payload)
```

List everything stored for one `phraser_key`:

```python
entries = store.find(
    phraser_key='phrase-123',
    model_name='wav2vec2',
)

for metadata in entries:
    print(metadata.output_type, metadata.layer, metadata.collar)
```

Use collar matching when an exact collar is not available:

```python
metadata = store.find_one(
    phraser_key='phrase-123',
    collar=160,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
    match='nearest',
)
```

Find stored output or compute it:

```python
def compute_hidden_state():
    return [[0.1, 0.2], [0.3, 0.4]]


metadata, created = store.find_or_compute(
    phraser_key='phrase-123',
    collar=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
    compute=compute_hidden_state,
)

print(created)
```

Delete one stored output:

```python
deleted = store.delete(
    phraser_key='phrase-123',
    collar=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
)

print(deleted.storage_status if deleted else None)
```

List all outputs for one tag:

```python
entries = store.find_by_tag('exp-a')

for metadata in entries:
    print(metadata.phraser_key, metadata.layer, metadata.tags)
```

Find records by multiple tags:

```python
entries = store.find_by_tags(['exp-a', 'speaker-01'], match='all')
tags = store.list_tags()
```

Find records by `phraser` label:

```python
entries = store.find_by_label('hello', model_name='wav2vec2')
```

Store and query in batches:

```python
created = store.put_many([
    {
        'phraser_key': 'phrase-123',
        'collar': 150,
        'model_name': 'wav2vec2',
        'output_type': 'hidden_state',
        'layer': 7,
        'data': [[0.1, 0.2]],
        'tags': ['exp-a'],
    },
])

results = store.find_many([
    {
        'phraser_key': 'phrase-123',
        'collar': 150,
        'model_name': 'wav2vec2',
        'output_type': 'hidden_state',
        'layer': 7,
    },
])

payloads = store.load_many([
    {
        'phraser_key': 'phrase-123',
        'collar': 150,
        'model_name': 'wav2vec2',
        'output_type': 'hidden_state',
        'layer': 7,
    },
], strict=False)
```

Run maintenance checks:

```python
report = store.verify_integrity()
plans = store.compact_shards(dry_run=True)
journal = store.compaction_journal()
stats = store.shard_stats()
```

## Metadata Contract

`Metadata` contains internal and operational fields, but the stable public
contract is limited to `echoframe.STABLE_METADATA_FIELDS`:

- `phraser_key`
- `collar`
- `model_name`
- `output_type`
- `layer`
- `storage_status`
- `shard_id`
- `dataset_path`
- `shape`
- `dtype`
- `tags`
- `created_at`
- `deleted_at`

`to_vector_version` is intentionally treated as a debug/provenance field rather
than a stable contract field. It may be present on `Metadata`, but downstream
code should not rely on it as part of the long-term API surface.

## Docs

The design notes and suggested storage approach are in
[docs/approach.md](docs/approach.md).
