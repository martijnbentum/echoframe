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

The first implementation exposes `echoframe.Store` and
`echoframe.Metadata`.

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
    collar_ms=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
    data=[[0.1, 0.2], [0.3, 0.4]],
    to_vector_version='debug-only-version',
)

print(metadata.entry_id)
print(metadata.dataset_path)
```

Check whether output is already available:

```python
exists = store.exists(
    phraser_key='phrase-123',
    collar_ms=150,
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
    collar_ms=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
)
```

List everything stored for one `phraser_key`:

```python
entries = store.find(
    phraser_key='phrase-123',
    model_name='wav2vec2',
)

for metadata in entries:
    print(metadata.output_type, metadata.layer, metadata.collar_ms)
```

Use collar matching when an exact collar is not available:

```python
metadata = store.find_one(
    phraser_key='phrase-123',
    collar_ms=160,
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
    collar_ms=150,
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
    collar_ms=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
)

print(deleted.storage_status if deleted else None)
```

## Docs

The design notes and suggested storage approach are in
[docs/approach.md](docs/approach.md).
