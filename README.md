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

The public package exports `echoframe.Store`,
`echoframe.EchoframeMetadata`, and
`echoframe.STABLE_METADATA_FIELDS`.

```python
from echoframe import Store

store = Store('cache')
```

For hidden-state retrieval, `Store` now exposes two typed loaders:

- `store.load_embedding(echoframe_key)`
- `store.load_embeddings(echoframe_keys)`
- `store.phraser_key_to_embedding(phraser_key, model_name, layer, collar=500)`
- `store.phraser_keys_to_embeddings(phraser_keys, model_name, layer,
  collar=500)`

`load_embedding(...)` returns one `Embedding` object. `load_embeddings(...)`
returns an `Embeddings` collection for multiple `echoframe_key` values.
`phraser_key_to_embedding(...)` and `phraser_keys_to_embeddings(...)` are
convenience helpers that derive hidden-state `echoframe_key` values from
`phraser` inputs.

This is an intentional API shift from the earlier loader design. The old
multi-layer `load_embeddings(...)`, `load_many_embeddings(...)`,
`TokenEmbeddings`, and `frame_aggregation` loader arguments are no longer part
of the current embedding retrieval API.

## Examples

Open a store and register models:

```python
from echoframe import Store

store = Store('cache')
record = store.register_model(
    'wav2vec2',
    huggingface_id='facebook/wav2vec2-base',
    language='en',
    size='base',
)

print(record.model_id)
print(store.load_model_metadata('wav2vec2').huggingface_id)
```

Import several model definitions from a JSON file:

```python
records = store.import_models('models.json')
print([record.model_name for record in records])
```

`models.json` should contain a JSON list:

```json
[
  {
    "model_name": "wav2vec2",
    "huggingface_id": "facebook/wav2vec2-base",
    "language": "en",
    "size": "base"
  },
  {
    "model_name": "bert-base-uncased",
    "local_path": "/models/bert-base-uncased"
  }
]
```

Store hidden states for a `phraser` object key:

```python
from echoframe.metadata import metadata_class_for_output_type

store = Store('cache')
store.register_model('wav2vec2')

phraser_key = b'phrase-123'.ljust(22, b'\0')
echoframe_key = store.make_echoframe_key(
    'hidden_state',
    model_name='wav2vec2',
    phraser_key=phraser_key,
    collar=150,
    layer=7,
)
metadata_cls = metadata_class_for_output_type('hidden_state')
metadata = metadata_cls(
    phraser_key=phraser_key,
    collar=150,
    model_name='wav2vec2',
    layer=7,
    tags=['exp-a', 'speaker-01'],
    echoframe_key=echoframe_key,
)

stored = store.save(
    echoframe_key,
    metadata,
    [[0.1, 0.2], [0.3, 0.4]],
)

print(stored.dataset_path)
```

Find stored output or compute it:

```python
def compute_hidden_state():
    return [[0.1, 0.2], [0.3, 0.4]]


metadata, created = store.find_or_compute(
    phraser_key=b'phrase-123'.ljust(22, b'\0'),
    collar=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
    compute=compute_hidden_state,
)

print(created)
```

Load a stored output:

```python
payload = store.load(echoframe_key)
payload = store.metadata_to_payload(stored)
payloads = store.metadatas_to_payloads([stored], strict=True)
```

Load hidden-state frames for one object and layer:

```python
payload = store.load_object_frames(
    phraser_key,
    model_name='wav2vec2',
    layer=7,
    collar=150,
)

all_payloads = store.load_object_frames(
    phraser_key,
    model_name='wav2vec2',
    layer=7,
    collar=None,
)

for metadata, payload in store.iter_object_frames(
    phraser_key,
    model_name='wav2vec2',
    layer=7,
):
    print(metadata.collar, payload)
```

Load one typed embedding:

```python
embedding_key = store.make_echoframe_key(
    'hidden_state',
    model_name='wav2vec2',
    phraser_key=phraser_key,
    layer=7,
    collar=150,
)
embedding = store.load_embedding(embedding_key)

print(embedding.shape)
print(embedding.layer)
print(embedding.data)
```

Load typed embeddings for several echoframe keys:

```python
embedding_key_a = store.make_echoframe_key(
    'hidden_state',
    model_name='wav2vec2',
    phraser_key=phraser_key_a,
    layer=7,
    collar=150,
)
embedding_key_b = store.make_echoframe_key(
    'hidden_state',
    model_name='wav2vec2',
    phraser_key=phraser_key_b,
    layer=7,
    collar=150,
)

embeddings = store.load_embeddings([embedding_key_a, embedding_key_b])

print(embeddings.count)
print(embeddings.phraser_keys)
stacked = embeddings.to_numpy()
```

Load typed embeddings from phraser keys with the convenience helpers:

```python
embedding = store.phraser_key_to_embedding(
    phraser_key,
    model_name='wav2vec2',
    layer=7,
    collar=150,
)

embeddings = store.phraser_keys_to_embeddings(
    [phraser_key_a, phraser_key_b],
    model_name='wav2vec2',
    layer=7,
    collar=150,
)
```

Current `Embeddings` behavior:

- all items must share one `model_name`, `output_type`, and `layer`
- duplicate `phraser_key` values are rejected
- invalid keys are skipped with a printed message
- `to_numpy()` only works when all payload shapes match exactly
- variable-length frame payloads are not automatically aggregated

List everything stored for one `phraser_key`:

```python
entries = store.find_phraser(phraser_key)

for metadata in entries:
    print(metadata.output_type, metadata.layer, metadata.collar)
```

Delete one stored output:

```python
deleted = store.delete(
    phraser_key,
    collar=150,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
)

print(deleted.storage_status if deleted else None)
```

List outputs by tag:

```python
entries = store.find_by_tag('exp-a')
entries = store.find_by_tags(['exp-a', 'speaker-01'], match='all')
tags = store.list_tags()
```

Store and query in batches:

```python
created = store.save_many([
    {
        'echoframe_key': echoframe_key,
        'metadata': metadata,
        'data': [[0.1, 0.2]],
    },
])

metadatas = store.load_many_metadata([echoframe_key])
payloads = store.load_many([echoframe_key], keep_missing=True)
```

Run maintenance checks:

```python
report = store.verify_integrity()
plans = store.compact_shards(dry_run=True)
journal = store.compaction_journal()
stats = store.shard_stats()
```

## Store Config

Each store keeps a `config.json` file next to `index.lmdb`. It stores:

- registered model metadata keyed by `model_name`

Registered model metadata records contain:

- `model_id`
- `local_path`
- `huggingface_id`
- `language`
- `size`
- `model_name`

## Metadata Contract

`EchoframeMetadata` contains internal and operational fields, but the stable
public
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

## Docs

The design notes and suggested storage approach are in
[docs/approach.md](docs/approach.md).

The intentional embedding loader API change is documented in
[docs/NOTE_embedding_loader_api_2026-04-24.md](
docs/NOTE_embedding_loader_api_2026-04-24.md).
