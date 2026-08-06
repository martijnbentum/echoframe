# echoframe

`echoframe` is a small Python package for storing intermediate model artifacts
and acoustic features on disk. The intended scope is hidden states, attention
outputs, derived artifacts such as codebooks, CNN feature-extractor output,
and frame-aligned features such as MFCCs, with support for both temporary
caches and long-lived experiment stores.

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

### Development

Activate the versioned git hooks once per clone:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook bumps the patch version in `pyproject.toml` on every
commit, unless the commit already stages a version change (a manual
minor/major bump wins).

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

For acoustic features, the public helpers are:

- `store_mfcc(segment, store, tags=None, verbose=False)`
- `store_mfcc_batch(segments, store, tags=None, verbose=True)`
- `store.phraser_key_to_acoustic_feature(phraser_key, feature_name)`
- `store.phraser_keys_to_acoustic_features(phraser_keys, feature_name)`

Acoustic features use `output_type='acoustic_feature'` and a `feature_name`
such as `'mfcc'`. They do not require a registered model, model name, layer,
or collar.

For CNN feature-extractor output (the convolutional stage of a Wav2Vec2-style
model, before its transformer layers), `Store` exposes the same typed-loader
shape as hidden states:

- `store.load_cnn_feature(echoframe_key)`
- `store.load_cnn_features(echoframe_keys)`
- `store.phraser_key_to_cnn_feature(phraser_key, model_name, collar=500)`
- `store.phraser_keys_to_cnn_features(phraser_keys, model_name, collar=500)`

CNN features use `output_type='cnn'` and, like `codebook_indices`, have no
`layer` — a CNN feature is one fixed conv-stage payload per segment, not one
per transformer layer.

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
from echoframe.metadata import EchoframeMetadata

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
metadata = EchoframeMetadata(
    echoframe_key,
    store=store,
    model_name='wav2vec2',
    tags=['exp-a', 'speaker-01'],
)

stored = store.save(
    echoframe_key,
    metadata,
    [[0.1, 0.2], [0.3, 0.4]],
)

print(stored.dataset_path)
```

Load a stored output:

```python
payload = store.load(echoframe_key)
payload = store.metadata_to_payload(stored)
payloads = store.metadatas_to_payloads([stored])
```

Store and load MFCCs already computed by a `phraser` segment:

```python
from echoframe import store_mfcc

store_mfcc(segment, store)
mfcc = store.phraser_key_to_acoustic_feature(segment.key, 'mfcc')

key = store.make_echoframe_key(
    'acoustic_feature',
    feature_name='mfcc',
    phraser_key=segment.key,
)
metadata = store.load_metadata(key)

print(metadata.output_type)   # acoustic_feature
print(metadata.feature_name)  # mfcc
```

No `mfcc` pseudo-model registration is needed. The metadata has no
`model_name`, `model_id`, `layer`, or `collar`.

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

Store and load a CNN feature-extractor payload the same way, using
`output_type='cnn'` (no `layer`):

```python
cnn_key = store.make_echoframe_key(
    'cnn',
    model_name='wav2vec2',
    phraser_key=phraser_key,
    collar=150,
)
metadata = EchoframeMetadata(cnn_key, store=store, model_name='wav2vec2')
store.save(cnn_key, metadata, [[0.1, 0.2], [0.3, 0.4]])

cnn_feature = store.load_cnn_feature(cnn_key)
print(cnn_feature.shape)
print(cnn_feature.data)

cnn_feature = store.phraser_key_to_cnn_feature(
    phraser_key,
    model_name='wav2vec2',
    collar=150,
)
```

`CNNFeature` supports the same frame-slicing API as `Embedding` (`to_frames`,
`slice_time`, `slice_segment`, `middle_frame_time`, `aggregate_time`, ...),
since CNN feature-extractor output is frame-aligned the same way hidden
states are — just without a `layer`. `CNNFeature.sub_feature(...)` returns a
`SlicedCNNFeature` (the `CNNFeature` analogue of `SlicedEmbedding`), and
`CNNFeature.sub_features(object_class, ...)` slices to every descendant of a
class at once, exactly like `Embedding.sub_embeddings(...)`.

Bind a live phraser store so phraser segments can reach their stored embeddings
directly. `attach_phraser_store` registers the source and sets a back-reference
(`phraser_store.echoframe_store = store`):

```python
store.attach_phraser_store('cgn-main', phraser_store)

word = phraser_store.words.get(label='hello')
embedding = word.embedding('wav2vec2', layer=7)   # one Embedding
embedding.data
```

Slice a stored embedding to a descendant phraser object (word, syllable, or
phone) with `Embedding.sub_embedding(...)`. This is useful when a hidden state
is stored at a coarser level, such as a phrase:

```python
phrase_embedding = word.phrase.embedding('wav2vec2', layer=7)
phone_embedding = phrase_embedding.sub_embedding(phone)   # SlicedEmbedding

phone_embedding.data           # payload rows sliced to the phone
phone_embedding.parent_class   # 'Phrase'
phone_embedding.object_class   # 'Phone'
```

`sub_embedding(phraser_object, aggregate=None)` returns a `SlicedEmbedding`
view: `aggregate=None` keeps the 2D rows, while `'mean'` or `'middle'` return a
1D vector. A `SlicedEmbedding` exposes `parent_embedding`,
`parent_phraser_key`, `parent_collar`, `parent_class`, `object_class`,
`phraser_object`, `data`, `rows`, `model_name`, `output_type`, and `layer`. On
the phraser side, `segment.embedding(model_name, layer, fallback=True)` walks
ancestors when nothing is stored for the segment itself and returns the nearest
ancestor embedding already sliced to the segment.

Slice to every descendant of a class at once with
`Embedding.sub_embeddings(object_class, aggregate=None)`. It returns a list of
`SlicedEmbedding`s, one per descendant phraser object of that class:

```python
phone_embeddings = phrase_embedding.sub_embeddings('phone')        # list
word_means = phrase_embedding.sub_embeddings('word', aggregate='mean')
```

`object_class` is the descendant segment type (`'word'`, `'syllable'`, or
`'phone'`; singular or plural, case-insensitive) and must be a descendant of
the embedding's own phraser class, otherwise a `ValueError` is raised. A
descendant whose span falls outside the stored payload raises the same
`no frames overlap` error as `sub_embedding`. Both `Embedding` and
`SlicedEmbedding` expose their own phraser class as `object_class` and show it
in their `repr` (e.g. `Embedding(shape=..., layer=7, class=Phrase)`);
`phraser_object` and `object_class` resolve lazily, so constructing an
`Embedding` does not require the phraser store to be reachable.

Current `Embeddings` behavior:

- all items must share one `model_name`, `output_type`, and `layer`
- duplicate `phraser_key` values are rejected
- invalid keys are skipped with a logged warning
- `to_numpy()` only works when all payload shapes match exactly
- variable-length frame payloads are not automatically aggregated

List everything stored for one `phraser_key`:

```python
entries = store.find_phraser(phraser_key)

for metadata in entries:
    print(metadata.output_type, metadata.layer, metadata.collar)
```

Delete stored outputs, either by echoframe key or by phraser key plus
filters:

```python
store.delete(echoframe_key)
store.delete_many([echoframe_key_a, echoframe_key_b])

store.delete_phraser_key(
    phraser_key,
    model_name='wav2vec2',
    output_type='hidden_state',
    layer=7,
    collar=150,
)
```

`delete_phraser_key(...)` deletes every stored output that matches the given
filters (`collar_match` can be `'exact'`, `'min'`, `'max'`, or `'nearest'`)
and prints how many records were deleted (silence with `verbose=False`).
Acoustic features can be deleted without a model name by passing
`output_type='acoustic_feature'` and `feature_name='mfcc'`.

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

`compute_embeddings_batch(...)` submits prepared results to a background
writer as one `save_many()` chunk per configured batch. The configured
`batch_size` is also the storage group size; when it is `None`, storage writes
use groups of 32 segment results. `store_queue_size` controls how many complete
write groups can wait while inference continues.

Run maintenance checks:

```python
report = store.verify_integrity()
plans = store.compact_shards(dry_run=True)
journal = store.compaction_journal()
stats = store.shard_stats()
```

### Transfer hidden states by model

Copy every hidden-state payload for one model into a new empty store:

```python
from echoframe import Store
from echoframe.transfer import copy_hidden_states_for_model

source = Store('full-cache')
destination = Store('wav2vec2-cache')

result = copy_hidden_states_for_model(
    source,
    destination,
    model_name='wav2vec2',
)
print(result['copied_count'])
```

Use `move_hidden_states_for_model(...)` with the same arguments to delete the
copied entries from the source after the destination has been verified. The
selection is always all `hidden_state` records for `model_name`; other models
and output types remain untouched.

The destination must have no stored entries, registered models, registered
phraser sources, or shard files. The transfer registers the selected model
with a destination-local model ID, rebuilds its echoframe keys, and copies
only the phraser source paths referenced by the selected records. Live
phraser store objects are not copied. A move keeps the source model and
phraser registrations, but deletes the complete source hidden-state shard
files directly without compaction.

### Relocate a complete store

Relocate a complete store on the same filesystem:

```python
from echoframe import Store
from echoframe.transfer import move_store

store = Store('cache')
# Finish work and close every Store instance using this path.
store.close()

result = move_store('cache', '/data/echoframe/cache')
moved_store = Store(result['destination_path'])
# Use the moved store, then close it normally.
moved_store.close()
```

`move_store(...)` relocates the complete directory with an atomic filesystem
rename. The exact destination path must not exist, and all processes using the
source must close it first. Its parent directory must already exist.
Cross-filesystem moves are not supported.

Stored shard and dataset references remain valid because they are relative to
the store root. Registered model paths and phraser store paths are preserved
unchanged because they refer to external resources. The function verifies
integrity before and after relocation and returns file, byte, and integrity
details.

### Overwrites and garbage

Re-saving an existing echoframe key writes the new payload to the current
active shard and deliberately leaves the old bytes behind in their original
shard file: saves stay fast and overwriting a key is an edge case. The
leftover bytes are counted as reclaimable garbage —
`store.overview(include_garbage=True)` reports them per shard and
`store.compact_shards()` rewrites only shards that hold enough garbage to be
worth reclaiming.

## Store Config

Each store keeps a `config.json` file next to `index.lmdb`. It stores:

- registered model metadata keyed by `model_name`
- registered phraser sources keyed by `phraser_source_id`

Registered model metadata records contain:

- `model_id`
- `local_path`
- `huggingface_id`
- `language`
- `size`
- `model_name`

### Concurrency

A store expects a single writer process at a time. The active shard cursor
lives in process memory, so two processes writing to the same store can pick
the same `.h5` shard and corrupt it with concurrent appends. Concurrent
readers are fine (LMDB handles that), and multiple writers can safely target
different stores. If you need parallel feature extraction, write to separate
stores or serialize the writes yourself.

### Logging

Warnings (skipped keys, failed deletes, missing phraser paths) go through
Python's `logging` under the `echoframe` logger and are visible on stderr by
default. Batch progress prints to stdout and is controlled per call with
`verbose=True/False`. To tune warning output:

```python
import logging

logging.getLogger('echoframe').setLevel(logging.ERROR)  # silence warnings
logging.basicConfig(level=logging.DEBUG)  # include per-key diagnostics
```

### Phraser Source Migration

Older metadata may have `phraser_source_id=None`. Register the phraser source
and backfill those records explicitly:

```python
from echoframe import Store

store = Store('cache')
store.register_phraser_store('cgn-main', '/data/cgn_lmdb')
updated_count = store.backfill_phraser_source_id('cgn-main')

print(updated_count)
```

Only metadata records without a `phraser_source_id` are updated.

## Metadata Contract

`EchoframeMetadata` contains internal and operational fields, but the stable
public contract is limited to `echoframe.STABLE_METADATA_FIELDS`:

- `model_name`
- `output_type`
- `shard_id`
- `dataset_path`
- `shape`
- `tags`
- `created_at`

## Docs

The design notes and suggested storage approach are in
[docs/approach.md](docs/approach.md).

The intentional embedding loader API change is documented in
[docs/NOTE_embedding_loader_api_2026-04-24.md](
docs/NOTE_embedding_loader_api_2026-04-24.md).
