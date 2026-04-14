## Feature 1: Typed Store Loaders

### Requirements

- `echoframe.Store` should expose:
  - `load_embeddings(...)`
  - `load_many_embeddings(...)`
  - `load_codebook(...)`
  - `load_many_codebooks(...)`
- `load_embeddings(...)` should support the same `layers` and
  `frame_aggregation` API as `phraser.segment_embeddings.get_embeddings(...)`
  supports today.
- `load_embeddings(...)` should build `Embeddings` objects exactly as today for:
  - single-layer requests
  - multi-layer requests
  - unaggregated frame-major data
  - aggregated `mean` and `centroid` data
- `load_many_embeddings(...)` should return `TokenEmbeddings`.
- `load_codebook(...)` should return `Codebook`.
- `load_many_codebooks(...)` should return `TokenCodebooks`.
- `load_many_*` methods should deduplicate repeated lookup identities while
  preserving first-seen order.
- The public methods should live on `Store`, but object-building logic may live
  in helper functions or helper modules to keep `store.py` small.

### Tests

- `load_embeddings(...)` returns the same `Embeddings` structure as the current
  `phraser` path for:
  - single layer, no aggregation
  - multi-layer, no aggregation
  - single layer with `mean`
  - multi-layer with `centroid`
- `load_many_embeddings(...)` returns `TokenEmbeddings` in first-seen order.
- `load_many_embeddings(...)` deduplicates repeated requests before loading.
- `load_codebook(...)` returns a store-bound `Codebook`.
- `load_many_codebooks(...)` returns `TokenCodebooks`.
- `load_many_codebooks(...)` deduplicates repeated requests before loading.


## Feature 2: Segment Orchestration In `echoframe.segment_features`

### Requirements

- Add `echoframe.segment_features` as the orchestration module for segment-based
  retrieval.
- The module should expose:
  - `get_embeddings(...)`
  - `get_embeddings_batch(...)`
  - `get_codebook_indices(...)`
  - `get_codebook_indices_batch(...)`
- The module should also expose public
  `segment_to_echoframe_key(segment)`.
- Public APIs should accept segment objects directly and assume the segment
  protocol used today:
  - `segment.key`
  - `segment.start`
  - `segment.end`
  - `segment.audio.filename`
  - optional `segment.audio.duration`
- The orchestration layer should own:
  - segment validation
  - collar application
  - store lookup
  - compute-on-miss
  - frame selection
- On cache hit, the orchestration layer should use the typed store loaders.
- On cache miss, it should compute via `to_vector`, store artifacts, then load
  via the typed store loaders.
- `to-vector` remains unchanged in this refactor.

### Tests

- `segment_to_echoframe_key(...)` preserves the current behavior for byte and
  string keys.
- `get_embeddings(...)` cache-hit path works with `model=None`.
- `get_embeddings(...)` cache-miss path requires a loaded model object.
- `get_embeddings_batch(...)` returns `TokenEmbeddings`.
- `get_codebook_indices(...)` cache-hit path works with `model=None`.
- `get_codebook_indices(...)` cache-miss path requires a loaded model object.
- `get_codebook_indices_batch(...)` returns `TokenCodebooks`.
- Segment validation and collar/window handling preserve current behavior.


## Feature 3: Reduce `phraser.segment_embeddings` To Forwarders

### Requirements

- `phraser.segment_embeddings` should become a thin wrapper around
  `echoframe.segment_features`.
- The wrapper should keep the current public function names:
  - `get_embeddings(...)`
  - `get_embeddings_batch(...)`
  - `get_codebook_indices(...)`
  - `get_codebook_indices_batch(...)`
- `phraser.segment_embeddings` should stop owning:
  - segment validation
  - collar/window logic
  - compute-and-store logic
  - typed object construction
- `segment_to_echoframe_key(...)` should no longer be defined in `phraser`.

### Tests

- Existing `phraser` tests for segment embedding/codebook retrieval continue to
  pass through the forwarding layer.
- Wrapper tests confirm the `phraser` module delegates to
  `echoframe.segment_features`.
