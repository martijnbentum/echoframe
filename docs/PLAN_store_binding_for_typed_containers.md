# Store Binding For Typed Containers

## Summary

Add a consistent store-access pattern for typed container objects in
`echoframe/embeddings.py` and `echoframe/codebooks.py`.

The goal is to let `Embeddings`, `TokenEmbeddings`, `Codebook`, and
`TokenCodebooks` recover the `Store` needed to load linked metadata and payloads
without forcing callers to keep a separate store handle alive.

Agreed design choices:

- support both explicit binding and lazy reopening
- use `path` as the canonical store locator
- do not add a separate store `name` or `db_name` field
- expose `.store` on the objects; no extra metadata convenience API in this
  feature
- allow manually-constructed detached objects with no `path`; `.store` should
  raise clearly in that case
- loaders should set `path` by default without eagerly binding a store

---

## Feature 1: Store Locator Metadata On Typed Containers

### Requirements

- Add an optional `path` attribute to `Embeddings`
- Add an optional `path` attribute to `TokenEmbeddings`
- Add an optional `path` attribute to `Codebook`
- Add an optional `path` attribute to `TokenCodebooks`
- `path` must represent the `Store.root` directory, because `Store(root)` is
  sufficient to reopen the same LMDB and shard layout
- `path` must be optional so detached in-memory objects remain constructible in
  tests and ad hoc code
- `path` should be excluded from repr noise if that keeps reprs compact
- `path` should not affect value equality if the repo wants these objects to
  remain mostly data-oriented containers

### Tests

- `Embeddings` accepts construction with no `path`
- `Embeddings` accepts construction with `path`
- `TokenEmbeddings` accepts construction when child tokens carry `path`
- `Codebook` accepts construction with no `path`
- `TokenCodebooks` accepts construction when child tokens carry `path`
- repr output stays compact and does not dump store internals

---

## Feature 2: Explicit Binding And Lazy `.store` Access

### Requirements

- Keep an explicit `bind_store(store)` API on `Embeddings`
- Keep an explicit `bind_store(store)` API on `TokenEmbeddings`
- Keep `bind_store(store)` on `Codebook`
- Add `bind_store(store)` on `TokenCodebooks`
- Add a `.store` property on all four container types
- `.store` must return the explicitly bound store when one is already attached
- `.store` must otherwise lazily construct `Store(path)` when `path` is known
- `.store` must cache the resulting `Store` object on the container so repeated
  access returns the same object instance
- `.store` must raise `ValueError` with a clear message when neither a bound
  store nor a `path` is available
- The implementation must not depend on LMDB env caching alone; the container
  should retain its own bound or lazily-created store reference
- The default loader behavior should be path-only, not eager binding
- `bind_store(store)` remains an explicit opt-in for callers that want an
  already-attached store

### Tests

- `bind_store(store)` makes `.store` return that exact object
- `.store` lazily opens a store from `path` when no store is bound
- a second `.store` access returns the same cached store object
- `.store` raises on detached objects with no `path`
- `TokenEmbeddings.store` works when only the collection knows the `path`
- `TokenCodebooks.store` works when only the collection knows the `path`

---

## Feature 3: Loader Propagation From `Store`

### Requirements

- `typed_loaders.load_embeddings()` must populate `Embeddings.path` from the
  source store
- `typed_loaders.load_many_embeddings()` must populate `TokenEmbeddings.path`
  from the source store
- `typed_loaders.load_many_embeddings()` must also ensure each child
  `Embeddings` carries the same `path`
- `typed_loaders.load_codebook()` must populate `Codebook.path` from the source
  store
- `typed_loaders.load_many_codebooks()` must populate `TokenCodebooks.path`
  from the source store
- `typed_loaders.load_many_codebooks()` must also ensure each child `Codebook`
  carries the same `path`
- Loaders should not eagerly call `bind_store(store)` by default
- `Embeddings.layer(n)` must preserve `path` and any bound store on the derived
  object
- `TokenEmbeddings.layer(n)` must preserve `path` and any bound store on the
  derived object

### Tests

- `Store.load_embeddings(...)` returns an `Embeddings` with `path` set to
  `store.root`
- `Store.load_embeddings(...)` should not have to bind the store eagerly
- `Store.load_many_embeddings(...)` returns a `TokenEmbeddings` with `path` set
  to `store.root`
- `Store.load_many_embeddings(...)` also gives each child token
  `path=store.root`
- `Store.load_codebook(...)` returns a `Codebook` with `path` set to
  `store.root`
- `Store.load_many_codebooks(...)` returns a `TokenCodebooks` with `path` set
  to `store.root`
- `Store.load_many_codebooks(...)` also gives each child token
  `path=store.root`
- loaded objects can resolve `.store` lazily from `path`
- `Embeddings.layer(n)` preserves store access behavior
- `TokenEmbeddings.layer(n)` preserves store access behavior

---

## Feature 4: Collection-Level Path Consistency Rules

### Requirements

- `TokenEmbeddings` must validate store-location compatibility across tokens
  when token-level store metadata is present
- `TokenCodebooks` must validate store-location compatibility across tokens
  when token-level store metadata is present
- If some child tokens are detached and the collection is given a usable
  collection-level `path`, the collection may still be valid
- The collection should prefer one shared store contract rather than silently
  mixing tokens from different store roots
- Validation errors should be explicit when token paths conflict

### Tests

- `TokenEmbeddings` accepts tokens from the same `path`
- `TokenEmbeddings` rejects tokens with conflicting paths
- `TokenCodebooks` accepts tokens from the same `path`
- `TokenCodebooks` rejects tokens with conflicting paths
- collection-level binding still works for detached child tokens

---

## Feature 5: Linked Artifact Access In Codebook Containers

### Requirements

- `Codebook.codebook_matrix` must use `.store`, not a separate ad hoc store
  field lookup path
- `Codebook.codebook_matrix` must continue to cache the loaded matrix payload
- `TokenCodebooks` should expose `.store` consistently even if it does not yet
  add new lazy-linked artifact helpers
- Existing codebook behavior should remain unchanged apart from the new path and
  store-binding contract
- Codebook loaders should rely on `path` by default; callers may still
  explicitly bind a store when they want to avoid the first lazy reopen

### Tests

- `Codebook.codebook_matrix` still lazy-loads once and caches the payload
- `Codebook.codebook_matrix` works after explicit `bind_store(store)`
- `Codebook.codebook_matrix` works after lazy store resolution from `path`
- `Codebook.codebook_matrix` raises clearly when detached and unbound

---

## Feature 6: Regression Coverage For Existing Container Behavior

### Requirements

- Existing validation for dims, layers, and frame aggregation must keep working
- Existing deduplication behavior in `TokenEmbeddings` and `TokenCodebooks`
  must keep working
- New store-related fields must not break current repr expectations unless the
  tests are intentionally updated to include the new fields
- The feature should avoid changing public loader signatures unless necessary

### Tests

- existing `tests/test_embeddings.py` cases still pass after adding store/path
  support
- existing codebook tests still pass after adding store/path support
- add focused regression tests only where store/path behavior intersects with
  existing token deduplication and layer selection

---

## Implementation Notes

- A small shared helper may be worthwhile for:
  - storing optional `_store`
  - resolving `.store`
  - normalizing `path`
- `Store.root` is a `Path`; decide whether container `path` stores a `Path` or
  a string and keep it consistent across constructors, reprs, and tests
- If equality should ignore store bindings, use dataclass fields with
  `compare=False` for cached store references
- Keep this feature scoped to typed containers and loader propagation; do not
  expand it into metadata helper methods yet

## Deferred Note

- Current collection-level `bind_store()` implementations also bind child
  tokens/codebooks in place.
- This is acceptable for the current scope, but it is not a fully isolated
  value-object model when child objects are shared across collections.
- If that becomes a real use case later, revisit collection binding semantics
  rather than extending this feature now.
