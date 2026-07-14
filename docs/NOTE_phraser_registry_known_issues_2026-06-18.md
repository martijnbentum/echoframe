# Phraser store registry — known issues

Recorded 2026-06-18, after the phraser-store refactor (extraction of
`StoreConfig`, dict-based `PhraserStoreRegistry`, `Store.model_registry` /
`Store.phraser_registry`).

## 1. Shared config.json read-modify-write race (#5)

`ModelRegistry` and `PhraserStoreRegistry` both persist into the same
`config.json` via the shared `StoreConfig`. Each mutating call does:

    config = self._config.read()   # whole graph from disk
    ... mutate one section ...
    self._config.write(config)     # whole graph back to disk

The write itself is atomic (temp file + `os.replace`), and because each call
reads the full graph fresh and writes it back, sequential calls preserve the
other section. But the read→mutate→write sequence is **not** locked, so two
concurrent registrations (e.g. a model registration and a phraser-store
registration from different threads/processes) can interleave such that the
last writer clobbers the other's change.

This is safe for the current single-threaded, single-process usage. If that
assumption changes, add a file lock (or a single serialized config owner)
around the read-modify-write in `StoreConfig`.

A second, related hazard (found 2026-07-14): `StoreConfig.write` builds its
temp file name from the pid only (`config.json.<pid>.tmp`). Two threads in one
process therefore share the same temp path, so concurrent writes could publish
a torn `config.json` — defeating the atomic temp-file-then-replace dance. The
fix is a thread-unique temp name (e.g. add `threading.get_ident()`), but it
only matters together with the locking fix above: a unique temp name without
the lock still loses updates to last-writer-wins clobbering.

Status of the single-threaded assumption (verified 2026-07-14): `StoreWriter`
(`utils_segment_features.py`) now runs `store.save_many` on a background
thread during batch extraction, so the process is no longer single-threaded.
However, that thread's path (`save_many` -> HDF5 storage + LMDB index) never
touches `config.json`, and key building (`load_model_id`) is read-only — it
raises for unregistered models rather than auto-registering. Config writes
happen only in explicit main-thread registration calls, so both hazards in
this section remain unreachable. If registration ever becomes concurrent,
apply the file lock and the thread-unique temp name together.

## 2. Naming inconsistency: `phraser_source` vs `phraser_store` (#6)

The live API was renamed to `phraser_store` vocabulary
(`register_phraser_store`, `attach_phraser_store`, `get_phraser_store`,
`PhraserStoreRegistry`), but two persisted names deliberately kept the older
`source` wording:

- the `config.json` key `phraser_sources` (now maps `source_id -> path`)
- the `EchoframeMetadata` field `phraser_source_id`

Keeping `phraser_source_id` was an explicit decision (it is a stable, persisted
metadata field; renaming it is a data migration). The `phraser_sources` config
key now holds plain path strings rather than source records, so the "sources"
wording reads slightly oddly. Renaming either is optional and would require a
config / metadata migration; left as-is on purpose.

## 3. Pending test coverage for the store lifecycle

The phraser-store lifecycle API added after the refactor is only partially
covered:

- `test_get_store_opens_registered_phraser_store` exercises the lazy open path
  (through `load_phraser_object`), so the `load_store` first-open branch is
  covered.
- **Not yet covered:** `close_phraser_stores()`, `open_phraser_stores()`, and
  the fail-loud-on-closed behaviour of `load_store` (a closed cached store must
  raise with the "call open_phraser_stores()" hint rather than silently
  reopen). The `load_phraser_store` rename (from `get_phraser_store`) also has
  no direct test.

When picking tests back up, add a close -> fail-loud -> reopen -> load
round-trip against a real phraser store, and assert `close_phraser_stores()` /
`open_phraser_stores()` return the expected counts and leave never-opened
sources untouched.
