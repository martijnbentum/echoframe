# Design Review — phraser + echoframe — 2026-06-16

## Should phraser and echoframe be merged?

**No — but the dependency structure needs fixing.**

The two packages have genuinely separate roles:

- **phraser** is a linguistic annotation store. It holds Audio, Speaker,
  Phrase, Word, Phone, and Syllable objects in LMDB, with a query API for
  navigating the hierarchy. Its dependencies are DSP-oriented (librosa,
  scipy, scikit-learn). It has value without any ML model outputs.

- **echoframe** is a model output store. It holds floating-point arrays
  (hidden states, attention, codebook indices) in HDF5 shards, indexed by
  a binary key that embeds model identity, output type, and — optionally —
  a phraser segment key. Its dependencies are storage-oriented (lmdb, h5py).

These are different layers of the same research pipeline and both have
users that don't need the other.

### The problem is the dependency direction, not the coupling

The coupling between them is real and appropriate: echoframe uses phraser
keys as identifiers, and the binary key schema hard-codes the phraser key
length (22 bytes). That is a thin, well-defined interface and not a reason
to merge.

The problem is that both packages depend on each other:

- `echoframe/pyproject.toml` declares `phraser` as a hard dependency
- `phraser/segment_embeddings.py` imports `echoframe`

This creates a circular dependency. Neither package can be installed or
understood without the other, even though most of phraser has nothing to do
with echoframe and echoframe stores without phraser-linked data don't need
phraser at all.

### Recommended fix

1. **Make phraser an optional extra in echoframe.**
   Phraser is only used in `PhraserStoreRegistry.get_store()` and
   `_load_phraser_models_module()`, both lazy imports. Formalise this:
   `pip install echoframe` works without phraser; `pip install
   echoframe[phraser]` pulls it in.

2. **Move `phraser.segment_embeddings` to echoframe.**
   The module is pure forwarding onto `echoframe.segment_features` — it exists
   only so that phraser users can call echoframe without knowing about echoframe.
   It belongs in echoframe, not phraser. Phraser should have no echoframe import
   at all. Users who want embedding retrieval import echoframe directly.

These two changes resolve the circular dependency without any merging and
without breaking the key-level coupling, which is fine.

---

## Design issues

### Structural

**1. Circular package dependency**
`echoframe` hard-depends on `phraser`; `phraser.segment_embeddings` imports
`echoframe`. Neither can be installed independently. See recommendation above.

**2. Phraser key global uniqueness is unenforced**
The binary echoframe key embeds the raw phraser_key bytes with no
phraser_source_id component. Collision-free operation relies on phraser keys
being globally unique across all phraser stores. This is documented as an
assumption but nothing in code enforces or checks it. If phraser key
generation changes, echoframe metadata would silently collide.

### Data integrity

**3. `_delete_payload` failure leaves metadata and payload inconsistent**
In `store.py`, if the HDF5 delete fails the exception is swallowed and the
LMDB metadata record is still deleted. Result: payload on disk with no index
entry. The safe order is payload first, then metadata index. Currently
reversed, and the failure is silently ignored via `print`.

**4. Silent failures via `print`**
Four places in `store.py` use `print` where behaviour should be explicit:
- `load_many` / `load_many_frames`: print a warning when keys are missing
  and return a shorter list — callers cannot detect or count misses.
- `_delete_payload`: swallows exceptions silently.
- `delete_phraser_key`: prints on success — chatty and untestable.

### Caching

**5. `store.metadatas` is permanently cached and never invalidated**
Set once via `hasattr` guard, never cleared after `save()`, `delete()`, or
`save_many()`. `store.overview()` calls `self.metadatas`, so repeated calls
after writes silently return stale counts and records.

**6. Global LMDB env cache**
`_ENV_CACHE` in `lmdb_helper.py` is module-level. Once a path is opened with
a given `map_size` it is locked in for the process lifetime. This breaks
multiprocessing (child processes inherit stale handles), leaks between tests,
and prevents reopening with a larger `map_size`.

### API / usability

**7. `store_root='echoframe'` default is a relative path**
`phraser/segment_embeddings.py` defaults `store_root` to the string
`'echoframe'`, which resolves relative to the caller's working directory.
Calling from the wrong directory silently creates or opens the wrong store.

**8. `find_by_label` scans all metadata**
`index.all_metadatas()` loads every LMDB record before label filtering.
Phraser lookups are now batched by source, so the dominant cost is gone.
The LMDB scan degrades linearly at large scale — a `by_label_db` secondary
index would fix it but requires label resolution at write time or an explicit
rebuild step.

**9. `phraser_source_id=None` resolution is fragile**
Legacy metadata without `phraser_source_id` only resolves when exactly one
phraser source is registered. With zero or multiple sources, resolution fails
with a generic error. `backfill_phraser_source_id()` exists to migrate legacy
records but is not discoverable and provides no warning at resolution time.
`phraser_source_id_to_phraser_source` should infer the single source with a
deprecation warning, and raise an actionable error pointing to
`backfill_phraser_source_id()` when ambiguous.

### Minor

**10. Compaction serialises echoframe keys as hex strings**
`compaction.py` round-trips keys through hex JSON in the compaction journal.
Works correctly but is inconsistent with the binary-first key design
everywhere else.
