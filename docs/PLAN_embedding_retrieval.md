# Embedding Retrieval — Feature Plan (echoframe)

Features that live in echoframe. See the corresponding plan in phraser
for F2–F5 (segment retrieval, frame selection, batch compute).

---

## F1 — `Embeddings` class

**What:** A named array container in `echoframe/embeddings.py`.
No dependency on phraser segments or echoframe storage — only numpy.
Imported by phraser's retrieval layer as the return type.

**Design:** `@dataclass(frozen=True)` — value object, never mutated in
place; new instances are derived via methods.

**Fields:**
- `data: np.ndarray`
- `dims: tuple[str, ...]` — axis labels, length must equal `data.ndim`
- `layers: tuple[int, ...] | None` — layer indices (e.g. `(3, 6, 12)`)
  when `'layers'` is in `dims`; `None` otherwise

**Valid `dims` combinations:**
- `('layers', 'frames', 'embed_dim')` — multiple layers, all frames
- `('frames', 'embed_dim')` — single layer, all frames
- `('layers', 'embed_dim')` — multiple layers, aggregated
- `('embed_dim',)` — single layer, aggregated

**Requirements:**
- `.shape` property delegates to `data.shape`
- Validated on init: `len(dims) == data.ndim`; `'layers'` in `dims`
  iff `layers` is not `None`; `len(layers)` matches the layers axis
  size when present
- `.layer(n)` — returns a new `Embeddings` with only layer `n` (looked
  up by value in `self.layers`, not by position); drops `'layers'`
  from `dims` and sets `layers=None`; raises `ValueError` if `n` is
  not in `self.layers`
- `Embeddings.concat(items, axis)` — concatenates a list of
  `Embeddings` along the named axis using `np.concatenate`; all items
  must share identical `dims`; when axis is `'layers'`, the `layers`
  tuples are merged in order; raises `ValueError` if axis is not
  present in `dims` or `dims` differ across items
- `__add__(other)` — sugar for `Embeddings.concat([self, other],
  axis='frames')`; raises `ValueError` if `'frames'` is not in `dims`

**Tests** (`tests/test_embeddings.py`)**:**
- init raises if `len(dims) != data.ndim`
- init raises if `'layers'` in `dims` but `layers is None`
- init raises if `layers` is set but `'layers'` not in `dims`
- init raises if `len(layers)` does not match the layers axis size
- `.layer(6)` on `layers=(3, 6, 12)` returns shape without layers
  axis and `layers=None`
- `.layer(99)` raises `ValueError`
- `concat` along `'frames'` merges frame counts, preserves other dims
- `concat` along `'layers'` merges `layers` tuples and layer axis size
- `concat` with mismatched `dims` raises `ValueError`
- `concat` with axis not in `dims` raises `ValueError`
- `__add__` on shapes without `'frames'` raises `ValueError`
- round-trip: `concat` then `.layer()` returns original sub-array

---

## F6 — `accessed_at` + recency eviction

**What:** Track last access per entry and evict entries not accessed
within a configurable window when over storage budget.

**Requirements:**
- Add `accessed_at` field to `EchoframeMetadata`; updated on every
  `load`/`load_many`
- `ECHOFRAME_RECENCY_WINDOW_DAYS` (default 30) and
  `ECHOFRAME_STORAGE_BUDGET_GB` in `.env`
- `Store.evict_by_recency()` — soft-deletes entries with `accessed_at`
  older than window, oldest-first, until under budget; skips entries
  with no `accessed_at`

**Tests** (`tests/test_eviction.py`)**:**
- `load` updates `accessed_at`
- `evict_by_recency` skips entries within the window
- Entries older than window are deleted oldest-first until budget is met
- Budget already met → no deletions

---

## Notes

- `Embeddings` is exported from `echoframe/__init__.py` so phraser can
  import it without knowing the internal module path
- F1 has no echoframe storage dependency — it can be tested in
  isolation without LMDB or HDF5
- F6 depends on F1 only indirectly; they can be implemented in either
  order
