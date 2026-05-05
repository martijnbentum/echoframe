# output_storage.py review — 2026-05-05

Summary of findings from iterative review of `echoframe/output_storage.py`
and its test suite.

## Issues resolved during review

### HDF5 file opened/closed per write (fixed)
`store_many()` previously called `store()` in a loop, opening and closing
the HDF5 file for each item. Now items are grouped by `(model_name,
output_type)` and written in one `h5.File` open per group.

### Active shard found via linear scan on every write (fixed)
`_active_shard_id()` scanned shards from index 1 upward on every write.
Replaced by `_cached_active_shard_id()` backed by `active_shard_ids` dict.
The expensive scan now only runs on cold start or rollover.

### `estimated_item_sizes` caching first item size for all items (fixed)
`_group_store_items_by_shard` originally cached the first item's size per
`(model, output_type)` key and reused it for all subsequent items. For
variable-length embeddings this caused inaccurate rollover projection.
Now `_estimated_item_size(item)` is called per item.

### Batch rollover silently broken (fixed)
All items in a `store_many()` batch previously routed to a single shard
regardless of batch size, violating `max_shard_size_bytes`. Fixed by
projecting cumulative item size and calling `_next_cached_active_shard_id`
when the threshold is crossed.

### `np.asarray` fallback in `_estimated_item_size` (fixed)
Removed the `np.asarray(data).nbytes` fallback that allocated a full array
to measure size. Now returns `0` for unknown types without allocation.

## Remaining open items

### No compression on `create_dataset()`
`_store_item_in_handle` (line 312) and `compact_shard_to` (line 206) call
`create_dataset()` without compression. Adding
`compression='gzip', compression_opts=4` would reduce shard sizes by
30–60% for typical float model outputs. Both call sites must be updated
together — compaction that rewrites without compression undoes the benefit.

### `FakeGroup.create_dataset` does not accept `**kwargs`
`tests/helpers.py:107` — the fake only accepts `(name, data)`. Adding
compression to production code will immediately raise `TypeError` in every
test using `make_fake_store`. Fix: add `**kwargs` to the signature.

### Stat call on every `store()` via `_refresh_cached_shard_size`
`store_with_shard()` calls `_refresh_cached_shard_size()` after every write
to track the shard's actual byte size. This is one stat per `store()` call.
Unavoidable if accurate rollover detection is required, but could be skipped
when the running estimate is well below the threshold.

### `_refresh_cached_shard_size` does a reverse linear scan
Iterates all `active_shard_ids` values to find a matching `shard_id`.
In practice the number of `(model, output_type)` pairs is small, so impact
is negligible. A reverse map `shard_id → key` would make it O(1).

### Cache-based rollover path in `store()` untested
`FakeH5Module` creates 0-byte files so `shard_size()` always returns 0.
The `cached['byte_size'] >= max_shard_size_bytes` branch in
`_cached_active_shard_id` is never triggered. The test for rollover
(`test_store_many_rolls_over_by_estimated_item_size`) only exercises the
pre-write estimate path inside `_group_store_items_by_shard`.

### Test gaps
- No test for `store_many` with two different `(model, output_type)` pairs
  asserting two separate file opens.
- No test for multiple rollovers (3+ shards) within a single `store_many`.
- No unit test for `_estimated_item_size` with `bytes` or unknown data types.
- `test_store_many_batches_writes_by_shard`: assertions run outside the
  `tempfile.TemporaryDirectory` context manager.

## LMDB (not changed, recommendation outstanding)
Opening the LMDB environment with `writemap=True, map_async=True` reduces
per-transaction overhead by eliminating the copy-on-write path and deferring
fsync. Safe on macOS/Linux unless the process crashes mid-write. Change site:
`lmdb_helper.py:34`.
