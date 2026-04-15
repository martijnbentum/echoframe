# Migration Feature 3: Store Model Metadata Lookup

## Summary

Wire the store-owned model registry from migration feature 2 into the actual
store lookup/write path for `model_metadata`, using the new binary key helpers
from migration feature 1.

This feature is intentionally narrow. It makes model metadata queryable through
the new key protocol, but it does not yet migrate the current artifact metadata
path for ordinary records such as `EchoframeMetadata`, `entry_id`, or
`object_key`.

## Feature 3: Store Model Metadata Lookup

### Requirements

- Add store-facing APIs to write and read `model_metadata` records using the new
  key helpers.
- Resolve model metadata from the store-owned registry state introduced in
  migration feature 2.
- Use the canonical `model_metadata` key shape from
  `PLAN_binary_echoframe_key_schema.md` for persisted model metadata records.
- Add a secondary model-name lookup path using the v1 scan-key layout
  `model_name_hash, echoframe_key`.
- Keep the store as the source of truth for model metadata after seed import or
  explicit model registration.
- Keep the same store path for both explicit single-model registration and
  seed-file import so they exercise the same persistence logic.
- Preserve the raw `model_name`, assigned `model_id`, and associated metadata
  in the stored record.
- Do not switch current artifact storage, lookup, or compaction behavior over
  to the new `echoframe_key` model yet.
- Do not change the current `Store.put`, `Store.find`, `LmdbIndex`, or
  `Hdf5ShardStore` path for ordinary artifact records in this feature.

### Tests

- registering one model writes a `model_metadata` record that can be read back
  from the store
- importing seed definitions writes the expected model metadata records into
  the store
- looking up a model by name resolves to the same stored `model_metadata`
  record every time
- duplicate registration of the same model does not create conflicting model
  metadata records
- model metadata records remain readable after seed files are no longer
  consulted
- the existing artifact-path tests continue to pass unchanged for current
  `entry_id` / `object_key` behavior

### Notes

- This should be the first migration feature that exercises the new key
  protocol in live store code.
- It should stay focused on `model_metadata` only.
- Do not add a second migration feature to this file.
