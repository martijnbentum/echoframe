# Migration Feature 4: Ordinary Artifact Echoframe-Key Bridge

## Summary

Introduce the first migration step for ordinary artifact records by adding a
parallel `echoframe_key` lookup path while keeping the current
`EchoframeMetadata` / `entry_id` / `identity_key` / `object_key` storage and
read behavior intact.

This feature is intentionally narrow. It starts the transition for ordinary
artifact records after the model-registry and `model_metadata` work from
migration features 1-3, but it does not replace the current artifact identity
model yet.

## Feature 4: Ordinary Artifact Echoframe-Key Bridge

### Requirements

- Add a parallel `echoframe_key` path for ordinary artifact records.
- Ordinary artifact metadata should be able to derive or expose an
  `echoframe_key` without changing the current `entry_id` identity.
- Persist a secondary lookup from `echoframe_key` to the existing ordinary
  artifact record.
- Keep the current `entry_id`-based primary record path for ordinary artifacts
  unchanged in this feature.
- Make `Store.load_with_echoframe_key` resolve ordinary records through the new
  `echoframe_key` path.
- Keep `Store.put`, `Store.find`, `Store.load`, `LmdbIndex`, and
  `Hdf5ShardStore` on the existing ordinary artifact identity path for now,
  except for the new `echoframe_key` bridge lookup.
- Keep tag, shard, payload, and compaction behavior on the current legacy
  artifact path in this feature.
- Do not change the `model_metadata` registry path or any store-owned registry
  behavior in this feature.

### Tests

- writing an ordinary artifact record stores data that can be loaded by the new
  `echoframe_key` bridge
- the `echoframe_key` bridge resolves to the same ordinary record as the legacy
  `entry_id` path
- ordinary artifact writes still continue to support the existing
  `entry_id`/`object_key` behavior unchanged
- `Store.load_with_echoframe_key` returns the expected ordinary record through
  the new bridge path
- tag and shard related ordinary artifact tests continue to pass unchanged
- model registry and `model_metadata` tests remain unchanged in this feature

### Notes

- Keep this feature focused on the first ordinary-artifact bridge only.
- Do not migrate the full artifact metadata model yet.
- Do not add additional migration features to this file.
