# Migration Feature 7: Ordinary Artifact Runtime Cleanup

## Summary

Remove the remaining ordinary-artifact dependency on the legacy
`entry_id` / `identity_key` / `object_key` path after migration/rekeying has
already been completed in a previous feature.

This feature is intentionally narrow. It is the final cleanup step that makes
ordinary-artifact runtime lookup use `echoframe_key` only. Model registry and
`model_metadata` behavior stay unchanged.

## Feature 7: Ordinary Artifact Runtime Cleanup

### Requirements

- Ordinary-artifact runtime reads and writes should use `echoframe_key` only
  for all supported records.
- Remove ordinary-artifact runtime fallback to `entry_id` / `identity_key` /
  `object_key` paths.
- `Store.load_with_echoframe_key` should remain the supported ordinary-artifact
  lookup path.
- New ordinary-artifact writes should not create legacy
  `entry_id` / `object_key` records.
- Existing `model_metadata` registry behavior should remain unchanged.
- The model registry, seed import, and `model_metadata` lookup paths should
  not be modified in this feature.

### Tests

- ordinary-artifact runtime lookups resolve through `echoframe_key` only
- runtime code no longer consults the legacy `entry_id` / `object_key` path
- new ordinary-artifact writes do not create legacy-path records
- ordinary-artifact tag and shard lookups still round-trip through the
  `echoframe_key` path
- `model_metadata` tests continue to pass unchanged

### Notes On Scope

- This feature assumes the ordinary-artifact bridge, primary-path switch, and
  any required rekeying have already been completed in earlier features.
- Do not change the binary key protocol in this feature.
- Do not add multiple migration features to this file.
