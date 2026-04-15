# Migration Feature 6: Migrate Remaining Ordinary Artifacts

## Summary

Add a one-time migration/rekey helper for remaining ordinary-artifact records
so legacy records can be moved onto the `echoframe_key` path introduced in the
earlier migration steps.

This feature is intentionally narrow. It performs the data migration step, but
it does not remove the legacy runtime fallback yet. Ordinary-artifact records
remain readable through the existing compatibility path during this feature,
and `model_metadata` / model-registry behavior stay unchanged.

## Feature 6: Migrate Remaining Ordinary Artifacts

### Requirements

- Add a one-time helper to migrate or rekey remaining ordinary-artifact
  records onto the `echoframe_key` path.
- Preserve payloads during migration.
- Preserve tag links during migration.
- Preserve shard links during migration.
- Preserve related metadata during migration.
- Keep the existing runtime fallback in place during this feature.
- Do not remove the legacy ordinary-artifact runtime fallback yet.
- Keep the migration helper focused on ordinary-artifact records only.
- `model_metadata` lookup and store-owned model registry behavior should remain
  unchanged in this feature.

### Notes

- This feature is the data-migration step before the final runtime cleanup.
- It should prepare the store for the later removal of the legacy path, but it
  should not perform that removal yet.
- Do not change the model registry or `model_metadata` path in this feature.
- Do not add another migration feature to this file.

### Tests

- migrating ordinary-artifact records preserves payload bytes
- migrating ordinary-artifact records preserves tag links
- migrating ordinary-artifact records preserves shard links
- migrating ordinary-artifact records preserves related metadata
- migrated records remain readable through the existing runtime fallback
- the migration helper handles already-migrated records predictably
- `model_metadata` and store registry tests remain unchanged in this feature
