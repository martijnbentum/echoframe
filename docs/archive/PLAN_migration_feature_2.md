# Migration Feature 2: Store-Owned Model Registry

## Summary

Build the store-owned model registry behavior on top of the binary key
protocol foundation from [PLAN_migration_feature_1.md](./PLAN_migration_feature_1.md).

This feature is intentionally narrow. It introduces explicit model
registration/import behavior and store-owned model metadata, but it does not
switch the existing artifact metadata/index/storage path over to the new
`echoframe_key` model yet.

The purpose of this feature is to make model identity and model registry state
live inside echoframe storage in a controlled way, so later migration steps can
move current artifact metadata onto the new key protocol without inventing the
registry behavior at the same time.

## Feature 2: Store-Owned Model Registry

### Requirements

- Add store-owned model registry behavior on top of the helper code from
  migration feature 1.
- Introduce a store-facing API to register one model explicitly.
- Introduce a store-facing API to import model seed definitions from the
  checked-in helper files in `data/`.
- Keep the seed files as helper input only; the store remains the source of
  truth after import.
- Persist model registry state inside the store so model identity can be
  resolved without depending on the seed files later.
- Model registry and model metadata state should survive closing and reopening
  the store.
- Store model metadata records should keep the raw `model_name` payload, the
  assigned `model_id`, and the associated metadata needed for later lookup.
- Store model metadata records should be addressable through the new
  `model_metadata` key shape from the canonical schema plan.
- Duplicate model registration requests should resolve consistently instead of
  silently creating conflicting ids.
- Re-importing identical seed definitions should be idempotent or otherwise
  predictably stable, rather than creating ambiguous duplicate registry state.
- The registration/import path should validate required model metadata fields
  before persisting anything.
- This feature should not change the current artifact metadata path that still
  uses `EchoframeMetadata`, `entry_id`, `identity_key`, and `object_key`.
- This feature should not switch current `Store.put`, `Store.find`,
  `LmdbIndex`, or `Hdf5ShardStore` artifact behavior over to `echoframe_key`
  yet.

### Notes

- The new registry behavior should be treated as store-owned protocol state,
  not as a loose convenience cache.
- Seed import should initialize or extend the store registry in a predictable
  way, but it should not become the runtime source of truth.
- The artifact metadata path stays on the old model for now so this feature can
  land without a simultaneous identity migration.
- Any new API should be explicit about whether it is registering one model or
  loading a seed file.

### Tests

- registering one valid model stores model metadata in the store and assigns a
  stable `model_id`
- closing and reopening the store still allows the model metadata to be
  resolved
- importing a valid seed file populates the store registry with the expected
  models
- registering the same model twice does not create a second conflicting
  registry entry
- importing the same seed definitions again is idempotent or otherwise
  predictably stable
- importing duplicate model names or duplicate model ids raises clearly
- invalid model metadata input is rejected before any store write
- model metadata records remain readable from the store after the importing
  process completes
- the existing `Store.put`, `Store.find`, `Store.load`, `LmdbIndex`, and
  payload storage tests continue to pass unchanged for the current artifact
  path

### Notes On Scope

- Do not migrate current artifact records to the new binary `echoframe_key`
  yet.
- Do not change the current entry-id/object-key indexing path in this feature.
- Do not add multiple migration features to this file.
