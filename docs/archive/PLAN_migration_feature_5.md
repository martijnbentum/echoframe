# Migration Feature 5: Ordinary Artifact Primary Path Switch

## Summary

Complete the first ordinary-artifact migration step by switching new ordinary
artifact writes and primary reads onto the `echoframe_key` path introduced by
the bridge feature, while keeping the legacy `entry_id` / `object_key` path
available for existing records during the transition.

This feature is intentionally narrow. It builds on the model-registry work,
`model_metadata` lookup, and the ordinary-artifact bridge from migration
features 1-4, but it does not remove legacy read support yet.

## Feature 5: Ordinary Artifact Primary Path Switch

### Requirements

- Ordinary artifact records should use `echoframe_key` as the primary identity
  path for new writes.
- Ordinary artifact loads should prefer the `echoframe_key` path for new
  records.
- The existing legacy `entry_id` / `object_key` path should remain readable for
  records that were already stored before the switch.
- `Store.put`, `Store.find`, `Store.load`, `LmdbIndex`, and `Hdf5ShardStore`
  should use the new ordinary-artifact `echoframe_key` path for new writes and
  primary lookups.
- The existing bridge from `echoframe_key` to ordinary records should remain in
  place for compatibility with already stored data.
- Tag and shard-related ordinary artifact indexing should follow the new
  `echoframe_key` path for new records.
- Ordinary artifact payload handling should remain unchanged except for the key
  path used to address the record.
- `model_metadata` and store-owned registry behavior should remain unchanged in
  this feature.

### Notes

- This feature is the first real cutover for ordinary artifact records after
  the bridge exists.
- Legacy data should still load, but new writes should stop depending on the
  old artifact identity path.
- Do not remove the old `entry_id` / `object_key` path in this feature.

### Tests

- writing a new ordinary artifact stores it under the `echoframe_key` primary
  path
- loading a newly written ordinary artifact resolves through the new
  `echoframe_key` path
- existing legacy ordinary-artifact records still load correctly through the
  legacy path
- the `echoframe_key` bridge continues to resolve to the same ordinary record
  as the legacy path for existing data
- tag and shard lookup for new ordinary artifacts uses the new
  `echoframe_key` path
- ordinary-artifact tests that do not depend on the storage path continue to
  pass unchanged
- `model_metadata` and model-registry tests remain unchanged in this feature

### Notes On Scope

- Do not migrate `model_metadata` again in this feature.
- Do not remove legacy reads yet.
- Do not add multiple migration features to this file.
