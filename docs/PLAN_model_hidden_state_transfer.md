# Model Hidden-State Transfer Plan

## Feature: Copy hidden states for one model

### Requirements

- Add `echoframe.transfer.copy_hidden_states_for_model`.
- Require distinct source and destination stores.
- Require the destination to be empty before making any changes.
- Select every `hidden_state` record for the requested model and no other
  output type or model.
- Copy the source model registration while allowing the destination to assign
  its own model ID.
- Copy only the phraser source registrations referenced by the selected
  records. Persist paths, but do not copy live phraser store objects.
- Rebuild echoframe keys using the destination model ID.
- Preserve tags, creation timestamps, phraser source IDs, phraser keys,
  collars, and layers.
- Copy payloads in bounded batches and verify the destination index and shard
  integrity before returning.
- Leave the source store unchanged.

### Tests

- Copies only the requested model's hidden states.
- Rebuilds keys when source and destination model IDs differ.
- Preserves payloads and stable user metadata.
- Copies all and only referenced phraser registrations.
- Rejects a non-empty destination, identical stores, invalid batch sizes,
  unknown models, and models without hidden states.
- Leaves the source unchanged when copying fails.

## Feature: Move hidden states for one model

### Requirements

- Add `echoframe.transfer.move_hidden_states_for_model`.
- Perform and verify the complete copy before deleting source data.
- Confirm every source shard selected for deletion contains only the selected
  model's hidden-state records.
- Remove the selected source index records in one transaction.
- Delete the now-unreferenced shard files directly without compaction.
- Clear source storage shard caches after deleting files.
- Keep source model and phraser registrations.
- Return copied, deleted, shard, and phraser-source counts.

### Tests

- Removes source metadata and shard files only after a successful copy.
- Keeps source model and phraser registrations.
- Does not invoke compaction.
- Refuses shard deletion when a shard contains an unselected entry.
- Leaves source entries and shard files intact when destination verification
  fails.

## Documentation

### Requirements

- Document both functions as model-selected operations.
- State that `copy` is non-destructive and `move` deletes complete source
  shards after verification.
- State that phraser registrations are copied by path and remain in the source.

### Tests

- Keep examples short and runnable with two `Store` objects.
