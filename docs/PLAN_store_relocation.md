# Store Relocation Plan

## Feature: Move a complete store

### Requirements

- Add `echoframe.transfer.move_store(source_path, destination_path)`.
- Accept filesystem paths only; do not accept or return an open `Store`.
- Require the source path to be an existing Echoframe store directory.
- Require the exact destination path not to exist.
- Require the destination parent directory to exist.
- Reject a destination located inside the source directory.
- Require the source LMDB environment to be closed in the current process.
- Document that callers must ensure no other process has the store open.
- Validate source integrity before moving.
- Move the complete directory with one same-filesystem atomic rename.
- Do not implement cross-filesystem copy, hashing, or fallback behavior.
- Reopen the store at the destination and verify its index and shard
  references.
- Preserve `config.json`, model IDs, shard IDs, dataset paths, compaction
  state, model paths, and phraser registrations without rewriting them.
- Return a report with source and destination paths, file and byte counts, and
  the destination integrity report.

### Tests

- Moves a real store and removes the source directory.
- Preserves payloads, metadata, model IDs, model configuration, phraser
  registrations, and shard references.
- Moves an empty store.
- Rejects an open source store in the current process.
- Rejects missing or malformed source paths.
- Rejects an existing destination and a destination inside the source.
- Leaves the source untouched when pre-move integrity validation fails.
- Confirms the destination can be reopened and written after relocation.
- Confirms no copy or hashing fallback is used.

## Feature: Report open LMDB environments

### Requirements

- Add `lmdb_helper.env_is_open(path)`.
- Resolve paths consistently with the existing LMDB environment cache.
- Return whether the current process has a cached environment for the path.
- Do not attempt to detect other processes.

### Tests

- Reports false before opening an environment.
- Reports true while any shared reference remains open.
- Reports false after the final reference closes.

## Documentation

### Requirements

- Document the path-only API and closed-store requirement.
- State that relocation is same-filesystem only.
- Explain that model and phraser paths remain unchanged because they refer to
  external resources.
- Show reopening the returned destination path with `Store`.

### Tests

- Keep the example short and use an exact non-existing destination root.
