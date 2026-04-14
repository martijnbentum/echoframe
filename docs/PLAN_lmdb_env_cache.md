## Goal

Prevent repeated `Store(root)` construction in one process from reopening the
same LMDB environment path and failing.

## Feature 1: Internal LMDB Environment Cache

### Requirements

- Repeated `Store(root)` construction in one process must not fail because the
  LMDB environment for that path is already open.
- The public `Store(root)` API must stay unchanged.
- LMDB environment caching must be internal to `echoframe`.
- Cache keys must be based on the resolved on-disk LMDB path so equivalent
  paths reuse the same environment.
- Different store roots must still get different environments.
- No new public reset/close lifecycle API is required in this iteration.

### Tests

- Creating `Store(root)` twice in one process does not raise.
- Both store instances can read and write successfully through the reused
  environment.
- Equivalent paths such as `root` and `root / '.'` reuse the same environment.
- Different roots do not reuse the same environment.

### Open Questions

- None currently.

## Suggested Implementation Order

1. Add internal LMDB env reuse in `lmdb_helper.open_env(...)`.
2. Add focused tests around repeated `Store(...)` construction and path
   equivalence.
3. Run the relevant `echoframe` test slice.
