# Migration Feature 1: Binary Key Protocol Foundation

## Summary

Introduce the new binary key protocol as standalone helper code before
switching the existing `metadata.py`, `index.py`, `store.py`, or storage
callers.

This feature is the first migration step because it creates the new protocol
surface in isolation:

- `echoframe_key` becomes the canonical artifact-key concept in helper code
- `phraser_key` remains the source key concept
- `struct_helper.py` owns binary layout definitions
- `key_helper.py` owns packing, unpacking, hashing, and registry helpers
- seed-definition files in `data/` provide the input shape for model registry
  protocol data

The goal is to make the new key model available and testable without changing
runtime behavior in the current LMDB index or payload store.

## Feature 1: Add Binary Key Protocol Helpers

### Requirements

- Add `echoframe/struct_helper.py`
- Add `echoframe/key_helper.py`
- `struct_helper.py` should define the fixed-width binary layouts for the v1
  canonical key families from
  `docs/PLAN_binary_echoframe_key_schema.md`
- `key_helper.py` should provide deterministic pack/unpack helpers for:
  - `hidden_state` and `attention`
  - `codebook_indices`
  - `codebook_matrix`
  - `model_metadata`
- `key_helper.py` should provide deterministic secondary scan-key builders for
  the v1 lookup patterns:
  - tag lookup
  - model-name lookup
  - output-type lookup
- `key_helper.py` should provide the shared `model_name_hash` helper using the
  canonical `blake2b` 8-byte digest contract
- helper code should use the stable binary field sizes from the canonical plan:
  - `model_id`: `uint16`
  - `output_type_id`: `uint8`
  - `layer`: `uint8`
  - `collar`: `uint16`
  - `phraser_key`: fixed `22` bytes for segment keys
  - `model_name_hash`: `8` bytes
- helper code should treat binary layout as big-endian and lexicographically
  ordered
- add seed-definition loading and validation for model registry data in
  `data/`
- seed loading should validate structure, required fields, and duplicate ids or
  names, but it should not yet persist anything into the store
- the existing metadata, index, and payload code paths should remain unchanged
  in this feature

### Tests

- `struct_helper.py` returns the expected struct formats for each canonical key
  family
- canonical key packers produce fixed-width byte strings of the expected size
- canonical key unpackers reconstruct the expected components
- `model_name_hash` is deterministic and returns exactly `8` bytes
- secondary scan-key builders return the expected ordered binary prefixes
- registry seed loading accepts the checked-in seed format and rejects duplicate
  model names or duplicate model ids
- seed loading validates required fields and record shape without touching
  runtime store state
- current `metadata.py`, `store.py`, and `index.py` behavior remains unchanged
  by this feature

### Notes

- This is intentionally the first migration feature because it establishes the
  new protocol boundary without forcing a metadata/index/storage rewrite.
- The feature should be implemented so later migration steps can swap current
  call sites over one at a time.
- Do not add additional migration features to this file.
