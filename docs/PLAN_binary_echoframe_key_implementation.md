# Binary Echoframe Key Implementation

## Summary

Implement the binary key protocol defined in
[PLAN_binary_echoframe_key_schema.md](./PLAN_binary_echoframe_key_schema.md).

The schema plan is the fixed reference for naming, key layouts, field widths,
output types, and scan-key strategy. This document covers only the
implementation sequence.

Constraints:

- no backward compatibility requirements
- migration covers code only, not existing databases
- helpers build (pack) and interpret (unpack) keys and return byte strings
  only; the store is responsible for loading and writing

---

## Feature 1: Binary Key Protocol Helpers

### Summary

Introduce the new binary key protocol as standalone helper code before
touching any store, index, or metadata code.

### Requirements

- Add `echoframe/struct_helper.py`
- Add `echoframe/key_helper.py`
- Add `echoframe/model_registry_loader.py`
- `struct_helper.py` should define the fixed-width binary layouts for all
  output types from the schema plan:
  - `hidden_state` and `attention`: `uint16 model_id, uint8 output_type_id,
    uint8 layer, 22-byte phraser_key, uint16 collar`
  - `codebook_indices`: `uint16 model_id, uint8 output_type_id, 22-byte
    phraser_key, uint16 collar`
  - `codebook_matrix`: `uint16 model_id, uint8 output_type_id`
  - `model_metadata`: `8-byte model_name_hash, uint8 output_type_id`
- `key_helper.py` should provide deterministic pack and unpack helpers for
  each output type
- `key_helper.py` should provide the `model_name_hash` helper using `blake2b`
  with `digest_size=8` and `model_name.encode('utf-8')` as input
- `key_helper.py` should provide secondary scan-key builders for the v1
  lookup patterns:
  - tag: `tag_hash, echoframe_key`
  - output-type: `output_type_id, echoframe_key`
- secondary scan-key builders return byte strings only; writing entries into
  the store is a store responsibility
- all binary layouts should be big-endian and fixed-width
- output types should use a fixed stable rank map in code
- `model_registry_loader.py` should load and validate seed definitions from
  `data/`
- seed loading should validate structure, required fields, and duplicate model
  names, but should not persist anything into the store
- existing store, index, and metadata code paths should remain unchanged

### Tests

- `struct_helper.py` returns the expected struct formats for each output type
- pack helpers produce fixed-width byte strings of the expected size for each
  output type
- unpack helpers reconstruct the expected field values for each output type
- `model_name_hash` is deterministic and returns exactly 8 bytes
- secondary scan-key builders return the expected ordered binary prefixes for
  tag and output-type lookups
- seed loading accepts the checked-in seed format
- seed loading rejects duplicate model names
- seed loading validates required fields and record shape

---

## Feature 2: Store-Owned Model Registry and `model_metadata` Integration

### Summary

Build the store-owned model registry and wire `model_metadata` reads and
writes into the store using the binary key helpers from feature 1.

### Requirements

- Add a store-facing API to register one model explicitly
- Add a store-facing API to import model seed definitions from `data/`; the
  entire seed file must be validated before any record is written — if any
  model name already exists in the store the import fails with a clear error
  and no records are written
- Persist `model_metadata` records using the canonical key shape from the
  schema plan: `8-byte model_name_hash, uint8 output_type_id`
- Stored `model_metadata` records should include the raw `model_name`, the
  assigned `model_id`, and any associated metadata needed for later lookup
- The store should assign and persist stable `model_id` values; `model_id` is
  store-internal and never supplied by the user
- Duplicate `model_name` registration must fail explicitly
- Add a store-facing lookup by model name: compute `model_name_hash` and
  retrieve the `model_metadata` record directly by its canonical key
  `model_name_hash, output_type_id`; the returned record's `model_id` field
  can then be used to prefix-scan the primary `echoframe_key` index for
  artifact records of other output types (those are wired in features 3–5)
- Model registry state should survive closing and reopening the store
- Seed files are helper input only; the store is the source of truth after
  import
- Validation of required fields should happen before any store write

### Tests

- registering one valid model stores a `model_metadata` record and assigns a
  stable `model_id`
- the stored `model_metadata` record is readable after closing and reopening
  the store
- importing a valid seed file populates the store with the expected
  `model_metadata` records
- registering a duplicate `model_name` raises clearly
- duplicate model names in a seed file raise clearly
- importing a seed file where any model name already exists in the store
  raises clearly and writes no records
- invalid model metadata is rejected before any store write
- looking up a model by name retrieves the `model_metadata` record directly
  via its `model_name_hash, output_type_id` key

---

## Feature 3: `hidden_state` and `attention` Store Integration

### Summary

Switch store read and write paths for `hidden_state` and `attention` output
types to use `echoframe_key` as the primary key.

Depends on feature 2: a valid `model_id` from the model registry is required
to pack an `echoframe_key` for these output types.

### Requirements

- `Store.put` for `hidden_state` and `attention` records should use the
  canonical `echoframe_key` layout: `uint16 model_id, uint8 output_type_id,
  uint8 layer, 22-byte phraser_key, uint16 collar`
- `Store.find` and `Store.load` for these output types should use the
  `echoframe_key` path
- Write secondary scan-key entries (tag, output-type) when storing a record
- `model_metadata` and model registry behavior should remain unchanged

### Tests

- writing a `hidden_state` record stores it under the canonical `echoframe_key`
- writing an `attention` record stores it under the canonical `echoframe_key`
- loading a written record resolves through the `echoframe_key` path
- secondary tag and output-type scan-key entries are written alongside the
  primary record
- `model_metadata` and model-registry tests remain unchanged

---

## Feature 4: `codebook_indices` Store Integration

### Summary

Switch store read and write paths for `codebook_indices` to use
`echoframe_key` as the primary key.

Depends on feature 2: a valid `model_id` from the model registry is required
to pack an `echoframe_key` for this output type.

### Requirements

- `Store.put` for `codebook_indices` records should use the canonical
  `echoframe_key` layout: `uint16 model_id, uint8 output_type_id, 22-byte
  phraser_key, uint16 collar`
- `Store.find` and `Store.load` for `codebook_indices` should use the
  `echoframe_key` path
- Write secondary scan-key entries (tag, output-type) when storing a record
- All other output type behavior should remain unchanged

### Tests

- writing a `codebook_indices` record stores it under the canonical
  `echoframe_key`
- loading a written record resolves through the `echoframe_key` path
- secondary tag and output-type scan-key entries are written alongside the
  primary record
- `hidden_state`, `attention`, `model_metadata`, and model-registry tests
  remain unchanged

---

## Feature 5: `codebook_matrix` Store Integration

### Summary

Switch store read and write paths for `codebook_matrix` to use `echoframe_key`
as the primary key.

Depends on feature 2: a valid `model_id` from the model registry is required
to pack an `echoframe_key` for this output type.

### Requirements

- `Store.put` for `codebook_matrix` records should use the canonical
  `echoframe_key` layout: `uint16 model_id, uint8 output_type_id`
- `Store.find` and `Store.load` for `codebook_matrix` should use the
  `echoframe_key` path
- `codebook_matrix` is model-scoped and not taggable; no secondary scan-key
  entries are written when storing a `codebook_matrix` record
- All other output type behavior should remain unchanged

### Tests

- writing a `codebook_matrix` record stores it under the canonical
  `echoframe_key`
- loading a written record resolves through the `echoframe_key` path
- all other output type tests remain unchanged
