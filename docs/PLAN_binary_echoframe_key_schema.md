# Binary Echoframe Key Schema

## Summary

Define the binary key schema for echoframe. This plan is design-first and
covers naming, ownership, output types, binary layouts, helper responsibilities,
and scan-key strategy. It does not cover implementation sequence, migration
steps, or tests — those live in
[PLAN_binary_echoframe_key_implementation.md](./PLAN_binary_echoframe_key_implementation.md).

Agreed direction:

- `echoframe_key` is the canonical artifact identifier and the default scan key
- `phraser_key` remains the source key
- keys are binary, explicit, and ordered like `phraser`
- extra scan keys exist separately for targeted lookups
- `hidden_state` and `attention` share one key schema
- `codebook_indices` is segment-scoped and has no `layer`
- `codebook_matrix` is model-scoped

---

## Feature 1: Key Vocabulary

### Requirements

- `echoframe_key` is the canonical artifact-key term
- `phraser_key` is the source-key term
- `echoframe_key` is the default scan key
- additional scan keys exist as secondary indexes for targeted lookup patterns

---

## Feature 2: Registry and Rank Sources

### Requirements

- the store is the source of truth for model registry state
- seed definitions in `data/` are helper files only
- echoframe defines a `model_metadata` output type
- `model_metadata` has its own `echoframe_key`
- canonical `model_metadata` identity is based on `hash(model_name)` plus
  `output_type='model_metadata'`
- `model_metadata` payloads include `model_id` and raw `model_name`
- `model_id` is not part of the canonical `model_metadata` key
- the store assigns and persists stable `model_id` values; `model_id` is
  store-internal and never supplied by the user
- duplicate `model_name` registration must fail explicitly
- echoframe supports checked-in human-readable seed definitions in `data/`
- users can load seed definitions into a store explicitly
- users can add one model through a store-facing API
- output types use a fixed rank map in code
- the output type set is:
  - `hidden_state`
  - `attention`
  - `codebook_indices`
  - `codebook_matrix`
  - `model_metadata`

---

## Feature 3: Output Type Key Schemas

### Requirements

- `hidden_state` and `attention` use the key schema:
  `model_id, output_type_id, layer, phraser_key, collar`
- `codebook_indices` uses the key schema:
  `model_id, output_type_id, phraser_key, collar`
- `codebook_matrix` uses the key schema:
  `model_id, output_type_id`
- `model_metadata` uses the key schema:
  `model_name_hash, output_type_id`
- segment-based keys embed raw `phraser_key` bytes rather than reparsing them
  into subfields

---

## Feature 4: Binary Field Layout

### Requirements

- all binary layouts are big-endian and fixed-width
- field widths are explicit and stable:
  - `model_id`: `uint16`
  - `output_type_id`: `uint8`
  - `layer`: `uint8`
  - `collar`: `uint16`
  - `phraser_key`: fixed `22` bytes for segment-based keys
  - `model_name_hash`: `8` bytes
- `model_name_hash` is computed with `blake2b` and `digest_size=8` over
  `model_name.encode('utf-8')`
- the binary layout for `hidden_state` and `attention`:
  `uint16 model_id, uint8 output_type_id, uint8 layer, 22-byte phraser_key,
  uint16 collar`
- the binary layout for `codebook_indices`:
  `uint16 model_id, uint8 output_type_id, 22-byte phraser_key, uint16 collar`
- the binary layout for `codebook_matrix`:
  `uint16 model_id, uint8 output_type_id`
- the binary layout for `model_metadata`:
  `8-byte model_name_hash, uint8 output_type_id`

---

## Feature 5: Helper Modules

### Requirements

- echoframe adds `struct_helper.py` for binary layout definitions
- echoframe adds `key_helper.py` for packing, unpacking, hashing, and
  scan-key building
- echoframe adds `model_registry_loader.py` for loading and validating seed
  definitions from `data/`
- helpers follow the phraser approach:
  - explicit field ordering
  - explicit binary layouts
  - builder/unpacker separation
- key-building is centralized in helpers; writing to the store is a store
  responsibility

---

## Feature 6: Scan-Key Strategy

### Requirements

- `echoframe_key` is the canonical artifact key and the default scan key
- secondary scan keys are explicit secondary indexes and do not become
  alternate canonical identities
- secondary scan keys apply to `hidden_state`, `attention`, and
  `codebook_indices` only — `model_metadata` and `codebook_matrix` are
  excluded
- `codebook_matrix` is model-scoped and not taggable
- v1 secondary scan keys are:
  - tag: `tag_hash, echoframe_key`
  - output-type: `output_type_id, echoframe_key`
- model-name-based lookup resolves through the model registry
  (`model_name → model_id`) followed by a prefix scan on the primary
  `echoframe_key` index — a separate model-name secondary scan key is not
  needed

---

## Non-Goals For This Plan

- implementation sequence and migration steps
- tests
- codebook-matrix dedup
- batch failure handling
