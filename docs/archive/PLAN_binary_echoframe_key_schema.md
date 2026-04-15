# Binary Echoframe Key Schema

## Summary

Define a new key model for echoframe before touching the current metadata,
LMDB, or storage implementation.

This plan is intentionally design-first. It focuses on naming, ownership,
schema families, registries, and helper responsibilities. It does not yet cover
implementation details, migration steps, or tests.

Agreed direction:

- `echoframe_key` becomes the one canonical artifact identifier
- `phraser_key` remains the source key
- keys should be binary, explicit, and ordered like `phraser`
- `echoframe_key` should be a sensible default scan path, not the answer to
  every query pattern
- extra scan keys can exist separately for tags and other targeted lookups
- `hidden_state` and `attention` share one schema family
- `codebook_indices` is segment-scoped and has no `layer`
- `codebook_matrix` is model-scoped

---

## Feature 1: Key Vocabulary

### Requirements

- `echoframe_key` is the only canonical artifact-key term in the new design
- `phraser_key` remains the source-key term
- `echoframe_key` replaces the current `identity_key`, `object_key`, and
  `entry_id` concepts as the canonical artifact-identity term
- `echoframe_key` also serves as the default scan key
- additional scan keys may exist as secondary indexes for targeted lookup
  patterns

### Notes

- This feature is about naming and conceptual boundaries only
- It should establish a clear language before binary encoding or migration is
  discussed

---

## Feature 2: Registry And Rank Sources

### Requirements

- the store should be the source of truth for model registry state
- seed definitions in `data/` should be helper files only
- echoframe should define a `model_metadata` artifact family
- `model_metadata` should have its own `echoframe_key`
- canonical `model_metadata` identity should be based on
  `hash(model_name)` plus `output_type='model_metadata'`
- `model_metadata` payloads should include `model_id`
- `model_metadata` payloads should include raw `model_name`
- `model_id` should not be part of the canonical `model_metadata` key
- the store should assign and persist stable `model_id` values
- once assigned, a `model_id` must not be reused for a different model
- echoframe should support checked-in human-readable seed definitions in
  `data/`
- users should be able to load seed definitions into a store explicitly
- users should be able to add one model explicitly through a store-facing API
- output types should use a fixed rank map in code rather than a registry file
- the initial output-type set should include:
  - `hidden_state`
  - `attention`
  - `codebook_indices`
  - `codebook_matrix`
  - `model_metadata`

### Notes

- This feature defines where model ids come from and who owns them
- Seed definitions should help initialize stores, but they should not be the
  runtime source of truth
- After seed definitions are loaded, store contents remain authoritative even if
  the seed files later differ
- It does not yet define binary field sizes or struct layouts

---

## Feature 3: Canonical Key Families

### Requirements

- `hidden_state` and `attention` should use the key schema:
  `model_id, output_type_id, layer, phraser_key, collar`
- `codebook_indices` should use the key schema:
  `model_id, output_type_id, phraser_key, collar`
- `codebook_matrix` should use the key schema:
  `model_id, output_type_id`
- `model_metadata` should use the key schema:
  `model_name_hash, output_type_id`
- segment-based keys should embed raw `phraser_key` bytes rather than
  reparsing them into subfields

### Notes

- This feature defines the grouped canonical schemas before binary field layout
- It should stay at the ordered-field level, not exact binary packing yet

---

## Feature 4: Binary Field Layout

### Requirements

- echoframe should use fixed-width binary layouts inspired by `phraser`
- echoframe should use big-endian binary layouts, matching `phraser`
- field widths should be explicit and stable
- segment `phraser_key` values should be treated as fixed-width `22`-byte
  values and embedded directly in segment-based echoframe keys
- output types should use a fixed stable rank map in code
- `model_name_hash` should be computed deterministically from
  `model_name.encode('utf-8')`
- `model_name_hash` should use `blake2b` with `digest_size=8`
- `model_name_hash` should use a fixed 8-byte digest
- the binary layout for `hidden_state` and `attention` should be:
  `uint16 model_id, uint8 output_type_id, uint8 layer, 22-byte phraser_key,
  uint16 collar`
- the binary layout for `codebook_indices` should be:
  `uint16 model_id, uint8 output_type_id, 22-byte phraser_key, uint16 collar`
- the binary layout for `codebook_matrix` should be:
  `uint16 model_id, uint8 output_type_id`
- the binary layout for `model_metadata` should be:
  `8-byte model_name_hash, uint8 output_type_id`

### Notes

- This feature is about binary representation only
- It should turn the grouped schemas into concrete fixed-width layouts
- Hash algorithm choice should be treated as protocol, not implementation
  detail

---

## Feature 5: Helper Modules

### Requirements

- echoframe should add a dedicated `struct_helper.py`
- echoframe should add a dedicated `key_helper.py`
- these helpers should be modeled on the phraser approach:
  - explicit field ordering
  - explicit binary layouts
  - builder/unpacker separation
- key-building responsibilities should be centralized in helper modules rather
  than distributed through metadata or storage code

### Notes

- This feature is about ownership of the schema logic
- It should make the key model inspectable and disciplined before migration

---

## Feature 6: Scan-Key Strategy

### Requirements

- `echoframe_key` should remain the canonical artifact key
- `echoframe_key` should also remain the default scan key
- separate scan keys should be treated as explicit secondary indexes
- secondary scan keys should not become alternate canonical identities
- v1 secondary scan keys should cover:
  - tags
  - model-name-based lookup
  - output-type-based lookup
- the v1 model-name secondary scan key should use:
  `model_name_hash, echoframe_key`
- the v1 output-type secondary scan key should use:
  `output_type_id, echoframe_key`
- the v1 tag secondary scan key should use:
  `tag_hash, echoframe_key`

### Notes

- This feature defines the relationship between the canonical key and secondary
  indexes
- It should avoid forcing the canonical key to satisfy every query pattern
- Secondary scan keys should follow the pattern:
  `lookup_component, echoframe_key`

---

## Non-Goals For This Plan

- no implementation yet
- no tests yet
- no migration of the current metadata model yet
- no redesign of the current LMDB databases yet
- no codebook-matrix dedup implementation yet
- no batch failure-handling implementation here
