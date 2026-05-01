# Binary Echoframe Key Implementation Restart

## Summary

Resume the binary-key rollout from the point where implementation stopped:

- Feature 1 is implemented: binary key helpers and seed loading
- Feature 2 is implemented: store-owned model registry and `model_metadata`
  persistence
- Features 3-5 are not implemented yet

The next phase should start with metadata refactoring before changing the
ordinary artifact store path. The goal is to stop treating one metadata class
as if every output type shares the same identity and payload shape.

## Stage 1: Metadata Refactor

### Requirements

- Keep `EchoframeMetadata` as the shared base class
- Add output-type-specific subclasses:
  - `HiddenStateMetadata`
  - `AttentionMetadata`
  - `CodebookIndicesMetadata`
  - `CodebookMatrixMetadata`
  - `ModelMetadata`
- Move shared storage concerns into the base class:
  - `storage_status`
  - `shard_id`
  - `dataset_path`
  - `shape`
  - `dtype`
  - `tags`
  - `created_at`
  - `deleted_at`
- Keep current ordinary-artifact identity behavior working during this stage
- Make metadata copy helpers preserve the concrete subclass
- Make `from_dict()` dispatch to the concrete subclass by `output_type`
- `ModelMetadata` should already support future model payload fields:
  - `model_name`
  - `model_id`
  - `local_path`
  - `huggingface_id`
  - `language`

### Notes

- This stage is a preparatory refactor only.
- It should not yet change `Store.put`, `LmdbIndex`, `lmdb_helper`, or
  `Hdf5ShardStore` to use binary `echoframe_key` as the primary key.
- The purpose is to make the metadata model match the key-schema plan before
  the ordinary-artifact path is switched over.

## Stage 2: Key Helper Dispatch

### Requirements

- Add a public `pack_echoframe_key(...)` dispatcher in `key_helper.py`
- Add a public `unpack_echoframe_key(echoframe_key)` dispatcher
- Keep pack/unpack logic strict and phraser-like
- Validate segment `phraser_key` inputs rather than hashing arbitrary values
- Keep key construction in helpers only

### Notes

- This stage should still be helper-only.
- The store should not infer or repair invalid key inputs.

## Stage 3: Ordinary Artifact Metadata Values

### Requirements

- Switch ordinary artifact metadata values to the new concrete subclasses
- Let decoded key fields live in the value where useful for convenience
- Keep metadata serialization/deserialization driven by `output_type`

### Notes

- This is the last stage before changing the store/index primary-key path.
- It should still avoid a mixed refactor where value-shape and primary-key
  switching happen in the same patch.

## Stage 4: Store and Index Primary-Key Switch

### Requirements

- Change `Store.put` to accept `echoframe_key` directly
- Use `key_helper` as the canonical place for building keys by output type
- Make LMDB primary records keyed by binary `echoframe_key`
- Migrate output types in order:
  - `hidden_state`
  - `attention`
  - `codebook_indices`
  - `codebook_matrix`

### Notes

- Existing databases are out of scope.
- The code should only target the new binary-key path.
