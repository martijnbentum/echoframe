## Token Embeddings Plan

### Summary

The current `Embeddings` class mixes two concerns:

- one embedding payload for one speech token
- tensor-axis operations such as layer and frame concatenation

The next iteration should make single-token embeddings explicit and add a new
multi-token container built from a list of single-token embeddings.

This plan assumes:

- no backward compatibility is required
- scope is limited to `echoframe/embeddings.py`
- the main target shapes are aggregated token embeddings, not frame-first
  sequences

### Terminology

- `token`: one speech item / token / object section
- `single-token embedding`: one embedding payload for one token
- `multi-token embedding`: a collection of token embeddings
- `aggregated`: frames were reduced into one embedding per token, optionally
  one embedding per layer per token

### Design Direction

- Keep `Embeddings` as the single-token object
- Add a new multi-token class, working name `TokenEmbeddings`
- Use `echoframe_key` as the required identifier field on single-token
  embeddings
- Construct the multi-token class from a list of single-token `Embeddings`
- Make aggregated token shapes the primary first-class case
- Allow frame-based shapes, including ragged multi-token storage, but do not
  center the API on frame concatenation

## Feature 1: Redefine `Embeddings` As A Single-Token Object

### Requirements

- `Embeddings` must represent exactly one token
- `Embeddings` must require `echoframe_key`
- `Embeddings` must continue to store `data`, `dims`, and optional `layers`
- `Embeddings` must support these primary shapes:
  - `('embed_dim',)`
  - `('layers', 'embed_dim')`
- `Embeddings` may also support frame-based shapes for one token:
  - `('frames', 'embed_dim')`
  - `('layers', 'frames', 'embed_dim')`
- `Embeddings` must support an optional aggregation field for frame reduction
- The aggregation field must be optional and named, for example
  `aggregation='mean'`
- If `layers` is present in `dims`, `layers` metadata must be required
- If `layers` metadata is provided, `layers` must be present in `dims`
- `__repr__` must remain compact and must not preview NumPy array contents
- `to_numpy()` must return the underlying NumPy array for single-token
  embeddings

### Notes

- Aggregation refers specifically to aggregation over frames
- Aggregation should not be inferred only from dims
- A shape like `('embed_dim',)` may have `aggregation=None` or
  `aggregation='mean'`, depending on whether provenance is known

### Tests

- Reject construction when `echoframe_key` is missing or empty
- Accept `('embed_dim',)` with `echoframe_key`
- Accept `('layers', 'embed_dim')` with matching `layers`
- Accept `('frames', 'embed_dim')` with `echoframe_key`
- Accept `('layers', 'frames', 'embed_dim')` with matching `layers`
- Reject `layers` length mismatch
- Reject `layers` metadata when `layers` is not in `dims`
- Reject `layers` in `dims` when `layers` metadata is missing
- Confirm `aggregation` is stored when provided
- Confirm `to_numpy()` returns the same array contents
- Confirm `repr()` includes shape, dims, layers, aggregation, and
  `echoframe_key` without array preview

## Feature 2: Add A Multi-Token Container Class

### Requirements

- Add a new class, preferred working name `TokenEmbeddings`
- `TokenEmbeddings` must be constructed from a list of single-token
  `Embeddings`
- Construction must require at least one token
- Construction must validate that all entries are `Embeddings`
- Construction must validate that all entries have unique `echoframe_key`
- Construction must preserve token order
- Construction must expose token count
- Construction must expose token keys via `echoframe_keys`
- `TokenEmbeddings` must support two storage modes:
  - stacked NumPy storage for uniform token shapes
  - list-backed storage for ragged token shapes
- The primary first-class uniform cases are:
  - `[tokens, embed_dim]`
  - `[tokens, layers, embed_dim]`
- Ragged frame-based tokens must be allowed through list-backed storage
- All tokens in one `TokenEmbeddings` object must share the same `layers`
  labels when layers are present
- Aggregation metadata must be compatible across all tokens in one collection
- The class `repr()` must summarize token count and shape mode without dumping
  payload contents

### Notes

- `TokenEmbeddings` should model a collection, not just one extra axis
- A list-backed implementation is needed for frame-variable tokens
- The first implementation can determine stacked vs ragged mode internally

### Tests

- Build from a list of aggregated single-token embeddings of shape
  `('embed_dim',)`
- Build from a list of aggregated single-token embeddings of shape
  `('layers', 'embed_dim')`
- Expose ordered `echoframe_keys`
- Reject empty input
- Reject non-`Embeddings` inputs
- Reject duplicate `echoframe_key` values
- Reject incompatible dims across tokens
- Reject incompatible `layers` labels across tokens
- Reject incompatible aggregation values across tokens
- Confirm stacked construction produces `[tokens, embed_dim]`
- Confirm stacked construction produces `[tokens, layers, embed_dim]`
- Confirm ragged frame-based input is accepted and stored without padding
- Confirm `repr()` does not show NumPy array previews

## Feature 3: Layer Selection For Single And Multi-Token Objects

### Requirements

- `Embeddings.layer(n)` must continue to select one layer from a single-token
  object
- Selecting one layer from `('layers', 'embed_dim')` must return
  `('embed_dim',)`
- Selecting one layer from `('layers', 'frames', 'embed_dim')` must return
  `('frames', 'embed_dim')`
- `TokenEmbeddings.layer(n)` must be added for the uniform layered cases
- Selecting one layer from `[tokens, layers, embed_dim]` must return another
  `TokenEmbeddings` object with `[tokens, embed_dim]`
- The result must preserve token order, `echoframe_keys`, and aggregation
  metadata
- `TokenEmbeddings.layer(n)` does not need to support ragged frame-based
  layouts in v1
- Unsupported ragged layer selection must raise `NotImplementedError`

### Tests

- Single-token layered aggregated selection returns one-token
  `('embed_dim',)` data
- Single-token layered frame selection returns one-token
  `('frames', 'embed_dim')` data
- Multi-token `[tokens, layers, embed_dim]` layer selection returns another
  multi-token object
- Result preserves `echoframe_keys`
- Result preserves selected data values
- Reject unknown layer id
- Ragged layered multi-token selection raises `NotImplementedError`

## Feature 4: Multi-Token Numeric Export

### Requirements

- `Embeddings.to_numpy()` must return the single-token array
- `TokenEmbeddings.to_numpy()` must return a stacked NumPy array when the
  collection is uniform
- For uniform aggregated tokens, `to_numpy()` must return:
  - shape `(tokens, embed_dim)` or
  - shape `(tokens, layers, embed_dim)`
- For ragged frame-based collections, `to_numpy()` must raise
  `NotImplementedError`
- The error message should clearly state that variable-length frame collections
  do not have a single NumPy representation

### Tests

- `Embeddings.to_numpy()` returns the original single-token array
- Uniform `TokenEmbeddings.to_numpy()` returns `[tokens, embed_dim]`
- Uniform layered `TokenEmbeddings.to_numpy()` returns
  `[tokens, layers, embed_dim]`
- Ragged frame-based `TokenEmbeddings.to_numpy()` raises
  `NotImplementedError`

## Feature 5: Explicit Combination APIs Instead Of Frame-Centric `+`

### Requirements

- Do not make `__add__` central to the new design
- Prefer explicit APIs over implicit operator behavior
- Provide a collection constructor or helper on the new class rather than
  reusing frame-centric concatenation semantics
- If concatenation helpers remain, they must be explicit about whether they
  combine:
  - tokens
  - layers
  - frames
- The new design should emphasize token collection and layer selection, not
  frame concatenation

### Tests

- Construct a multi-token object from a list of `Embeddings`
- Combine compatible aggregated token embeddings without using `+`
- Reject ambiguous combination requests with clear errors

## Feature 6: Validation And Error Messages

### Requirements

- Validation errors must use clear `ValueError` messages
- Errors should explain whether incompatibility is about:
  - dims
  - layer labels
  - aggregation mode
  - `echoframe_key`
  - ragged vs uniform export
- The distinction between single-token and multi-token misuse should be clear

### Tests

- Missing `echoframe_key` raises a clear `ValueError`
- Duplicate `echoframe_key` in multi-token input raises a clear `ValueError`
- Mixed dims in multi-token input raises a clear `ValueError`
- Mixed layer labels in multi-token input raises a clear `ValueError`
- Unsupported ragged operation raises a clear `NotImplementedError`

## Implementation Order

1. Redefine `Embeddings` around the single-token contract
2. Add `echoframe_key`, `aggregation`, and `to_numpy()` to `Embeddings`
3. Add `TokenEmbeddings` with list-of-`Embeddings` construction
4. Support uniform aggregated token stacking
5. Add layered multi-token selection to `[tokens, layers, embed_dim]`
6. Add ragged frame-based storage mode
7. Add `to_numpy()` failure behavior for ragged collections
8. Update tests to reflect the new single-token and multi-token split

## Out Of Scope For This Plan

- Store integration
- Serialization to disk
- Automatic conversion from echoframe metadata objects
- Rich per-token metadata beyond required `echoframe_key`
- Ragged multi-token layer selection beyond explicit `NotImplementedError`
- Final naming cleanup if a better class name than `TokenEmbeddings` emerges

## Open Naming Decisions

- Keep `TokenEmbeddings` as the class name, or rename to a more neutral
  collection name if needed
- Confirm the final aggregation field name:
  - `aggregation`
  - `frame_aggregation`
  - similar
