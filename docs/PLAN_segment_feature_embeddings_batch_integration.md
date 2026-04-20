# Plan: Use `to-vector` Batch Embeddings In `echoframe.segment_features`

## Summary

`echoframe.segment_features` does not currently use the batch functionality
implemented in `to-vector` for embeddings. The current
`get_embeddings_batch(...)` implementation loops over `get_embeddings(...)`
one segment at a time, and the compute path eventually calls
`to_vector.filename_to_vector(...)` per segment.

`to-vector` already exposes a batch entry point,
`filename_batch_to_vector(...)`, which:

- accepts parallel lists of filenames, starts, and ends
- loads all requested windows
- batches inference across those windows
- splits outputs back into one output object per requested segment
- preserves per-item metadata like filename, start, and end

This makes embeddings a good candidate for a real batch integration inside
`echoframe.segment_features`.

This plan now assumes a stricter batch contract than the current code:

- `get_embeddings_batch(...)` should do real batch orchestration
- invalid segments should be filtered before inference
- filtered segments should be reported with warnings
- post-preflight compute or storage failures should raise
- backward compatibility with the old `on_error` behavior is not required

## Current State

### In `echoframe`

- `get_embeddings_batch(...)` iterates over segments and calls
  `get_embeddings(...)` for each item.
- `get_embeddings(...)` checks whether requested layers are already present in
  the store.
- When embeddings are missing, `_compute_and_store_embeddings(...)` computes
  outputs for a single collared segment window via
  `to_vector.filename_to_vector(...)`.
- After inference, it:
  - validates that `hidden_states` exist
  - derives frames from the model outputs
  - selects frame indices fully inside the original segment
  - slices requested layers
  - stores each layer in the echoframe store
- there is no dedicated batch preflight validation module yet
- `TokenEmbeddings` does not currently expose a built-in failure sidecar such
  as `_failed_metadatas`

### In `to-vector`

- `filename_batch_to_vector(...)` already supports batch inference over many
  `(audio_filename, start, end)` windows.
- `batch_helper.handle_batch(...)` splits work into smaller sub-batches using a
  total-sample budget (`batch_minutes`).
- The Hugging Face batch helper slices padded model output back into per-item
  outputs, so each returned object still works with per-segment frame
  selection.

## Recommended Refactor

### 1. Make `get_embeddings_batch(...)` a real batch orchestrator

Refactor `get_embeddings_batch(...)` into this flow:

1. Resolve context for all requested segments.
2. Preflight-validate those segment contexts for batch suitability.
3. Batch-compute only the valid segments whose embeddings are missing.
4. Store those newly computed embeddings.
5. Load results from the store and return `TokenEmbeddings`.

The first phase should use the same helpers already used by the single-item
path:

- `_require_segments(...)`
- `_normalise_layers(...)`
- `_segment_context(...)`
- `_embeddings_missing(...)`

This keeps context resolution and cache decisions consistent between the single
and batch paths while still allowing batch-only preflight filtering.

### 2. Add batch-only preflight validation in a helper module

Add a small generic helper module for segment batch validation. It should not
be embeddings-specific, because the same checks should be reusable later for
other segment batch flows.

Initial helper responsibilities:

- define a default minimum segment duration constant
- validate that the resolved `audio_filename` exists
- validate that the segment duration is at least the minimum duration
- return enough metadata to report filtered items clearly

Initial reporting behavior:

- filtered items should trigger a warning
- if this stays clean with the typed container design, filtered item metadata
  may also be attached on `TokenEmbeddings` via `_failed_metadatas`
- warnings are the default reporting requirement; `_failed_metadatas` is an
  optional extension rather than the core contract

Single-item `get_embeddings(...)` should not use this helper yet. Single-item
calls can continue to fail directly.

### 3. Compute only the missing valid subset

Do not batch every requested segment blindly. First separate valid segments
into:

- already cached in the store
- missing one or more requested layers

For the valid missing subset, build:

- `audio_filenames`
- `starts`
- `ends`

Then issue one call to:

- `to_vector.filename_batch_to_vector(...)`

Pass:

- `model=compute_model`
- `gpu=gpu`
- `numpify_output=True`
- `batch_minutes=batch_minutes`

`get_embeddings_batch(...)` should expose `batch_minutes=None` publicly and
forward it unchanged so `to-vector` defaults still apply when unspecified.

### 4. Split storage logic out of the single-item compute helper

The current `_compute_and_store_embeddings(...)` mixes two responsibilities:

- inference
- post-processing and storage

Extract the second half into a reusable helper, for example:

- `_store_embeddings_from_outputs(outputs, col_start_ms, orig_start_ms,
  orig_end_ms, collar, layers, model_name, phraser_key, store, tags)`

That helper should:

- validate `outputs.hidden_states`
- create frames with `frame.make_frames_from_outputs(...)`
- select fully overlapping frames for the original segment
- collect frame indices
- validate requested layers against the number of hidden states
- slice the hidden-state arrays
- write metadata and data into the store

With that split:

- the existing single-item path can remain:
  - `filename_to_vector(...)`
  - `_store_embeddings_from_outputs(...)`
- the new batch path can do:
  - `filename_batch_to_vector(...)`
- loop over returned outputs
  - `_store_embeddings_from_outputs(...)` per item

This keeps the output interpretation identical between single and batch
execution.

### 5. Replace `on_error` with preflight filtering plus fail-fast compute

The old batch API exposes `on_error='skip'` and `on_error='raise'`. This plan
does not preserve that behavior.

New behavior:

- invalid segments are filtered out before inference
- filtered segments are reported
- batch inference runs only on the preflight-approved missing subset
- if batch inference fails, raise
- if post-processing or storage fails for any returned item, raise

Rationale:

- the main recoverable failures observed so far are input-shape issues such as
  missing files or too-short segments
- those can be filtered before the `to-vector` batch call
- once the batch starts, unexpected failures should be treated as real errors
  rather than silently skipped

### 6. Keep the public return shape unchanged where practical

After storing any newly computed items, continue loading data from the store the
same way the current implementation does. This keeps these behaviors stable:

- return type remains `TokenEmbeddings`
- cached and newly computed items are handled uniformly
- `frame_aggregation` remains a store-load concern rather than a compute-time
  concern

If filtered-item metadata can be attached without making `TokenEmbeddings`
awkward, `_failed_metadatas` may be added there. If `_failed_metadatas`
introduces noticeable container complexity, stop and ask for feedback rather
than forcing it in. Warnings remain the required reporting path either way.

## Suggested Implementation Steps

1. Add a generic segment batch validation helper module with:
   - default minimum duration
   - audio filename existence checks
   - duration checks
   - reporting metadata for filtered items
2. Extract a reusable helper that stores embeddings from one `to-vector`
   output object.
3. Refactor `get_embeddings_batch(...)` to:
   - collect segment contexts
   - preflight-filter invalid items
   - warn about filtered items
   - identify valid missing items
   - call `to_vector.filename_batch_to_vector(...)` once for the missing subset
   - store results per item
   - load final results from the store
   - optionally attach `_failed_metadatas` if that stays simple
4. Leave `get_embeddings(...)` on the single-item path.
5. Defer tests for this first pass so the behavior change can land first.

If making the new helper generic for embeddings and codebook-related batch
flows starts introducing a lot of branching logic, stop and ask for feedback
before pushing that abstraction further.

## Requirements

- `get_embeddings_batch(...)` must expose `batch_minutes=None`
- `get_embeddings_batch(...)` must not loop through `get_embeddings(...)`
- batch preflight validation must live in a generic helper module
- batch preflight must filter at least:
  - missing audio files
  - segments shorter than the helper's default minimum duration
- only the batch path should use the new preflight helper
- cached valid segments must still be returned
- only valid missing segments should be sent to `filename_batch_to_vector(...)`
- unexpected failures after preflight should raise immediately
- filtered segments must be reported with warnings
- `_failed_metadatas` on `TokenEmbeddings` is allowed only if it does not make
  the container design awkward
- ask for feedback if the generic helper starts accumulating substantial
  branching for different output types
- ask for feedback if `_failed_metadatas` starts adding substantial complexity
  to `TokenEmbeddings`

## Tests

Testing is intentionally deferred for the first pass. When tests are added
after the refactor, update the current suite to match the new contract:

- remove the assertion that `get_embeddings_batch(...)` reuses
  `get_embeddings(...)`
- assert that `filename_batch_to_vector(...)` is used for multiple missing
  valid segments
- assert that invalid segments are filtered before the batch call
- assert that cached valid segments are still returned
- assert that filtered segments are warned about
- assert that post-preflight compute failures raise
- assert that frame selection and hidden-state slicing remain identical to the
  single-item storage path

## Scope Boundary

This plan applies cleanly to embeddings only.

`get_codebook_indices_batch(...)` does not have the same ready-made path yet,
because the current codebook flow in `echoframe.segment_features` depends on
single-item `to_vector.filename_to_codebook_artifacts(...)`, and the inspected
`to-vector` batching support is for embeddings rather than codebook artifacts.

If codebook batching is needed later, that likely requires new batch support in
`to-vector` first, or a separate refactor in `echoframe`.
