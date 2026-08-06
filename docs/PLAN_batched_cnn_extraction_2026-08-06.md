# Add batched CNN-only (feature-extractor-only) extraction to to_vector

STATUS: implemented in to_vector at commit 6fdd764 ("Add batched CNN-only
(feature-extractor-only) extraction"), pushed to origin/main. Kept here as a
record for echoframe's cnn output_type Phase B, which is the consumer of this
work — see docs/PLAN_binary_echoframe_key_schema.md-style siblings and the
in-conversation Phase A/B history.

## Context

Repo: `/Users/martijn.bentum/repos/to-vector` (github: `martijnbentum/to_vector`). This
is a standalone package, pulled into other projects (e.g. `echoframe`) as a git
dependency — it has no awareness of those downstream projects and none is needed
here.

`to_vector` wraps Wav2Vec2/WavLM/HuBERT/SpidR models to extract embeddings from
audio. The **CNN feature-extractor stage** is the convolutional stack that runs
*before* the transformer/encoder layers in these architectures. `to_vector` already
supports extracting *only* that stage for a single audio segment
(`filename_to_cnn`/`audio_to_cnn`), which is meaningfully cheaper than the full
forward pass because it skips the transformer entirely. There is currently **no
batched equivalent** — the batch embedding path (`iter_filename_batch_to_vector`)
always runs the full model forward, even when a caller only wants the CNN stage.

Goal: add a batched, CNN-only extraction path — `filename_batch_to_cnn` /
`iter_filename_batch_to_cnn` — that processes multiple audio files with genuine
GPU-batched throughput through the conv stack alone, without ever invoking the
transformer/encoder layers. This mirrors the existing `filename_batch_to_vector` /
`iter_filename_batch_to_vector` batching architecture (multi-batch coordination,
GPU batch-size estimation, background audio prefetching) but swaps in a
CNN-only forward.

## What already exists (read these first)

- `to_vector/to_embeddings.py`
  - `filename_to_cnn`/`audio_to_cnn` (lines ~63-76, ~134-161) — the **single-segment**
    CNN-only path this work batches. `audio_to_cnn` explicitly raises for SpidR
    models: `'audio_to_cnn() is not implemented for SpidR models yet. Check whether
    the convolutional frontend can be called directly on the SpidR model.'` — **keep
    this same restriction in the batched version**, don't attempt to solve SpidR
    here.
  - `filename_batch_to_vector`/`iter_filename_batch_to_vector` (lines ~27-60) — the
    **batched full-forward** path this new work parallels structurally.
- `to_vector/hf_batch_helper.py`
  - `audio_batch_to_outputs` (lines 7-28) — batches inputs, then always runs
    `model(**inputs, output_hidden_states=True)` (line 16), then slices per item.
  - `inputs_to_cnn(inputs, model)` (lines 65-74) — **this already does the batched
    CNN-only forward you need**: takes already-batched/padded `inputs`, calls
    `model.feature_extractor(input_values)` (or `model.wav2vec2.feature_extractor`
    for `*ForPreTraining`-wrapped checkpoints) in one call across the whole batch,
    and returns a transposed, detached tensor of shape `(batch, frames, channels)`.
    Today it's only ever called *after* the full forward already ran (line 21, as a
    hubert-only backfill). The new work is mostly: call this **instead of** running
    the full model, and build correct per-item slicing/lengths around it.
  - `compute_output_lengths(inputs, outputs, model)` (lines 30-47) — derives
    per-item valid frame counts. The `attention_mask` + `model._get_feat_extract_
    output_lengths(...)` branch (lines 40-42) needs no hidden_states and is directly
    reusable. Only the no-`attention_mask` fallback (`default_length = int(
    hidden_states[0].shape[1])`, line 35) depends on `outputs.hidden_states`, which
    won't exist in the CNN-only path — needs a CNN-specific twin using
    `extract_features.shape[1]` instead.
  - `slice_outputs` (lines 50-62) — per-item slicing template; the CNN-only version
    is a subset of this (no `hidden_states` field to slice).
- `to_vector/batch_helper.py`
  - `iter_handle_batching` (lines 45-82) — full orchestration: resolves filenames,
    validates `starts`/`ends`, calls `load.prepare_model`, computes `model_type` via
    `model_registry.model_to_type`, estimates GPU batch size
    (`compute_embedding_batch_size`), splits into batches (`split`), prefetches
    audio on a background thread (`make_audio_queue`), and for each batch calls
    `single_batch_to_outputs` then `numpify`, yielding items in order.
  - `single_batch_to_outputs` (lines 141-146) — dispatches to
    `spidr_batch_helper.audio_batch_to_outputs` or `hf_batch_helper.
    audio_batch_to_outputs` based on `model_type`.
  - **`numpify` (lines 149-159) is UNSAFE for CNN-only outputs** — it
    unconditionally does `for hidden_state in outputs.hidden_states:`, which raises
    `TypeError` if `hidden_states` is `None` (the shape used for CNN-only outputs,
    per `filename_to_cnn`'s `BaseModelOutput(hidden_states=None)` at
    `to_embeddings.py:74`). **Do not reuse this function as-is on CNN-only
    outputs.** The single-segment `audio_to_cnn` sidesteps this by doing its own
    `.detach().cpu().numpy()` conversion inline (`to_embeddings.py:160`) instead of
    calling `batch_helper.numpify`. Follow that same pattern for the batched path —
    convert to numpy inside the new CNN-only slicing function, don't touch the
    shared `numpify`.
  - `compute_embedding_batch_size`/`compute_item_bytes` (lines 84-108) — GPU batch
    size heuristic, tuned via `estimated_embedding_mb_per_second = 2.0` for
    full-hidden-state outputs. CNN-only tensors are much smaller, so reusing this
    as-is will under-batch (safe, just conservative/suboptimal) — **fine to reuse
    unchanged for correctness; tuning a separate, smaller per-second estimate for
    CNN-only batches is an optional follow-up, not required for this task.**
- `to_vector/__init__.py` — public API export list/pattern to extend (see below).
- `tests/test_helpers.py` — `FakeHuggingFaceModel` (wraps a fixed `outputs` value,
  callable via `__call__`), `FakeSpidrModel` (search for it) — test doubles to
  reuse. Note `FakeHuggingFaceModel` has no built-in `.feature_extractor`; tests set
  it directly, e.g. `model.feature_extractor = mock.Mock(return_value=...)`.
- `tests/test_to_embeddings.py` — existing conventions to mirror:
  `test_hf_batch_helper_splits_huggingface_batch_outputs` (mocks
  `to_vector.hf_batch_helper.load.prepare_feature_extractor`, builds a
  `FakeHuggingFaceModel`, mocks `_get_feat_extract_output_lengths`, asserts sliced
  per-item outputs) and `test_audio_to_cnn_raises_clear_error_for_spidr` (asserts
  the exact SpidR-unsupported error message via `assertRaisesRegex`).

## What to build

### 1. `to_vector/hf_batch_helper.py`

- `compute_cnn_output_lengths(inputs, extract_features, model)` — twin of
  `compute_output_lengths`, but the no-`attention_mask` fallback uses
  `int(extract_features.shape[1])` instead of `hidden_states[0].shape[1]`. The
  `attention_mask` branch (using `model._get_feat_extract_output_lengths`) is
  identical logic — consider extracting a shared helper if that avoids
  duplicating that branch, but only if it stays clean; don't force it.
- `audio_batch_to_cnn(audio_arrays, model, model_type)` — batched CNN-only
  extraction for one already-formed batch of raw audio arrays:
  1. `feature_extractor = load.prepare_feature_extractor(model)`;
     `gpu = load.model_is_on_gpu(model)`.
  2. Build padded batch `inputs` exactly like `audio_batch_to_outputs` (lines
     11-14): `feature_extractor(arrays, sampling_rate=16_000,
     return_tensors='pt', padding=True)`, moved to `'cuda'` if `gpu`.
  3. `extract_features = inputs_to_cnn(inputs, model)` — the batched conv-only
     forward. **Do not call `model(**inputs, ...)` anywhere in this path.**
  4. `output_lengths = compute_cnn_output_lengths(inputs, extract_features, model)`.
  5. Build one `BaseModelOutput(hidden_states=None)` per item, with
     `.extract_features = extract_features[index:index+1, :output_length]
     .detach().cpu().numpy()` (numpy conversion inline — see the `numpify`
     warning above) and `.model_type = model_type`.
  6. Return the list of items, in input order.

### 2. `to_vector/batch_helper.py`

- `single_batch_to_cnn_outputs(audio_arrays, model, model_type)` — dispatch
  mirroring `single_batch_to_outputs` (lines 141-146), but:
  - if `model_type == 'spidr'`: raise the same-style error as `audio_to_cnn`
    (adapt the message to name the batch entry point, e.g.
    `'audio_batch_to_cnn() is not implemented for SpidR models yet...'`).
  - otherwise: `hf_batch_helper.audio_batch_to_cnn(audio_arrays, model, model_type)`.
- `iter_handle_cnn_batching(filenames, starts=None, ends=None, model=None,
  gpu=False, batch_size=None)` — orchestration mirroring `iter_handle_batching`
  (lines 45-82), with these differences:
  - Call `load.prepare_model` + `model_registry.model_to_type` **first**, and if
    `model_type == 'spidr'`, raise immediately (before entering the batching/
    prefetch-thread machinery) — cheaper failure than `single_batch_to_outputs`'s
    per-batch dispatch check, and matches `audio_to_cnn`'s early-raise behavior.
  - Reuse `_check_batch_values`, `compute_embedding_batch_size`, `_check_batch_size`,
    `split`, `make_audio_queue` as-is.
  - Per batch, call `single_batch_to_cnn_outputs` instead of
    `single_batch_to_outputs`, and **skip the `numpify` call entirely** — items are
    already numpy (see step 1.5 above). Do not pass CNN-only items through
    `batch_helper.numpify`.
  - No `numpify_output` parameter — CNN-only outputs are always returned as numpy
    (matches `audio_to_cnn`'s unconditional numpy return; there's no torch-output
    caller for the single-segment CNN path either, so don't invent one here).
- Skip adding a `handle_cnn_batching` list-wrapper equivalent — `handle_batching`
  (lines 37-42) itself has **zero call sites anywhere in the repo** (verified via
  grep; not even used by `__init__.py` or tests), so it's already dead/unused code.
  Don't extend a pattern nobody calls.

### 3. `to_vector/to_embeddings.py`

- `filename_batch_to_cnn(audio_filenames, starts=None, ends=None, model=None,
  gpu=False, batch_size=None)` — mirrors `filename_batch_to_vector` (lines 27-45):
  materializes `iter_filename_batch_to_cnn(...)` into a list, validates the
  returned count matches input count.
- `iter_filename_batch_to_cnn(audio_filenames, starts=None, ends=None, model=None,
  gpu=False, batch_size=None)` — mirrors `iter_filename_batch_to_vector` (lines
  48-60): thin wrapper yielding from `batch_helper.iter_handle_cnn_batching(...)`.

### 4. `to_vector/__init__.py`

Add `filename_batch_to_cnn` and `iter_filename_batch_to_cnn` to both the
`from .to_embeddings import (...)` block and `__all__`, keeping the existing
(roughly alphabetical) ordering convention.

## Explicitly out of scope

- SpidR batched CNN-only extraction — raise clearly, don't implement (matches
  existing single-segment `audio_to_cnn` restriction).
- Tuning `estimated_embedding_mb_per_second` specifically for CNN-only batch sizing
  — reuse the existing (conservative) heuristic unchanged.
- Anything in the `echoframe` repo — that's a separate downstream consumer and not
  this task; this plan is scoped entirely to `to_vector`.

## Tests to add

- `tests/test_to_embeddings.py` (add to the existing `ToEmbeddingsTests` class,
  next to `test_hf_batch_helper_splits_huggingface_batch_outputs` and
  `test_audio_to_cnn_raises_clear_error_for_spidr`, whose exact patterns to mirror):
  - `hf_batch_helper.audio_batch_to_cnn` splits a batch correctly: mock
    `to_vector.hf_batch_helper.load.prepare_feature_extractor`, build a
    `FakeHuggingFaceModel`, set `model.feature_extractor = mock.Mock(return_value=
    <tensor>)`, mock `model._get_feat_extract_output_lengths`, assert per-item
    `extract_features` shapes/values, that `hidden_states` is `None` on every
    returned item, and that **the model itself was never called** (assert
    `model.outputs`/call-count was untouched, or wrap `model.__call__` in a
    `mock.Mock` side-channel) — this is the core regression to guard, not just
    output shape.
  - `compute_cnn_output_lengths` no-`attention_mask` fallback matches
    `extract_features.shape[1]`.
  - `batch_helper.single_batch_to_cnn_outputs` raises the SpidR "not implemented"
    error, mirroring `test_audio_to_cnn_raises_clear_error_for_spidr`'s
    `assertRaisesRegex` pattern exactly.
- `tests/test_public_api.py`:
  - Add `'filename_batch_to_cnn'` and `'iter_filename_batch_to_cnn'` to the name
    list in `test_public_api_exports_main_helpers` (lines 9-27).
  - Add a test mirroring `test_iter_filename_batch_to_vector_yields_outputs`
    (lines 40-50) exactly, but patching
    `to_vector.to_embeddings.batch_helper.iter_handle_cnn_batching` and calling
    `to_vector.iter_filename_batch_to_cnn(...)` — note this existing test's
    `assert_called_once_with` pins the **exact positional-arg order** passed
    through from the public function to the batch_helper call; match that
    convention for the new function's signature/passthrough too.
- Confirm the full existing suite still passes: `.venv/bin/python -m pytest
  tests/ -q` (repo has its own `.venv`, same convention as `echoframe`).

## Verification

- Run the new tests plus the full existing suite; no regressions.
- Manually sanity-check (script or REPL) that `iter_filename_batch_to_cnn` returns
  items whose `.extract_features` numpy shapes match calling `audio_to_cnn` on the
  same files one at a time (same values, batched vs. unbatched) — this is the
  correctness bar, not just "it runs."
- Confirm via a mock/spy that the transformer forward (`model.__call__`) is never
  invoked anywhere in the new CNN-only call path — this is the entire point of the
  work, so it needs an explicit regression test, not just an implicit assumption.

## House conventions observed in this repo (follow them, don't introduce new ones)

- Docstrings: short, single-purpose, sometimes with a colon-aligned arg list (see
  `to_embeddings.py` functions), sometimes a single line (see some
  `hf_batch_helper.py` functions) — match whichever style the immediately
  surrounding function in the same file uses.
- `if x: return`/`if x: raise` one-liners where the body is a single short
  statement (see `to_embeddings.py:104`, `:151`, etc.).
- Public function naming: `filename_to_X` (single-segment, disk), `audio_to_X`
  (single-segment, in-memory array), `filename_batch_to_X` (batch, disk, eager
  list), `iter_filename_batch_to_X` (batch, disk, generator) — the new CNN batch
  functions must follow this exactly, matching the existing `_to_vector` family.
- **Never bump the version manually** — `.githooks/pre-commit` runs
  `scripts/bump_pyproject_version.py` and stages `pyproject.toml` automatically on
  every commit.
- Commit only what's requested/approved; this plan does not authorize pushing —
  stop after implementation + tests are green and report back.
