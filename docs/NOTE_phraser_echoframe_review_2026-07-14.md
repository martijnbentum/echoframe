# Phraser/Echoframe Review Handoff - Updated 2026-07-30

This note was created from a review on 2026-07-14 and refreshed on
2026-07-30. It records which findings have been resolved and which still
need follow-up.

## Repository State Reviewed

- `phraser`: `/Users/martijn.bentum/repos/phraser`
  - branch: `main`, aligned with `origin/main`
  - refreshed HEAD: `b599253 Return MFCC features in frame-major order`
  - 37 commits ahead of the originally reviewed `936523c`
  - existing user edit: `docs/PLAN_segment_linking.md`
- `echoframe`: `/Users/martijn.bentum/repos/echoframe`
  - branch: `main`, aligned with `origin/main`
  - refreshed code HEAD: `a06b9d5 Remove private MFCC cache detection`
  - 9 commits ahead of the originally reviewed `4d9dd18`

## Change Summary Since The Original Review

### phraser

- The TextGrid staging and persistence policies remain in place.
- Segment linking and save validation were substantially tightened.
- Duplicate phrase identities are rejected within a
  `save_phrase_trees(...)` batch.
- TextGrid `existing='append'` still bypasses existence matching and can
  persist phrases that compare equal.
- MFCC extraction and batching were added.
- MFCC payloads are now returned in frame-major order.

Important current files:

- `phraser/segment.py`
- `phraser/textgrid_loader.py`
- `phraser/save_validation.py`
- `phraser/audio/mfcc.py`
- `phraser/audio/batch.py`
- `tests/test_position_on_ingest.py`
- `tests/test_audio_mfcc.py`

### echoframe

- Refcounted LMDB lifecycle support and `Store.close()` were added.
- Feature-computation helpers now require an explicit store.
- `acoustic_feature` was added as a first-class output type.
- MFCC storage no longer requires pseudo-model registration or fabricated
  model, layer, and collar fields.
- Acoustic feature storage relies only on the public `segment.mfcc`
  property, not phraser's private cache state.

Important current files:

- `echoframe/acoustic_features.py`
- `echoframe/compaction.py`
- `echoframe/lmdb_helper.py`
- `echoframe/metadata.py`
- `echoframe/output_storage.py`
- `echoframe/store.py`
- `tests/test_acoustic_features.py`
- `tests/test_store_end_to_end.py`

## Finding Status

### 1. echoframe LMDB environment leak - Resolved

Original severity: high.

Resolved by `c1b6e4b Add refcounted Store.close() to fix LMDB env fd leak`.

Current behavior:

- `_ENV_CACHE` entries hold a reference count.
- `LmdbIndex.close()` releases owned environment references.
- `Store.close()` exposes lifecycle management publicly.
- Equal-path stores remain usable until every owner closes.
- Repeated close calls are safe.
- Injected environments are not closed by the index.

Coverage is in `tests/test_store_end_to_end.py`, including cache eviction,
shared-environment lifetime, idempotent close, and injected-environment
behavior. The full echoframe suite now passes without file-descriptor
failures.

### 2. phraser global Phrase equality remains TextGrid identity - Open

Severity: medium. This may be intentional, but it broadens behavior beyond the
TextGrid import path.

`phraser/segment.py` treats two `Phrase` objects as equal when
`(audio_id, speaker_id, start)` match. This supports TextGrid replacement and
upsert matching, but it also affects all generic phrase comparison, hashing,
list membership, and sibling navigation.

Current evidence:

- `Segment.__eq__` and `Segment.__hash__` use the subclass
  `IDENTITY_FIELDS`.
- `Phrase.IDENTITY_FIELDS` is still
  `{'audio_id', 'speaker_id', 'start'}`.
- `find_matching_textgrid_phrases()` still matches with Phrase equality.
- `existing='append'` deliberately bypasses matching and uses
  `store.save_many(items)`.
- `test_save_textgrid_items_append_does_not_check_existing` confirms that
  two same-identity phrases can be persisted.
- `next_sibling` and `prev_sibling` still use `siblings.index(self)`.
- The newer `save_phrase_trees(...)` duplicate-identity guard applies only
  within that save path and does not remove the persisted-append case.

Why this is risky:

- `existing='append'` can intentionally create multiple phrases with the same
  `(audio_id, speaker_id, start)`.
- Those duplicates now compare equal even if labels, end times, filenames, and
  identifiers differ.
- Code that does `siblings.index(self)`, set/dict membership, or deduplication
  may pick the wrong phrase in the presence of duplicates.

Suggested direction:

- Decide whether global `Phrase.__eq__` and `__hash__` should represent
  object identity or TextGrid matching identity.
- Add a dedicated helper for TextGrid matching, for example
  `textgrid_phrase_identity(phrase)` or `textgrid_phrases_match(a, b)`.
- Add tests for appended duplicate phrases and `next_sibling`/`prev_sibling` so
  the intended semantics are explicit.

### 3. fully unreferenced shard documentation remains ambiguous - Open

Severity: low to medium.

`verify_integrity()` now reports `unreferenced_shard_files`, but it
intentionally does not mark `ok=False` for those files. This is reasonable
if they are treated as waste rather than data loss.

The nuance is that README says overwrite leftovers are reported by
`store.overview(include_garbage=True)` and reclaimed by
`store.compact_shards()`. That is true for garbage inside indexed shards,
but not for fully unreferenced shard files. The code comment in
`find_unreferenced_shard_files()` says these files are invisible to
`list_shards()` and `compact_shards()`.

Current evidence:

- `verify_integrity()` returns `unreferenced_shard_files`.
- Those files intentionally do not make the integrity result fail.
- `find_unreferenced_shard_files()` states that these files are invisible
  to `list_shards()` and `compact_shards()`.
- The README's overwrite section still describes only garbage inside
  indexed shards.

Suggested direction:

- Clarify the README: indexed shard garbage is handled by overview/compaction;
  fully unreferenced shard files are surfaced by `verify_integrity()`.
- Consider including `unreferenced_shard_files` in
  `overview(include_integrity=True)` documentation.
- Consider an explicit cleanup helper for unreferenced shard files, but only if
  the project wants automatic cleanup of crash leftovers.

## Verification Refreshed On 2026-07-30

### phraser

```bash
cd /Users/martijn.bentum/repos/phraser
.venv/bin/python -B -m pytest -p no:cacheprovider -q
```

Observed result:

```text
248 passed, 37 subtests passed, 99 warnings
```

### echoframe

```bash
cd /Users/martijn.bentum/repos/echoframe
.venv/bin/python -m pytest -q
```

Observed result:

```text
357 passed, 62 subtests passed
```

## Practical Next Steps

1. Decide whether `phraser` global `Phrase` equality should remain TextGrid
   identity-based or move to a dedicated TextGrid matching helper.
2. Add sibling-navigation coverage for persisted same-identity phrases.
3. Clarify `echoframe` docs for indexed garbage versus fully unreferenced shard
   files.
4. Decide whether echoframe should offer explicit cleanup for fully
   unreferenced shard files.
