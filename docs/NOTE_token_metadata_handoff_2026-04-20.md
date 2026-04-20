# Token metadata handoff

## Completed

The typed container API was updated in `echoframe/embeddings.py` and
`echoframe/codebooks.py`.

Implemented behavior:
- `Embeddings.metadatas` is now a lazy property that loads metadata through
  `self.store.load_many_metadata(self.echoframe_keys)`.
- `Codebook.metadata` is now a lazy property that loads the indices metadata
  through `self.store.load_metadata(self.echoframe_key)`.
- `TokenEmbeddings.echoframe_keys` now returns a flat tuple with all
  `echoframe_keys` from all embedding tokens.
- `TokenEmbeddings.metadatas` now returns a flat tuple built from each token's
  `metadatas` property.
- `TokenCodebooks.metadatas` now returns a flat tuple built from each token's
  `metadata` property.

No metadata constructor parameters were kept. The source of truth remains the
stored `echoframe_keys` and the bound store/path.

## Open dedupe question

The token collection classes still use their old token-level dedupe logic.

Current behavior:
- `TokenEmbeddings` dedupes tokens by `token.echoframe_key`, which is the
  first key of a token.
- `TokenCodebooks` dedupes tokens by `token.echoframe_key`.

Why this is still open:
- `TokenEmbeddings.echoframe_keys` is now a flat aggregate of all keys, but
  collection dedupe still happens before that flattening.
- For multi-layer embeddings, deduping only on the first key may be too weak
  if two tokens could share the first key but differ elsewhere.
- The user suggested `phraser_key` may be the right conceptual identity, but
  this has not been implemented or validated yet.

Recommended follow-up:
- Revisit whether token dedupe is still wanted at all.
- If it is wanted for embeddings, compare token identity using the full
  `token.echoframe_keys` tuple, or explicitly derive identity from the
  intended token concept such as `phraser_key`.
- Keep the change narrow and add focused tests before changing dedupe.

## Test follow-up

Tests were not updated yet in this pass.

Files most directly affected:
- `tests/test_embeddings.py`
- `tests/test_embeddings_multikey.py`
- `tests/test_codebook_indices.py`
- `tests/test_store_typed_loaders.py`
- `tests/test_segment_features.py`

Recommended assertions to add:
- `Embeddings.metadatas` loads one metadata object per key.
- `Codebook.metadata` loads the indices metadata object.
- `TokenEmbeddings.echoframe_keys` returns a flat tuple containing all keys
  from all tokens.
- `TokenEmbeddings.metadatas` returns a flat tuple containing all metadata
  objects from all tokens.
- `TokenCodebooks.metadatas` returns one metadata per token.
- Detached objects without a bound store or path still raise `ValueError`
  through the existing lazy store access path.

## Verification status

Automated verification was blocked in the current shell context:
- `.venv/bin/python` was not available from the package subdirectory context at
  the time of the check.
- The available system `python3` did not have `pytest` or `numpy`, so targeted
  test execution could not run.

Recommended next command from the repo root:
- `.venv/bin/python -m pytest tests/test_embeddings.py`
- `.venv/bin/python -m pytest tests/test_codebook_indices.py`
- `.venv/bin/python -m pytest tests/test_store_typed_loaders.py`
