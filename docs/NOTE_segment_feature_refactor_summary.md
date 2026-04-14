## Segment Feature Refactor Summary

The segment retrieval refactor moved the segment-specific orchestration out of
`phraser` and into `echoframe`.

### What Changed

- `echoframe.Store` now exposes typed loaders:
  - `load_embeddings(...)`
  - `load_many_embeddings(...)`
  - `load_codebook(...)`
  - `load_many_codebooks(...)`
- `echoframe.segment_features` now owns:
  - segment validation
  - `segment_to_echoframe_key(...)`
  - collar application
  - store lookup
  - compute-on-miss
  - frame selection
  - typed object loading after storage
- batch segment retrieval now routes through the typed batch loaders.

### Why

- `Embeddings` and `Codebook` construction now live closer to storage and
  retrieval semantics.
- `phraser.segment_embeddings` can stay as a thin wrapper instead of owning the
  orchestration logic.
- typed loading stays explicit instead of overloading generic `load(...)`
  behavior.

### Current Limitations

- batch compute still iterates segment by segment before typed batch loading
- `segment_features.py` still has some duplication between embeddings and
  codebook paths
- the module now intentionally depends on `frame` and `to-vector` output
  conventions

### Likely Follow-Up

- reduce duplication inside `segment_features.py`
- add an explicit duplicate-segment batch test at the orchestration layer
- consider deduplicating batch compute before the store/load step if it becomes
  worthwhile
