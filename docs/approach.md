# Approach

Keep the first version narrow and explicit.

Expose a small public API around three concepts:

1. `CacheKey`
2. `CacheEntry`
3. `DiskCache`

Suggested cache key fields:

- model identifier
- layer index
- artifact kind: `hidden_states`, `attention`, `codebook`
- input hash
- dtype
- shape
- optional metadata such as split, prompt id, or timestamp

Suggested storage layout:

```text
cache_root/
  model_name/
    artifact_kind/
      layer_000/
        <input_hash>.pt
        <input_hash>.json
```

Suggested behavior:

- Store tensor-like payloads in a binary format that preserves dtype and shape.
- Store metadata separately in JSON for easy indexing and debugging.
- Support `temporary` mode for disposable run caches.
- Support `persistent` mode for experiment artifacts you want to reuse.
- Make cache writes atomic by writing to a temp file and renaming.
- Add optional TTL or cleanup helpers only after the core write/read path is
  stable.

## Format Choices

For a pragmatic first version:

- Use `torch.save` if the project is PyTorch-only.
- Use `numpy.save` or `np.savez_compressed` if you want a lighter dependency
  boundary.
- Use JSON sidecars for metadata and provenance.

If this grows into large-scale dataset tooling, move later to a chunked format
such as `zarr`, but that is probably unnecessary for the first release.

## Initial Roadmap

- Add a disk cache class with `put`, `get`, `exists`, and `delete`.
- Add key normalization and input hashing.
- Add tests for temporary vs persistent storage behavior.
- Add one end-to-end example showing hidden-state caching during a forward pass.
