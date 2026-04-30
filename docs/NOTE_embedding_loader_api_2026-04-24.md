# Embedding Loader API Note (2026-04-24)

This note records the intentional embedding retrieval API shift introduced in
the current worktree.

## Summary

Hidden-state loading now uses two store methods:

- `Store.load_embedding(echoframe_key)`
- `Store.load_embeddings(echoframe_keys)`

Phraser convenience helpers are also available:

- `Store.phraser_key_to_embedding(phraser_key, model_name, layer, collar=500)`
- `Store.phraser_keys_to_embeddings(phraser_keys, model_name, layer,
  collar=500)`

The new shape is:

- one `echoframe_key` returns one `Embedding`
- many `echoframe_key` values return `Embeddings`
- `phraser_key`-based loading is an explicit convenience layer

## Removed Or Replaced Surface

The older typed embedding loader surface is no longer current:

- `load_embeddings(...)` as one-key multi-layer retrieval
- `load_many_embeddings(...)`
- `EmbeddingLoadRequest`
- `TokenEmbeddings`
- `frame_aggregation` as a store loader argument
- typed loader support for requesting several layers in one call

## Current Behavior

- `Embedding` validates one stored hidden-state payload and its metadata
- `Embeddings` is a collection of `Embedding` objects for one shared layer
- all items in one `Embeddings` object must share:
  `model_name`, `output_type`, and `layer`
- duplicate `phraser_key` values raise `ValueError`
- invalid requested keys are skipped with a printed message
- if every requested key is skipped, loading raises `ValueError`
- `Embeddings.to_numpy()` only works when every payload has the same shape

## Constraint To Keep In Mind

This API does not yet replace the old multi-layer or frame-aggregation
behavior. For variable-length frame payloads across several phraser objects,
`Embeddings.to_numpy()` may raise `NotImplementedError` instead of returning a
stacked array.

## README

The README should describe the new loader contract and avoid examples that rely
on the removed multi-layer or `frame_aggregation` embedding API.
