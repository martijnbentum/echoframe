# Resilient Codebook Batch Indices

## Summary

Keep batch codebook-index computation small and resilient:

- return `TokenCodebooks`
- continue after per-item failures
- optionally stop after collecting enough successful tokens via `max_tokens=None`
- log failures to a fresh file for each call
- report how many tokens were created out of the input count

## Feature 1: Resilient Batch Processing

### Requirements

- `get_codebook_indices_batch(...)` must keep processing later items when one item fails
- the function must still return `TokenCodebooks`
- add `max_tokens=None`
- when `max_tokens` is set, stop once that many successful tokens have been collected
- failures must be written to a fresh log file created for that call
- at the end of execution, report how many tokens were created out of the total input count
- when `max_tokens` is set, include that limit in the final report
- if failures occurred, emit a warning that summarizes the failures and includes the log file path

### Tests

- a batch with one failed item still returns successful tokens from later items
- a batch with `max_tokens=2` stops after 2 successes even if more candidates remain
- a batch with failures creates a fresh log file for that call
- the final report includes created-count and input-count information
- when `max_tokens` is set, the final report includes the limit
- failures trigger a warning that includes the log file path

