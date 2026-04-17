## Coding Style State 2026-04-18

Current state:
- `echoframe/model_registry.py` was cleaned up and committed in `be9c904`.
- `echoframe/store.py` had a substantial style cleanup pass and
  `tests/test_store_io.py` currently passes.
- `scripts/check_style.py` now checks more than before, including:
  - wrapped `from x import (...)`
  - quote style
  - missing docstrings and parameter blocks
  - small multi-line dict literals
  - direct nested calls
  - simple `if ...: return ...` layout
  - multi-line inline `raise SomeError(...)` with message text

Important limitation:
- The newer hard rules in `scripts/check_style.py` are still too broad for
  repo-wide enforcement.
- The script is useful as a detector right now, but not yet a realistic full
  gating step for the entire repository.

Current practical split:
- Agent should judge:
  - importance ordering of methods/functions
  - whether parameter docs are really needed
  - naming quality of intermediate variables
  - whether readability actually improved
- Script should judge:
  - explicit structural rules with low ambiguity
  - obvious wrapped import / dict / raise / call-shape violations

What was learned from `store.py`:
- Direct `outer(inner(...))` calls should usually be split into named steps
  when they are long or multi-line.
- Small dict literals that fit should stay on one line.
- Multi-line comprehensions should be expanded into loops.
- Simple `if ...: return ...` should be one line only if the combined line
  fits under the hard width.
- Multi-line `raise ValueError(...)` with inline message text should be turned
  into:
  - `message = ...`
  - `raise ValueError(message)`

Recommended next step:
1. Narrow `scripts/check_style.py` further before treating it as a hard gate.
2. Use concrete examples from files like `store.py`, `index.py`,
   `output_storage.py`, and `codebooks.py` to decide which patterns are truly
   hard failures and which should remain agent-reviewed.
3. Only after that, continue broader cleanup in another module.

Useful files:
- `docs/propose_agent_coding_check.md`
- `scripts/check_style.py`
- `echoframe/store.py`
- `echoframe/model_registry.py`
