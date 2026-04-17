## Proposed Agent Coding Check

### Agent Role

- Organize modules and classes by importance.
- Keep the main public API near the top.
- Push helper functions and helper methods toward the bottom.
- Within classes, keep dunder methods first.
- After dunder methods, keep the most important public methods near the top.
- Place less important public methods below the main user-facing methods.
- Keep private helpers near the bottom.
- Judge which public methods and functions are important enough to need
  parameter explanations.
- Use compact parameter blocks when they help the caller understand inputs.
- Prefer clear intermediate variables with good names when nested expressions
  start to hide meaning.
- In wrapped calls, keep as much as practical on the first line.
- When a call wraps, prefer one continuation line over splitting arguments
  across multiple continuation lines when that still fits cleanly.
- Do not insert a newline before an argument that still fits on the current
  continuation line.
- When refactoring for style, re-check behavior carefully, especially around
  control flow such as `continue`, `break`, and early returns.

### Script Role

- Enforce the hard width limit.
- Flag wrapped `from x import (...)` imports.
- Flag obvious double-quote cases where single quotes would work.
- Flag missing docstrings on public methods and functions, including `__init__`.
- Flag malformed compact parameter blocks.
- Flag uses of `*args` and `**kwargs`.
- Flag small multi-line dict literals that should stay on one line when they
  fit within the hard width.
- Flag direct nested function calls when they are long or multi-line and should
  be split into named intermediate steps.
- Flag simple `if ...: return ...` blocks that should stay on one line when the
  combined line fits within the hard width.
- Flag multi-line comprehensions that should be expanded into explicit loops.
- Flag wrapped call layouts that leave avoidable trailing space on an earlier
  line or split arguments across extra continuation lines without need.
- Warn on larger multi-line dict literals returned directly instead of being
  built first in a variable.

### Current Interpretation

- The script should enforce structural and lexical rules that are cheap and
  predictable.
- The script should not try to decide which public methods are most important.
- The agent should decide importance, readability, naming quality, and whether
  a parameter block adds real value.
- The agent should treat script findings as a minimum bar, not as a complete
  style review.

### Feedback

- The current `check_style.py` script is useful for surfacing style drift, but
  some rules are still too broad when turned into hard failures across the full
  repo.
- The strongest pattern so far is:
  mechanical structure belongs in the script,
  importance and readability belong to the agent.
- A good rule for the script is:
  only enforce checks that are explicit, local, and low-ambiguity.
- A good rule for the agent is:
  when the script is quiet, still review ordering, naming, and whether code
  reads in the intended style rather than merely passing checks.
