# echoframe

`echoframe` is a small Python package for storing intermediate model artifacts
on disk. The intended scope is hidden states, attention outputs, and derived
artifacts such as codebooks, with support for both temporary caches and
long-lived experiment stores.

## Install

```bash
uv pip install git+https://git@github.com/martijnbentum/echoframe.git
```

After installation, import it as:

```python
import echoframe
```

## Docs

The design notes and suggested storage approach are in
[docs/approach.md](/Users/martijn.bentum/echoframe/docs/approach.md).
