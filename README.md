# echoframe

`echoframe` is a small Python package for storing intermediate model artifacts
on disk. The intended scope is hidden states, attention outputs, and derived
artifacts such as codebooks, with support for both temporary caches and
long-lived experiment stores.

It treats `phraser` as the source of truth for object metadata and stores only
metadata about model outputs plus pointers to payloads.

## Install

```bash
uv pip install git+https://git@github.com/martijnbentum/echoframe.git
```

After installation, import it as:

```python
import echoframe
```

## API

The first implementation exposes `echoframe.Store` and
`echoframe.Metadata`.

```python
from echoframe import Store

store = Store('cache')
```

## Docs

The design notes and suggested storage approach are in
[docs/approach.md](/Users/martijn.bentum/echoframe/docs/approach.md).
