"""awgraph — Semantic Python Code Graph.

A lightweight, portable Python AST indexer with call graphs, embeddings support,
and hybrid keyword+semantic search. Originally developed as part of AitherOS's
CodeGraph faculty, now available as a public package.

## What it does

- Real AST parsing (not regex)
- Extracts functions, classes, and methods into chunks
- Builds call graphs (what calls what, what's called by what)
- Hybrid search: keyword (FTS5) + semantic (embeddings)
- SQLite storage with WAL mode for concurrent access
- Optional NumPy for fast vector operations

## What it doesn't do

- PDFs, web pages, or "universal" anything — just Python code
- Embeddings by default — bring your own or use plugin hooks
- Network access by default — embeddings API must be provided externally
- Persistence across instances — in-memory by design; SQLite for durability

## Usage

```python
from awgraph import CodeGraph

# Create an indexer
graph = CodeGraph()

# Index a directory
await graph.index_codebase("/path/to/code")

# Query
chunks = await graph.query("rate limiter", limit=10)

for chunk in chunks:
    print(f"{chunk.name}: {chunk.calls}")
```

## Architecture

- **graph.py**: CodeGraph engine, AST parsing, call graphs, chunking
- **store.py**: SQLite+FTS5 backend for storage and search
- **registry.py**: Multi-root manager for indexing external repos
- **base.py**: Minimal faculty graph base class
- **logging.py**: Standard Python logging shim
- **degradation.py**: Optional dependency tracking

Plugin support via awgraph.plugins (not yet public).
"""

from __future__ import annotations

# LAZY on purpose (PEP 562). `awgraph.plugins` documents that a host must
# configure its hooks BEFORE the engine is imported, because the engine snapshots
# them at import time. Eagerly importing `awgraph.graph` here made that
# impossible: `import awgraph.plugins` executes this file first, so the engine was
# always already imported and every hook was silently ignored — configure()
# returned cleanly, the registry reported the hooks installed, and the engine ran
# on defaults. Measured while binding the platform: the HTTP client stayed httpx
# and _HAS_EMBEDDING_ENGINE stayed False with all four hooks reporting active.
#
# It also makes `import awgraph` cheap for anything that only wants the plugin
# surface or the version.
from awgraph.base import BaseFacultyGraph, GraphSyncConfig

_LAZY = {
    "CodeGraph": "awgraph.graph",
    "CodeChunk": "awgraph.graph",
    "CodeGraphRegistry": "awgraph.registry",
    "get_codegraph_registry": "awgraph.registry",
    "CodeGraphStore": "awgraph.store",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'awgraph' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))

__version__ = "1.4.3"

__all__ = [
    "CodeGraph",
    "CodeChunk",
    "CodeGraphRegistry",
    "CodeGraphStore",
    "BaseFacultyGraph",
    "GraphSyncConfig",
    "get_codegraph_registry",
]
