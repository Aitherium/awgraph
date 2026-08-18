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

from awgraph.graph import CodeGraph, CodeChunk
from awgraph.registry import CodeGraphRegistry, get_codegraph_registry
from awgraph.store import CodeGraphStore
from awgraph.base import BaseFacultyGraph, GraphSyncConfig

__version__ = "1.1.0"

__all__ = [
    "CodeGraph",
    "CodeChunk",
    "CodeGraphRegistry",
    "CodeGraphStore",
    "BaseFacultyGraph",
    "GraphSyncConfig",
    "get_codegraph_registry",
]
