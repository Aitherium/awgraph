# Changelog — awgraph

All notable changes to awgraph are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-08-18

### Added

- **Initial public release** of awgraph as a standalone package
- **CodeGraph engine**: Real AST parsing of Python code into semantic chunks
- **Call graphs**: Automatic extraction of call relationships (what calls what)
- **Hybrid search**: Full-text search (FTS5) + semantic search support (embeddings optional)
- **SQLite backend**: Durable storage with WAL mode for concurrent read access
- **CodeGraphRegistry**: Multi-root indexing with tenant isolation
- **Memory optimization**: Float32 vectors (array('f')) instead of Python lists
- **Optional NumPy support**: Accelerated vector operations when available
- **No external hard dependencies**: httpx only; numpy is optional

### Changed

- **Extracted from AitherOS**: Originally part of AitherOS's CodeGraph faculty
- **Removed AitherOS dependencies**: Dropped internal imports from lib.*, lib.core.*, lib.agents.*
- **Stubbed optional features**: Strata persistence, TLS with internal CA, internal embeddings engine

### Known limitations

- Registry is in-memory only (no Strata persistence)
- Embeddings must be provided externally or via plugin hooks
- No vLLM reranking (disabled in public version)
- No AitherOS infrastructure assumptions (paths, config files, etc.)

### Under the hood

- **4850 lines of core indexing logic** extracted and vetted for public use
- **Minimal base class** (`BaseFacultyGraph`) provides plugin architecture
- **Standard Python logging** instead of internal AitherChronicle
- **Plain httpx** instead of internal AsyncClient (with AitherNet CA)

## Benchmarks (measured 2026-07)

On AitherOS/lib/faculties (scope, 200 commits, 15 sampled, 2400 chunks, 800 embedded):

| Retriever | Recall@10 | Avg tokens | Notes |
|-----------|-----------|-----------|-------|
| Grep | 0.933 | 350k | Module-level code, comments, strings |
| Graph (semantic) | 0.867 | ~1.5k | Function/method/class bodies only |
| UNION (graph + grep) | 1.000 | ~50k | Graph-first, grep fallback |

Graph and grep are **complementary**, not rivals. Graph excels on intent queries; grep handles structural searches.

## Roadmap (future versions)

- Plugin architecture for custom embeddings (awgraph.plugins)
- Optional Strata persistence for AitherOS deployments
- Support for other languages via parser plugins (Java, Go, Rust, etc.)
- Streaming API for large codebases (100M+ lines)
- GraphQL query interface (optional)

---

For the full history before public release, see TECH_DEBT.md in the AitherOS monorepo.
