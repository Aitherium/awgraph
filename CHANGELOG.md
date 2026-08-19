# Changelog — awgraph

All notable changes to awgraph are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.3.0] - 2026-08-19

### Added

- **Multi-language and document indexing** via the new `multilang` extra
  (`pip install awgraph[multilang]`). awgraph was Python-only: it parsed with
  CPython's `ast` and discovered files with `rglob("*.py")`, so every `.ts`,
  `.go`, `.rs`, `.cs` file — and every `.md` — was absent from the index. On a
  mixed repository that is most of the repository, and the failure was SILENT:
  the index built, queries returned, and the answers came from whatever fraction
  of the tree happened to be Python.

  Symbol-bearing languages are chunked by SYMBOL through
  [repowise](https://pypi.org/project/repowise/) (75 extensions, 42 languages),
  whose ids survive a symbol MOVING — which is why the id, not a body hash, is
  what a chunk's identity is derived from. Symbol-less files (markdown,
  asciidoc, json, yaml, toml, ini, csv, html, css, xml, sql) are chunked by
  SECTION: prose by heading, everything else by an overlapping line window.

  Verified end to end rather than asserted — indexing a mixed tree and querying
  it returns `Rollback timeout@RUNBOOK.md`, `validateToken@auth.ts` and
  `StartServer@server.go`.

  The extra is optional and absence is not an error: parsing falls back to
  Python and records `multi-language indexing unavailable` in `parse_errors`,
  so "why is my .ts file missing" has an answer instead of an empty graph.

- **`discover_files(root, extensions=..., exclude_dirs=...)`** — the discovery
  policy is now a parameter. `extensions` defaults to exactly what the parser
  can handle, so discovery and parsing cannot drift into indexing files nothing
  can read. `exclude_dirs=[]` relies on `.gitignore` alone.

### Fixed

- **`.gitignore` was never honoured.** An *include* glob switches ripgrep's
  ignore-file handling OFF, and discovery passed `-g "*.py"` — so every ignored
  build artifact and vendored tree was indexed as source, invisibly, because
  those files genuinely exist. Extensions are now filtered in Python and only
  NEGATED globs are passed.

- **The exclude list excluded nothing** whenever the indexer ran from a
  directory other than the tree being indexed — the normal case for a library.
  ripgrep anchors a glob at the CURRENT WORKING DIRECTORY, not the search root,
  so `!node_modules/**` matched nothing. Now anchored with `**/`.

- **Symbol names carried the indexing machine's directory layout.** repowise
  derives `qualified_name` from the file path, so an absolute path produced
  `C.Users.me.tmp.probe.alpha` instead of `probe.alpha` — unfindable by its own
  name, and a local-path leak into an index that may be shared.

- `code_index` reported "awgraph parses Python only" for every empty result. It
  now distinguishes an empty tree from a missing `multilang` extra, which are
  different problems that previously read identically.

## [1.2.1] - 2026-08-18

### Fixed

- **`awgraph mcp` was broken on every clean install of 1.2.0.** The module was
  written against the `mcp` 1.x server API (`Server` with `@list_tools()` /
  `@call_tool()` decorators), which **2.0 removed** — so `build_server()` raised
  `AttributeError: 'Server' object has no attribute 'list_tools'`. It passed its
  tests locally, where 1.x happened to be installed, and failed for anyone whose
  pip resolved `mcp>=1.0` to 2.0, which by then was everyone.

  Rewritten for the 2.x `MCPServer` API and the dependency **pinned to `mcp>=2.0`**
  so the tested version and the shipped version are the same one. The tests now
  drive the server's own `list_tools()` / `call_tool()` rather than reaching into
  its internal handler registry, which is what coupled them to one SDK generation
  and let a broken build look green.

  Caught by installing the built wheel into a clean venv and running it — the
  check that a source-tree test structurally cannot perform.

## [1.2.0] - 2026-08-18

### Added

- **An MCP server: `awgraph mcp`.** One line of config and any MCP coding agent
  (Claude Code, Cursor, Windsurf, Zed) gains structural code search — symbols,
  signatures, callers and callees — instead of pasting grep output into its own
  context:

      pip install "awgraph[mcp]"
      {"mcpServers": {"awgraph": {"command": "awgraph", "args": ["mcp"]}}}

  Five tools: `code_index`, `code_search`, `code_callers`, `code_calls`,
  `code_stats`. Every design choice fights one failure — an agent that cannot
  tell "nothing matched" from "nothing is indexed" concludes the repository does
  not contain what it is looking for and stops. So an unindexed search names
  `code_index`, an unknown symbol says so, a tree with no Python says so, and
  `code_stats` always reports embedding coverage, saying "keyword-only" at 0%.
  Indexing is never implicit: auto-indexing would turn a missing index into a
  multi-minute pause that reads as a hung tool call.
- **`awgraph.plugins` — the host integration points this package has advertised
  since 1.0.0 while the module did not exist** and `plugins/` was an empty
  directory. `configure(logger_factory=..., async_client=..., embedding_engine=...,
  degradation_registry=...)` lets a host inject its own implementations instead
  of forking the engine. An unknown hook name RAISES rather than being ignored: a
  typo'd `embeddings_engine=` that silently did nothing would look exactly like a
  host that never configured one.

### Changed

- **`import awgraph` is now lazy** (PEP 562). This is not tidiness — it is what
  makes the hooks work at all. The engine snapshots its hooks at import time, and
  the previous eager `from awgraph.graph import ...` in `__init__` meant
  `import awgraph.plugins` loaded the engine BEFORE `configure()` could run.
  Measured: `configure()` returned cleanly and `active_hooks()` reported all four
  installed while the engine was still on httpx with embeddings off. It also
  makes importing the package cheap for anything that only wants the version or
  the plugin surface.

## [1.1.0] - 2026-08-18

### Added

- **A command line interface.** `pip install awgraph` now puts an `awgraph`
  command on PATH: `index`, `query`, `callers`, `calls`, `stats`, `selftest`.
  Until now the only way to use the package was to write async Python, which is
  a higher bar than the tool deserves.
- `awgraph selftest` indexes a throwaway package, retrieves from it, and asserts
  a **control query** (about something absent from the corpus) does not match as
  widely as the real one. Without that control the test passes on a retriever
  that returns everything, which is the failure mode a naive smoke test cannot
  see.
- `--json` on every read command, and meaningful exit codes: 0 success, 1 a real
  negative answer, 2 could not run. A caller can distinguish "nothing matched"
  from "there is no index".
- `awgraph stats` reports embedding coverage unconditionally, including 0%.

### Fixed

- **The index cache no longer writes into the repository being indexed.** It
  previously fell back to `<repo>/Library/Data/codegraph`, which pollutes a
  stranger's checkout and simply fails on a read-only or CI tree. Caches now go
  to `AWGRAPH_CACHE_DIR`, else the platform user-cache directory, keyed by a
  digest of the absolute repo path so two checkouts never share an index.

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
