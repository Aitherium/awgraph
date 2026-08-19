"""An MCP server for awgraph — one config line, any coding agent.

    pip install "awgraph[mcp]"

then point a client at `awgraph mcp`:

    {"mcpServers": {"awgraph": {"command": "awgraph", "args": ["mcp"]}}}

That works for Claude Code, Cursor, Windsurf, Zed, and anything else speaking MCP
over stdio. The agent gains structural code search — symbols, signatures, callers
and callees — instead of pasting grep output into its own context.

WHY THIS EXISTS
---------------
The engine was previously reachable only two ways: as a Python import, or through
a full agent runtime. Both are a large ask for someone who just wants their
editor's agent to stop grepping. MCP is the format coding agents already consume,
so onboarding becomes one line of JSON instead of an integration.

DESIGN NOTES THAT MATTER
------------------------
**The index is per-repository and cached on disk**, so `code_index` is a one-time
cost per project and every later call is a load. The cache lives outside the
repository, so pointing this at someone else's checkout never writes into it.

**A missing index is reported, never silently built.** An agent calling
`code_search` against an unindexed repo gets a message telling it to run
`code_index` first, rather than a multi-minute pause that reads as a hang and
usually gets killed — leaving no index and no explanation.

**Empty results say which kind of empty they are.** "No matches" and "no index"
are different answers with different fixes, and an agent that cannot tell them
apart concludes the repository does not contain what it is looking for and stops.

SDK VERSION
-----------
Written against the `mcp` 2.x `MCPServer` API and pinned to it. 1.x exposed a
different surface (`Server` with `@list_tools()` / `@call_tool()` decorators)
which 2.0 removed — code written for one raises `AttributeError` on the other.
That is not hypothetical: the first version of this module was written against a
1.x installed locally, passed its tests, and failed on a clean install because
pip resolved 2.0. Pinning is what keeps the tested version and the shipped
version the same one.
"""

from __future__ import annotations

import json
import os
from typing import Any

_INSTALL_HINT = (
    "The MCP server needs the `mcp` package: pip install \"awgraph[mcp]\". "
    "Raised rather than degraded, because an MCP server that starts and serves "
    "no tools looks to the client exactly like a server with nothing to offer."
)


def _rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except (ValueError, OSError):
        return path


async def _open(root: str, build: bool):
    from awgraph.graph import CodeGraph, _load_chunk_cache, _save_chunk_cache

    graph = CodeGraph(root_path=root, auto_index=False)
    if build:
        await graph.index_codebase(root)
        _save_chunk_cache(graph, root)
        return graph, True
    loaded = _load_chunk_cache(graph, root)
    return graph, bool(loaded and graph.chunks)


def _fmt(chunks: list, root: str) -> str:
    if not chunks:
        return ("No matches. The index exists, so this is a real negative answer "
                "rather than a missing index.")
    return "\n".join(json.dumps({
        "name": getattr(c, "name", ""),
        "type": getattr(getattr(c, "chunk_type", None), "value", ""),
        "file": _rel(str(getattr(c, "source_path", "") or ""), root),
        "line": getattr(c, "start_line", 0),
        "signature": (getattr(c, "signature", "") or "").strip()[:300],
        "calls": list(getattr(c, "calls", []) or [])[:15],
        "called_by": list(getattr(c, "called_by", []) or [])[:15],
    }) for c in chunks)


def _no_index(root: str) -> str:
    return (f"No index for {root}. Run the `code_index` tool on this path first "
            "(one-time per repository; cached on disk outside the repo). Not "
            "built automatically here because indexing a large tree takes "
            "minutes and would read to you as a hung call.")


def build_server():
    """Construct the MCP server. Raises ImportError if `mcp` is absent."""
    try:
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - exercised by a bare install
        raise ImportError(_INSTALL_HINT) from exc

    server = MCPServer(
        name="awgraph",
        instructions=(
            "Structural code search over a Python repository. Call code_index "
            "once per repo, then code_search / code_callers / code_calls. "
            "Results are symbols with file and line, not file text."),
    )

    @server.tool(
        name="code_index",
        description=("Index a repository so the other tools can answer it. "
                     "One-time per repo; cached on disk outside the repo."),
    )
    async def code_index(path: str) -> str:
        """Parse a repository into a symbol index.

        Args:
            path: Repository root. An absolute path is preferred.
        """
        root = os.path.abspath(path)
        if not os.path.isdir(root):
            return f"Not a directory: {root}"
        graph, _ = await _open(root, build=True)
        n = len(graph.chunks)
        if n == 0:
            # The reason matters and differs: with the multilang extra this
            # really is an empty tree, without it the files may be there and
            # simply unparseable. Reporting "Python only" in both cases sent a
            # user with a TypeScript repo looking for a bug that did not exist.
            from awgraph import multilang  # noqa: PLC0415

            ok, why = multilang.available()
            if ok:
                langs = len(multilang.supported_extensions())
                return (f"Indexed 0 chunks from {root} — nothing indexable found "
                        f"({langs} extensions supported, documents included).")
            return (f"Indexed 0 chunks from {root} — no importable Python found, "
                    f"and multi-language indexing is unavailable ({why}). "
                    "Install awgraph[multilang] to index other languages and docs.")
        return f"Indexed {n:,} chunks from {root}."

    @server.tool(
        name="code_search",
        description=("Find code by MEANING as well as keyword. Returns symbols "
                     "with file, line, signature, calls and callers."),
    )
    async def code_search(query: str, path: str, limit: int = 10) -> str:
        """Search the index.

        Args:
            query: Natural language description or a symbol name.
            path: Repository root that was indexed.
            limit: Maximum results.
        """
        root = os.path.abspath(path)
        graph, ok = await _open(root, build=False)
        if not ok:
            return _no_index(root)
        return _fmt(await graph.hybrid_query(query, max_results=limit), root)

    async def _edges(symbol: str, path: str, edge: str) -> str:
        root = os.path.abspath(path)
        graph, ok = await _open(root, build=False)
        if not ok:
            return _no_index(root)
        matches = [c for c in graph.chunks.values()
                   if getattr(c, "name", "") == symbol
                   or str(getattr(c, "name", "")).split(".")[-1] == symbol]
        if not matches:
            return (f"No symbol named {symbol!r} in the index. It may be defined "
                    "outside this repository, or the index may predate it.")
        names: list = []
        seen = set()
        for c in matches:
            for nm in getattr(c, edge, []) or []:
                if nm not in seen:
                    seen.add(nm)
                    names.append(nm)
        by_name = {getattr(c, "name", ""): c for c in graph.chunks.values()}
        text = _fmt([by_name[n] for n in names if n in by_name], root)
        external = [n for n in names if n not in by_name]
        if external:
            # Reported rather than dropped: showing only resolvable edges
            # understates the fan-out and hides third-party coupling.
            text += ("\n\nOutside this repository (stdlib or third party): "
                     + ", ".join(external[:25]))
        return text

    @server.tool(name="code_callers", description="What calls this symbol.")
    async def code_callers(symbol: str, path: str) -> str:
        """Find callers of a symbol.

        Args:
            symbol: Function, method or class name.
            path: Repository root that was indexed.
        """
        return await _edges(symbol, path, "called_by")

    @server.tool(name="code_calls", description="What this symbol calls.")
    async def code_calls(symbol: str, path: str) -> str:
        """Find what a symbol calls.

        Args:
            symbol: Function, method or class name.
            path: Repository root that was indexed.
        """
        return await _edges(symbol, path, "calls")

    @server.tool(
        name="code_stats",
        description=("What is in the index, including embedding coverage — "
                     "0% means search is keyword-only."),
    )
    async def code_stats(path: str) -> str:
        """Report index contents and embedding coverage.

        Args:
            path: Repository root that was indexed.
        """
        root = os.path.abspath(path)
        graph, ok = await _open(root, build=False)
        if not ok:
            return _no_index(root)
        total = len(graph.chunks)
        embedded = sum(1 for c in graph.chunks.values()
                       if getattr(c, "embedding", None) is not None)
        by_type: dict[str, int] = {}
        for c in graph.chunks.values():
            key = getattr(getattr(c, "chunk_type", None), "value", "?")
            by_type[key] = by_type.get(key, 0) + 1
        payload: dict[str, Any] = {
            "root": root, "chunks": total, "by_type": by_type,
            "embedded_chunks": embedded,
            "embedding_coverage": round(embedded / total, 4) if total else 0.0,
        }
        note = ("" if embedded else
                " NOTE: no embeddings — search is keyword-only, which is silent "
                "at query time (results still look confident).")
        return json.dumps(payload) + note

    return server


def main(argv: list[str] | None = None) -> int:
    """Serve over stdio. Blocks until the client disconnects."""
    import asyncio
    import sys

    try:
        server = build_server()
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        asyncio.run(server.run_stdio_async())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
