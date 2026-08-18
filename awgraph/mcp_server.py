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
the full agent runtime. Both are a large ask for someone who just wants their
editor's agent to stop grepping. An MCP server is the format coding agents
already consume, so onboarding becomes one line of JSON instead of an
integration.

DESIGN NOTES THAT MATTER
------------------------
**The index is per-repository and cached on disk**, so `code_index` is a
one-time cost per project and every later call is a load. The cache lives outside
the repository (see `awgraph.graph._get_data_path`), so pointing this at someone
else's checkout never writes into it.

**A missing index is reported, never silently built.** An agent calling
`code_search` against an unindexed repo gets a message telling it to run
`code_index` first, rather than a multi-minute pause that reads as a hang and
usually gets killed.

**Empty results say so explicitly.** "No matches" and "no index" are different
answers with different fixes, and an agent that cannot tell them apart will
conclude the repository does not contain what it is looking for.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
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
    out = []
    for c in chunks:
        path = _rel(str(getattr(c, "source_path", "") or ""), root)
        out.append(json.dumps({
            "name": getattr(c, "name", ""),
            "type": getattr(getattr(c, "chunk_type", None), "value", ""),
            "file": path,
            "line": getattr(c, "start_line", 0),
            "signature": (getattr(c, "signature", "") or "").strip()[:300],
            "calls": list(getattr(c, "calls", []) or [])[:15],
            "called_by": list(getattr(c, "called_by", []) or [])[:15],
        }))
    return "\n".join(out)


def _no_index(root: str) -> str:
    return (f"No index for {root}. Run the `code_index` tool on this path first "
            "(one-time per repository; the result is cached on disk outside the "
            "repo). Not built automatically here because indexing a large tree "
            "takes minutes and would read to you as a hung call.")


def build_server():
    """Construct the MCP server. Raises ImportError if `mcp` is absent."""
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover - exercised by a bare install
        raise ImportError(_INSTALL_HINT) from exc

    server = Server("awgraph")

    @server.list_tools()
    async def list_tools() -> list:
        path = {"type": "string",
                "description": "Repository root (absolute path preferred)."}
        return [
            Tool(name="code_index",
                 description=("Index a repository so the other tools can answer. "
                              "One-time per repo; cached on disk outside it."),
                 inputSchema={"type": "object",
                              "properties": {"path": path},
                              "required": ["path"]}),
            Tool(name="code_search",
                 description=("Find code by MEANING as well as keyword. Returns "
                              "symbols with file, line, signature, calls and "
                              "callers — not raw file text."),
                 inputSchema={"type": "object", "properties": {
                     "query": {"type": "string",
                               "description": "Natural language or symbol name."},
                     "path": path,
                     "limit": {"type": "integer", "default": 10}},
                     "required": ["query", "path"]}),
            Tool(name="code_callers",
                 description="What calls this symbol.",
                 inputSchema={"type": "object", "properties": {
                     "symbol": {"type": "string"}, "path": path},
                     "required": ["symbol", "path"]}),
            Tool(name="code_calls",
                 description="What this symbol calls.",
                 inputSchema={"type": "object", "properties": {
                     "symbol": {"type": "string"}, "path": path},
                     "required": ["symbol", "path"]}),
            Tool(name="code_stats",
                 description=("What is in the index, including embedding "
                              "coverage — 0% means search is keyword-only."),
                 inputSchema={"type": "object",
                              "properties": {"path": path},
                              "required": ["path"]}),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list:
        root = os.path.abspath(str(arguments.get("path") or "."))

        if name == "code_index":
            if not os.path.isdir(root):
                return [TextContent(type="text", text=f"Not a directory: {root}")]
            graph, _ = await _open(root, build=True)
            n = len(graph.chunks)
            if n == 0:
                return [TextContent(type="text", text=(
                    f"Indexed 0 chunks from {root} — no importable Python found. "
                    "awgraph parses Python only."))]
            return [TextContent(type="text",
                                text=f"Indexed {n:,} chunks from {root}.")]

        graph, ok = await _open(root, build=False)
        if not ok:
            return [TextContent(type="text", text=_no_index(root))]

        if name == "code_search":
            hits = await graph.hybrid_query(str(arguments.get("query", "")),
                                            max_results=int(arguments.get("limit", 10)))
            return [TextContent(type="text", text=_fmt(hits, root))]

        if name in ("code_callers", "code_calls"):
            symbol = str(arguments.get("symbol", ""))
            edge = "called_by" if name == "code_callers" else "calls"
            matches = [c for c in graph.chunks.values()
                       if getattr(c, "name", "") == symbol
                       or str(getattr(c, "name", "")).split(".")[-1] == symbol]
            if not matches:
                return [TextContent(type="text", text=(
                    f"No symbol named {symbol!r} in the index. It may be defined "
                    "outside this repository, or the index may predate it."))]
            names, seen = [], set()
            for c in matches:
                for nm in getattr(c, edge, []) or []:
                    if nm not in seen:
                        seen.add(nm)
                        names.append(nm)
            by_name = {getattr(c, "name", ""): c for c in graph.chunks.values()}
            resolved = [by_name[n] for n in names if n in by_name]
            external = [n for n in names if n not in by_name]
            text = _fmt(resolved, root)
            if external:
                text += ("\n\nOutside this repository (stdlib or third party): "
                         + ", ".join(external[:25]))
            return [TextContent(type="text", text=text)]

        if name == "code_stats":
            total = len(graph.chunks)
            embedded = sum(1 for c in graph.chunks.values()
                           if getattr(c, "embedding", None) is not None)
            by_type: dict = {}
            for c in graph.chunks.values():
                key = getattr(getattr(c, "chunk_type", None), "value", "?")
                by_type[key] = by_type.get(key, 0) + 1
            note = ("" if embedded else
                    " NOTE: no embeddings — search is keyword-only, which is "
                    "silent at query time (results still look confident).")
            return [TextContent(type="text", text=json.dumps({
                "root": root, "chunks": total, "by_type": by_type,
                "embedded_chunks": embedded,
                "embedding_coverage": round(embedded / total, 4) if total else 0.0,
            }) + note)]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


def main(argv: list[str] | None = None) -> int:
    """Serve over stdio. Blocks until the client disconnects."""
    try:
        server = build_server()
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    async def run() -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
