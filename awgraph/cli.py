"""awgraph command line interface.

The library is async and returns dataclasses; most people meeting a tool for the
first time want a terminal command that prints file:line. This module is the thin
translation layer between the two, and deliberately holds no retrieval logic of
its own — anything it could decide differently from ``CodeGraph`` would be a
second implementation to keep in step.

    awgraph index .                     # parse + persist an index for this repo
    awgraph query "retry with backoff"  # ask it a question in English
    awgraph callers send_request        # who calls this
    awgraph calls send_request          # what does this call
    awgraph stats                       # what is in the index
    awgraph selftest                    # prove the install works

Exit codes follow the house rule that silence is not a pass: 0 success, 1 a real
negative answer (no match), 2 the command could not run at all.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, List, Optional

from awgraph import __version__

EXIT_OK = 0
EXIT_EMPTY = 1
EXIT_DEAD = 2


def _fail(msg: str) -> int:
    print("awgraph: " + msg, file=sys.stderr)
    return EXIT_DEAD


def _rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except (ValueError, OSError):
        return path


async def _open_graph(root: str, build: bool):
    """Return ``(graph, usable)`` for ``root``.

    ``build`` distinguishes ``index`` (parse the tree) from every read command
    (load the persisted cache). A read that silently re-indexed would turn a
    missing index into a multi-minute pause that looks like a hang.
    """
    from awgraph.graph import CodeGraph, _load_chunk_cache

    graph = CodeGraph(root_path=root, auto_index=False)
    if build:
        await graph.index_codebase(root)
        return graph, True
    loaded = _load_chunk_cache(graph, root)
    return graph, bool(loaded and graph.chunks)


def _chunk_row(c: Any, root: str) -> dict:
    return {
        "name": getattr(c, "name", ""),
        "type": getattr(getattr(c, "chunk_type", None), "value", ""),
        "path": _rel(str(getattr(c, "source_path", "") or ""), root),
        "line": getattr(c, "start_line", 0),
        "signature": getattr(c, "signature", "") or "",
    }


def _print_rows(rows: List[dict], as_json: bool) -> int:
    if as_json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK if rows else EXIT_EMPTY
    if not rows:
        print("no matches")
        return EXIT_EMPTY
    for r in rows:
        loc = (r["path"] + ":" + str(r["line"])) if r["path"] else "(unknown)"
        kind = ("[" + r["type"] + "]") if r["type"] else ""
        print(loc + "  " + kind + " " + r["name"])
        if r["signature"]:
            print("    " + r["signature"].strip()[:160])
    return EXIT_OK


async def cmd_index(args: argparse.Namespace) -> int:
    import time

    from awgraph.graph import _get_data_path, _save_chunk_cache

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        return _fail("not a directory: " + root)

    t0 = time.perf_counter()
    graph, _ = await _open_graph(root, build=True)
    elapsed = time.perf_counter() - t0
    n = len(graph.chunks)
    if n == 0:
        print("indexed 0 chunks from " + root + " — no importable Python found",
              file=sys.stderr)
        return EXIT_EMPTY
    _save_chunk_cache(graph, root)
    print("indexed {0:,} chunks from {1} in {2:.1f}s".format(n, root, elapsed))
    print("cache: " + os.path.dirname(_get_data_path(root, "x")))
    return EXIT_OK


async def cmd_query(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.path)
    graph, ok = await _open_graph(root, build=False)
    if not ok:
        return _fail("no index for " + root + " — run: awgraph index " + args.path)
    chunks = await graph.hybrid_query(args.text, max_results=args.k)
    return _print_rows([_chunk_row(c, root) for c in chunks], args.json)


async def _edges(args: argparse.Namespace, direction: str) -> int:
    root = os.path.abspath(args.path)
    graph, ok = await _open_graph(root, build=False)
    if not ok:
        return _fail("no index for " + root + " — run: awgraph index " + args.path)

    matches = [c for c in graph.chunks.values()
               if getattr(c, "name", "") == args.symbol]
    if not matches:
        matches = [c for c in graph.chunks.values()
                   if str(getattr(c, "name", "")).split(".")[-1] == args.symbol]
    if not matches:
        print("no symbol named " + repr(args.symbol) + " in the index")
        return EXIT_EMPTY

    names: List[str] = []
    for c in matches:
        edge = "called_by" if direction == "callers" else "calls"
        names.extend(getattr(c, edge, []) or [])

    by_name = {getattr(c, "name", ""): c for c in graph.chunks.values()}
    seen = set()
    rows: List[dict] = []
    for nm in names:
        if nm in seen:
            continue
        seen.add(nm)
        hit = by_name.get(nm)
        if hit is not None:
            rows.append(_chunk_row(hit, root))
        else:
            # An edge whose target is outside the indexed tree (a stdlib or
            # third-party call). Reported rather than dropped: silently showing
            # only resolvable edges understates the fan-out.
            rows.append({"name": nm, "type": "external", "path": "",
                         "line": 0, "signature": ""})
    return _print_rows(rows, args.json)


async def cmd_stats(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.path)
    graph, ok = await _open_graph(root, build=False)
    if not ok:
        return _fail("no index for " + root + " — run: awgraph index " + args.path)

    by_type: dict = {}
    embedded = 0
    for c in graph.chunks.values():
        key = getattr(getattr(c, "chunk_type", None), "value", "?")
        by_type[key] = by_type.get(key, 0) + 1
        if getattr(c, "embedding", None) is not None:
            embedded += 1
    total = len(graph.chunks)
    coverage = round(embedded / total, 4) if total else 0.0
    out = {
        "root": root,
        "chunks": total,
        "by_type": by_type,
        "files": len(getattr(graph, "by_file", {}) or {}),
        "embedded_chunks": embedded,
        "embedding_coverage": coverage,
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return EXIT_OK

    print("root:   " + root)
    print("chunks: {0:,} across {1:,} files".format(total, out["files"]))
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print("  {0:<10} {1:,}".format(k, v))
    # Coverage prints unconditionally, including 0%. Without an embedding backend
    # hybrid_query silently degrades to keyword scoring and still returns ten
    # confident results, so "is the semantic half actually on?" must be
    # answerable without reading the source.
    print("embeddings: {0:,}/{1:,} chunks ({2:.1f}%)".format(
        embedded, total, coverage * 100))
    if embedded == 0:
        print("  note: no embeddings — queries are keyword-only "
              "(see README: embeddings)")
    return EXIT_OK


_SELFTEST_SRC = '''
class BackoffPolicy:
    """Backoff policy for flaky calls."""

    def next_delay(self, attempt: int) -> float:
        """Exponential delay with a ceiling."""
        return min(2.0 ** attempt, 30.0)


def send_request(url: str) -> str:
    """Send one request, retrying on failure."""
    policy = BackoffPolicy()
    return _do_send(url, policy)


def _do_send(url: str, policy: BackoffPolicy) -> str:
    return url
'''


async def cmd_selftest(_args: argparse.Namespace) -> int:
    """Index a throwaway package and prove retrieval really returns it."""
    import tempfile

    failures: List[str] = []
    with tempfile.TemporaryDirectory() as td:
        os.environ["AWGRAPH_CACHE_DIR"] = os.path.join(td, ".cache")
        pkg = os.path.join(td, "pkg")
        os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(pkg, "client.py"), "w", encoding="utf-8") as fh:
            fh.write(_SELFTEST_SRC)

        graph, _ = await _open_graph(td, build=True)
        names = {getattr(c, "name", "") for c in graph.chunks.values()}
        for want in ("send_request", "BackoffPolicy", "BackoffPolicy.next_delay"):
            if want not in names:
                failures.append("chunk missing: " + want)

        hits = await graph.hybrid_query("retry a request when it fails",
                                        max_results=5)
        if not hits:
            failures.append("hybrid_query returned nothing for a matching question")
        elif not any(str(getattr(h, "source_path", "") or "") for h in hits):
            failures.append("results carry no source_path — locations unusable")

        # A CONTROL, because the assertion above passes on a retriever that
        # returns everything. Ask about something the corpus does not contain;
        # it must not outrank the real question.
        noise = await graph.hybrid_query(
            "kubernetes ingress tls certificate rotation", max_results=5)
        if hits and len(noise) >= len(hits):
            failures.append(
                "control query matched as widely as the real one "
                "({0} vs {1}) — scoring on noise".format(len(noise), len(hits)))

    if failures:
        for f in failures:
            print("FAIL: " + f, file=sys.stderr)
        return EXIT_DEAD
    print("SELF-TEST PASS — indexed, retrieved, and the control query stayed quiet")
    return EXIT_OK


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--path", default=".", help="repository root (default: .)")
    p.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="awgraph",
        description="A code knowledge graph for agents: symbols, call paths, "
                    "and hybrid keyword+semantic search.",
    )
    ap.add_argument("--version", action="version",
                    version="awgraph " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index",
                             help="parse a repository and persist its index")
    p_index.add_argument("path", nargs="?", default=".")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="ask the index a question in English")
    p_query.add_argument("text")
    p_query.add_argument("-k", type=int, default=10, help="results to return")
    _add_common(p_query)
    p_query.set_defaults(func=cmd_query)

    p_callers = sub.add_parser("callers", help="what calls this symbol")
    p_callers.add_argument("symbol")
    _add_common(p_callers)
    p_callers.set_defaults(func=lambda a: _edges(a, "callers"))

    p_calls = sub.add_parser("calls", help="what this symbol calls")
    p_calls.add_argument("symbol")
    _add_common(p_calls)
    p_calls.set_defaults(func=lambda a: _edges(a, "calls"))

    p_stats = sub.add_parser("stats", help="what is in the index")
    _add_common(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    p_self = sub.add_parser("selftest",
                            help="prove this install indexes and retrieves")
    p_self.set_defaults(func=cmd_selftest)

    sub.add_parser("mcp", help="serve over MCP stdio for a coding agent")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "mcp":
        # Not routed through asyncio.run below: the MCP server owns its own loop
        # and blocks until the client disconnects.
        from awgraph.mcp_server import main as mcp_main

        return mcp_main()
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # a CLI must not hand a user a traceback
        return _fail(type(exc).__name__ + ": " + str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
