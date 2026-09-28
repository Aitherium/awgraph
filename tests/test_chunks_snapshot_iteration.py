"""Read paths iterate a SNAPSHOT of ``CodeGraph.chunks``, never the live dict.

Measured 2026-09-27 in aither-cognition-advanced (free-threaded 3.14t): the boot
index was still inserting chunks while ``/codegraph/search`` scored them, so
``_score_chunks_against_tokens`` raised "dictionary changed size during
iteration" after ~60 s and the request 500'd. The mutation here is simulated
deterministically: reading one chunk's ``name`` inserts another chunk, which is
exactly what a concurrent indexer does to a live iteration.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awgraph.graph import ChunkType, CodeChunk, CodeGraph  # noqa: E402


def _chunk(cid: str, name: str) -> CodeChunk:
    return CodeChunk(
        id=cid, name=name, chunk_type=ChunkType.FUNCTION,
        source_path="pkg/mod.py", start_line=1, end_line=2,
        signature="def {}()".format(name), docstring="", body_preview="",
    )


class _MutatingChunk(CodeChunk):
    """A chunk whose ``name`` read inserts a new chunk into its graph."""

    graph: CodeGraph | None = None

    def __getattribute__(self, attr):
        if attr == "name":
            g = object.__getattribute__(self, "graph")
            if g is not None and "late" not in g.chunks:
                g.chunks["late"] = _chunk("late", "late_arrival")
        return object.__getattribute__(self, attr)


def test_scoring_survives_a_concurrent_insert():
    g = CodeGraph(root_path=".", auto_index=False)
    g.chunks["a"] = _chunk("a", "alpha_search")
    m = _MutatingChunk(
        id="m", name="search_mutator", chunk_type=ChunkType.FUNCTION,
        source_path="pkg/mod.py", start_line=1, end_line=2,
        signature="def search_mutator()", docstring="", body_preview="",
    )
    m.graph = g
    g.chunks["m"] = m
    g.chunks["b"] = _chunk("b", "beta")

    results = g._score_chunks_against_tokens(["search"], None, None)

    names = {c.name for _, c in results}
    assert "alpha_search" in names and "search_mutator" in names
    assert "late" in g.chunks  # the insert really happened mid-scan
