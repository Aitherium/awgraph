"""The embed pass must survive a graph that is being mutated underneath it.

Measured 2026-09-11 inside aither-cognition-advanced: its service re-indexes on
its own schedule (a filesystem watcher), and `embed_chunks` died with
`RuntimeError: dictionary changed size during iteration` at its first loop --
embedding a LIVE service's graph was impossible, and the error names a Python
internal rather than the concurrent writer.

The test is deterministic rather than thread-timed: the mutation is performed
from inside the apply-cached loop itself (via `_as_f32`, which that loop calls
for every cached chunk), so the old code iterating the live dict always sees
the size change and the snapshot never does.
"""
from __future__ import annotations

import asyncio
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awgraph import graph as g  # noqa: E402


def _chunk(cid: str, name: str) -> g.CodeChunk:
    return g.CodeChunk(id=cid, name=name, chunk_type=g.ChunkType.FUNCTION,
                       source_path=f"/x/{name}.py", start_line=1, end_line=2)


def _graph_with(n: int) -> g.CodeGraph:
    cg = g.CodeGraph.__new__(g.CodeGraph)  # no root on disk needed
    cg.chunks = {f"func_f{i}_deadbeef": _chunk(f"func_f{i}_deadbeef", f"f{i}")
                 for i in range(n)}
    return cg


def test_embed_pass_survives_a_writer_mutating_mid_loop(monkeypatch, tmp_path):
    cg = _graph_with(200)
    first_id = next(iter(cg.chunks))

    # A cache entry for one chunk, HMAC-valid so the pass trusts it and calls
    # _as_f32 inside the apply loop.
    cache = tmp_path / "e.pkl"
    with open(cache, "wb") as fh:
        pickle.dump({first_id: [0.5] * 8}, fh)
    g._write_pickle_hmac(str(cache))

    real_as_f32 = g._as_f32
    fired = []

    def mutating_as_f32(vec):
        if not fired:  # exactly once, from inside the iteration
            fired.append(True)
            cg.chunks["func_new_cafe"] = _chunk("func_new_cafe", "brand_new")
        return real_as_f32(vec)

    async def fake_embed(texts, model=None, is_query=False):
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(g, "_as_f32", mutating_as_f32)
    monkeypatch.setattr(g, "_embed_texts", fake_embed)
    monkeypatch.setattr(g, "_can_embed_queries", lambda: True)

    stats = asyncio.run(cg.embed_chunks(model="stub", batch_size=64,
                                        cache_path=str(cache), force=False))
    assert fired, "the mutation never landed inside the loop -- test is vacuous"
    assert stats["total"] >= 200
