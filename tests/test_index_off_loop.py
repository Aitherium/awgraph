"""index_codebase must not run its O(N·edges) phases on the event loop.

Measured 2026-09-29 on genesis (standard CPython 3.14, ~28K chunks): the full
index ran `_backfill_called_by` synchronously inside the coroutine, held the
loop for >240 s, gunicorn's heartbeat missed, and the worker was SIGABRTed with
the stack `_bg_index -> index_codebase -> _backfill_called_by`. The incremental
path already offloaded it; the full path did not.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awgraph.graph import CodeGraph  # noqa: E402


def test_full_index_backfill_and_centrality_run_off_the_loop(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def f():\n    return g()\n\ndef g():\n    return 1\n",
                                   encoding="utf-8")
    cg = CodeGraph()
    seen: dict[str, int] = {}
    orig_backfill, orig_centrality = cg._backfill_called_by, cg._compute_centrality

    def backfill():
        seen["backfill"] = threading.get_ident()
        return orig_backfill()

    def centrality():
        seen["centrality"] = threading.get_ident()
        return orig_centrality()

    monkeypatch.setattr(cg, "_backfill_called_by", backfill)
    monkeypatch.setattr(cg, "_compute_centrality", centrality)

    async def run():
        loop_thread = threading.get_ident()
        await cg.index_codebase(str(tmp_path))
        return loop_thread

    loop_thread = asyncio.run(run())
    assert "backfill" in seen and "centrality" in seen, seen
    assert seen["backfill"] != loop_thread, "backfill ran ON the event loop thread"
    assert seen["centrality"] != loop_thread, "centrality ran ON the event loop thread"
    # and the work still happened: g is called by f
    g = [c for c in cg.chunks.values() if c.name == "g"]
    assert g and any("f" in n for n in g[0].called_by), [c.called_by for c in g]
