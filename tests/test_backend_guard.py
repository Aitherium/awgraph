"""The query-embedding guard must know every lane, not just the first two.

Measured 2026-09-10: a 402k-chunk index fully embedded by the distilled student,
`AITHER_CODEGRAPH_EMBED_URL` set, and `awgraph query` still answered
keyword-only — `semantic_query()` returned [] on a guard that only knew the
EmbeddingEngine and a local vLLM. Nothing said so: the CLI returned ten
plausible keyword hits, which is exactly how a silent degradation looks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awgraph import graph  # noqa: E402


def test_guard_accepts_the_code_service_lane(monkeypatch):
    monkeypatch.setattr(graph, "_HAS_EMBEDDING_ENGINE", False)
    monkeypatch.setattr(graph, "_detect_vllm", lambda: None)
    monkeypatch.setattr(graph, "_code_embed_enabled", lambda: True)
    assert graph._can_embed_queries() is True, (
        "the code-specialized service was configured and the guard still said "
        "no backend — every query degrades to keyword-only, silently"
    )


def test_guard_reports_false_with_no_lane_at_all(monkeypatch):
    monkeypatch.setattr(graph, "_HAS_EMBEDDING_ENGINE", False)
    monkeypatch.setattr(graph, "_detect_vllm", lambda: None)
    monkeypatch.setattr(graph, "_code_embed_enabled", lambda: False)
    assert graph._can_embed_queries() is False


def test_guard_still_accepts_the_engine_and_vllm_lanes(monkeypatch):
    monkeypatch.setattr(graph, "_HAS_EMBEDDING_ENGINE", True)
    monkeypatch.setattr(graph, "_detect_vllm", lambda: None)
    monkeypatch.setattr(graph, "_code_embed_enabled", lambda: False)
    assert graph._can_embed_queries() is True

    monkeypatch.setattr(graph, "_HAS_EMBEDDING_ENGINE", False)
    monkeypatch.setattr(graph, "_detect_vllm", lambda: "http://vllm:8000")
    assert graph._can_embed_queries() is True


def test_keyword_only_fallback_warns_once_embeddings_exist(monkeypatch, caplog):
    """Embedded chunks + no query lane must SAY SO, not silently answer.

    Measured 2026-09-10: a session ran the whole ground-truth set with
    AITHER_CODEGRAPH_EMBED_URL unset -- ten plausible keyword hits per query and
    no signal anywhere that the 402k-chunk index was never consulted.
    """
    import asyncio
    import logging

    monkeypatch.setattr(graph, "_can_embed_queries", lambda: False)

    class _Chunk:
        embedding = None

    class _CG:
        _has_embeddings_cached = True
        chunks = {"a": _Chunk()}

        def classify_query(self, q):
            return 0.5, 0.5, "balanced"

        async def query(self, q, **kw):
            return []

    with caplog.at_level(logging.WARNING, logger=graph.logger.name):
        asyncio.run(graph.CodeGraph.hybrid_query(_CG(), "any question"))
    assert "KEYWORD-ONLY" in caplog.text, caplog.text
