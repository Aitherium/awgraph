"""An index is searched in the embedding space it was built in, or not at all.

Before these tests awgraph recorded nothing about the model behind its vectors:

* a query embedded by a different model of the SAME width was scored against the
  index and ten meaningless cosines came back as an answer;
* a different width raised numpy's ``matmul`` ValueError out of ``hybrid_query``
  (and, without numpy, ``zip()`` silently scored a truncated vector);
* an incremental ``embed_chunks`` added new-model vectors beside old-model ones;
* ``AITHER_EMBED_SPACE`` -- the platform's one switch -- was never read, so the
  engine was asked for ``nomic-embed-text`` in the code-embed space;
* a lane that produced no query vector downgraded to keyword-only with nothing
  logged above debug (a detected vLLM said "can embed" while nothing could).

Each rule below has a control showing what the unguarded path does.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awgraph import cli  # noqa: E402
from awgraph import graph as g  # noqa: E402
from awgraph import space  # noqa: E402
from awgraph.graph import (  # noqa: E402
    CodeGraph,
    _hydrate_embeddings,
    _save_chunk_cache,
    _write_pickle_hmac,
)
from awgraph.space import EmbeddingSpaceMismatch  # noqa: E402

_SRC = '''
class Retry:
    """Retry a flaky call with backoff."""

    def run(self, attempt: int) -> int:
        """Run one attempt of the retry loop."""
        return attempt + 1


def backoff_delay(attempt: int) -> float:
    """Exponential backoff delay for a retry."""
    return min(2.0 ** attempt, 30.0)
'''

NOMIC = {"model": "nomic-embed-text", "dim": 4,
         "prefix_scheme": space.prefix_scheme(""), "lane": "host-engine", "space": None}


def _indexed(tmp: str) -> CodeGraph:
    (Path(tmp) / "retry.py").write_text(_SRC, encoding="utf-8")
    graph = CodeGraph(root_path=tmp, auto_index=False)
    asyncio.run(graph.index_codebase(tmp))
    assert graph.chunks, "fixture produced no chunks"
    return graph


def _give_vectors(graph: CodeGraph, dim: int = 4) -> None:
    for n, chunk in enumerate(graph.chunks.values()):
        chunk.embedding = [1.0 + n] + [0.5] * (dim - 1)
    graph._has_embeddings_cached = True


def _write_cache(path: Path, vectors: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)
    _write_pickle_hmac(str(path))


@pytest.fixture(autouse=True)
def _fresh_warning_state(monkeypatch):
    monkeypatch.setattr(g, "_WARNED_AT", {})
    monkeypatch.setattr(g, "_UNKNOWN_SPACE_WARNED", set())
    monkeypatch.setattr(g, "_detect_vllm", lambda: None)
    monkeypatch.setattr(g, "_EMBED_SPACE", None)
    monkeypatch.setattr(g, "_DEFAULT_EMBED_MODEL", "nomic-embed-text")


def _code_lane(monkeypatch, model: str, dim: int, calls: list | None = None):
    """The code-service lane, answering ``dim``-wide vectors without a network."""
    monkeypatch.setattr(g, "_HAS_EMBEDDING_ENGINE", False)
    monkeypatch.setattr(g, "_CODE_EMBED_URL", "http://lane.invalid/v1/embeddings")
    monkeypatch.setattr(g, "_CODE_EMBED_MODEL", model)

    async def _fake(texts, is_query):
        if calls is not None:
            calls.append((list(texts), is_query))
        return [[0.3] * dim for _ in texts]

    monkeypatch.setattr(g, "_embed_via_code_service", _fake)


# ── the platform space is honoured ──────────────────────────────────────────

def test_declared_space_resolves_model_and_width():
    ce = {"AITHER_EMBED_SPACE": "ce1024"}
    assert space.declared_space(ce) == space.SPACE_CODE_EMBED
    assert space.declared_space({"AITHER_EMBED_SPACE": "aither-code-embed"}) == "aither-code-embed"
    assert space.platform_model("aither-code-embed", ce) == "aither-code-embed"
    assert space.known_dim("aither-code-embed", "aither-code-embed", ce) == 1024
    # a nomic id in the code-embed space is retired config, not a choice
    assert space.resolve_model("nomic-embed-text", "aither-code-embed", ce) == "aither-code-embed"
    assert space.resolve_model("", "aither-code-embed", ce) == "aither-code-embed"
    assert space.resolve_model("my-model", "aither-code-embed", ce) == "my-model"
    assert space.known_dim("nomic-embed-text", "nomic", {"AITHER_EMBED_SPACE": "nomic"}) == 768


def test_no_env_keeps_todays_defaults():
    """Control: a standalone user with no env sees exactly the old names."""
    assert space.declared_space({}) is None
    assert space.platform_model(None, {}) == "nomic-embed-text"
    assert space.resolve_model(None, None, {}) == "nomic-embed-text"
    assert space.resolve_model("nomic-embed-text", None, {}) == "nomic-embed-text"
    # ...and awgraph does not guess a width from a model name without a declaration
    assert space.known_dim("aither-code-embed", None, {}) is None


def test_unknown_space_value_refuses_to_guess():
    with pytest.raises(ValueError, match="AITHER_EMBED_SPACE"):
        space.declared_space({"AITHER_EMBED_SPACE": "nomic-v2"})


def _import_probe(extra_env: dict) -> str:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AITHER_EMBED", "AITHER_CODEGRAPH", "AITHER_PLATFORM"))}
    env.update(extra_env)
    code = ("import awgraph.graph as g; a = g._active_space(); "
            "print(g._EMBED_SPACE, g._DEFAULT_EMBED_MODEL, repr(g._CODE_EMBED_MODEL), "
            "a['model'], a['dim'], a['lane'])")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]), timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_engine_import_follows_the_env_space():
    """The real import path: the env key alone moves the lane's model and width."""
    got = _import_probe({
        "AITHER_EMBED_SPACE": "aither-code-embed",
        "AITHER_CODEGRAPH_EMBED_URL": "http://lane.invalid/v1/embeddings",
        "AITHER_CODEGRAPH_EMBED_MODEL": "nomic-embed-text",  # stale config
    })
    assert got == ("aither-code-embed aither-code-embed 'aither-code-embed' "
                   "aither-code-embed 1024 code-service"), got


def test_engine_import_without_env_is_unchanged():
    """Control: no AITHER_EMBED_SPACE -> no space, the old default name, no lane
    conjured from nothing, and a configured model left exactly as written."""
    assert _import_probe({}) == "None nomic-embed-text '' None None none"
    got = _import_probe({
        "AITHER_CODEGRAPH_EMBED_URL": "http://lane.invalid/v1/embeddings",
        "AITHER_CODEGRAPH_EMBED_MODEL": "nomic-embed-text",
    })
    assert got == "None nomic-embed-text 'nomic-embed-text' nomic-embed-text None code-service", got


def test_host_engine_is_asked_for_the_space_model(monkeypatch):
    asked = []

    class _Engine:
        async def embed_batch(self, texts, model=None):
            asked.append(model)
            return [[0.1] * 4 for _ in texts]

    monkeypatch.setattr(g, "_code_embed_enabled", lambda: False)
    monkeypatch.setattr(g, "_HAS_EMBEDDING_ENGINE", True)
    monkeypatch.setattr(g, "get_embedding_engine", lambda: _Engine())

    asyncio.run(g._embed_texts(["x"]))                      # control: no space declared
    monkeypatch.setattr(g, "_EMBED_SPACE", "aither-code-embed")
    asyncio.run(g._embed_texts(["x"]))
    asyncio.run(g._embed_texts(["x"], model="nomic-embed-text"))
    assert asked == ["nomic-embed-text", "aither-code-embed", "aither-code-embed"], asked


# ── a search across spaces is refused ───────────────────────────────────────

def test_model_mismatch_refuses_and_names_both(monkeypatch):
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        _give_vectors(graph, 4)
        _code_lane(monkeypatch, "aither-code-embed", 4, calls)
        graph._index_space = dict(NOMIC)

        with pytest.raises(EmbeddingSpaceMismatch) as err:
            asyncio.run(graph.hybrid_query("retry backoff"))
        msg = str(err.value)
        assert "nomic-embed-text" in msg and "aither-code-embed" in msg, msg
        assert "embed_chunks(force=True)" in msg, msg
        assert calls == [], "refused only after spending an embedding request"
        with pytest.raises(EmbeddingSpaceMismatch):
            asyncio.run(graph.semantic_query("retry backoff"))
        with pytest.raises(EmbeddingSpaceMismatch):
            asyncio.run(graph._rerank_by_embedding(
                "retry backoff", list(graph.chunks.values()), top_k=2))


def test_control_unrecorded_index_same_width_is_still_searched(monkeypatch):
    """Control for the refusal above: the SAME vectors and the SAME foreign-model
    query, with no record on the index. This is the old behaviour -- a cross-space
    cosine returned as an answer -- and it is what an unrecorded index still does."""
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        _give_vectors(graph, 4)
        _code_lane(monkeypatch, "aither-code-embed", 4, calls)
        assert graph._index_space is None

        scored = asyncio.run(graph.semantic_query("retry backoff"))
        assert scored, "an unknown-space index must keep answering as it did"
        assert calls and calls[0][1] is True


@pytest.mark.parametrize("numpy_on", [True, False])
def test_width_mismatch_refuses_and_names_both(monkeypatch, numpy_on):
    if not numpy_on:
        monkeypatch.setattr(g, "_HAS_NUMPY", False)
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        _give_vectors(graph, 4)
        _code_lane(monkeypatch, "aither-code-embed", 6)

        with pytest.raises(EmbeddingSpaceMismatch) as err:
            asyncio.run(graph.hybrid_query("retry backoff"))
        msg = str(err.value)
        assert "4-d" in msg and "6-d" in msg, msg
        assert "aither-code-embed" in msg and "unrecorded model" in msg, msg
        assert "embed_chunks(force=True)" in msg, msg
        with pytest.raises(EmbeddingSpaceMismatch):
            asyncio.run(graph._rerank_by_embedding(
                "retry backoff", list(graph.chunks.values()), top_k=2))


def test_control_unguarded_width_mismatch_was_truncated_or_crashed():
    """What the scoring code does with no guard in front of it: zip() silently
    scores the first four of six components, numpy raises a bare shape error."""
    import numpy as np

    row, query = [1.0, 0.5, 0.5, 0.5], [0.3] * 6
    assert sum(a * b for a, b in zip(row, query)) > 0          # a "similarity"
    with pytest.raises(ValueError):
        np.asarray([row], dtype=np.float32) @ np.asarray(query, dtype=np.float32)


def test_declared_width_refuses_a_recorded_index(monkeypatch):
    """A record of 768-d against a declared 1024-d space is proven on paper."""
    index = dict(NOMIC, model="host-engine", dim=768)
    active = {"model": "aither-code-embed", "dim": 1024, "prefix_scheme": "x"}
    verdict, why = space.compare(index, active)
    assert verdict == space.MISMATCH and "768-d" in why and "1024-d" in why


def test_mixed_widths_in_one_index_are_refused(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        _give_vectors(graph, 4)
        # the LAST row, so the first-vector width check cannot be what catches it
        list(graph.chunks.values())[-1].embedding = [0.5] * 6
        _code_lane(monkeypatch, "aither-code-embed", 4)
        with pytest.raises(EmbeddingSpaceMismatch) as err:
            asyncio.run(graph.hybrid_query("retry backoff"))
        msg = str(err.value)
        assert "mixes embedding spaces" in msg and "4-d" in msg and "6-d" in msg, msg


def test_query_prefix_change_is_reported_not_refused():
    index = dict(NOMIC, prefix_scheme=space.prefix_scheme("Represent: "))
    active = dict(NOMIC, prefix_scheme=space.prefix_scheme("Instruct: x\nQuery: "))
    verdict, why = space.compare(index, active)
    assert verdict == space.MATCH and "prefix" in why


# ── the index records its space ─────────────────────────────────────────────

def test_embed_pass_records_the_space_and_a_reload_enforces_it(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        cache = Path(tmp) / "emb.pkl"
        _code_lane(monkeypatch, "aither-code-embed", 4)
        monkeypatch.setenv("AITHER_CODEGRAPH_QDRANT_PERSIST", "0")
        stats = asyncio.run(graph.embed_chunks(cache_path=str(cache)))
        assert stats["new"] == len(graph.chunks) and stats["model"] == "aither-code-embed"

        record = space.read_record(str(cache))
        assert record and record["model"] == "aither-code-embed" and record["dim"] == 4
        assert record["prefix_scheme"].startswith("doc:plain|query:prefix-sha256:")

        reopened = _indexed(tmp)
        assert _hydrate_embeddings(reopened, cache_path=str(cache)) == len(reopened.chunks)
        assert reopened._index_space["model"] == "aither-code-embed"
        assert asyncio.run(reopened.semantic_query("retry backoff"))     # same space: answers

        _code_lane(monkeypatch, "some-other-embedder", 4)                # same width, other model
        with pytest.raises(EmbeddingSpaceMismatch) as err:
            asyncio.run(reopened.hybrid_query("retry backoff"))
        assert "aither-code-embed" in str(err.value) and "some-other-embedder" in str(err.value)

        # ...and an incremental pass may not ADD the other model's vectors either
        with pytest.raises(EmbeddingSpaceMismatch, match="add vectors"):
            asyncio.run(reopened.embed_chunks(cache_path=str(cache)))
        # force is the documented rebuild, and it re-records
        stats = asyncio.run(reopened.embed_chunks(cache_path=str(cache), force=True))
        assert stats["new"] == len(reopened.chunks)
        assert space.read_record(str(cache))["model"] == "some-other-embedder"


def test_a_cache_rewritten_by_another_tool_reads_as_unknown(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        cache = Path(tmp) / "emb.pkl"
        _write_cache(cache, {cid: [0.5] * 4 for cid in graph.chunks})
        assert space.write_record(str(cache), NOMIC)
        assert space.read_record(str(cache))["model"] == "nomic-embed-text"
        _write_cache(cache, {cid: [0.5] * 5 for cid in graph.chunks})   # not awgraph's writer
        assert space.read_record(str(cache)) is None


def test_old_index_loads_as_unknown_with_one_warning(monkeypatch, caplog):
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "emb.pkl"
        first = _indexed(tmp)
        _write_cache(cache, {cid: [0.5, 0.1, 0.2, 0.3] for cid in first.chunks})
        _code_lane(monkeypatch, "aither-code-embed", 4)

        with caplog.at_level(logging.WARNING, logger=g.logger.name):
            assert _hydrate_embeddings(first, cache_path=str(cache)) == len(first.chunks)
            second = _indexed(tmp)
            assert _hydrate_embeddings(second, cache_path=str(cache)) == len(second.chunks)
        warned = [r for r in caplog.records if "UNKNOWN space" in r.getMessage()]
        assert len(warned) == 1, [r.getMessage() for r in caplog.records]
        assert first._index_space is None and second._index_space is None
        # behaves as it did: searched, not refused
        assert asyncio.run(second.hybrid_query("retry backoff"))
        assert g.space_status(second)["match"] == "unknown"


def test_unknown_cache_is_not_stamped_and_new_width_is_never_stored(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        ids = list(graph.chunks)
        cache = Path(tmp) / "emb.pkl"
        _write_cache(cache, {ids[0]: [0.5] * 4})                  # an old, unrecorded cache
        monkeypatch.setenv("AITHER_CODEGRAPH_QDRANT_PERSIST", "0")

        _code_lane(monkeypatch, "aither-code-embed", 6)            # the new model is wider
        with pytest.raises(EmbeddingSpaceMismatch, match="add vectors"):
            asyncio.run(graph.embed_chunks(cache_path=str(cache)))
        widths = {len(c.embedding) for c in graph.chunks.values() if c.embedding is not None}
        assert widths == {4}, f"a second width reached the index: {widths}"

        _code_lane(monkeypatch, "aither-code-embed", 4)            # same width: as before
        again = _indexed(tmp)
        stats = asyncio.run(again.embed_chunks(cache_path=str(cache)))
        assert stats["new"] == len(again.chunks) - 1
        assert space.read_record(str(cache)) is None, (
            "the active space was stamped on vectors of unknown origin")


# ── a downgrade is said out loud, and never papered over with zeros ─────────

class _DeadClient:
    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *a, **kw):
        raise OSError("connection refused")


def test_unreachable_endpoint_is_a_logged_keyword_only_downgrade(monkeypatch, caplog):
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        _give_vectors(graph, 4)
        monkeypatch.setattr(g, "_HAS_EMBEDDING_ENGINE", False)
        monkeypatch.setattr(g, "_CODE_EMBED_URL", "http://lane.invalid/v1/embeddings")
        monkeypatch.setattr(g, "_CODE_EMBED_MODEL", "aither-code-embed")
        monkeypatch.setattr(g, "AsyncClient", _DeadClient)

        assert asyncio.run(g._embed_texts(["q"], is_query=True)) == [None], (
            "an unreachable lane must yield NO vector, never a zero vector")
        with caplog.at_level(logging.WARNING, logger=g.logger.name):
            hits = asyncio.run(graph.hybrid_query("retry backoff"))
        assert hits, "keyword search still answers"
        text = caplog.text
        assert "KEYWORD-ONLY" in text and "no query vector" in text, text
        assert graph._query_embed_cache == {}, "a failed embed left a vector in the cache"


def test_detected_vllm_without_a_lane_says_nothing_was_embedded(monkeypatch, caplog):
    """`_can_embed_queries()` says yes for a detected vLLM; no lane embeds through
    one. Control: before, this returned [None] with not one line logged."""
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        _give_vectors(graph, 4)
        monkeypatch.setattr(g, "_HAS_EMBEDDING_ENGINE", False)
        monkeypatch.setattr(g, "_code_embed_enabled", lambda: False)
        monkeypatch.setattr(g, "_detect_vllm", lambda: "http://vllm.invalid:8000")
        assert g._can_embed_queries() is True

        with caplog.at_level(logging.WARNING, logger=g.logger.name):
            assert asyncio.run(g._embed_texts(["q"], is_query=True)) == [None]
            assert asyncio.run(graph.hybrid_query("retry backoff"))
        assert "no embedding lane can embed text" in caplog.text
        assert "KEYWORD-ONLY" in caplog.text


# ── `awgraph stats` answers it without reading the source ───────────────────

def test_stats_prints_index_space_against_active_space(monkeypatch, capsys):
    import argparse

    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        _save_chunk_cache(graph, tmp)
        cache = Path(tmp) / "default_emb.pkl"
        _write_cache(cache, {cid: [0.25] * 4 for cid in graph.chunks})
        assert space.write_record(str(cache), NOMIC)
        monkeypatch.setattr(g, "_embedding_cache_path", lambda: str(cache))
        args = argparse.Namespace(path=tmp, json=False)

        _code_lane(monkeypatch, "aither-code-embed", 4)
        assert asyncio.run(cli.cmd_stats(args)) == cli.EXIT_OK
        out = capsys.readouterr().out
        line = next(ln for ln in out.splitlines() if ln.startswith("embedding space:"))
        assert "index = nomic-embed-text 4-d" in line, line
        assert "active = aither-code-embed" in line and "match: NO" in line, line
        assert "REFUSED until the index is rebuilt" in out

        _code_lane(monkeypatch, "nomic-embed-text:latest", 4)     # one model, two served ids
        assert asyncio.run(cli.cmd_stats(args)) == cli.EXIT_OK
        line = next(ln for ln in capsys.readouterr().out.splitlines()
                    if ln.startswith("embedding space:"))
        assert "match: yes" in line, line

        args.json = True
        assert asyncio.run(cli.cmd_stats(args)) == cli.EXIT_OK
        import json

        status = json.loads(capsys.readouterr().out)["embedding_space"]
        assert status["match"] == "yes" and status["index"]["model"] == "nomic-embed-text"


def test_cli_query_exits_dead_on_a_space_mismatch(monkeypatch, capsys):
    import argparse

    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        _save_chunk_cache(graph, tmp)
        cache = Path(tmp) / "default_emb.pkl"
        _write_cache(cache, {cid: [0.25] * 4 for cid in graph.chunks})
        assert space.write_record(str(cache), NOMIC)
        monkeypatch.setattr(g, "_embedding_cache_path", lambda: str(cache))
        _code_lane(monkeypatch, "aither-code-embed", 4)
        args = argparse.Namespace(path=tmp, text="retry backoff", k=5, json=False,
                                  hints=None, local=True)
        assert asyncio.run(cli.cmd_query(args)) == cli.EXIT_DEAD
        err = capsys.readouterr().err
        assert "nomic-embed-text" in err and "aither-code-embed" in err, err


def test_registry_says_which_root_was_refused(monkeypatch, caplog):
    """The multi-root registry catches every per-root error at debug; a refusal
    there made the root vanish from the answer with nothing said."""
    from awgraph import registry

    monkeypatch.setattr(registry, "_REFUSAL_LOGGED_AT", {})
    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        registry._log_root_failure("repo-a", EmbeddingSpaceMismatch("refusing: nomic vs code"))
        registry._log_root_failure("repo-a", EmbeddingSpaceMismatch("refusing: nomic vs code"))
        registry._log_root_failure("repo-b", KeyError("flaky"))          # control: stays debug
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 1 and "repo-a was NOT searched" in warned[0], warned


# ── the host code index is another producer's collection ───────────────────

EMBED_DIM = 4          # read by the engine from the host CodeIndex's module (this one)
_UPSERTS: list = []


class _HostCodeIndex:
    async def upsert_chunks(self, chunks, scope=None):
        _UPSERTS.append(len(chunks))
        return len(chunks)


class _HostScope:
    def __init__(self, **kw):
        self.kw = kw


def _fold(monkeypatch, graph, caplog) -> str:
    _UPSERTS.clear()
    monkeypatch.setattr(g, "_WARNED_AT", {})
    monkeypatch.setattr(g._plugins, "code_index", lambda: (lambda: (_HostCodeIndex, _HostScope)))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=g.logger.name):
        asyncio.run(graph._persist_to_qdrant())
    return caplog.text


def test_vectors_of_another_space_are_not_folded_into_the_host_index(monkeypatch, caplog):
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed(tmp)
        _give_vectors(graph, 4)

        # control: the platform's own model at the collection's width folds as before
        _code_lane(monkeypatch, "nomic-embed-text", 4)
        assert "NOT folded" not in _fold(monkeypatch, graph, caplog) and _UPSERTS

        # same width, another model: stored as given, scored against platform queries
        _code_lane(monkeypatch, "some-other-embedder", 4)
        text = _fold(monkeypatch, graph, caplog)
        assert not _UPSERTS, "a foreign model's vectors reached the platform collection"
        assert "some-other-embedder" in text and "nomic-embed-text" in text, text

        # the declared space moves the platform model with it
        monkeypatch.setattr(g, "_EMBED_SPACE", "aither-code-embed")
        _code_lane(monkeypatch, "aither-code-embed", 4)
        assert "NOT folded" not in _fold(monkeypatch, graph, caplog) and _UPSERTS

        # another width: said at WARNING, not left to the store's debug-logged 400
        monkeypatch.setattr(sys.modules[__name__], "EMBED_DIM", 6)
        text = _fold(monkeypatch, graph, caplog)
        assert not _UPSERTS and "4-d" in text and "6-d" in text, text

        # a host engine names no model: only the width can refuse it
        monkeypatch.setattr(sys.modules[__name__], "EMBED_DIM", 4)
        monkeypatch.setattr(g, "_EMBED_SPACE", None)
        monkeypatch.setattr(g, "_CODE_EMBED_URL", "")
        monkeypatch.setattr(g, "_HAS_EMBEDDING_ENGINE", True)
        assert "NOT folded" not in _fold(monkeypatch, graph, caplog) and _UPSERTS


# ── a bridge is never a search path ─────────────────────────────────────────

def test_no_module_wires_an_embedding_bridge_into_search():
    """A registered (index space -> active space) bridge is for identity joins;
    bridged top-10 overlaps native top-10 by 37-39%. Nothing here may call one."""
    pkg = Path(g.__file__).resolve().parent
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any("bridge" in n.lower() for n in names), (path.name, names)
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("bridge_path", "apply_bridge"), path.name
