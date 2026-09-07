"""A persisted embedding pass must be visible to every read command.

Regression for 1.4.4: ``embed_chunks()`` wrote 304,429 vectors to the cache and
``awgraph stats`` still reported 0% because nothing on the read path opened the
file. These tests pin the read path: hydration attaches persisted vectors,
refuses a tampered cache, and the CLI's ``_open_graph`` performs it.
"""

import asyncio
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awgraph.graph import (  # noqa: E402
    CodeGraph,
    _hydrate_embeddings,
    _save_chunk_cache,
    _write_pickle_hmac,
)

_SRC = '''
class Retry:
    """Retry a flaky call."""

    def run(self, attempt: int) -> int:
        """Run one attempt."""
        return attempt + 1
'''


def _indexed_root(tmp: str) -> CodeGraph:
    (Path(tmp) / "retry.py").write_text(_SRC, encoding="utf-8")
    graph = CodeGraph(root_path=tmp, auto_index=False)
    asyncio.run(graph.index_codebase(tmp))
    assert graph.chunks, "fixture produced no chunks"
    return graph


def _write_cache(path: Path, vectors: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)
    _write_pickle_hmac(str(path))


def test_hydrate_attaches_persisted_vectors_by_chunk_id():
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed_root(tmp)
        assert all(c.embedding is None for c in graph.chunks.values())
        cache = Path(tmp) / "emb.pkl"
        _write_cache(cache, {cid: [0.5] * 8 for cid in graph.chunks})
        applied = _hydrate_embeddings(graph, cache_path=str(cache))
        assert applied == len(graph.chunks)
        assert all(c.embedding is not None for c in graph.chunks.values())
        assert graph.embedding_coverage == 1.0
        # Idempotent: a second pass has nothing left to attach.
        assert _hydrate_embeddings(graph, cache_path=str(cache)) == 0


def test_hydrate_ignores_a_tampered_or_missing_cache():
    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed_root(tmp)
        missing = Path(tmp) / "nope.pkl"
        assert _hydrate_embeddings(graph, cache_path=str(missing)) == 0
        cache = Path(tmp) / "emb.pkl"
        _write_cache(cache, {cid: [0.5] * 8 for cid in graph.chunks})
        with open(cache, "ab") as f:  # bytes appended after the HMAC was written
            f.write(b"tampered")
        assert _hydrate_embeddings(graph, cache_path=str(cache)) == 0
        assert all(c.embedding is None for c in graph.chunks.values())


def test_cli_open_graph_hydrates_from_the_default_cache(monkeypatch):
    """The read path the CLI actually takes: chunk cache on disk + embedding
    cache at the package-keyed default path -> stats sees the vectors."""
    from awgraph import cli
    from awgraph import graph as graph_mod

    with tempfile.TemporaryDirectory() as tmp:
        graph = _indexed_root(tmp)
        _save_chunk_cache(graph, tmp)
        cache = Path(tmp) / "default_emb.pkl"
        _write_cache(cache, {cid: [0.25] * 8 for cid in graph.chunks})
        monkeypatch.setattr(graph_mod, "_embedding_cache_path", lambda: str(cache))

        reopened, ok = asyncio.run(cli._open_graph(tmp, build=False))
        assert ok
        assert reopened.chunks
        embedded = sum(1 for c in reopened.chunks.values() if c.embedding is not None)
        assert embedded == len(reopened.chunks), "read path did not hydrate the persisted pass"


if __name__ == "__main__":
    test_hydrate_attaches_persisted_vectors_by_chunk_id()
    test_hydrate_ignores_a_tampered_or_missing_cache()
    print("ok (run under pytest for the monkeypatch case)")
