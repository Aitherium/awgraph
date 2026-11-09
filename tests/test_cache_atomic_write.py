"""The chunk cache is written beside and swapped in, never rewritten in place.

Measured 2026-11-08: a query worker opened the cache while `awgraph index` was
writing it and logged "Chunk cache corrupted: Ran out of input".
"""

from __future__ import annotations

import asyncio
import os

from awgraph import cli as awcli
from awgraph import graph as G  # noqa: N812


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    return str(root)


def test_a_failed_save_leaves_the_previous_cache_whole(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    assert awcli.main(["index", root]) == 0
    path = G._get_data_path(root, "codegraph_chunks.pkl")
    before = open(path, "rb").read()
    graph, ok = asyncio.run(awcli._open_graph(root, build=False, hydrate=False))
    assert ok

    def _boom(obj, fh, protocol=None):
        fh.write(b"\x80\x05partial")  # a writer that dies half way
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(G.pickle, "dump", _boom)
        G._save_chunk_cache(graph, root)  # logs and returns; never raises
    assert open(path, "rb").read() == before  # the old cache is untouched
    assert G._verify_pickle_hmac(path)
    assert not [f for f in os.listdir(os.path.dirname(path)) if ".tmp" in f]  # cleaned up
    again, ok = asyncio.run(awcli._open_graph(root, build=False, hydrate=False))
    assert ok and set(again.chunks) == set(graph.chunks)


def test_a_save_leaves_no_temp_files_and_a_valid_hmac(tmp_path):
    root = _repo(tmp_path)
    assert awcli.main(["index", root]) == 0
    path = G._get_data_path(root, "codegraph_chunks.pkl")
    assert G._verify_pickle_hmac(path)
    leftovers = [f for f in os.listdir(os.path.dirname(path)) if f.endswith((".tmp", ".tmp.hmac"))]
    assert leftovers == []


def test_a_reader_during_the_write_sees_the_old_cache_whole(tmp_path, monkeypatch):
    """Simulated reader: it opens the cache while the writer is mid-dump."""
    root = _repo(tmp_path)
    assert awcli.main(["index", root]) == 0
    path = G._get_data_path(root, "codegraph_chunks.pkl")
    graph, ok = asyncio.run(awcli._open_graph(root, build=False, hydrate=False))
    seen = {}
    real_dump = G.pickle.dump

    def _dump_with_a_reader(obj, fh, protocol=None):
        fh.write(b"\x80\x05")  # the writer is half way through
        fh.flush()
        reader = G.CodeGraph(root_path=root, auto_index=False)
        seen["ok"] = G._load_chunk_cache(reader, root)
        seen["n"] = len(reader.chunks)
        fh.seek(0)
        fh.truncate()
        real_dump(obj, fh, protocol=protocol)

    with monkeypatch.context() as m:
        m.setattr(G.pickle, "dump", _dump_with_a_reader)
        G._save_chunk_cache(graph, root)
    assert seen == {"ok": True, "n": len(graph.chunks)}  # never a truncated file
    assert G._verify_pickle_hmac(path)
