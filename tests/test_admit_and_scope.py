"""Host admission, single flight, generated-file excludes and the leaner load (2026-11-08).

Measured that day: a 10.8 GB query worker and a 3.4 GB reindex of the SAME root ran at
once with the host at 93-96% commit; 6.1% of the 532,965 chunks were generated files.
"""

from __future__ import annotations

import asyncio
import os
import pickle

from awgraph import _admit
from awgraph import cli as awcli
from awgraph import graph as G  # noqa: N812

# ── admission ─────────────────────────────────────────────────────────────


def test_admit_defers_at_or_above_the_limit_and_says_why():
    why = _admit.admit("awgraph index", pct_fn=lambda: 94.0, limit=85.0)
    assert why.startswith("deferred: host commit 94%")
    assert "awgraph index" in why
    assert _admit.admit("x", pct_fn=lambda: 85.0, limit=85.0)  # at the limit = deferred


def test_admit_runs_below_the_limit_when_unreadable_or_disabled():
    assert _admit.admit("x", pct_fn=lambda: 50.0, limit=85.0) == ""
    assert _admit.admit("x", pct_fn=lambda: None, limit=85.0) == ""
    assert _admit.admit("x", pct_fn=lambda: 99.0, limit=0.0) == ""


def test_host_commit_is_measured_on_this_platform():
    pct = _admit.host_commit_pct()
    assert pct is not None and 0.0 < pct <= 100.0


def test_limit_comes_from_the_env(monkeypatch):
    monkeypatch.setenv(_admit.COMMIT_ENV, "70")
    assert _admit.commit_limit_pct() == 70.0
    monkeypatch.setenv(_admit.COMMIT_ENV, "junk")
    assert _admit.commit_limit_pct() == _admit.DEFAULT_COMMIT_PCT


# ── single flight ─────────────────────────────────────────────────────────


def test_full_load_lock_is_exclusive_and_names_its_holder(tmp_path):
    a = _admit.FullLoadLock("r", "index", cache_dir=str(tmp_path))
    b = _admit.FullLoadLock("r", "graph-worker", cache_dir=str(tmp_path))
    assert a.acquire() == ""
    busy = b.acquire()
    assert busy.startswith("busy:") and "index pid %d" % os.getpid() in busy
    assert b.held_elsewhere()
    a.release()
    assert not b.held_elsewhere()
    assert b.acquire() == ""
    b.release()


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("def alpha_beta():\n    return 1\n", encoding="utf-8")
    return str(root)


def test_index_defers_on_host_commit_and_writes_nothing(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path)
    monkeypatch.setenv(_admit.COMMIT_ENV, "0.001")  # every real host is above this
    assert awcli.main(["index", root]) == 0
    out = capsys.readouterr().out
    assert out.startswith(awcli.SKIPPED_PREFIX) and "host commit" in out
    assert not os.path.exists(G._get_data_path(root, "codegraph_chunks.pkl"))


def test_index_skips_while_another_full_load_holds_the_root(tmp_path, capsys):
    root = _repo(tmp_path)
    held = _admit.FullLoadLock(root, "graph-worker")
    assert held.acquire() == ""
    try:
        assert awcli.main(["index", root]) == 0
        out = capsys.readouterr().out
        assert out.startswith(awcli.SKIPPED_PREFIX) and "busy" in out
        assert not os.path.exists(G._get_data_path(root, "codegraph_chunks.pkl"))
    finally:
        held.release()
    assert awcli.main(["index", root]) == 0  # free again: it builds
    assert os.path.exists(G._get_data_path(root, "codegraph_chunks.pkl"))
    assert not _admit.FullLoadLock(root, "probe").held_elsewhere()  # released after


# ── no vector hydrate without a query lane ────────────────────────────────


def test_query_does_not_hydrate_vectors_without_a_query_lane(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    assert awcli.main(["index", root]) == 0
    calls = []
    monkeypatch.setattr(G, "_hydrate_embeddings", lambda cg, *a: calls.append(1) or 0)
    monkeypatch.setattr(G, "_can_embed_queries", lambda: False)
    assert awcli.main(["query", "alpha_beta", "--path", root]) in (0, 1)
    assert calls == []
    monkeypatch.setattr(G, "_can_embed_queries", lambda: True)
    assert awcli.main(["query", "alpha_beta", "--path", root]) in (0, 1)
    assert calls == [1]


# ── generated files are not indexed ───────────────────────────────────────


_GENERATED = [
    ("tokenizer.json", "{\"v\": [" + "1," * 600_000 + "1]}\n", "data file"),
    ("package-lock.json", "{}\n", "lockfile"),
    ("vendor/three.min.js", "var a=1;\n", "minified"),
    ("bundle.cjs", "x" * 6000 + "\n", "a line over"),
]
_WRITTEN = [
    ("small.json", "{\"a\": 1}\n"),
    ("app.js", "function f() {\n  return 1;\n}\n"),
    ("NOTES.md", "# t\n" + "word " * 2000 + "\n"),  # long prose lines are not minified
    ("nb.ipynb", "{" + "a" * 6000 + "}\n"),  # notebook outputs: long, not a bundle
    ("page.html", "<script>" + "x" * 6000 + "</script>\n"),
]


def test_generated_files_are_skipped():
    for name, content, expect in _GENERATED:
        why = G.generated_reason(name, content, content.split("\n"))
        assert expect in why, name


def test_written_files_are_kept():
    for name, content in _WRITTEN:
        assert G.generated_reason(name, content, content.split("\n")) == "", name


def test_a_big_data_file_yields_no_chunks_and_says_why(tmp_path):
    f = tmp_path / "tokenizer.json"
    f.write_text("{\n" + "".join("  \"t%d\": %d,\n" % (i, i) for i in range(80_000)) + "}\n",
                 encoding="utf-8")
    fg = G.parse_file_sync(str(f))
    assert fg.chunks == []
    assert any(e.startswith("skipped: data file") for e in fg.parse_errors)


# ── the load ──────────────────────────────────────────────────────────────


def test_load_interns_repeated_strings_and_round_trips(tmp_path):
    root = _repo(tmp_path)
    assert awcli.main(["index", root]) == 0
    path = G._get_data_path(root, "codegraph_chunks.pkl")
    with open(path, "rb") as fh:
        cache = pickle.load(fh)
    rows = list(cache["chunks"].values())
    proto = rows[0]
    for i in range(3):  # equal paths/calls as DISTINCT objects, as the parser makes them
        row = dict(proto, id="function_x%d" % i, name="x%d" % i,
                   source_path="".join(list(proto["source_path"])),
                   calls=["".join(list("logger.info"))])
        cache["chunks"][row["id"]] = row
    with open(path, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    G._write_pickle_hmac(path)
    graph, ok = asyncio.run(awcli._open_graph(root, build=False, hydrate=False))
    assert ok
    xs = [graph.chunks["function_x%d" % i] for i in range(3)]
    assert xs[0].source_path is xs[1].source_path is xs[2].source_path
    assert xs[0].calls[0] is xs[1].calls[0]
    assert len(graph.chunks) == len(rows) + 3
    assert graph.chunks[proto["id"]].name == proto["name"]


def test_cli_caps_blas_threads_unless_the_caller_chose(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    awcli.main(["stats", "--path", str(tmp_path)])
    assert os.environ["OPENBLAS_NUM_THREADS"] == awcli.BLAS_THREADS == "2"
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    awcli.main(["stats", "--path", str(tmp_path)])
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"


def test_a_probe_never_overwrites_the_holders_note_and_release_removes_it(tmp_path):
    a = _admit.FullLoadLock("r", "index", cache_dir=str(tmp_path))
    probe = _admit.FullLoadLock("r", "probe", cache_dir=str(tmp_path))
    assert a.acquire() == ""
    a.release()
    assert not os.path.exists(a.owner_path)  # release removes the holder's note
    assert not probe.held_elsewhere()
    assert not os.path.exists(probe.owner_path)  # the probe wrote none
    assert a.acquire() == ""
    other = _admit.FullLoadLock("r", "graph-worker", cache_dir=str(tmp_path))
    assert "index pid" in other.acquire()
    a.release()
