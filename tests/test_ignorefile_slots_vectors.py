"""`.awgraphignore`, slotted chunks, mmap vectors and the host-budget gate (2026-11-08).

Measured on the host repo that day: 35,545 chunks under scratch dirs and the eight
language mirrors of one API reference; ~1.5 KB of instance dict per chunk; a
2.6 GB pickle of boxed floats that cost ~9 GB to hydrate.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import sys
from pathlib import Path

from awgraph import _admit, ignorefile
from awgraph import graph as G  # noqa: N812

REPO = Path(__file__).resolve().parents[4]


# -- .awgraphignore -----------------------------------------------------------


def test_ignore_rules_anchor_dirs_negate_and_globstar():
    rules = ignorefile.parse([
        "# comment", "", "/*tmp*/", ".lane-backups/", "docs/**/gen/", "*.snap",
        "!keep.snap",
    ])
    assert ignorefile.ignored(".tmp_hosted/a.py", rules)
    assert ignorefile.ignored("tmp_kaggle_adapter/tokenizer.json", rules)
    assert not ignorefile.ignored("AitherOS/tmp_tool/a.py", rules)  # anchored at root
    assert not ignorefile.ignored(".tmp_vision_test.png", rules)  # a file, not a dir
    assert ignorefile.ignored("a/b/.lane-backups/x.py", rules)  # unanchored: any depth
    assert ignorefile.ignored("docs/x/y/gen/z.md", rules)
    assert ignorefile.ignored("src/a.snap", rules)
    assert not ignorefile.ignored("src/keep.snap", rules)  # the last match wins


def test_discovery_honours_the_roots_awgraphignore(tmp_path):
    for rel in ("src/a.py", ".tmp_x/b.py", "src/.lane-backups/c.py"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / ".awgraphignore").write_text("/*tmp*/\n.lane-backups/\n", encoding="utf-8")
    files, _ = asyncio.run(G.discover_files(tmp_path, extensions={".py"}))
    rels = sorted(os.path.relpath(str(f), str(tmp_path)).replace("\\", "/") for f in files)
    assert rels == ["src/a.py"]


def test_the_repos_own_excludes_are_pinned():
    rules = ignorefile.load(REPO)
    assert rules, "the repository root must carry a .awgraphignore"
    for rel in (".tmp_hosted/x.py", ".audit-tmp/y.json", "tmp_kaggle_adapter/tokenizer.json",
                ".tmp_stage2/z.ts", ".lane-backups/a.py",
                ".RESEARCH/.CLAUDE-MANAGED-AGENTS/api/python/beta.md",
                ".RESEARCH/.CLAUDE-MANAGED-AGENTS/api/typescript/messages.md"):
        assert ignorefile.ignored(rel, rules), rel
    for rel in (".RESEARCH/.CLAUDE-MANAGED-AGENTS/api/beta.md", "AitherOS/lib/core/x.py",
                "awdk/adk/graph_worker.py", "AitherOS/dev/tools/tmp_cleanup.py"):
        assert not ignorefile.ignored(rel, rules), rel


# -- slotted chunks -----------------------------------------------------------


def _chunk(**kw):
    base = dict(id="function_x", name="x", chunk_type=G.ChunkType.FUNCTION,
                source_path="m.py", start_line=1, end_line=2)
    base.update(kw)
    return G.CodeChunk(**base)


def test_a_chunk_keeps_its_fields_in_slots_not_an_instance_dict():
    c = _chunk(calls=["a"], embedding=[0.1, 0.2])
    assert "source_path" in G.CodeChunk.__slots__
    assert vars(c) == {}  # every field and the vector live in slots
    assert list(c.embedding) == [0.1, 0.2] and c.calls == ["a"]
    c.extra_note = "hosts may still hang attributes"  # the lazy __dict__ takes it
    assert vars(c) == {"extra_note": "hosts may still hang attributes"}


def test_a_slotted_chunk_pickles_with_its_vector_and_matrix_binding_works():
    np = G.np
    c = _chunk(embedding=G._as_f32([1.0, 2.0]))
    back = pickle.loads(pickle.dumps(c))  # the parse pool ships chunks between processes
    assert back.name == "x" and list(back.embedding) == [1.0, 2.0]
    mat = np.array([[3.0, 4.0]], dtype=np.float32)
    c._bind_matrix_row(mat, 0)
    assert list(c.embedding) == [3.0, 4.0]
    c.embedding = None
    assert c.embedding is None  # an assignment supersedes the matrix binding


# -- mmap vectors -------------------------------------------------------------


def _graph_with(ids):
    cg = G.CodeGraph(root_path=".", auto_index=False)
    for cid in ids:
        cg.chunks[cid] = _chunk(id=cid, name=cid)
    return cg


def _write_pickle(path, vecs):
    with open(path, "wb") as fh:
        pickle.dump(vecs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    G._write_pickle_hmac(path)


def test_hydrate_writes_an_mmap_sidecar_once_then_never_reads_the_pickle(tmp_path,
                                                                         monkeypatch):
    path = str(tmp_path / "codegraph_embeddings.pkl")
    _write_pickle(path, {"a": [0.5] * 8, "b": [0.25] * 8})
    assert G._hydrate_embeddings(_graph_with(["a", "b"]), path) == 2  # old format read
    assert os.path.exists(path + ".ids.json")

    def _no_pickle(*a, **kw):
        raise AssertionError("the boxed-float pickle was read again")

    monkeypatch.setattr(G.pickle, "load", _no_pickle)
    np = G.np
    cg = _graph_with(["a", "b", "c"])
    assert G._hydrate_embeddings(cg, path) == 2
    vec = cg.chunks["a"].embedding
    assert isinstance(vec, np.memmap) and float(vec[0]) == 0.5
    assert cg.chunks["c"].embedding is None


def test_a_stale_sidecar_is_ignored(tmp_path):
    path = str(tmp_path / "codegraph_embeddings.pkl")
    _write_pickle(path, {"a": [0.5] * 8})
    assert G._write_vector_sidecar(path, {"a": [0.5] * 8})
    _write_pickle(path, {"a": [0.75] * 8, "b": [0.1] * 8})  # rewritten by an older writer
    os.utime(path, (1, 1))
    assert G._load_vector_sidecar(path) is None
    cg = _graph_with(["a", "b"])
    assert G._hydrate_embeddings(cg, path) == 2
    assert float(cg.chunks["a"].embedding[0]) == 0.75  # from the pickle, then re-written
    assert G._load_vector_sidecar(path) is not None


# -- one host admission ledger ------------------------------------------------


class _Holder:
    def __init__(self):
        self.released = 0

    def start(self, *a, **kw):
        return self

    def release(self):
        self.released += 1


class _Budget:
    def __init__(self, result=None, boom=False):
        self.calls, self.result, self.boom = [], result, boom

    def acquire(self, gb, label, wait_s=0):
        self.calls.append((gb, label, wait_s))
        if self.boom:
            raise RuntimeError("gate broken")
        return self.result


def test_the_host_budget_gate_decides_when_present():
    h = _Holder()
    gate = _Budget((h, "admitted"))
    why, holder = _admit.admit_job("awgraph index", 4.0, "awgraph-index",
                                   pct_fn=lambda: 99.0, budget=gate)
    assert why == "" and holder is h  # the gate admitted: no second 85% check
    assert gate.calls == [(4.0, "awgraph-index", 0)]
    why, holder = _admit.admit_job("awgraph index", 4.0, "awgraph-index",
                                   pct_fn=lambda: 10.0, budget=_Budget((None, "queue busy")))
    assert holder is None and why.startswith("deferred: host budget: queue busy")


def test_without_a_gate_or_with_a_broken_one_the_commit_check_decides(monkeypatch):
    monkeypatch.setenv(_admit.COMMIT_ENV, "85")
    why, holder = _admit.admit_job("x", 4.0, "x", pct_fn=lambda: 99.0, budget=None)
    assert holder is None and why.startswith("deferred: host commit 99%")
    why, holder = _admit.admit_job("x", 4.0, "x", pct_fn=lambda: 10.0,
                                   budget=_Budget(boom=True))
    assert why == "" and holder is not None
    holder.release()


def test_the_gate_module_is_found_by_env_path(tmp_path, monkeypatch):
    mod = tmp_path / "host_budget.py"
    mod.write_text("def acquire(gb, label, wait_s=0):\n    return None, 'fake'\n",
                   encoding="utf-8")
    monkeypatch.setenv(_admit.HOST_BUDGET_ENV, str(mod))
    monkeypatch.delitem(sys.modules, "aither_host_budget", raising=False)
    _admit._HOST_BUDGET.clear()
    try:
        why, holder = _admit.admit_job("x", 1.0, "x", pct_fn=lambda: 1.0)
        assert holder is None and "fake" in why
    finally:
        _admit._HOST_BUDGET.clear()
        sys.modules.pop("aither_host_budget", None)
