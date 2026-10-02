"""Inside ANY container the index parses in threads, never a process pool.

aither-worker aborted inside libuv (`uv__io_poll -> abort`, exit 139) three times while a
full CodeGraph index was parsing with a ProcessPoolExecutor from a worker thread of the
live uvloop service. The pool was only chosen because the container test looked for
`/.dockerenv`, which podman does not create.
"""
from __future__ import annotations

import os

from awgraph import graph


def _exists_only(*present):
    return lambda p: p in present


def test_podman_is_a_container(monkeypatch):
    monkeypatch.delenv("container", raising=False)
    monkeypatch.setattr(os.path, "exists", _exists_only("/run/.containerenv"))
    assert graph._in_container() is True


def test_docker_is_a_container(monkeypatch):
    monkeypatch.delenv("container", raising=False)
    monkeypatch.setattr(os.path, "exists", _exists_only("/.dockerenv"))
    assert graph._in_container() is True


def test_the_container_env_var_counts(monkeypatch):
    monkeypatch.setattr(os.path, "exists", _exists_only())
    monkeypatch.setenv("container", "podman")
    assert graph._in_container() is True


def test_a_bare_host_is_not_a_container(monkeypatch):
    monkeypatch.delenv("container", raising=False)
    monkeypatch.setattr(os.path, "exists", _exists_only())
    assert graph._in_container() is False


def test_the_index_asks_the_helper_and_honours_the_override():
    src = open(graph.__file__, encoding="utf-8").read()
    start = src.index("    async def index_codebase(")
    body = src[start:src.index("await asyncio.to_thread(self._backfill_called_by)")]
    assert "_in_docker = _in_container()" in body
    assert 'os.path.exists("/.dockerenv")' not in body
    assert 'os.environ.get("AWGRAPH_PARSE_POOL") in ("thread", "process")' in body
