"""The process guard ends an awgraph process whose caller is gone, that ran too long, or
that holds too much memory; it never ends a healthy one."""

import os
import subprocess
import sys
import time

from awgraph import _guard as G  # noqa: N812


def _r(**kw):
    base = dict(ppid=123, started=0.0, now=10.0, max_s=900.0, max_gb=16.0,
                alive=lambda pid: True, mem=lambda: 1.0)
    base.update(kw)
    return G.reason(**base)


def test_healthy_process_keeps_running():
    assert _r() == ""


def test_dead_parent_ends_it():
    assert "parent" in _r(alive=lambda pid: False)


def test_wall_clock_limit_ends_it_and_zero_disables():
    assert "AWGRAPH_MAX_SECONDS" in _r(now=901.0)
    assert _r(now=10_000.0, max_s=0.0) == ""


def test_memory_limit_ends_it_unreadable_memory_does_not():
    assert "AWGRAPH_MAX_RSS_GB" in _r(mem=lambda: 17.0)
    assert _r(mem=lambda: None) == ""
    assert _r(mem=lambda: 99.0, max_gb=0.0) == ""


def test_long_lived_commands_have_no_wall_clock_limit(monkeypatch):
    seen = {}
    fake = type("T", (), {"start": lambda self: None})()
    monkeypatch.setattr(G.threading, "Thread",
                        lambda target, name, daemon: seen.setdefault("t", fake))
    monkeypatch.setenv("AWGRAPH_MAX_SECONDS", "5")
    assert G.install("index") is not None
    monkeypatch.setenv("AWGRAPH_NO_GUARD", "1")
    assert G.install("stats") is None


def test_this_process_reads_its_own_memory_and_parent():
    gb = G.private_gb()
    assert gb is not None and 0 < gb < 64
    assert G.parent_alive(os.getppid())
    assert not G.parent_alive(0)


def test_real_orphan_is_ended_when_its_parent_dies(tmp_path):
    """Fail direction on a real process tree: a guarded child of a parent that exits."""
    marker = tmp_path / "child.pid"
    child_code = (
        "import os, time, sys; sys.path.insert(0, %r);"
        "from awgraph import _guard; _guard.install('stats', interval=0.2);"
        "open(%r, 'w').write(str(os.getpid())); time.sleep(60)"
    ) % (os.path.dirname(os.path.dirname(G.__file__)), str(marker))
    parent_code = (
        "import subprocess, sys; subprocess.Popen([sys.executable, '-c', %r]); "
        "import time; time.sleep(1.5)"
    ) % child_code
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run([sys.executable, "-c", parent_code], timeout=30, creationflags=flags)
    deadline = time.time() + 15
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.1)
    pid = int(marker.read_text())
    while time.time() < deadline and G.parent_alive(pid):
        time.sleep(0.2)
    assert not G.parent_alive(pid), "orphaned guarded child survived its parent"


def test_guard_ends_the_process_when_stderr_is_a_full_pipe_nobody_reads():
    """Fail direction: the caller is gone and its stderr pipe is full and never read.

    The guard used to report with ``print(..., flush=True)``, which blocked on that pipe (and
    on ``sys.stderr``'s lock, held by the main thread's own blocked write), so ``os._exit``
    never ran: a query held 9.7 GB for 45 minutes past every limit. The exit never waits on
    the message.
    """
    child_code = (
        "import os, sys, threading, time; sys.path.insert(0, %r);"
        "os.environ['AWGRAPH_MAX_SECONDS'] = '0.5';"
        "from awgraph import _guard;"
        "t = threading.Thread(target=lambda: sys.stderr.write('x' * 400000), daemon=True);"
        "t.start(); time.sleep(0.3);"
        "_guard.install('query', interval=0.2); time.sleep(30)"
    ) % os.path.dirname(os.path.dirname(G.__file__))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # stderr piped and NEVER read: the 400 KB write fills the pipe and blocks
    p = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, creationflags=flags)
    try:
        deadline = time.time() + 15
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        assert p.poll() == G.EXIT_DEAD, "guard did not end a process blocked on stderr"
    finally:
        if p.poll() is None:
            p.kill()
