"""awgraph asks a host-shared graph server before loading the index itself.

The point is memory: a read command answered by the shared server must NOT load the
index in this process. ``cli._open_graph`` is the one place a read command loads it,
so it is replaced with a tripwire in the shared-path tests.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

_LOOPBACK = "127.0.0.1"

from awgraph import cli, shared  # noqa: E402


class _Fake(BaseHTTPRequestHandler):
    root = ""
    script: list = []  # queued (status, body) for /query; the last one repeats
    queries: list = []

    def log_message(self, *a):  # silence
        return

    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/codegraph/info":
            self._send(200, {"root": type(self).root, "warm": True})
        else:
            self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        type(self).queries.append(json.loads(self.rfile.read(n) or b"{}"))
        cls = type(self)
        status, body = cls.script[0] if len(cls.script) == 1 else cls.script.pop(0)
        self._send(status, body)


@pytest.fixture()
def server(tmp_path, monkeypatch):
    _Fake.root = str(tmp_path)
    _Fake.queries = []
    _Fake.script = [(200, {"hits": [{
        "name": "send_request", "path": os.path.join(str(tmp_path), "pkg", "client.py"),
        "line": 7, "signature": "def send_request(url)"}]})]
    httpd = HTTPServer((_LOOPBACK, 0), _Fake)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("AWGRAPH_SHARED", "1")
    monkeypatch.setenv("AWGRAPH_SHARED_URL", f"http://127.0.0.1:{httpd.server_port}")
    yield _Fake
    httpd.shutdown()


def _tripwire(*_a, **_k):
    raise AssertionError("the index was loaded locally although the shared server answered")


def test_query_is_answered_by_the_shared_server_without_a_local_load(
        server, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_open_graph", _tripwire)
    before = shared.ANSWERS["shared"]
    rc = cli.main(["query", "retry a request", "--path", str(tmp_path), "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "send_request"
    assert rows[0]["path"] == os.path.join("pkg", "client.py")  # relative to the root
    assert shared.ANSWERS["shared"] == before + 1
    assert server.queries[0]["q"] == "retry a request"


def test_mcp_code_search_formats_shared_rows():
    from awgraph.mcp_server import format_shared_rows

    text = format_shared_rows([{"name": "f", "type": "", "path": "a.py", "line": 3,
                                "signature": "def f()"}])
    assert json.loads(text)["file"] == "a.py"
    assert "No matches" in format_shared_rows([])


def test_a_server_holding_another_root_is_not_asked(server, tmp_path):
    server.root = str(tmp_path / "elsewhere")
    assert shared.query(str(tmp_path), "x", 5) is None
    assert server.queries == []


def test_a_warming_server_is_polled_until_it_answers(server, tmp_path):
    server.script = [(503, {"error": "loading"}), (503, {"error": "loading"}),
                     (200, {"hits": []})]
    naps = []
    assert shared.query(str(tmp_path), "x", 5, sleep=naps.append) == []
    assert len(naps) == 2


def test_local_flag_and_env_off_skip_the_server(server, tmp_path, monkeypatch):
    monkeypatch.setenv("AWGRAPH_SHARED", "0")
    assert shared.query(str(tmp_path), "x", 5) is None
    assert server.queries == []


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_with_no_server_the_query_still_answers_locally(tmp_path, monkeypatch, capsys):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "client.py").write_text(cli._SELFTEST_SRC, encoding="utf-8")
    # A busy host must not turn this into an admission skip: the index is tiny.
    monkeypatch.setenv("AWGRAPH_ADMIT_COMMIT_PCT", "100")
    assert cli.main(["index", str(tmp_path)]) == 0
    monkeypatch.setenv("AWGRAPH_SHARED", "1")
    monkeypatch.setenv("AWGRAPH_SHARED_URL", f"http://127.0.0.1:{_closed_port()}")
    loads = []
    real = cli._open_graph

    async def counting(*a, **k):
        loads.append(1)
        return await real(*a, **k)

    monkeypatch.setattr(cli, "_open_graph", counting)
    capsys.readouterr()
    rc = cli.main(["query", "send_request", "--path", str(tmp_path), "--json"])
    assert rc == 0
    assert loads == [1]
    assert any(r["name"] == "send_request" for r in json.loads(capsys.readouterr().out))


_DEFER = (503, {"error": "deferred", "deferred": True,
                "reason": "deferred: host commit 94% >= 85% (graph worker load)"})


def test_a_deferred_server_answers_with_the_reason_and_never_loads_locally(
        server, tmp_path, monkeypatch, capsys):
    server.script = [_DEFER]
    monkeypatch.setattr(cli, "_open_graph", _tripwire)
    rc = cli.main(["query", "retry a request", "--path", str(tmp_path), "--json"])
    out = capsys.readouterr()
    assert rc == cli.EXIT_DEAD  # no search ran: not a "no matches" exit 1
    assert json.loads(out.out) == []
    assert "host commit 94%" in out.err and "--local" in out.err


def test_local_flag_still_loads_when_the_server_deferred(server, tmp_path, monkeypatch):
    server.script = [_DEFER]
    loads = []

    async def counting(*a, **k):
        loads.append(1)
        return None, False

    monkeypatch.setattr(cli, "_open_graph", counting)
    cli.main(["query", "x", "--path", str(tmp_path), "--local"])
    assert loads == [1] and server.queries == []


def test_a_server_still_loading_past_the_wait_is_a_deferral(server, tmp_path, monkeypatch):
    server.script = [(503, {"error": "loading"})]
    monkeypatch.setenv("AWGRAPH_SHARED_WAIT_S", "0")
    got = shared.query(str(tmp_path), "x", 5, sleep=lambda s: None)
    assert isinstance(got, shared.Deferred) and "still loading" in got.reason


def test_a_server_error_that_is_not_a_deferral_falls_back_locally(server, tmp_path):
    server.script = [(502, {"error": "query_failed"})]
    assert shared.query(str(tmp_path), "x", 5) is None


def test_mcp_code_search_says_deferred_not_empty():
    from awgraph.mcp_server import format_deferred

    text = format_deferred("deferred: host commit 94%")
    assert "deferred, not empty" in text and "host commit 94%" in text
