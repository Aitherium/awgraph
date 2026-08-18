"""The MCP surface must actually answer, not merely construct.

A server that starts, lists tools, and returns nothing useful is indistinguishable
to a coding agent from a repository with nothing in it — the same silent-empty
failure the rest of this package is built to avoid.

These drive the server's OWN `list_tools()` / `call_tool()` entry points, which is
the path a client takes. An earlier version reached into `server.request_handlers`
to find handlers by request type; that coupled the tests to one SDK generation and
passed against a locally-installed `mcp` 1.x while the shipped module raised
`AttributeError` on every clean install, where pip resolved 2.0. The dependency is
pinned to 2.x now, so the tested version is the shipped version.
"""

from __future__ import annotations

import asyncio

import pytest

mcp_server = pytest.importorskip(
    "awgraph.mcp_server", reason="awgraph[mcp] not installed")
pytest.importorskip("mcp", reason="awgraph[mcp] not installed")

SRC = '''
class BackoffPolicy:
    """Backoff policy for flaky calls."""

    def next_delay(self, attempt: int) -> float:
        return min(2.0 ** attempt, 30.0)


def send_request(url: str) -> str:
    """Send one request, retrying on failure."""
    return _do_send(url)


def _do_send(url: str) -> str:
    return url
'''


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("AWGRAPH_CACHE_DIR", str(tmp_path / ".cache"))
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "client.py").write_text(SRC, encoding="utf-8")
    return str(tmp_path)


def _text(result) -> str:
    """Flatten a CallToolResult to text, whatever shape it carries."""
    content = getattr(result, "content", None)
    if content is None:
        return str(result)
    return "\n".join(getattr(c, "text", "") or "" for c in content)


def _call(name: str, **args) -> str:
    server = mcp_server.build_server()
    return _text(asyncio.run(server.call_tool(name, args)))


def test_server_declares_the_documented_tools():
    """The tool NAMES are the contract users paste into a client config."""
    server = mcp_server.build_server()
    # await: list_tools is a coroutine despite annotating `-> list[MCPTool]`.
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"code_index", "code_search", "code_callers", "code_calls",
            "code_stats"} <= names, f"missing tools: {sorted(names)}"


def test_search_before_index_says_so_rather_than_returning_empty(repo):
    """The distinction an agent cannot otherwise make.

    "No index yet" and "no matches" have different fixes. Returning empty for the
    first teaches the agent the repository does not contain what it is looking
    for — the wrong conclusion, and an expensive one.
    """
    out = _call("code_search", query="retry", path=repo)
    assert "No index" in out, out[:300]
    assert "code_index" in out, "the message must name the tool that fixes it"


def test_index_then_search_returns_real_symbols(repo):
    indexed = _call("code_index", path=repo)
    assert "Indexed" in indexed and "0 chunks" not in indexed, indexed[:300]

    found = _call("code_search", query="retry a request when it fails",
                  path=repo, limit=5)
    assert "send_request" in found, found[:400]
    assert "client.py" in found, "results must carry a file location"


def test_calls_resolves_edges(repo):
    _call("code_index", path=repo)
    out = _call("code_calls", symbol="send_request", path=repo)
    assert "_do_send" in out, out[:400]


def test_unknown_symbol_is_reported_not_silently_empty(repo):
    """The control: a real negative answer must be legible as one."""
    _call("code_index", path=repo)
    out = _call("code_callers", symbol="nonexistent_symbol_xyz", path=repo)
    assert "No symbol" in out, out[:300]


def test_stats_reports_embedding_coverage(repo):
    """0% coverage must be stated: keyword-only degradation is silent."""
    _call("code_index", path=repo)
    out = _call("code_stats", path=repo)
    assert "embedding_coverage" in out, out[:300]
    assert "keyword-only" in out, (
        "with no embeddings configured, stats must SAY search is keyword-only")


def test_index_of_a_tree_with_no_python_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("AWGRAPH_CACHE_DIR", str(tmp_path / ".cache"))
    (tmp_path / "README.md").write_text("no python here", encoding="utf-8")
    out = _call("code_index", path=str(tmp_path))
    assert "0 chunks" in out and "Python only" in out, out[:300]


def test_missing_mcp_package_message_names_the_fix():
    """The install hint is the whole error story for a bare install."""
    assert "awgraph[mcp]" in mcp_server._INSTALL_HINT
    assert "pip install" in mcp_server._INSTALL_HINT
