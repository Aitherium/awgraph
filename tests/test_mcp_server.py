"""The MCP surface must actually answer, not merely construct.

A server that starts, lists tools, and returns nothing useful is indistinguishable
to a coding agent from a repository with nothing in it — the same silent-empty
failure the rest of this package is built to avoid.
"""

from __future__ import annotations

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


def _handlers():
    """The registered list/call handlers, whatever the SDK names them."""
    server = mcp_server.build_server()
    return server


def test_server_declares_the_documented_tools():
    server = _handlers()
    assert server is not None
    # The tool NAMES are the contract a user pastes into their client config and
    # an agent calls by name; asserting them keeps a rename from silently
    # breaking every existing config.
    import asyncio

    names = asyncio.run(_list_tool_names(server))
    assert {"code_index", "code_search", "code_callers", "code_calls",
            "code_stats"} <= names, f"missing tools: {names}"


async def _list_tool_names(server) -> set:
    from mcp.types import ListToolsRequest

    for req_type, handler in server.request_handlers.items():
        if req_type is ListToolsRequest:
            result = await handler(ListToolsRequest(method="tools/list"))
            tools = result.root.tools if hasattr(result, "root") else result.tools
            return {t.name for t in tools}
    raise AssertionError("the server registered no tools/list handler")


async def _call(server, name: str, args: dict) -> str:
    from mcp.types import CallToolRequest, CallToolRequestParams

    for req_type, handler in server.request_handlers.items():
        if req_type is CallToolRequest:
            result = await handler(CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name=name, arguments=args)))
            content = (result.root.content if hasattr(result, "root")
                       else result.content)
            return "\n".join(getattr(c, "text", "") for c in content)
    raise AssertionError("the server registered no tools/call handler")


def test_search_before_index_says_so_rather_than_returning_empty(repo):
    """The distinction an agent cannot otherwise make.

    "No index yet" and "no matches" have different fixes. Returning empty for the
    first teaches the agent the repository does not contain what it is looking
    for, which is the wrong conclusion and an expensive one.
    """
    import asyncio

    server = _handlers()
    out = asyncio.run(_call(server, "code_search",
                            {"query": "retry", "path": repo}))
    assert "No index" in out, out[:200]
    assert "code_index" in out, "the message must name the tool that fixes it"


def test_index_then_search_returns_real_symbols(repo):
    import asyncio

    server = _handlers()
    indexed = asyncio.run(_call(server, "code_index", {"path": repo}))
    assert "Indexed" in indexed and "0 chunks" not in indexed, indexed[:200]

    found = asyncio.run(_call(server, "code_search",
                              {"query": "retry a request when it fails",
                               "path": repo, "limit": 5}))
    assert "send_request" in found, found[:300]
    assert "client.py" in found, "results must carry a file location"


def test_callers_resolves_edges(repo):
    import asyncio

    server = _handlers()
    asyncio.run(_call(server, "code_index", {"path": repo}))
    out = asyncio.run(_call(server, "code_calls",
                            {"symbol": "send_request", "path": repo}))
    assert "_do_send" in out, out[:300]


def test_unknown_symbol_is_reported_not_silently_empty(repo):
    """The control: a real negative answer must be legible as one."""
    import asyncio

    server = _handlers()
    asyncio.run(_call(server, "code_index", {"path": repo}))
    out = asyncio.run(_call(server, "code_callers",
                            {"symbol": "nonexistent_symbol_xyz", "path": repo}))
    assert "No symbol" in out, out[:200]


def test_stats_reports_embedding_coverage(repo):
    """0% coverage must be stated, because keyword-only degradation is silent."""
    import asyncio

    server = _handlers()
    asyncio.run(_call(server, "code_index", {"path": repo}))
    out = asyncio.run(_call(server, "code_stats", {"path": repo}))
    assert "embedding_coverage" in out, out[:200]
    assert "keyword-only" in out, (
        "with no embeddings configured, stats must SAY search is keyword-only")


def test_index_of_a_tree_with_no_python_says_so(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setenv("AWGRAPH_CACHE_DIR", str(tmp_path / ".cache"))
    (tmp_path / "README.md").write_text("no python here", encoding="utf-8")
    server = _handlers()
    out = asyncio.run(_call(server, "code_index", {"path": str(tmp_path)}))
    assert "0 chunks" in out and "Python only" in out, out[:200]


def test_missing_mcp_package_message_names_the_fix():
    """The install hint is the whole error-handling story for a bare install."""
    assert "awgraph[mcp]" in mcp_server._INSTALL_HINT
    assert "pip install" in mcp_server._INSTALL_HINT
