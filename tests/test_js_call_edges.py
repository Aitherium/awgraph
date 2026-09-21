"""`callers` must answer for the JavaScript/TypeScript family, not only Python.

Measured on a 387,098-chunk index whose embeddings were 100% complete:

    awgraph callers spaceUrl           -> "no matches", exit 1, 30-48 s
    rg -n "\\bspaceUrl\\b" --type ts   -> 64 lines, 1.2 s

The symbol was IN the index (two chunks, a .ts definition and a .tsx use) and
carried zero edges, because the non-Python parse path deliberately built none.
So the one question the tool exists to answer returned a confident negative on
an index that held the answer.

These tests pin the extractor's contract at the level a wrong answer would be
worst: a definition must never be reported as its own caller, and a name
mentioned in a comment or a string must never become an edge.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awgraph import cli, jscalls, multilang  # noqa: E402


def _run(argv):
    args = cli.build_parser().parse_args(argv)
    return asyncio.run(args.func(args))


class _Chunk:
    """The subset of CodeChunk the edge derivation reads."""

    def __init__(self, cid, name, path, body):
        self.id = cid
        self.name = name
        self.source_path = path
        self.body_preview = body
        self.calls = []
        self.called_by = []
        self.chunk_type = None
        self.start_line = 1
        self.signature = ""


# ── the extractor ────────────────────────────────────────────────────────


def test_a_definition_is_not_its_own_caller():
    """The failure mode that would make the feature worse than absent."""
    src = "export function spaceUrl(handle: string): string {\n  return handle;\n}\n"
    assert jscalls.extract_calls(src, "spaceUrl") == []
    assert jscalls.extract_calls(src, None) == [], (
        "`function spaceUrl(` was read as a CALL to spaceUrl, so every "
        "definition would list itself among its own callers"
    )


def test_a_real_call_is_found():
    src = 'import { spaceUrl } from "./u";\nconst href = spaceUrl(handle, tab);\n'
    assert "spaceUrl" in jscalls.extract_calls(src, "ShareSheet")


def test_comments_and_strings_are_not_calls():
    src = (
        "function useIt() {\n"
        "  // spaceUrl(handle) -- described, not called\n"
        "  /* alsoNot(x) */\n"
        '  const msg = "neitherThis(y)";\n'
        "  const t = `norThat(z)`;\n"
        "  return realCall(1);\n"
        "}\n"
    )
    found = jscalls.extract_calls(src, "useIt")
    assert found == ["realCall"], found


def test_template_substitutions_are_code():
    """`${ ... }` holds real calls; the literal text around it does not."""
    src = "function f() {\n  return `a/${slugOf(x)}/b ${'notACall(1)'}`;\n}\n"
    found = jscalls.extract_calls(src, "f")
    assert found == ["slugOf"], found


def test_control_flow_keywords_are_not_calls():
    src = (
        "function g(xs) {\n"
        "  if (xs) { for (const x of xs) { switch (x) { default: break; } } }\n"
        "  try { doWork(); } catch (e) { return await later(); }\n"
        "}\n"
    )
    found = jscalls.extract_calls(src, "g")
    assert found == ["doWork", "later"], found


def test_member_calls_resolve_to_the_method_name():
    src = "function h(c) {\n  return c.client.sendRequest(1);\n}\n"
    assert jscalls.extract_calls(src, "h") == ["sendRequest"]


def test_construction_is_an_edge():
    src = "function mk() {\n  return new Renderer();\n}\n"
    assert jscalls.extract_calls(src, "mk") == ["Renderer"]


def test_regex_literal_is_not_scanned():
    src = "function r(s) {\n  return /notACall\\(x\\)/.test(s);\n}\n"
    found = jscalls.extract_calls(src, "r")
    assert "notACall" not in found, found
    assert "test" in found


def test_minified_bodies_contribute_no_edges():
    """A bundle's one-letter names would bury every real answer."""
    src = "function z(){" + "a(1);b(2);" * 200 + "}"
    assert len(max(src.split("\n"), key=len)) > jscalls.MAX_LINE_FOR_EDGES
    assert jscalls.extract_calls(src, "z") == []


def test_min_js_files_are_not_js_sources():
    assert jscalls.is_js_source("src/lib/spaces-url.ts")
    assert jscalls.is_js_source("src/components/Share.tsx")
    assert not jscalls.is_js_source("vendor/model-viewer.min.js")
    assert not jscalls.is_js_source("README.md")


# ── resolution into edges ────────────────────────────────────────────────


def test_callers_lists_every_caller_file_and_not_the_definition():
    """The end-to-end shape of the reported gap, in miniature."""
    from awgraph import symbols

    definition = _Chunk(
        "c1", "spaceUrl", "src/lib/spaces-url.ts",
        "export function spaceUrl(handle: string): string {\n  return `/s/${handle}`;\n}\n",
    )
    caller_tsx = _Chunk(
        "c2", "ShareSheet", "src/components/ShareSheet.tsx",
        'export function ShareSheet(p) {\n  const href = spaceUrl(p.handle);\n'
        "  return href;\n}\n",
    )
    caller_ts = _Chunk(
        "c3", "buildMenu", "src/lib/menu.ts",
        "export function buildMenu(h) {\n  return [spaceUrl(h)];\n}\n",
    )
    unrelated = _Chunk(
        "c4", "Unrelated", "src/lib/other.ts",
        "export function Unrelated() {\n  return somethingElse();\n}\n",
    )

    calls, called_by = symbols.derive_js_edges(
        [definition, caller_tsx, caller_ts, unrelated]
    )

    assert set(called_by.get("c1", [])) == {"ShareSheet", "buildMenu"}, called_by
    assert "spaceUrl" not in called_by.get("c1", []), (
        "the definition was listed among its own callers"
    )
    assert "c4" not in called_by
    assert "spaceUrl" in calls["c2"] and "spaceUrl" in calls["c3"]


def test_a_short_name_is_not_bound_across_files():
    """One-and two-character names are not resolvable by name alone."""
    from awgraph import symbols

    definition = _Chunk("d", "g", "a.ts", "export function g() {\n  return 1;\n}\n")
    caller = _Chunk("u", "useG", "b.ts", "export function useG() {\n  return g();\n}\n")
    _calls, called_by = symbols.derive_js_edges([definition, caller])
    assert "d" not in called_by, (
        "a one-letter name was bound across files; on a real tree that is a "
        "four-figure fan-in of coincidences"
    )


def test_a_short_name_still_binds_inside_one_file():
    """Same-file resolution is unambiguous, so the length rule must not apply."""
    from awgraph import symbols

    definition = _Chunk("d", "g", "a.ts", "function g() {\n  return 1;\n}\n")
    caller = _Chunk("u", "useG", "a.ts", "function useG() {\n  return g();\n}\n")
    _calls, called_by = symbols.derive_js_edges([definition, caller])
    assert called_by.get("d") == ["useG"], called_by


def test_a_very_common_name_is_not_bound_across_files():
    from awgraph import symbols

    defs = [
        _Chunk("d%d" % i, "handle", "d%d.ts" % i,
               "export function handle() {\n  return %d;\n}\n" % i)
        for i in range(jscalls.MAX_DEFINITION_CANDIDATES + 1)
    ]
    caller = _Chunk("u", "useIt", "u.ts",
                    "export function useIt() {\n  return handle();\n}\n")
    _calls, called_by = symbols.derive_js_edges(defs + [caller])
    assert called_by == {}, (
        "a name with more definitions than the ceiling was bound anyway"
    )
    # The unresolved call is still REPORTED, which is the honest half.
    assert _calls["u"] == ["handle"]


def test_python_chunks_are_left_alone():
    """The derivation must not invent edges for a language it cannot read."""
    from awgraph import symbols

    py = _Chunk("p", "send_request", "client.py",
                "def send_request(url):\n    return _do_send(url)\n")
    calls, called_by = symbols.derive_js_edges([py])
    assert calls == {} and called_by == {}


# ── end to end, through the real parser and CLI ──────────────────────────

requires_multilang = pytest.mark.skipif(
    not multilang.available()[0], reason="multilang extra not installed"
)

_DEF_TS = """export function spaceUrl(handle: string, tab?: string): string {
  const slug = handle.toLowerCase();
  return tab ? `/s/${slug}/${tab}` : `/s/${slug}`;
}
"""

_CALLER_TSX = """import { spaceUrl } from "../lib/spaces-url";

export function ShareSheet(props: { handle: string }) {
  const href = spaceUrl(props.handle, "about");
  return href;
}
"""

_CALLER_TS = """import { spaceUrl } from "./spaces-url";

export function buildMenu(handle: string): string[] {
  // spaceUrl is also named in this comment, which must change nothing
  return [spaceUrl(handle)];
}
"""


@requires_multilang
def test_callers_answers_for_a_typescript_symbol(tmp_path, monkeypatch, capsys):
    """The reported gap, end to end: index a .ts tree, ask who calls."""
    monkeypatch.setenv("AWGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    lib = tmp_path / "src" / "lib"
    comp = tmp_path / "src" / "components"
    lib.mkdir(parents=True)
    comp.mkdir(parents=True)
    (lib / "spaces-url.ts").write_text(_DEF_TS, encoding="utf-8")
    (lib / "menu.ts").write_text(_CALLER_TS, encoding="utf-8")
    (comp / "ShareSheet.tsx").write_text(_CALLER_TSX, encoding="utf-8")

    assert _run(["index", str(tmp_path)]) == cli.EXIT_OK

    capsys.readouterr()
    rc = _run(["callers", "spaceUrl", "--path", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == cli.EXIT_OK, "callers still returns a negative for a .ts symbol: " + out
    assert "ShareSheet" in out, out
    assert "buildMenu" in out, out
    assert "spaces-url.ts:1" not in out, (
        "the definition was reported as one of its own callers:\n" + out
    )
