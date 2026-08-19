"""Multi-language and document indexing, via repowise.

WHY THIS EXISTS
---------------
awgraph's engine is Python-only: ``parse_file_sync`` routes to CPython's ``ast``
and file discovery globs ``*.py``, so every ``.ts``, ``.go``, ``.rs``, ``.cs``
file — and every ``.md`` — is invisible to the index. On a mixed repository that
is not a corner case, it is most of the repository, and the failure is silent:
the index builds, queries return, and the answers are simply drawn from a
fraction of the tree. A retrieval system that quietly sees a third of your code
is worse than one that refuses, because nothing about the result says so.

Rather than grow a tree-sitter stack here, this adapts repowise, which parses
**75 extensions across 42 languages** and emits symbol ids of the form
``path::qualified_name``. Those ids are stable under movement, which is what
makes them usable as chunk identity rather than as a position that shifts on
every edit above it.

DOCUMENTS ARE KEPT, WHICH IS THE DIFFERENCE FROM A MERGE ENGINE
---------------------------------------------------------------
repowise classifies markdown, asciidoc, json, yaml, toml, ini, csv, text, html,
css, xml and sql as *symbol-less*: it recognises the language but extracts no
functions or classes from it. A semantic version-control layer must DROP those
— it needs a symbol to have an identity to merge. A retrieval index must not:
"where is the deploy runbook" and "which config sets the timeout" are ordinary
questions, and the answer lives in exactly those files.

So symbol-bearing languages are chunked by SYMBOL, and symbol-less ones are
chunked by SECTION — markdown by heading, everything else by a size window.
Both become ordinary chunks the existing retrieval path already knows how to
score, so nothing downstream needs to learn a new shape.

OPTIONAL, ALWAYS
----------------
repowise is a third-party dependency and this package must not hard-require it.
``available()`` returns ``(False, why)`` when it is absent and every caller
falls back to Python-only. Missing repowise degrades coverage; it must never
turn ``import awgraph`` into an ImportError.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

_PARSER = None
_UNAVAILABLE: Optional[str] = None

#: Languages repowise recognises but extracts no symbols from. A merge engine
#: drops these; a retrieval index chunks them as prose instead (see module doc).
SYMBOL_LESS_LANGUAGES: Set[str] = {
    "markdown", "asciidoc", "json", "yaml", "toml", "ini", "csv", "text",
    "html", "css", "xml", "sql",
}

#: Chunked by heading rather than by byte window — a heading is a real semantic
#: boundary, and splitting prose mid-sentence produces chunks that embed badly.
_HEADING_LANGUAGES: Set[str] = {"markdown", "asciidoc"}

#: Fallback window for symbol-less files with no headings (yaml, json, csv…).
#: Overlap exists so a fact spanning the boundary survives in one whole piece.
_WINDOW_LINES = 120
_WINDOW_OVERLAP = 20

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_ADOC_HEADING = re.compile(r"^(={1,6})\s+(.*\S)\s*$")


#: Directories skipped by default. GENERIC ONLY, on purpose: these are build
#: output, dependency trees and VCS metadata that no repository wants indexed.
#: Anything specific to one deployment's layout belongs in a caller's
#: `exclude_dirs`, not here — a published package that hardcodes another
#: project's folder names is shipping one team's tree layout to every user, and
#: it silently skips a directory a stranger may legitimately have.
DEFAULT_EXCLUDE_DIRS: Tuple[str, ...] = (
    ".git", "__pycache__", "node_modules", ".venv", "venv", "site-packages",
    "dist", "build", ".next", ".turbo", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".worktrees",
)

#: Extensions indexed when the caller names none. Python always; the rest only
#: when repowise is installed, because indexing a file nothing can parse just
#: produces empty chunks.
def index_extensions(extra: Optional[Set[str]] = None,
                     only: Optional[Set[str]] = None) -> Set[str]:
    """The extension set to discover, as a caller-overridable policy.

    `only` replaces the default set outright; `extra` adds to it. Both are
    normalised to a leading dot so ``{"ts"}`` and ``{".ts"}`` behave the same —
    a config that silently matched nothing because of a missing dot is the kind
    of quiet miss this whole module exists to remove.
    """
    def _norm(items: Optional[Set[str]]) -> Set[str]:
        return {e if e.startswith(".") else "." + e for e in (items or set()) if e}

    if only:
        return _norm(only)
    return supported_extensions() | _norm(extra)


def available() -> Tuple[bool, str]:
    """(usable, why-not). Never raises: absence is a fallback, not a failure."""
    global _PARSER, _UNAVAILABLE
    if _PARSER is not None:
        return True, ""
    if _UNAVAILABLE is not None:
        return False, _UNAVAILABLE
    try:
        from repowise.core.ingestion import ASTParser  # noqa: PLC0415

        _PARSER = ASTParser()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
        _UNAVAILABLE = f"{type(exc).__name__}: {exc}"
        return False, _UNAVAILABLE


def _extension_map() -> Dict[str, str]:
    try:
        from repowise.core.ingestion import EXTENSION_TO_LANGUAGE  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return {}
    return dict(EXTENSION_TO_LANGUAGE)


def language_for(path: str) -> Optional[str]:
    """repowise's language for a path, or None when it does not handle it.

    Unlike a merge engine's version of this function, symbol-less languages are
    RETURNED rather than filtered out — the caller decides whether it wants
    symbols or sections, and dropping them here would make documents
    unreachable with no way for a caller to opt in.
    """
    ok, _ = available()
    if not ok:
        return None
    _, _, ext = (path or "").rpartition(".")
    if not ext:
        return None
    return _extension_map().get("." + ext.lower())


def supported_extensions() -> Set[str]:
    """Every extension this adapter can index, ``.py`` included."""
    ok, _ = available()
    if not ok:
        return {".py"}
    return set(_extension_map()) | {".py"}


def is_document(path: str) -> bool:
    """True when the file is prose/config rather than symbol-bearing code."""
    lang = language_for(path)
    return lang is not None and lang in SYMBOL_LESS_LANGUAGES


def _path_prefix(path: str) -> str:
    """The dotted prefix repowise builds a qualified_name from.

    repowise derives ``qualified_name`` from the FILE PATH, so an ABSOLUTE path
    yields names like ``C.Users.me.tmp.probe.alpha`` instead of
    ``probe.alpha``. Names are the thing a person searches for, so a name
    carrying the indexing machine's directory layout is not a cosmetic problem:
    it makes the symbol unfindable by its own name, and it leaks local paths
    into an index that may be shared.
    """
    stem = re.sub(r"\.[^./\\]+$", "", path or "")
    return re.sub(r"[\\/]+", ".", stem).lstrip(".") + "."


def _strip_path_prefix(name: str, prefix: str) -> str:
    """Drop the path-derived prefix, keeping the symbol's own dotted path.

    Falls back to the raw name when it does not match: a future repowise that
    stops prefixing must not have its names mangled by a guess.
    """
    if prefix and len(prefix) > 1 and name.startswith(prefix):
        stripped = name[len(prefix):]
        if stripped:
            return stripped
    # Drive-letter and leading-separator forms vary; also try the last path
    # component only, which is what a relative path would have produced.
    tail = prefix.rstrip(".").rsplit(".", 1)[-1]
    if tail and name.startswith(tail + "."):
        return name[len(tail) + 1:]
    return name


def parse_symbols(content: bytes, path: str) -> List[dict]:
    """Symbols in `content`, or [] when it cannot parse.

    Returns name/kind/start_line/end_line/language/symbol_id dicts. The
    symbol_id is repowise's ``path::qualified_name``, which survives a move —
    that is why it is carried through instead of a body hash.
    """
    ok, _ = available()
    if not ok:
        return []
    lang = language_for(path)
    if not lang or lang in SYMBOL_LESS_LANGUAGES:
        return []
    try:
        from repowise.core.ingestion import FileInfo  # noqa: PLC0415

        info = FileInfo(
            path=path, abs_path=path, language=lang, size_bytes=len(content),
            git_hash="", last_modified=0.0, is_test=False, is_config=False,
            is_api_contract=False, is_entry_point=False,
        )
        parsed = _PARSER.parse_file(info, content)
    except Exception:  # noqa: BLE001
        # One unparseable file must degrade to "no symbols", never take the
        # whole index down. A parser crash on a single file is not a reason to
        # lose the other twenty thousand.
        return []

    prefix = _path_prefix(path)
    out: List[dict] = []
    for sym in (getattr(parsed, "symbols", None) or []):
        name = getattr(sym, "qualified_name", None) or getattr(sym, "name", "")
        if not name:
            continue
        out.append({
            "name": _strip_path_prefix(str(name), prefix),
            "kind": str(getattr(sym, "kind", "") or "symbol"),
            "start_line": int(getattr(sym, "start_line", 0) or 0),
            "end_line": int(getattr(sym, "end_line", 0) or 0),
            "language": lang,
            "symbol_id": str(getattr(sym, "id", "") or f"{path}::{name}"),
        })
    return out


def _heading_sections(lines: List[str], pattern: "re.Pattern[str]") -> List[dict]:
    """Split prose at headings. Text before the first heading is its own chunk."""
    marks: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            marks.append((i, m.group(2)))

    sections: List[dict] = []
    if not marks:
        return sections
    if marks[0][0] > 0:
        sections.append({"name": "(preamble)", "start_line": 1, "end_line": marks[0][0]})
    for idx, (line_no, title) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        sections.append({"name": title, "start_line": line_no + 1, "end_line": end})
    return sections


def chunk_document(text: str, path: str) -> List[dict]:
    """Sections for a symbol-less file: headings where it has them, else windows.

    Returns name/kind/start_line/end_line/language dicts, the same shape
    :func:`parse_symbols` returns, so one caller handles both.
    """
    lang = language_for(path) or "text"
    lines = text.split("\n")
    if not any(line.strip() for line in lines):
        return []

    sections: List[dict] = []
    if lang in _HEADING_LANGUAGES:
        pattern = _ADOC_HEADING if lang == "asciidoc" else _MD_HEADING
        sections = _heading_sections(lines, pattern)

    if not sections:
        step = max(1, _WINDOW_LINES - _WINDOW_OVERLAP)
        for start in range(0, len(lines), step):
            end = min(start + _WINDOW_LINES, len(lines))
            if not any(line.strip() for line in lines[start:end]):
                continue
            sections.append({
                "name": f"{path.rsplit('/', 1)[-1]}:{start + 1}",
                "start_line": start + 1,
                "end_line": end,
            })
            if end >= len(lines):
                break

    for s in sections:
        s["kind"] = "section"
        s["language"] = lang
        s["symbol_id"] = f"{path}::{s['name']}"
    return sections


def self_test() -> int:
    """Prove the adapter still works, and that it fails honestly when it cannot."""
    failures = 0
    ok, why = available()
    print(f"repowise available: {ok}" + ("" if ok else f" ({why})"))

    # Heading splitting must not depend on repowise at all.
    md = "intro text\n\n# One\nalpha\n\n## Two\nbeta\n"
    secs = chunk_document(md, "doc.md") if ok else _heading_sections(md.split("\n"), _MD_HEADING)
    names = [s["name"] for s in secs]
    if "One" not in names or "Two" not in names:
        print(f"self-test: markdown headings not found, got {names}")
        failures += 1
    if secs and secs[0]["name"] != "(preamble)":
        print("self-test: text before the first heading was dropped")
        failures += 1

    if ok:
        # A symbol-less language must yield sections, never symbols.
        if parse_symbols(b"# hi\n", "a.md"):
            print("self-test: markdown produced symbols")
            failures += 1
        if not is_document("a.md"):
            print("self-test: markdown not classified as a document")
            failures += 1
        if is_document("a.ts"):
            print("self-test: typescript misclassified as a document")
            failures += 1
        # The whole point: a non-Python language yields symbols.
        syms = parse_symbols(b"export function alpha() { return 1 }\n", "probe.ts")
        if not any(s["name"].endswith("alpha") for s in syms):
            print(f"self-test: no symbols from typescript, got {syms}")
            failures += 1
        exts = supported_extensions()
        if ".py" not in exts or len(exts) < 20:
            print(f"self-test: extension set looks wrong ({len(exts)})")
            failures += 1
        # A file repowise cannot parse degrades to [], it does not raise.
        if parse_symbols(b"\x00\x01 not source", "broken.ts") is None:
            print("self-test: unparseable input did not degrade to a list")
            failures += 1

    if not ok:
        # Everything below the heading check needs repowise. Reporting OK here
        # would be a pass over untested code -- the exact vacuous-green this
        # package's own gates exist to prevent.
        print("multilang self-test: NOT VERIFIED - repowise absent, "
              f"{len(names)} heading assertions ran, symbol path untested")
        return 2 if not failures else 1
    print("multilang self-test:", "OK" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    sys.exit(self_test())
