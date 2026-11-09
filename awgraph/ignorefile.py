"""`.awgraphignore`: paths a repository keeps out of ITS code graph, gitignore syntax.

`.gitignore` says what git does not track; a code graph also wants to skip tracked or
untracked trees that are not the project's code -- scratch dirs, backups, the eight
language mirrors of one vendored API reference. Measured 2026-11-08 on the host repo:
`.RESEARCH/.CLAUDE-MANAGED-AGENTS/api/<lang>/` alone was 21,500 chunks of near-copies
(prose 97% identical to the canonical top-level pages).

Supported, matched against the path RELATIVE to the root with `/` separators:
  * `#` comments and blank lines;
  * `!pattern` re-includes (the last matching line wins);
  * a trailing `/` matches directories only (everything below them);
  * a leading `/`, or a `/` inside the pattern, anchors it at the root; otherwise it
    matches at any depth;
  * `*` and `?` within one path segment, `**` across segments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Pattern, Tuple

FILENAME = ".awgraphignore"

Rule = Tuple[Pattern[str], bool]  # (regex, negated)


def _glob_to_re(body: str) -> str:
    out, i = [], 0
    while i < len(body):
        if body.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif body.startswith("**", i):
            out.append(".*")
            i += 2
        elif body[i] == "*":
            out.append("[^/]*")
            i += 1
        elif body[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(body[i]))
            i += 1
    return "".join(out)


def parse(lines: Iterable[str]) -> List[Rule]:
    rules: List[Rule] = []
    for raw in lines:
        line = raw.rstrip("\n\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        neg = line.startswith("!")
        pat = line[1:] if neg else line
        pat = pat.strip()
        is_dir = pat.endswith("/")
        body = pat.strip("/")
        if not body:
            continue
        anchored = pat.startswith("/") or "/" in body
        prefix = "^" if anchored else "^(?:.*/)?"
        tail = "/" if is_dir else "(?:/|$)"
        rules.append((re.compile(prefix + _glob_to_re(body) + tail), neg))
    return rules


def load(root: Path) -> List[Rule]:
    try:
        with open(os.path.join(str(root), FILENAME), encoding="utf-8") as fh:
            return parse(fh)
    except OSError:
        return []


def ignored(rel: str, rules: List[Rule]) -> bool:
    rel = rel.replace("\\", "/").lstrip("/")
    hit = False
    for rx, neg in rules:
        if rx.match(rel):
            hit = not neg
    return hit


def apply(root: Path, files: List[Path], rules: Optional[List[Rule]] = None) -> List[Path]:
    """``files`` minus what ``root``'s `.awgraphignore` excludes (unchanged without one)."""
    rules = load(root) if rules is None else rules
    if not rules:
        return files
    base = os.path.abspath(str(root))
    kept = []
    for f in files:
        try:
            rel = os.path.relpath(os.path.abspath(str(f)), base)
        except ValueError:  # another drive: not under the root, keep it
            kept.append(f)
            continue
        if not ignored(rel, rules):
            kept.append(f)
    return kept
