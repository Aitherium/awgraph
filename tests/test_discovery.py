"""Discovery contract: what gets indexed, and what must not.

These assertions were established by hand against real `rg` behaviour while
fixing discovery, and every one of them guards a bug that had SHIPPED and was
invisible:

  * discovery globbed `*.py`, so 40+ languages and every document were missing
    from the index while queries returned confident answers from the remainder;
  * an INCLUDE glob (`-g "*.py"`) switches ripgrep's `.gitignore` handling off,
    so ignored build output was indexed as source;
  * ripgrep anchors a glob at the CURRENT WORKING DIRECTORY, not the search
    root, so `!node_modules/**` excluded nothing whenever the indexer ran from
    anywhere other than the tree — the normal case for a library.

All three failed as a SILENCE: files that exist, an index that builds, results
that look fine. Nothing but an assertion on the discovered SET can see them,
which is why this file exists rather than a runbook.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awgraph import multilang  # noqa: E402
from awgraph.graph import default_exclude_dirs, discover_files  # noqa: E402

#: The ignore-file and glob guarantees below are ripgrep's. Without it discovery
#: falls back to a pure-Python walk that has no ignore-file handling at all, and
#: asserting them would be asserting a behaviour the code never claimed.
#: Decoration-time skipif, never a skip inside the body: a body-level skip fires
#: AFTER partial execution and reports a real failure as "skipped".
_HAS_RG = shutil.which("rg") is not None
_HAS_GIT = shutil.which("git") is not None
requires_rg = pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
requires_git = pytest.mark.skipif(not _HAS_GIT, reason="git not installed")


def _tree(root: Path) -> Path:
    """A small mixed repository: several languages, a document, a data file."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src/auth.ts").write_text("export function validateToken(){}\n", encoding="utf-8")
    (root / "src/server.go").write_text("package main\nfunc Start() {}\n", encoding="utf-8")
    (root / "src/util.py").write_text("def parse():\n    return 1\n", encoding="utf-8")
    (root / "src/conf.yaml").write_text("timeout: 30\n", encoding="utf-8")
    (root / "RUNBOOK.md").write_text("# Deploy\nrun it\n", encoding="utf-8")
    return root


def _names(paths) -> set:
    return {p.name for p in paths}


def _discover(root: Path, **kw) -> set:
    files, _ms = asyncio.run(discover_files(root, **kw))
    return _names(files)


@pytest.mark.skipif(not multilang.available()[0], reason="multilang extra not installed")
def test_discovers_more_than_python(tmp_path):
    """The whole point: a mixed tree must not be indexed as a Python tree."""
    found = _discover(_tree(tmp_path))
    assert "util.py" in found, found
    for name in ("auth.ts", "server.go", "RUNBOOK.md", "conf.yaml"):
        assert name in found, f"{name} missing from discovery: {found}"


def test_extensions_argument_narrows_discovery(tmp_path):
    """`extensions=` must actually restrict, not be accepted and ignored."""
    found = _discover(_tree(tmp_path), extensions={"py"})
    assert found == {"util.py"}, found


@requires_git
@requires_rg
def test_gitignored_files_are_not_indexed(tmp_path):
    """An ignored file is not part of the codebase.

    Guards the include-glob trap: `rg --files -g '*.ts'` returns ignored files,
    `rg --files` does not. Discovery therefore filters extensions in Python and
    passes only NEGATED globs.
    """
    _tree(tmp_path)
    (tmp_path / "src/generated.ts").write_text("export function gen(){}\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("generated.ts\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], capture_output=True, check=False)

    found = _discover(tmp_path)
    assert "auth.ts" in found, found
    assert "generated.ts" not in found, f"gitignored file was indexed: {found}"


@requires_git
@requires_rg
def test_exclude_dirs_argument_is_honoured(tmp_path):
    """`exclude_dirs=` must change the result in BOTH directions.

    Guards the cwd-anchoring trap: `!skipme/**` matched nothing when the
    process ran outside the tree, so the exclude list looked enforced and was
    inert. The pytest tmp_path is never the cwd, which is exactly the condition
    that exposed it.
    """
    root = _tree(tmp_path)
    (root / "skipme").mkdir()
    (root / "skipme/ignored.py").write_text("x = 1\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules/dep.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], capture_output=True, check=False)

    # Default list: node_modules is generic build/dependency noise, always out.
    assert "dep.py" not in _discover(root), "node_modules leaked into discovery"

    # A caller-supplied list REPLACES the default, so node_modules returns.
    custom = _discover(root, exclude_dirs=["skipme"])
    assert "ignored.py" not in custom, f"exclude_dirs had no effect: {custom}"
    assert "dep.py" in custom, "exclude_dirs did not replace the default list"

    # Empty list means "exclude nothing" — .gitignore alone decides.
    everything = _discover(root, exclude_dirs=[])
    assert "ignored.py" in everything and "dep.py" in everything, everything


def test_default_exclude_list_is_split_generic_from_local():
    """Generic skips are the package's; this tree's layout is a replaceable default.

    A published package that hardcodes another project's folder names silently
    skips directories a stranger legitimately has.
    """
    from awgraph.graph import EXTRA_EXCLUDE_DIRS

    generic = set(multilang.DEFAULT_EXCLUDE_DIRS)
    assert {"node_modules", ".git", "dist", "build"} <= generic
    # Nothing project-specific may sit in the generic set.
    assert not (generic & set(EXTRA_EXCLUDE_DIRS))
    assert set(default_exclude_dirs()) == generic | set(EXTRA_EXCLUDE_DIRS)
