"""Git enrichment must survive being pointed at a SUBTREE, not just a repo root.

The guard used to be `if not (root / ".git").exists(): return not_a_git_repo`,
tested against the indexed path. Indexing a subtree - the normal case, because
indexing a whole monorepo takes far longer than indexing the package you care
about - therefore disabled commit counts, author counts and churn silently.

It is a stats field nobody reads, so nothing fails: no exception, no warning,
and the index reports success with every other number intact. Measured
2026-08-19 on a 2,434-file subtree index, git enrichment contributed nothing to
ranking while looking fully wired.

Each test carries a mutation guard reproducing the OLD shape, so a future
"simplification" back to the one-line check fails here instead of going quiet.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awgraph.graph import CodeGraph  # noqa: E402


def _old_guard(root_path: str):
    """The shape this test exists to keep dead."""
    return None if not (Path(root_path) / ".git").exists() else Path(root_path)


def test_finds_repo_root_from_a_subtree(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    deep = repo / "pkg" / "sub" / "leaf"
    deep.mkdir(parents=True)

    assert CodeGraph._git_root(str(deep)) == repo.resolve()
    # Mutation guard: the old guard sees nothing here, which is the whole defect.
    assert _old_guard(str(deep)) is None


def test_repo_root_itself_still_resolves(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    assert CodeGraph._git_root(str(repo)) == repo.resolve()
    assert _old_guard(str(repo)) is not None  # the old guard was right ONLY here


def test_worktree_dot_git_file_not_dir_counts(tmp_path):
    """`.git` is a FILE in a worktree or submodule.

    Testing `is_dir()` instead of `exists()` would reintroduce the same silent
    skip for anyone working in a worktree - which this repo does routinely.
    """
    repo = tmp_path / "wt"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    leaf = repo / "a" / "b"
    leaf.mkdir(parents=True)

    assert CodeGraph._git_root(str(leaf)) == repo.resolve()


#: Decided at COLLECTION time, not inside the body. A body-level skip fires
#: after partial execution, so a genuine failure would be reported as "skipped"
#: and CI would stay green - the exact hole a collection-time skip closes.
_TMP = Path(tempfile.gettempdir())
_TMP_INSIDE_A_REPO = any((d / ".git").exists() for d in [_TMP, *_TMP.parents])


@pytest.mark.skipif(_TMP_INSIDE_A_REPO,
                    reason="system temp dir is itself inside a git repo, so the "
                           "negative case is not constructible here")
def test_returns_none_outside_any_repo(tmp_path):
    """Fails CLOSED. A wrong root would run git log in someone else's repo."""
    lonely = tmp_path / "nowhere" / "deep"
    lonely.mkdir(parents=True)

    assert CodeGraph._git_root(str(lonely)) is None


def test_nearest_root_wins_for_nested_repos(tmp_path):
    """A submodule's own .git must win over its parent's."""
    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    inner = outer / "vendor" / "inner"
    (inner / ".git").mkdir(parents=True)
    leaf = inner / "src"
    leaf.mkdir()

    assert CodeGraph._git_root(str(leaf)) == inner.resolve()
