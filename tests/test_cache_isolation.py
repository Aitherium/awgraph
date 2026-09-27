"""Guard: the test suite never writes into the real user awgraph cache."""

from __future__ import annotations

import os

from awgraph import graph


def _user_cache_root() -> str:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    return os.path.abspath(os.path.join(base, "awgraph"))


def test_cache_root_is_isolated(_isolated_awgraph_cache):
    root = os.path.abspath(graph._cache_root())
    assert root == os.path.abspath(str(_isolated_awgraph_cache))
    assert root != _user_cache_root()


def test_data_path_for_tmp_repo_stays_out_of_user_cache(tmp_path):
    path = os.path.abspath(graph._get_data_path(str(tmp_path), "index.json"))
    assert not path.startswith(_user_cache_root() + os.sep)
