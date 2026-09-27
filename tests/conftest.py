"""Shared fixtures for the awgraph test suite.

Every test indexes throwaway ``tmp*`` repositories. Without an override,
``awgraph.graph._cache_root()`` resolves to the real user cache
(``%LOCALAPPDATA%\\awgraph`` / ``~/.cache/awgraph``) and each run leaves one
``tmpXXXX-<digest>`` index directory behind there -- measured 2026-09-26: 30
leaked dirs, 3.6 GB. This autouse fixture points ``AWGRAPH_CACHE_DIR`` at a
per-test temporary directory so no test can write to the user's cache.
Tests that set their own ``AWGRAPH_CACHE_DIR`` still win (they run after).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_awgraph_cache(tmp_path_factory, monkeypatch):
    cache = tmp_path_factory.mktemp("awgraph-cache")
    monkeypatch.setenv("AWGRAPH_CACHE_DIR", str(cache))
    yield cache
