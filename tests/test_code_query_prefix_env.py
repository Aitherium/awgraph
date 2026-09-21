"""AITHER_CODEGRAPH_QUERY_PREFIX: a literal backslash-n in the env value becomes a real
newline, so a podman quadlet (which cannot carry a newline in Environment=) can hand
awgraph the exact instruction prefix an embedder was trained with.

Measured 2026-09-02: quadlet renders `Environment="X=a\\nb"` as `--env "X=a\\nb"` verbatim,
and a raw newline in the unit file is a parse error that unloads EVERY unit in the
directory (202 units went LoadState=not-found for ~12 minutes).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))


def _reload_with(monkeypatch, value):
    monkeypatch.setenv("AITHER_CODEGRAPH_QUERY_PREFIX", value)
    import awgraph.graph as g
    return importlib.reload(g)


def test_literal_backslash_n_becomes_a_newline(monkeypatch):
    g = _reload_with(monkeypatch, "Instruct: find the dir\\nQuery: ")
    assert g._CODE_QUERY_PREFIX == "Instruct: find the dir\nQuery: "


def test_a_real_newline_passes_through_unchanged(monkeypatch):
    g = _reload_with(monkeypatch, "Instruct: find the dir\nQuery: ")
    assert g._CODE_QUERY_PREFIX == "Instruct: find the dir\nQuery: "


def test_prefix_without_escapes_is_untouched(monkeypatch):
    g = _reload_with(monkeypatch, "Represent this query for searching relevant code: ")
    assert g._CODE_QUERY_PREFIX == "Represent this query for searching relevant code: "
