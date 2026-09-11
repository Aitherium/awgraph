"""A file that DISCOVERS but produces zero chunks is invisible to search.

Measured 2026-09-10 on the host repo: `.DEPLOYMENT/scripts/*.sh` were discovered
(once discovery was fixed to see dot-directories) and parsed without error, but
both scripts are straight-line -- no shell functions -- so `parse_symbols`
returned [] and the file contributed nothing to a 400k-chunk index. Two of six
known questions in that repo are answered by exactly those scripts, so they
could never hit however good the embedder was. The fallback is bounded by size:
a minified bundle must NOT explode into hundreds of windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awgraph import multilang  # noqa: E402
from awgraph.graph import SYMBOLLESS_FALLBACK_MAX_BYTES, parse_file_sync  # noqa: E402

requires_multilang = pytest.mark.skipif(
    not multilang.available()[0], reason="multilang extra not installed"
)


@requires_multilang
def test_functionless_shell_script_yields_a_chunk(tmp_path):
    """The exact artifact: a script whose only content is top-level commands."""
    script = tmp_path / "entrypoint.sh"
    script.write_text(
        "#!/bin/sh\nset -eu\nexec /app/llama-server -m /models/code-embed.gguf "
        "--embedding --port 8229\n",
        encoding="utf-8",
    )
    graph = parse_file_sync(str(script))
    assert graph.chunks, (
        "a function-less script produced no chunks -- it is discovered, parsed, "
        "and invisible to every query"
    )
    assert any("llama-server" in c.body_preview for c in graph.chunks), graph.chunks


@requires_multilang
def test_oversized_symbol_less_file_is_still_skipped(tmp_path):
    """The bound, in the other direction: generated/minified files stay out."""
    blob = "var a=1;" * (SYMBOLLESS_FALLBACK_MAX_BYTES // 4)
    big = tmp_path / "bundle.min.js"
    big.write_text(blob, encoding="utf-8")
    graph = parse_file_sync(str(big))
    assert not graph.chunks, "an oversized symbol-less file was windowed into the index"


@requires_multilang
def test_script_with_functions_still_chunks_by_symbol(tmp_path):
    """The fallback must not replace symbol chunking where symbols exist."""
    script = tmp_path / "tool.sh"
    script.write_text(
        "#!/bin/sh\nstart_server() {\n  exec llama-server --port 8229\n}\n",
        encoding="utf-8",
    )
    graph = parse_file_sync(str(script))
    assert graph.chunks, "a script WITH a function produced no chunks"
    names = {c.name for c in graph.chunks}
    assert any("start_server" in n for n in names), names
