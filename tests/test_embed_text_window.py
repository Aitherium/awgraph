"""The embed window must carry the payload's END when the preview is long.

Measured 2026-09-11 on the llama-server entrypoint chunk (a 2,361-char preview):
head-only text put the gold file at cosine rank 11 for the question it answers;
head 180 + "..." + tail 110 put it at rank 4 (+0.021, against a 0.0007 slot
margin). An entrypoint's whole answer is its `exec` line, and that is at the END.

The first test fails against the old builder by construction (the marker sits
past char 300). The second pins that short previews are byte-identical to
before, so this change re-embeds only the long-preview chunks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awgraph import graph as g  # noqa: E402


def _section(preview: str) -> "g.CodeChunk":
    return g.CodeChunk(
        id="section_probe_deadbeef", name="probe.sh:1",
        chunk_type=g.ChunkType.SECTION, source_path="x/probe.sh",
        start_line=1, end_line=48,
        signature="shell section probe.sh:1", body_preview=preview)


def test_a_long_preview_reaches_the_tail():
    preview = ("# a header comment line that says nothing searchable\n" * 40
               + "exec llama-server -m aither-code-embed.gguf\n")
    assert len(preview) > g._EMBED_PREVIEW_HEADTAIL_MIN
    text = g._embed_text_for_chunk(_section(preview))
    assert "exec llama-server" in text, (
        "the window must reach the payload's end -- an entrypoint's exec line is "
        "its whole answer, and head-only windowing hides it")


def test_a_short_preview_is_unchanged():
    preview = "# short header\nexec llama-server\n"
    text = g._embed_text_for_chunk(_section(preview))
    assert text == "shell section probe.sh:1" + "\n" + preview, (
        "a preview under the head+tail threshold must embed exactly as before, or "
        "this change quietly re-embeds the whole index")
