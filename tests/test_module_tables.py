"""A module-level DATA TABLE must be indexable, not just its first 120 chars.

Measured 2026-09-11 on the live index: `volunteer_batch_embed` sits at
EarnLedger.py:132 inside EARN_SOURCES, a ~107-line module-level dict. The module
chunk renders every constant through ast.unparse and truncates to 120 chars, so
the table's head was indexed and its tail was not — and NO chunk's text held the
literal. The question about it was unanswerable by any ranking, at every fusion
weight and in every rerank mode; it looked like a ranking gap for a day.

The first test pins BOTH halves: the tail is absent from the old-style module
preview (the reason this exists) and present in the table's own chunk. If a
future change makes the module preview large enough to carry the tail, that
assertion fails loudly instead of silently making the table chunk redundant.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awgraph import graph as g  # noqa: E402

BIG_TABLE = "EARN_SOURCES = {\n" + "".join(
    f'    "source_{i}": {{"reward": {i}, "note": "entry number {i}"}},\n'
    for i in range(40)
) + "}\n"


def _parse(src: str):
    return g.parse_file_sync("pkg/module_tables_probe.py", content=src)


def test_a_long_table_is_indexed_past_the_module_previews_budget():
    fg = _parse("class Thing:\n    pass\n\n\n" + BIG_TABLE +
                "\n\ndef helper():\n    return 1\n")
    module_chunk = next(c for c in fg.chunks if c.chunk_type == g.ChunkType.MODULE
                        and c.name == "module_tables_probe")
    assert "source_39" not in module_chunk.body_preview, (
        "the module preview now carries the tail -- the table chunk may be "
        "redundant; re-measure before deleting it")
    texts = [c.body_preview or "" for c in fg.chunks]
    assert any("source_39" in t for t in texts), "the table's tail is unindexed"

    table = next(c for c in fg.chunks if c.name == "EARN_SOURCES")
    assert table.chunk_type == g.ChunkType.MODULE
    assert table.end_line > table.start_line


def test_a_scalar_constant_gets_no_table_chunk():
    fg = _parse("PORT = 8194\nTIMEOUT = 30\n\n\ndef helper():\n    return 1\n")
    names = [c.name for c in fg.chunks]
    assert "PORT" not in names and "TIMEOUT" not in names, (
        "a scalar must not earn its own chunk -- thousands of `X = 1` lines "
        "would dilute every ranking")


# The real shape: a dict comprehension over calls whose FIRST argument is the
# registry key (EarnLedger's EARN_SOURCES). The key of interest is last, so its
# literal sits ~900 chars into the unparsed value -- outside the 300-char window
# embed_chunks actually embeds.
EARN_SHAPE = (
    "EARN_SOURCES = {\n    s.id: s\n    for s in [\n"
    + "".join(
        f'        EarnSource("source_{i}", "Title number {i}", '
        f'"A longer description of what entry {i} rewards.", base_award={i}),\n'
        for i in range(9)
    )
    + '        EarnSource("volunteer_batch_embed", "Embedded a verified batch", '
      '"A batch agreed with an independent peer.", base_award=10),\n'
    + "    ]\n}\n"
)


def test_table_keys_land_in_the_window_the_embedder_actually_reads():
    fg = _parse(EARN_SHAPE + "\n\ndef helper():\n    return 1\n")
    table = next(c for c in fg.chunks if c.name == "EARN_SOURCES")
    rendered = (table.body_preview or "").split("\n", 1)[-1]
    assert "volunteer_batch_embed" not in rendered[:300], (
        "fixture drifted: the key is now visible in the raw window, so this "
        "test no longer proves the keys line is doing the work")
    window = (table.signature or "") + "\n" + (table.body_preview or "")[:300]
    assert "volunteer_batch_embed" in window, (
        "embed_chunks builds its text from signature + body_preview[:300]; a "
        "registry key outside that window is invisible to the vector")
