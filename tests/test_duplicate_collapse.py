"""Identical chunks must take one result slot, not four.

Measured on a 387,098-chunk index: 71,257 chunks (18%) sit inside 29,168
groups whose bodies are byte-for-byte identical — vendored copies, mirrored
trees, and one instruction file duplicated across four directories. Because
identical text scores identically, those copies arrive adjacent at the top of
a result list. One question returned the same documentation section four
times, using four of its ten slots to say one thing, and the file that
actually answered it never appeared.

The collapse is by CONTENT, not by path, and it runs before the slice — so a
duplicate does not merely get dropped, it frees its slot for a different
answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awgraph.graph import DUPLICATE_MIN_BODY, _collapse_duplicate_bodies  # noqa: E402

LONG_A = "def alpha():\n    " + ("return compute_the_thing(1)  # a\n    " * 6)
LONG_B = "def beta():\n    " + ("return other_thing(2)  # b\n    " * 6)


class _Chunk:
    def __init__(self, cid, path, body):
        self.id = cid
        self.source_path = path
        self.body_preview = body

    def __repr__(self):
        return "<%s %s>" % (self.id, self.source_path)


def test_identical_bodies_collapse_to_one():
    chunks = [
        _Chunk("c1", "a/skill.md", LONG_A),
        _Chunk("c2", "b/skill.md", LONG_A),
        _Chunk("c3", "c/skill.md", LONG_A),
        _Chunk("c4", "d/skill.md", LONG_A),
    ]
    kept = _collapse_duplicate_bodies(chunks)
    assert [c.id for c in kept] == ["c1"], kept


def test_the_highest_ranked_copy_is_the_one_kept():
    """Order is rank order; the survivor must be the one the scorer preferred."""
    chunks = [
        _Chunk("winner", "a.md", LONG_A),
        _Chunk("loser", "b.md", LONG_A),
    ]
    assert [c.id for c in _collapse_duplicate_bodies(chunks)] == ["winner"]


def test_distinct_bodies_all_survive():
    chunks = [_Chunk("c1", "a.py", LONG_A), _Chunk("c2", "b.py", LONG_B)]
    assert [c.id for c in _collapse_duplicate_bodies(chunks)] == ["c1", "c2"]


def test_collapsing_frees_slots_for_different_answers():
    """The point of the fix: three copies must not spend three of four slots."""
    chunks = [
        _Chunk("dup1", "a.md", LONG_A),
        _Chunk("dup2", "b.md", LONG_A),
        _Chunk("dup3", "c.md", LONG_A),
        _Chunk("real", "answer.py", LONG_B),
    ]
    top = _collapse_duplicate_bodies(chunks)[:2]
    assert [c.id for c in top] == ["dup1", "real"], (
        "the real answer was still crowded out by copies of one file"
    )


def test_short_bodies_are_never_collapsed():
    """A one-line stub legitimately recurs; collapsing it hides real answers."""
    stub = "export default x;"
    assert len(stub) < DUPLICATE_MIN_BODY
    chunks = [_Chunk("c1", "a.ts", stub), _Chunk("c2", "b.ts", stub)]
    assert len(_collapse_duplicate_bodies(chunks)) == 2


def test_whitespace_only_differences_still_collapse():
    """Leading/trailing whitespace is not a different answer."""
    chunks = [
        _Chunk("c1", "a.md", LONG_A),
        _Chunk("c2", "b.md", "\n\n" + LONG_A + "  \n"),
    ]
    assert [c.id for c in _collapse_duplicate_bodies(chunks)] == ["c1"]


def test_a_chunk_with_no_body_is_passed_through():
    chunks = [_Chunk("c1", "a.md", ""), _Chunk("c2", "b.md", None)]
    assert len(_collapse_duplicate_bodies(chunks)) == 2


def test_empty_input_is_empty_output():
    assert _collapse_duplicate_bodies([]) == []
