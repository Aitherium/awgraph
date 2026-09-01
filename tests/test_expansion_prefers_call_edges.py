"""Context expansion must spend its budget CROSSING files, not filling one.

`_expand_context` had three steps per seed - parent class + every sibling
method, then up to 8 same-file neighbours, then call edges - and a shared
`structural_budget` of `max_expand // 2`. The two same-file steps ran first, so
on any crowded module the budget was gone before the only step that crosses a
file boundary by STRUCTURE ever executed.

Measured 2026-08-19 on a 2,434-file / 1.23M-line index over 40 real commit
tasks: at k=10 the arm returned a mean of 8.3 distinct files against grep's flat
10.0, fewer than 10 on 30 of 40 tasks, and 14 of the 19 tasks with a miss were
starved that way. The misses were files one call-edge from a scoring chunk and
sharing no vocabulary with the task at all.

Both tests reproduce the OLD ordering explicitly, so a revert fails here rather
than going quiet - the failure mode being guarded against is a silent narrowing
of results, which no other check can see.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awgraph.graph import ChunkType, CodeChunk, CodeGraph  # noqa: E402


def _chunk(cid: str, path: str, name: str, parent: str = "") -> CodeChunk:
    return CodeChunk(
        id=cid, name=name, chunk_type=ChunkType.METHOD if parent else ChunkType.FUNCTION,
        source_path=path, start_line=1, end_line=2,
        signature="def {}()".format(name), docstring="", body_preview="",
        parent_class=parent,
    )


def _graph_with_crowded_seed_file(n_siblings: int = 40, n_seeds: int = 5):
    """N seeds, each in a crowded module, each with a caller in another file.

    MULTIPLE seeds is the point, and getting this wrong once already produced a
    guard that proved nothing. With a single seed the old code still reached its
    call edges: every inner loop appends BEFORE testing the budget, so seed #1
    always got a sibling, a neighbour AND a caller. The starvation is across
    seeds - the outer loop tests the budget at the TOP, so once seed #1 spent it
    on same-file work, seeds #2..#N were skipped entirely and their cross-file
    call edges were never explored at all.
    """
    g = CodeGraph(root_path=".", auto_index=False)
    seeds, callers = [], {}
    for s in range(n_seeds):
        path = "/repo/crowded{}.py".format(s)
        cls = "Crowded{}".format(s)
        seed = _chunk("seed{}".format(s), path, "target{}".format(s), parent=cls)
        g.chunks[seed.id] = seed
        g.by_file[path] = [seed.id]
        g.by_class[cls] = [seed.id]
        g.by_name[cls] = []
        for i in range(n_siblings):
            c = _chunk("sib{}_{}".format(s, i), path, "sib{}_{}".format(s, i),
                       parent=cls)
            g.chunks[c.id] = c
            g.by_file[path].append(c.id)
            g.by_class[cls].append(c.id)

        caller = _chunk("caller{}".format(s), "/repo/probe{}.py".format(s),
                        "runs_the_check{}".format(s))
        g.chunks[caller.id] = caller
        g.by_file["/repo/probe{}.py".format(s)] = [caller.id]
        callers[seed.id] = caller
        seeds.append(seed.id)

    g.get_context_for_chunk = lambda cid: (  # noqa: E731
        {"callers": [callers[cid]], "callees": []} if cid in callers else
        {"callers": [], "callees": []})
    return g, seeds, callers


def test_every_seed_gets_its_call_edge_explored():
    g, seeds, callers = _graph_with_crowded_seed_file()
    out = asyncio.run(g._expand_context(set(seeds), query="", max_expand=10))
    got = {c.id for c in out}

    reached = [s for s in seeds if callers[s].id in got]
    assert len(reached) >= 4, (
        "only {} of {} seeds had their cross-file caller explored: {}".format(
            len(reached), len(seeds), sorted(got)))


def test_expansion_widens_rather_than_deepens():
    """The point of expansion is more FILES, not more chunks of one file."""
    g, seeds, _ = _graph_with_crowded_seed_file()
    out = asyncio.run(g._expand_context(set(seeds), query="", max_expand=10))

    from collections import Counter
    per_file = Counter(c.source_path for c in out)

    # Seed files are already hits, so they are exempt from the cap by design -
    # but expansion must not return only more of what we already had.
    assert len(per_file) > 1, (
        "expansion returned chunks from a single file: {}".format(dict(per_file)))
    new_files = [p for p in per_file if not p.startswith("/repo/crowded")]
    assert new_files, "expansion added no file that was not already a hit"


def test_per_file_cap_applies_to_new_files():
    """A non-hit file may not spend the whole budget either."""
    g = CodeGraph(root_path=".", auto_index=False)
    seed = _chunk("seed", "/repo/a.py", "seed_fn")
    g.chunks[seed.id] = seed
    g.by_file["/repo/a.py"] = ["seed"]

    callers = []
    for i in range(9):
        c = _chunk("c{}".format(i), "/repo/one_other.py", "c{}".format(i))
        g.chunks[c.id] = c
        callers.append(c)
    g.by_file["/repo/one_other.py"] = [c.id for c in callers]

    g.get_context_for_chunk = lambda cid: (  # noqa: E731
        {"callers": callers, "callees": []} if cid == "seed" else
        {"callers": [], "callees": []})

    out = asyncio.run(g._expand_context({"seed"}, query="", max_expand=12))
    from collections import Counter
    per_file = Counter(c.source_path for c in out)
    assert per_file.get("/repo/one_other.py", 0) <= CodeGraph._EXPAND_MAX_PER_FILE, (
        "one non-hit file took {} slots, cap is {}".format(
            per_file.get("/repo/one_other.py"), CodeGraph._EXPAND_MAX_PER_FILE))


def _old_order_expand(g, chunk_ids, max_expand):
    """The pre-fix phase 1, verbatim in shape: siblings, neighbours, THEN calls.

    A mutation guard is only worth having if it actually reproduces the defect.
    This asserts the old ordering really did starve the cross-file edge on this
    fixture, so the passing tests above are evidence rather than decoration.
    """
    expanded, seen = [], set(chunk_ids)
    budget = max_expand // 2 or max_expand
    for cid in list(chunk_ids):
        chunk = g.chunks.get(cid)
        if not chunk or len(expanded) >= budget:
            break
        if chunk.parent_class:
            for oid in g.by_class.get(chunk.parent_class, []):
                other = g.chunks.get(oid)
                if oid not in seen and other and other.chunk_type == ChunkType.METHOD:
                    seen.add(oid)
                    expanded.append(other)
                    if len(expanded) >= budget:
                        break
        for sid in g.by_file.get(chunk.source_path, [])[:8]:
            if sid not in seen and sid in g.chunks:
                seen.add(sid)
                expanded.append(g.chunks[sid])
                if len(expanded) >= budget:
                    break
        ctx = g.get_context_for_chunk(cid)
        for rel in ctx.get("callers", []) + ctx.get("callees", []):
            if rel.id not in seen:
                seen.add(rel.id)
                expanded.append(rel)
                if len(expanded) >= budget:
                    break
    return expanded


def test_mutation_old_order_starves_later_seeds():
    """The old order explored seed #1 and abandoned the rest."""
    g, seeds, callers = _graph_with_crowded_seed_file()
    old = _old_order_expand(g, set(seeds), max_expand=10)
    got = {c.id for c in old}
    reached = [s for s in seeds if callers[s].id in got]

    assert len(reached) <= 1, (
        "the mutation guard no longer reproduces the defect: the old order "
        "reached {} of {} seeds, so the tests above prove nothing".format(
            len(reached), len(seeds)))

    new = asyncio.run(g._expand_context(set(seeds), query="", max_expand=10))
    new_reached = [s for s in seeds if callers[s].id in {c.id for c in new}]
    assert len(new_reached) > len(reached), (
        "the fix must reach strictly more seeds than the old order "
        "({} vs {})".format(len(new_reached), len(reached)))
