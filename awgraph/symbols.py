"""A small, edge-only sidecar index, so ``callers`` costs a lookup, not a load.

WHY THIS EXISTS
---------------
``callers`` and ``calls`` need three things about one symbol: where it is, what
it calls, and what calls it. None of that is a vector. Yet every read command
went through the same door — deserialise the whole chunk cache, then attach
every persisted embedding — because that door was written for ``query``, which
genuinely does need the matrix.

Measured on a 387,098-chunk index: 19.9 s to rebuild the chunk objects and
34.7 s to hydrate a 2.6 GB embedding cache, in front of a 0.2 s symbol scan.
Fifty-five seconds of work to answer a question that touches neither structure.
A tool that takes most of a minute to say "no callers" is one nobody runs twice,
and the usage numbers said exactly that.

So the edges get their own store: one SQLite file, one row per symbol, the two
edge lists carried as JSON on that row. It is built from whatever already
exists — a freshly indexed graph, or the persisted chunk cache — and it is
rebuilt automatically whenever the cache it was derived from moves underneath
it. Nothing here is authoritative; it is a projection, and a stale projection is
detected rather than served.

The build is also where JavaScript/TypeScript edges are derived (see
``jscalls``), which is what lets an existing index gain them without being
re-parsed from source.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

from awgraph import jscalls
from awgraph.logging import get_logger

logger = get_logger(__name__)

#: Bumped whenever the table shape changes. A store written by an older schema
#: is rebuilt rather than queried, because a missing column would surface as an
#: empty answer — indistinguishable from "this symbol has no callers".
SCHEMA_VERSION = 2

#: How many edges are kept per symbol per direction. The TRUE count is stored
#: beside the list and reported, so a truncated answer says it was truncated —
#: a silently shortened caller list is the one failure mode worse than a slow
#: one, because it reads as a complete answer.
#:
#: The ceiling exists because name-matched call graphs have pathological nodes.
#: Measured on a 387k-chunk index: a vendored class named ``str`` collected
#: 19,018 callers from every ordinary ``str(...)`` in the tree, and edge lists
#: like it were 422 MB of a 600 MB store. Nobody reads the 19,018th caller.
MAX_STORED_EDGES = 500

STORE_FILENAME = "codegraph_symbols.sqlite"

#: Re-exported so the store and the in-process backfill cannot drift apart on
#: the one policy question that decides whether an edge is drawn at all.
MAX_DEFINITION_CANDIDATES = jscalls.MAX_DEFINITION_CANDIDATES

#: Ceiling on how many callers one symbol may collect from derived edges. A
#: utility named ``get`` in a hundred files would otherwise produce a list no
#: human reads, and it would be mostly wrong.
MAX_DERIVED_FAN_IN = 400

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS symbols (
    chunk_id  TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    short     TEXT NOT NULL,
    kind      TEXT,
    path      TEXT,
    line      INTEGER,
    signature TEXT,
    calls     TEXT,
    called_by TEXT,
    calls_n     INTEGER,
    called_by_n INTEGER
);
CREATE INDEX IF NOT EXISTS ix_symbols_name  ON symbols(name);
CREATE INDEX IF NOT EXISTS ix_symbols_short ON symbols(short);
"""


def store_path(root_path: str) -> str:
    """Where the sidecar for ``root_path`` lives (beside the chunk cache)."""
    from awgraph.graph import _get_data_path  # noqa: PLC0415 - avoids a cycle

    return _get_data_path(root_path, STORE_FILENAME)


def source_stamp(path: str) -> str:
    """A cheap identity for the file this store was derived from.

    Size and mtime rather than a digest: the chunk cache is hundreds of
    megabytes, and hashing it on every read would reintroduce the very cost
    this store exists to avoid.
    """
    try:
        st = os.stat(path)
    except OSError:
        return ""
    return "{0}:{1}".format(int(st.st_mtime), st.st_size)


def _read_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.Error:
        return {}
    return {str(k): str(v) for k, v in rows}


def is_fresh(path: str, derived_from: str) -> bool:
    """True when ``path`` is a usable store built from the current source."""
    if not os.path.exists(path):
        return False
    want = source_stamp(derived_from) if derived_from else ""
    try:
        conn = sqlite3.connect("file:{0}?mode=ro".format(path), uri=True)
    except sqlite3.Error:
        return False
    try:
        meta = _read_meta(conn)
        if meta.get("schema") != str(SCHEMA_VERSION):
            return False
        if want and meta.get("source_stamp") != want:
            return False
        return meta.get("complete") == "1"
    finally:
        conn.close()


# ── building ─────────────────────────────────────────────────────────────


def _short(name: str) -> str:
    return str(name or "").split(".")[-1]


def derive_js_edges(chunks: Iterable[Any]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return ``(calls_by_id, called_by_by_id)`` for JS-family chunks.

    Resolution is deliberately conservative. A derived call binds to a
    definition in the SAME file whenever one exists; otherwise it binds across
    files only when the name is distinctive enough to mean something and few
    enough definitions answer to it. Anything else is left as an unresolved
    call name, which the caller still reports — an unresolved edge is
    information, a wrong one is not.
    """
    js_chunks = [c for c in chunks if jscalls.is_js_source(getattr(c, "source_path", ""))]
    by_short: Dict[str, List[Any]] = {}
    for chunk in js_chunks:
        by_short.setdefault(_short(getattr(chunk, "name", "")), []).append(chunk)

    calls_by_id: Dict[str, List[str]] = {}
    called_by_by_id: Dict[str, List[str]] = {}
    fan_in: Dict[str, int] = {}

    for caller in js_chunks:
        body = getattr(caller, "body_preview", "") or ""
        if not body:
            continue
        names = jscalls.extract_calls(body, getattr(caller, "name", ""))
        if not names:
            continue
        calls_by_id[caller.id] = names
        caller_name = getattr(caller, "name", "")
        caller_path = getattr(caller, "source_path", "")
        for name in names:
            candidates = by_short.get(name)
            if not candidates:
                continue
            same_file = [c for c in candidates if getattr(c, "source_path", "") == caller_path]
            if same_file:
                targets = same_file
            elif (len(name) >= jscalls.MIN_CROSS_FILE_NAME
                    and len(candidates) <= MAX_DEFINITION_CANDIDATES):
                targets = candidates
            else:
                continue
            for callee in targets:
                if callee.id == caller.id:
                    continue  # a definition is never its own caller
                if fan_in.get(callee.id, 0) >= MAX_DERIVED_FAN_IN:
                    continue
                bucket = called_by_by_id.setdefault(callee.id, [])
                if caller_name and caller_name not in bucket:
                    bucket.append(caller_name)
                    fan_in[callee.id] = fan_in.get(callee.id, 0) + 1
    return calls_by_id, called_by_by_id


def build(chunks: Iterable[Any], path: str, derived_from: str = "") -> Dict[str, int]:
    """Write the sidecar for ``chunks`` to ``path``. Returns a small summary.

    Written to a temporary file and renamed, so a reader never sees a half-built
    store and an interrupted build leaves the previous one in place.
    """
    chunk_list = list(chunks)
    js_calls, js_called_by = derive_js_edges(chunk_list)

    # The temp name carries the pid: several agent sessions ask the same
    # repository the same question at once, and a shared ".building" file
    # means two processes writing one SQLite database. The loser used to
    # corrupt the winner's store, which then failed its freshness check and
    # was rebuilt — so the bug showed up only as "this is slow again".
    tmp = "{0}.{1}.building".format(path, os.getpid())
    for leftover in (tmp, tmp + "-journal"):
        if os.path.exists(leftover):
            try:
                os.unlink(leftover)
            except OSError as exc:
                # A previous build died and something still holds its temp
                # file. sqlite3.connect below will say so precisely; this is
                # logged rather than swallowed so the CAUSE is not invisible.
                logger.warning("[symbols] could not remove %s: %s", leftover, exc)

    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(_SCHEMA)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        rows = []
        n_edges = 0
        for chunk in chunk_list:
            cid = getattr(chunk, "id", "")
            name = getattr(chunk, "name", "") or ""
            calls = list(getattr(chunk, "calls", None) or []) or js_calls.get(cid, [])
            called_by = list(getattr(chunk, "called_by", None) or [])
            for extra in js_called_by.get(cid, []):
                if extra not in called_by:
                    called_by.append(extra)
            n_edges += len(calls) + len(called_by)
            rows.append((
                cid,
                name,
                _short(name),
                getattr(getattr(chunk, "chunk_type", None), "value", "") or "",
                str(getattr(chunk, "source_path", "") or ""),
                int(getattr(chunk, "start_line", 0) or 0),
                getattr(chunk, "signature", "") or "",
                json.dumps(calls[:MAX_STORED_EDGES]),
                json.dumps(called_by[:MAX_STORED_EDGES]),
                len(calls),
                len(called_by),
            ))
        conn.executemany(
            "INSERT OR REPLACE INTO symbols "
            "(chunk_id, name, short, kind, path, line, signature, calls, called_by, "
            "calls_n, called_by_n) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        meta = {
            "schema": str(SCHEMA_VERSION),
            "source_stamp": source_stamp(derived_from) if derived_from else "",
            "symbols": str(len(rows)),
            "js_edge_chunks": str(len(js_calls)),
            "complete": "1",
        }
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", list(meta.items())
        )
        conn.commit()
    finally:
        conn.close()

    if os.path.exists(path):
        try:
            os.unlink(path)
        except OSError as exc:
            # On Windows a concurrent reader holding the old store blocks the
            # unlink. os.replace below is the real attempt; this says what
            # preceded it if that one also fails.
            logger.debug("[symbols] could not unlink the previous store: %s", exc)
    try:
        os.replace(tmp, path)
    except OSError as exc:
        # Another process published its own build first, or still has the old
        # store open. Either way a valid store is (or will be) in place and
        # the caller falls back for this one call. Drop our copy rather than
        # leaving a few hundred megabytes of orphan behind.
        logger.warning("[symbols] could not publish the store: %s", exc)
        try:
            os.unlink(tmp)
        except OSError as cleanup_exc:
            logger.warning("[symbols] orphan build file left at %s: %s", tmp, cleanup_exc)
        raise
    return {
        "symbols": len(rows),
        "edges": n_edges,
        "js_edge_chunks": len(js_calls),
    }


# ── reading ──────────────────────────────────────────────────────────────


def _row_to_dict(row: sqlite3.Row, root: str) -> Dict[str, Any]:
    path = str(row["path"] or "")
    if root and path:
        try:
            path = os.path.relpath(path, root)
        except (ValueError, OSError) as exc:
            # Different drives on Windows, or a path that no longer resolves.
            # The absolute path is still a correct answer, so it is kept.
            logger.debug("[symbols] keeping absolute path for %s: %s", path, exc)
    return {
        "name": row["name"],
        "type": row["kind"] or "",
        "path": path,
        "line": row["line"] or 0,
        "signature": row["signature"] or "",
    }


def lookup(
    path: str, symbol: str, direction: str, root: str = ""
) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Resolve one edge question from the sidecar.

    Returns ``None`` when ``symbol`` is not in the store at all — the caller
    must distinguish "unknown symbol" from "known symbol, no edges", because
    only the second is a real negative answer.

    Otherwise returns ``(rows, total)``. ``total`` is how many edges the symbol
    actually has; it exceeds ``len(rows)`` when the list hit
    ``MAX_STORED_EDGES``, and the caller is expected to say so rather than
    present a truncated list as the whole answer.
    """
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect("file:{0}?mode=ro".format(path), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        matches = conn.execute(
            "SELECT * FROM symbols WHERE name = ?", (symbol,)
        ).fetchall()
        if not matches:
            matches = conn.execute(
                "SELECT * FROM symbols WHERE short = ?", (symbol,)
            ).fetchall()
        if not matches:
            return None

        column = "called_by" if direction == "callers" else "calls"
        # Truncation is decided by whether any single symbol's stored list hit
        # the ceiling — NOT by comparing counts. Two chunks share a name here
        # (an overload, a re-export), their edge lists overlap, and summing the
        # per-row counts made every ordinary answer look truncated.
        capped = any(int(row[column + "_n"] or 0) > MAX_STORED_EDGES for row in matches)
        stored_total = sum(int(row[column + "_n"] or 0) for row in matches)
        wanted: List[str] = []
        seen = set()
        for row in matches:
            for name in json.loads(row[column] or "[]"):
                if name not in seen:
                    seen.add(name)
                    wanted.append(name)
        if not wanted:
            return [], stored_total if capped else 0

        found: Dict[str, sqlite3.Row] = {}
        step = 400  # stay under SQLite's variable limit on a wide fan-out
        for start in range(0, len(wanted), step):
            batch = wanted[start:start + step]
            marks = ",".join("?" for _ in batch)
            for row in conn.execute(
                "SELECT * FROM symbols WHERE name IN ({0})".format(marks), batch
            ):
                found.setdefault(row["name"], row)

        out: List[Dict[str, Any]] = []
        for name in wanted:
            row = found.get(name)
            if row is not None:
                out.append(_row_to_dict(row, root))
            else:
                # Outside the indexed tree (a runtime or third-party call).
                # Reported rather than dropped: showing only resolvable edges
                # understates the fan-out.
                out.append({"name": name, "type": "external", "path": "",
                            "line": 0, "signature": ""})
        return out, (max(stored_total, len(out)) if capped else len(out))
    finally:
        conn.close()


def summary(path: str) -> Dict[str, str]:
    """Metadata for a store, or ``{}`` when it is absent or unreadable."""
    if not os.path.exists(path):
        return {}
    try:
        conn = sqlite3.connect("file:{0}?mode=ro".format(path), uri=True)
    except sqlite3.Error:
        return {}
    try:
        return _read_meta(conn)
    finally:
        conn.close()
