"""Which embedding space an index was built in, and which one is active now.

Vectors from two embedding models are not comparable. A query embedded by one
model scored against an index embedded by another either fails on width (768-d
against 1024-d) or, when the widths happen to agree, returns cosines that mean
nothing while every call still succeeds. So an index records the space it was
embedded in, and a search in a different space is REFUSED, never answered.

The platform switch is one environment key, ``AITHER_EMBED_SPACE`` -- the same
key and aliases as AitherOS ``lib/core/EmbeddingSpace.py`` and the SDK's
``adk/embeddings.py``, restated here because awgraph ships without either:

  unset                  today's standalone behaviour: awgraph names no model of
                         its own; the lane you configured decides.
  ``nomic``              ``nomic-embed-text``, 768-d.
  ``aither-code-embed``  ``aither-code-embed``, 1024-d; a request or a config
                         that still names a nomic model is served by the
                         platform model instead (that lane is retired there).

An unknown value raises, exactly as the platform does: guessing a space corrupts
an index silently.

ABOUT BRIDGES. The platform can register an orthonormal bridge file that maps
one space onto another (``EmbeddingSpace.bridge_path(src, dst)``). Do NOT use
one here to search an index from another space. It exists for identity joins
only; measured, the top-10 of a bridged search overlaps the native top-10 by
37-39%. A space mismatch in awgraph has one remedy: re-embed the index.

Pure functions and one small JSON file; no imports beyond the standard library.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Mapping, Optional, Tuple

SPACE_NOMIC = "nomic"
SPACE_CODE_EMBED = "aither-code-embed"
_SPACES = (SPACE_NOMIC, SPACE_CODE_EMBED)
_SPACE_ALIASES = ("code-embed", "ce1024", "aither-code-embed-1024")

NOMIC_MODEL = "nomic-embed-text"
CODE_EMBED_MODEL = "aither-code-embed"

#: Label for a host-installed engine when no space is declared: the host picks
#: the model and awgraph cannot name it. Only the measured width is checked then.
HOST_ENGINE_MODEL = "host-engine"

#: ``<embedding cache>.space.json`` -- written beside the vector cache.
RECORD_VERSION = 1
RECORD_SUFFIX = ".space.json"

MATCH = "match"
MISMATCH = "mismatch"
UNKNOWN = "unknown"

REBUILD_HINT = (
    "Rebuild the vectors in the active space with "
    "CodeGraph.embed_chunks(force=True), or point the environment back at the "
    "embedder the index was built with (AITHER_EMBED_SPACE, "
    "AITHER_CODEGRAPH_EMBED_MODEL)."
)


class EmbeddingSpaceMismatch(RuntimeError):
    """The index and the active embedder are in different spaces: search refused."""


def declared_space(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The space ``AITHER_EMBED_SPACE`` declares, or None when it is unset."""
    env = os.environ if environ is None else environ
    raw = (env.get("AITHER_EMBED_SPACE", "") or "").strip().lower()
    if not raw:
        return None
    if raw in _SPACE_ALIASES:
        raw = SPACE_CODE_EMBED
    if raw not in _SPACES:
        raise ValueError(
            f"AITHER_EMBED_SPACE={raw!r} is not one of {_SPACES} -- refusing to guess "
            "an embedding space (a wrong guess corrupts the index silently)"
        )
    return raw


def model_family(name: Optional[str]) -> str:
    """Served ids that are one model compare equal (``nomic-embed-text:latest``,
    ``nomic-ai/nomic-embed-text-v1.5``; a GGUF build of the same weights)."""
    n = (name or "").strip().lower()
    if not n:
        return ""
    if n.startswith("nomic") or "nomic-embed-text" in n:
        return NOMIC_MODEL
    if n.startswith((CODE_EMBED_MODEL, "code-embed")):
        return CODE_EMBED_MODEL
    return n


def platform_model(space: Optional[str],
                   environ: Optional[Mapping[str, str]] = None) -> str:
    """The model id a space serves. Unset keeps the pre-space default name."""
    env = os.environ if environ is None else environ
    if space == SPACE_CODE_EMBED:
        name = (env.get("AITHER_PLATFORM_EMBED_MODEL", "") or "").strip()
        if not name or model_family(name) == NOMIC_MODEL:
            name = CODE_EMBED_MODEL
        return name
    return NOMIC_MODEL


def resolve_model(requested: Optional[str], space: Optional[str],
                  environ: Optional[Mapping[str, str]] = None) -> str:
    """Model id to send: empty -> the space's model; in the code-embed space a
    nomic id is retired config and becomes the platform model too."""
    name = (requested or "").strip()
    if not name:
        return platform_model(space, environ)
    if space == SPACE_CODE_EMBED and model_family(name) == NOMIC_MODEL:
        return platform_model(space, environ)
    return name


def known_dim(model: Optional[str], space: Optional[str],
              environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """Width of ``model`` when the DECLARED space states it, else None.

    Only a declared space is believed. With none, awgraph does not guess a width
    from a model name (a wrong guess would refuse a healthy index): the width is
    measured from the vectors instead."""
    env = os.environ if environ is None else environ
    if not space or model_family(model) != model_family(platform_model(space, env)):
        return None
    raw = (env.get("AITHER_PLATFORM_EMBED_DIM", "") or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 1024 if space == SPACE_CODE_EMBED else 768


def prefix_scheme(query_prefix: Optional[str]) -> str:
    """Documents always embed plain; a query may carry an instruction prefix.

    The prefix is named by a digest so two schemes can be told apart without
    writing somebody's prompt into an index file."""
    if not query_prefix:
        return "doc:plain|query:plain"
    digest = hashlib.sha256(query_prefix.encode("utf-8", "replace")).hexdigest()[:12]
    return "doc:plain|query:prefix-sha256:" + digest


def describe(space: Optional[Mapping[str, Any]]) -> str:
    """One human phrase for a space: ``aither-code-embed 1024-d``."""
    if not space:
        return "unknown space (no space record)"
    model = space.get("model") or "no embedder"
    if model == HOST_ENGINE_MODEL:
        model = "host engine (model chosen by the host)"
    dim = space.get("dim")
    return "{0} {1}".format(model, "%d-d" % dim if dim else "(width not yet measured)")


def compare(index: Optional[Mapping[str, Any]],
            active: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    """``(verdict, reason)`` for an index space against the active one.

    Only a PROVEN difference is a mismatch: two named models of different
    families, or two known widths that differ. A query-prefix difference is
    reported and never refused -- documents embed plain under every scheme, so
    the stored vectors are the same; only the query side is tuned."""
    if not index:
        return UNKNOWN, "the index carries no space record"
    if not active or not active.get("model"):
        return UNKNOWN, "no embedding lane is active"
    i_model, a_model = model_family(index.get("model")), model_family(active.get("model"))
    named = bool(i_model and a_model) and HOST_ENGINE_MODEL not in (i_model, a_model)
    if named and i_model != a_model:
        return MISMATCH, "model {0} != {1}".format(index.get("model"), active.get("model"))
    i_dim, a_dim = index.get("dim"), active.get("dim")
    if i_dim and a_dim and int(i_dim) != int(a_dim):
        return MISMATCH, "width {0}-d != {1}-d".format(i_dim, a_dim)
    if not named:
        return UNKNOWN, "the host engine does not name its model; only width is checked"
    i_pre, a_pre = index.get("prefix_scheme"), active.get("prefix_scheme")
    if i_pre and a_pre and i_pre != a_pre:
        return MATCH, "query prefix changed since the index was built (documents unaffected)"
    return MATCH, ""


def mismatch_message(index: Optional[Mapping[str, Any]],
                     active: Optional[Mapping[str, Any]], reason: str,
                     action: str = "search") -> str:
    return (
        "refusing to {0} across embedding spaces -- the index was embedded with "
        "{1}, the active embedder is {2} ({3}). Vectors from two models are not "
        "comparable. {4}".format(
            action, describe(index), describe(active), reason, REBUILD_HINT)
    )


def record_path(cache_path: str) -> str:
    return cache_path + RECORD_SUFFIX


def _source_stamp(path: str) -> list:
    st = os.stat(path)
    return [st.st_mtime, float(st.st_size)]


def write_record(cache_path: str, space: Mapping[str, Any]) -> bool:
    """Record the space of the vector cache at ``cache_path``. Never raises.

    Stamped with the cache file (mtime, size): a cache rewritten by anything
    that does not know this record makes it stale, and a stale record is read
    as "unknown space" rather than trusted."""
    try:
        body: Dict[str, Any] = {
            "version": RECORD_VERSION,
            "model": space.get("model"),
            "dim": space.get("dim"),
            "prefix_scheme": space.get("prefix_scheme"),
            "lane": space.get("lane"),
            "space": space.get("space"),
            "source": _source_stamp(cache_path),
            "written_at": time.time(),
        }
        path = record_path(cache_path)
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(body, fh)
        os.replace(path + ".tmp", path)
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_record(cache_path: str) -> Optional[Dict[str, Any]]:
    """The CURRENT space record of ``cache_path``, else None (absent, unreadable,
    another version, or written for a different cache file)."""
    try:
        with open(record_path(cache_path), encoding="utf-8") as fh:
            info = json.load(fh)
        if not isinstance(info, dict) or info.get("version") != RECORD_VERSION:
            return None
        if info.get("source") != _source_stamp(cache_path):
            return None
        if not info.get("model"):
            return None
        return info
    except (OSError, ValueError, TypeError):
        return None


def drop_record(cache_path: str) -> None:
    """Remove a record that no longer describes the cache. Never raises."""
    try:
        os.remove(record_path(cache_path))
    except OSError:
        return


__all__ = [
    "CODE_EMBED_MODEL",
    "EmbeddingSpaceMismatch",
    "HOST_ENGINE_MODEL",
    "MATCH",
    "MISMATCH",
    "NOMIC_MODEL",
    "REBUILD_HINT",
    "SPACE_CODE_EMBED",
    "SPACE_NOMIC",
    "UNKNOWN",
    "compare",
    "declared_space",
    "describe",
    "drop_record",
    "known_dim",
    "mismatch_message",
    "model_family",
    "platform_model",
    "prefix_scheme",
    "read_record",
    "record_path",
    "resolve_model",
    "write_record",
]
