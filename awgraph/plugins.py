"""Host integration points for awgraph.

awgraph runs standalone with stdlib defaults: `logging` for logs, `httpx` for
HTTP, an in-process degradation registry, and **no embedding backend**. A host
application that already has richer versions of those can inject them here
instead of forking the engine — which is the point.

    import awgraph.plugins as plugins

    plugins.configure(
        logger_factory=my_get_logger,      # (name) -> logger
        async_client=MyAsyncClient,        # httpx.AsyncClient-compatible class
        embedding_engine=my_get_engine,    # () -> engine with .embed/.embed_batch
        degradation_registry=my_registry,  # object with register_ok/register_failed
    )

Call `configure()` BEFORE importing `awgraph.graph`, or re-import after, since
the engine reads these once at module import.

## Why this module exists

Its absence is what produced three independent copies of this engine. The
AitherOS monorepo needed a Chronicle logger, an internal-CA HTTP client and a
real embedding backend, so it forked the file and substituted the imports; the
ADK forked it again. Nothing compared them, and a fix in one reached the others
only if somebody remembered. The same shape in a sibling package left **14 of 17
modules drifted**, including one that had lost a guard and started recording
every function in a conflicted file as deleted.

Note that `awgraph/__init__.py` has advertised "Plugin support via
awgraph.plugins" since 1.0.0 while this module did not exist and `plugins/` was
an empty directory — an advertised capability that was simply absent.

## The embedding hook is the one that bites

`get_embedding_engine()` returns `None` by default, and the engine correctly
degrades to keyword-only scoring when it does. That degradation is SILENT: a
query still returns ten confident-looking results. A host that means to have
semantic search and forgets this hook gets a working-looking system with the
semantic half switched off — measured on this codebase, the difference is
**+0.121 recall at k=25**.

So `embeddings_configured()` exists to be asserted, not merely consulted. A host
that requires semantic search should fail its own startup on it rather than
discover the loss from a benchmark months later.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

__all__ = [
    "configure",
    "reset",
    "active_hooks",
    "embeddings_configured",
    "get_logger",
    "async_client_class",
    "embedding_engine",
    "degradation_registry",
]

# Every supported hook, with the reason a host would set it. A name not in here
# is rejected by configure(): a typo'd hook that silently did nothing would be
# indistinguishable from a host that never configured it, which is precisely the
# failure this module exists to prevent.
_SUPPORTED: Dict[str, str] = {
    "logger_factory": "(name) -> logger; default: logging.getLogger",
    "async_client": "httpx.AsyncClient-compatible class; default: httpx.AsyncClient",
    "embedding_engine": "() -> engine with .embed/.embed_batch; default: None",
    "degradation_registry": "object with register_ok/register_failed",
}

_HOOKS: Dict[str, Optional[Any]] = {name: None for name in _SUPPORTED}


def configure(**hooks: Any) -> None:
    """Install host implementations. Unknown names raise; None clears a hook."""
    unknown = sorted(set(hooks) - set(_SUPPORTED))
    if unknown:
        raise ValueError(
            "awgraph.plugins.configure() got unsupported hook(s): "
            + ", ".join(unknown)
            + ". Supported: "
            + ", ".join(sorted(_SUPPORTED))
            + ". (Rejected rather than ignored — a typo'd hook that silently did "
            "nothing looks exactly like never configuring one.)"
        )
    _HOOKS.update(hooks)


def reset() -> None:
    """Drop every hook. For tests, and for a host tearing itself down."""
    for name in _HOOKS:
        _HOOKS[name] = None


def active_hooks() -> Dict[str, bool]:
    """Which hooks a host has installed. For startup assertions and diagnostics."""
    return {name: _HOOKS[name] is not None for name in sorted(_SUPPORTED)}


def embeddings_configured() -> bool:
    """True when a host has supplied an embedding backend.

    Worth asserting at host startup: without it, hybrid search silently degrades
    to keyword-only and still returns confident results.
    """
    return _HOOKS["embedding_engine"] is not None


def get_logger(name: str):
    """The host's logger factory, else stdlib logging."""
    factory: Optional[Callable[[str], Any]] = _HOOKS["logger_factory"]
    if factory is not None:
        return factory(name)
    import logging

    return logging.getLogger(name)


def async_client_class():
    """The host's async HTTP client class, else `httpx.AsyncClient`.

    A host with an internal CA injects its own client here; without it, calls to
    an internal TLS endpoint fail certificate verification, which surfaces as the
    peer being unreachable rather than as a trust problem.
    """
    client = _HOOKS["async_client"]
    if client is not None:
        return client
    import httpx

    return httpx.AsyncClient


def embedding_engine():
    """The host's embedding engine, or None.

    None is a valid, supported configuration — the engine falls back to keyword
    scoring. It is also silent, which is why `embeddings_configured()` exists.
    """
    factory = _HOOKS["embedding_engine"]
    if factory is None:
        return None
    return factory()


def degradation_registry():
    """The host's degradation registry, else awgraph's own."""
    registry = _HOOKS["degradation_registry"]
    if registry is not None:
        return registry
    from awgraph.degradation import get_registry

    return get_registry()
