"""The host integration points must actually take effect.

`awgraph/__init__.py` advertised "Plugin support via awgraph.plugins" from 1.0.0
while the module did not exist and `plugins/` was an empty directory. This
asserts the capability is real, because an advertised hook that silently does
nothing is worse than no hook: a host configures it, believes the engine is using
its logger / client / embedder, and never finds out otherwise.
"""

from __future__ import annotations

# isort: skip_file
#
# The repo's ruff config knows `awgraph` as FIRST-party and wants it in its own
# block; the quality gate runs ruff `--isolated` (deliberately, so the ambient
# per-file-ignores cannot hide E501/F-rules) where `awgraph` reads as THIRD-party
# and belongs beside pytest. Those two layouts are mutually exclusive — one block
# or two — so no ordering satisfies both and this states the conflict instead of
# flip-flopping the file every time whichever gate ran last disagreed.
import logging

import pytest

import awgraph.plugins as plugins


@pytest.fixture(autouse=True)
def _clean_hooks():
    plugins.reset()
    yield
    plugins.reset()


def test_defaults_are_the_standalone_behaviour():
    """No host configured: stdlib logging, httpx, no embeddings."""
    import httpx

    assert isinstance(plugins.get_logger("x"), logging.Logger)
    assert plugins.async_client_class() is httpx.AsyncClient
    assert plugins.embedding_engine() is None
    assert plugins.embeddings_configured() is False
    assert plugins.degradation_registry() is not None


def test_logger_hook_is_used():
    seen = []

    def factory(name):
        seen.append(name)
        return logging.getLogger("host." + name)

    plugins.configure(logger_factory=factory)
    got = plugins.get_logger("CodeGraph")
    assert seen == ["CodeGraph"], "the host factory was never called"
    assert got.name == "host.CodeGraph"


def test_async_client_hook_is_used():
    class HostClient:
        pass

    plugins.configure(async_client=HostClient)
    assert plugins.async_client_class() is HostClient


def test_embedding_hook_is_used_and_reported():
    class Engine:
        pass

    made = Engine()
    plugins.configure(embedding_engine=lambda: made)
    assert plugins.embedding_engine() is made
    assert plugins.embeddings_configured() is True, (
        "a host installed an embedding engine and embeddings_configured() still "
        "said no — this is the flag hosts are told to assert at startup"
    )


def test_degradation_hook_is_used():
    class Registry:
        pass

    reg = Registry()
    plugins.configure(degradation_registry=reg)
    assert plugins.degradation_registry() is reg


def test_unknown_hook_raises_rather_than_being_ignored():
    """A typo'd hook must be loud.

    Silently accepting `embeddings_engine=` (note the s) would leave the host
    believing semantic search is on while the engine runs keyword-only — the
    exact silent degradation this module exists to make impossible.
    """
    with pytest.raises(ValueError) as excinfo:
        plugins.configure(embeddings_engine=lambda: None)
    message = str(excinfo.value)
    assert "embeddings_engine" in message, "the error must name the bad hook"
    assert "embedding_engine" in message, "the error must list the valid names"


def test_reset_clears_hooks():
    plugins.configure(embedding_engine=lambda: object())
    assert plugins.embeddings_configured() is True
    plugins.reset()
    assert plugins.embeddings_configured() is False
    assert all(v is False for v in plugins.active_hooks().values())


def test_engine_reads_the_hooks_at_import():
    """The control that matters: the ENGINE, not just the registry, uses them.

    Every assertion above could pass while `graph.py` still imported httpx and
    stdlib logging directly — which is what the three forks of this engine did.
    """
    import importlib

    class HostClient:
        pass

    plugins.configure(async_client=HostClient,
                      embedding_engine=lambda: "host-engine",
                      logger_factory=lambda name: logging.getLogger("host." + name))
    import awgraph.graph as graph

    importlib.reload(graph)
    try:
        assert graph.AsyncClient is HostClient, (
            "graph.py did not pick up the async_client hook"
        )
        assert graph.get_embedding_engine() == "host-engine", (
            "graph.py did not pick up the embedding_engine hook"
        )
        assert graph._HAS_EMBEDDING_ENGINE is True
        assert graph.logger.name == "host.CodeGraph"
    finally:
        plugins.reset()
        importlib.reload(graph)
