"""
Minimal base class for awgraph.

Provides the core interface a faculty graph implements, without the AitherOS
integrations (sync, provenance, integrity). This is what the public package
exposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GraphSyncConfig:
    """Stub for compatibility; sync is not available in the public package."""
    enabled: bool = False
    domain: str = ""
    batch_size: int = 20
    flush_interval: float = 5.0
    source_graph: str = ""
    provenance: bool = False
    provenance_node_type: str = "claim"


class BaseFacultyGraph:
    """
    Minimal base class for a faculty graph.

    In AitherOS, this also provides sync to AitherKnowledgeGraph, provenance
    emission, and integrity tracking. The public package drops those features,
    keeping only the core interface: query(), stats(), and graph_query().
    """

    _sync_config: GraphSyncConfig = GraphSyncConfig()
    _scope_level: str = "platform"

    def __init__(self):
        self._sync_config = GraphSyncConfig()

    def _queue_sync(self, node_data: Dict[str, Any], tenant_id: str = "platform") -> None:
        """No-op: sync to AitherKnowledgeGraph is not available in public package."""
        pass

    def _queue_deletion(self, node_id: str, tenant_id: str = "platform") -> None:
        """No-op: sync is not available in public package."""
        pass

    # SYNC on purpose: every call site is `self._flush_to_bus()` with no await.
    # Declaring it async made each call return a coroutine nobody awaits, which
    # Python reports only as a RuntimeWarning — so the no-op would have looked
    # like it ran while doing nothing, in a package whose whole promise is that
    # the stub is inert rather than silently broken.
    def _flush_to_bus(self, *args: Any, **kwargs: Any) -> None:
        """No-op: there is no event bus in the standalone package.

        This one is not decoration. Indexing calls it, so omitting it made the
        package import cleanly and then die with AttributeError on the FIRST
        index_codebase() call — the shape where a vendored package looks fine to
        every import check and is broken for the person who pip-installed it.
        An import test cannot catch this; only actually indexing something can.

        Subclasses in a host application may override it to re-attach their own
        event plumbing; nothing here needs to know that they exist.
        """
        return None

    def _emit_provenance(
        self,
        node_id: str,
        name: str,
        properties: Dict[str, Any],
        tenant_id: str = "platform",
    ) -> None:
        """No-op: provenance emission is not available in public package."""
        pass

    def graph_stats(self) -> Dict[str, Any]:
        """Inventory: how many nodes does this graph hold?

        Subclasses may override. Default derives counts from public attributes.
        """
        entities: Dict[str, int] = {}
        for name, val in vars(self).items():
            if name.startswith("_") or name.startswith("by_"):
                continue
            if not isinstance(val, (dict, list, set, tuple)):
                continue
            entities[name] = len(val)
        return {"nodes": sum(entities.values()), "containers": entities}

    async def graph_query(
        self, text: str, limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Uniform cross-graph query.

        Adapts this graph's search() or query() method by parameter name.
        Subclasses may override for a better native path.
        """
        import asyncio
        import inspect

        fn = getattr(self, "search", None) or getattr(self, "query", None)
        if not callable(fn):
            return []

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return []

        # Bind by parameter name (not position)
        kwargs = {}
        for p in sig.parameters.values():
            if p.name in ("query", "query_str", "text", "prompt", "search"):
                kwargs[p.name] = text
            elif p.name in ("limit", "max_results", "top_k", "k"):
                kwargs[p.name] = limit

        # Call sync or async
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(**kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, fn, **kwargs)
        except TypeError:
            # Fall back to no kwargs if binding failed
            try:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(text, limit)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: fn(text, limit))
            except Exception:
                return []
        except Exception:
            return []
