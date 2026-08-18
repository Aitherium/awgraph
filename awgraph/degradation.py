"""DegradationRegistry shim for awgraph.

Replaces lib.core.DegradationRegistry with a no-op that tracks optional
dependencies gracefully.
"""

from enum import Enum
from typing import Any, Optional


class SubsystemTier(Enum):
    """Severity of a subsystem failure."""
    CORE = "core"
    CRITICAL = "critical"
    COGNITIVE = "cognitive"
    AUXILIARY = "auxiliary"


class DegradationRegistry:
    """Track optional dependency availability.

    No-op in the public package; just silently tracks OK/failed states.
    """

    def __init__(self):
        self._ok: dict[str, Any] = {}
        self._failed: dict[str, Any] = {}

    def register_ok(self, name: str, module: str, tier: SubsystemTier) -> None:
        """Mark a subsystem as OK."""
        self._ok[name] = {"module": module, "tier": tier}

    def register_failed(
        self, name: str, module: str, error: Exception, tier: SubsystemTier,
    ) -> None:
        """Mark a subsystem as failed."""
        self._failed[name] = {"module": module, "error": str(error), "tier": tier}


_GLOBAL_REGISTRY: Optional[DegradationRegistry] = None


def get_registry() -> DegradationRegistry:
    """Get the global degradation registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = DegradationRegistry()
    return _GLOBAL_REGISTRY
