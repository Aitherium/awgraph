"""Logging shim for awgraph.

Replaces lib.agents.AitherChronicle with Python's standard logging.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name (standard Python logging)."""
    return logging.getLogger(name)
