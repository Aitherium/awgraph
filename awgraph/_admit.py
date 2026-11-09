"""Host admission and single flight for awgraph's full-index loads.

A full index (``awgraph index``) and a full load (a long-lived reader such as a query
worker) each cost GBs. Measured 2026-11-08 on the owner's desktop: a reindex at 3.4 GB
and rising ran beside a query worker holding 10 GB of the SAME root's index, with the
host's commit charge at 93-96% for hours (the machine has rebooted near 96%). Two rules:

* **admission** -- neither starts while the host's commit charge is at or above
  ``AWGRAPH_ADMIT_COMMIT_PCT`` (default 85; ``0`` disables). The caller defers with the
  reason instead of failing: the next scheduled run, or the next query, tries again.
* **single flight** -- at most one full load of a given root at a time, held by an OS
  file lock beside that root's cache. The OS drops the lock when the holder dies, so a
  crash never leaves it stale.

Commit charge = (commit limit - available commit) / commit limit, the same figure as
Win32_OperatingSystem (TotalVirtualMemorySize - FreeVirtualMemory) / TotalVirtualMemorySize.
Off Windows it is the share of RAM not available (/proc/meminfo).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Callable, Optional

COMMIT_ENV = "AWGRAPH_ADMIT_COMMIT_PCT"
DEFAULT_COMMIT_PCT = 85.0
logger = logging.getLogger("awgraph.admit")
LOCK_NAME = "fullload.lock"
OWNER_NAME = "fullload.owner.json"


def _commit_windows() -> Optional[float]:
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    if not st.ullTotalPageFile:  # the commit limit, despite the field's name
        return None
    return 100.0 * (st.ullTotalPageFile - st.ullAvailPageFile) / st.ullTotalPageFile


def _commit_proc() -> Optional[float]:
    vals = {}
    with open("/proc/meminfo", encoding="ascii") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                vals[key] = float(rest.split()[0])
    total, avail = vals.get("MemTotal"), vals.get("MemAvailable")
    if not total or avail is None:
        return None
    return 100.0 * (total - avail) / total


def host_commit_pct() -> Optional[float]:
    """The host's commit charge in percent, or None when it cannot be read."""
    try:
        if sys.platform == "win32":
            return _commit_windows()
        return _commit_proc()
    except (OSError, ValueError, AttributeError):
        return None


def commit_limit_pct() -> float:
    try:
        return float(os.environ.get(COMMIT_ENV, DEFAULT_COMMIT_PCT))
    except ValueError:
        return DEFAULT_COMMIT_PCT


def admit(what: str, pct_fn: Callable[[], Optional[float]] = host_commit_pct,
          limit: Optional[float] = None) -> str:
    """"" when ``what`` may start now, else the reason it is deferred.

    An unreadable commit charge admits: the guard must never be the reason nothing
    ever runs on a platform it cannot measure.
    """
    limit = commit_limit_pct() if limit is None else limit
    if limit <= 0:
        return ""
    pct = pct_fn()
    if pct is None or pct < limit:
        return ""
    return "deferred: host commit %.0f%% >= %g%% (%s) -- %s not started" % (
        pct, limit, COMMIT_ENV, what)


class FullLoadLock:
    """An exclusive, non-blocking, crash-safe lock on one root's full load."""

    def __init__(self, root: str, kind: str, cache_dir: Optional[str] = None):
        if cache_dir is None:
            from awgraph.graph import _get_data_path  # noqa: PLC0415 -- heavy module

            cache_dir = os.path.dirname(_get_data_path(root, LOCK_NAME))
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, LOCK_NAME)
        self.owner_path = os.path.join(cache_dir, OWNER_NAME)
        self.kind = kind
        self._fh = None

    def holder(self) -> str:
        """Who holds the lock, from the owner note (best effort)."""
        try:
            with open(self.owner_path, encoding="utf-8") as fh:
                o = json.load(fh)
            return "%s pid %s for %.0fs" % (o.get("kind"), o.get("pid"),
                                             time.time() - float(o.get("since") or 0))
        except (OSError, ValueError, TypeError):
            return "another process"

    def acquire(self, note: bool = True) -> str:
        """"" once held, else "busy: <holder>". Never blocks. ``note`` writes the owner
        note other callers quote; a probe passes False so it never overwrites it."""
        fh = open(self.path, "a+b")  # noqa: SIM115 -- held until release()
        try:
            fh.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return "busy: a full load of this root is running (%s)" % self.holder()
        self._fh = fh
        self._noted = note
        if not note:
            return ""
        try:
            with open(self.owner_path, "w", encoding="utf-8") as out:
                json.dump({"kind": self.kind, "pid": os.getpid(), "since": time.time()}, out)
        except OSError as exc:  # the note is diagnostics only; the lock is what is held
            logger.debug("awgraph: owner note not written: %s", exc)
        return ""

    def held_elsewhere(self) -> bool:
        """True while ANOTHER holder has the lock (probes by acquiring and releasing)."""
        if self._fh is not None:
            return False
        if self.acquire(note=False):
            return True
        self.release()
        return False

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        if getattr(self, "_noted", False):
            try:
                os.remove(self.owner_path)  # the note names a holder that is leaving
            except OSError as exc:
                logger.debug("awgraph: owner note not removed: %s", exc)
        try:
            fh.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError as exc:  # closing the handle below releases it regardless
            logger.debug("awgraph: unlock failed: %s", exc)
        finally:
            fh.close()

    def __enter__(self) -> "FullLoadLock":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
