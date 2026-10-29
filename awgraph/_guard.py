"""Process guard for the read commands: an awgraph process must never outlive its caller.

Measured 2026-10-29: an ``awgraph stats`` held 11.9 GB of private memory, idle, for 50
minutes after the agent that started it had moved on, and with a second ``query`` at
7.6 GB it pushed the host's commit charge to 93% (the host has hard-rebooted at 92%).
Loading the full index costs 8-12 GB, so a stray read is expensive and must end on its own.

A daemon thread ends the process (``os._exit``) when any of these holds:
  * the parent process is gone (the caller exited or was killed);
  * the wall clock passed ``AWGRAPH_MAX_SECONDS`` (default 900; ``0`` disables);
  * private memory passed ``AWGRAPH_MAX_RSS_GB`` (default 16; ``0`` disables).

Long-lived commands (``index``, ``mcp``) are not wall-clock limited by default; the
parent and memory checks still apply. Exit code 2, with a reason on stderr.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, Optional

EXIT_DEAD = 2
_UNLIMITED = frozenset({"index", "mcp"})


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def parent_alive(ppid: int) -> bool:
    """True while ``ppid`` is a running process. A ppid of 0/1 counts as gone."""
    if ppid <= 1:
        return False
    if sys.platform == "win32":
        return _alive_windows(ppid)
    return os.getppid() == ppid


def private_gb() -> Optional[float]:
    """This process's private memory in GB, or None when it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t),
                            ("PrivateUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32")
            # the -1 pseudo-handle: an int restype truncates it on 64-bit
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            gpmi = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            ok = gpmi(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return c.PrivateUsage / 2**30 if ok else None
        with open("/proc/self/statm", encoding="ascii") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, ValueError, AttributeError):
        return None


def reason(ppid: int, started: float, now: float, max_s: float, max_gb: float,
           alive: Callable[[int], bool], mem: Callable[[], Optional[float]]) -> str:
    """Why the process must end now, or "" to keep running. Pure given its callables."""
    if not alive(ppid):
        return "parent process %d is gone" % ppid
    if max_s > 0 and now - started > max_s:
        return "ran %.0f s, past AWGRAPH_MAX_SECONDS=%g" % (now - started, max_s)
    gb = mem()
    if max_gb > 0 and gb is not None and gb > max_gb:
        return "private memory %.1f GB, past AWGRAPH_MAX_RSS_GB=%g" % (gb, max_gb)
    return ""


def install(cmd: str, interval: float = 5.0) -> Optional[threading.Thread]:
    """Start the watchdog for ``cmd``. Returns the thread, or None when disabled."""
    if os.environ.get("AWGRAPH_NO_GUARD") == "1":
        return None
    ppid = os.getppid()
    started = time.monotonic()
    max_s = 0.0 if cmd in _UNLIMITED else _env_float("AWGRAPH_MAX_SECONDS", 900.0)
    max_gb = _env_float("AWGRAPH_MAX_RSS_GB", 16.0)

    def watch() -> None:
        while True:
            time.sleep(interval)
            why = reason(ppid, started, time.monotonic(), max_s, max_gb, parent_alive, private_gb)
            if why:
                print("awgraph: guard: ending %s - %s" % (cmd, why), file=sys.stderr, flush=True)
                os._exit(EXIT_DEAD)

    t = threading.Thread(target=watch, name="awgraph-guard", daemon=True)
    t.start()
    return t
