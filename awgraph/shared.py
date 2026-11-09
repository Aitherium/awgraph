"""Ask a host-shared graph server before loading the index in this process.

Loading a large index costs gigabytes and tens of seconds, and every ``awgraph query``
used to pay it again: one host with a dozen agent sessions held several full copies at
once. A long-running agent daemon on the same host already holds ONE warm copy of the
graph for its own recall. This module is the thin client of that copy.

Contract (the server side is any process that serves it):

    GET  {url}/v1/codegraph/info   -> {"root": "<abs root>"|null, "warm": bool}
    POST {url}/v1/codegraph/query  {"q": str, "k": int, "root": str}
         200 {"hits": [{"name", "path", "line", "signature"}]}
         409 the server holds a different root      -> answer locally
         503 {"error": "loading"} the graph is warming -> poll until the deadline
         503 {"error": "deferred", "reason"} the server put its load off -> Deferred

``None`` means "no shared server can answer" (absent, another root, an error) and the
caller loads the index itself, so awgraph still works alone. A ``Deferred`` means the
server is there and declined to load the index right now (host memory pressure, or
another full load of the same root), and the caller must NOT load it locally instead:
that is the load the server just refused to make. Never raises.

Environment:
    AWGRAPH_SHARED=0          never ask a shared server (always load locally)
    AWGRAPH_SHARED_URL        server base URL (default http://127.0.0.1:9001)
    AWGRAPH_SHARED_WAIT_S     how long to wait for a warming server (default 120)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Union

DEFAULT_URL = "http://127.0.0.1:9001"
#: Header and file the local daemon accepts as proof the caller is this machine's user.
LOCAL_TOKEN_HEADER = "X-Aither-Local-Token"
#: Answers served by a shared server in this process (tests assert on it).
ANSWERS = {"shared": 0}


class Deferred:
    """The shared server is up and put off loading the graph; ``reason`` says why."""

    def __init__(self, reason: str):
        self.reason = reason or "the shared graph server deferred its load"

    def __repr__(self) -> str:
        return "Deferred(%r)" % self.reason


#: What ``query`` returns: hits, a deferral, or None (answer locally).
Answer = Optional[Union[List[dict], Deferred]]


def enabled() -> bool:
    return os.environ.get("AWGRAPH_SHARED", "1").strip().lower() not in ("0", "false", "no")


def base_url() -> str:
    return (os.environ.get("AWGRAPH_SHARED_URL") or DEFAULT_URL).rstrip("/")


def _wait_s() -> float:
    try:
        return max(0.0, float(os.environ.get("AWGRAPH_SHARED_WAIT_S", "120")))
    except ValueError:
        return 120.0


def _headers() -> dict:
    try:
        tok = (Path.home() / ".aither" / "daemon-token").read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    return {LOCAL_TOKEN_HEADER: tok} if tok else {}


def _same_root(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def query(root: str, text: str, k: int,
          sleep: Callable[[float], None] = time.sleep) -> Answer:
    """Hits for ``text`` from the shared server, a ``Deferred``, or ``None`` to answer
    locally.

    Hit paths are returned RELATIVE to ``root`` when they sit under it.
    """
    if not enabled():
        return None
    try:
        import httpx
    except ImportError:
        return None
    url = base_url()
    deadline = time.monotonic() + _wait_s()
    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=0.5),
                          headers=_headers(), trust_env=False) as client:
            info = client.get(url + "/v1/codegraph/info")
            if info.status_code != 200:
                return None
            served = (info.json() or {}).get("root") or ""
            if not served or not _same_root(served, root):
                return None
            while True:
                resp = client.post(url + "/v1/codegraph/query",
                                   json={"q": text, "k": int(k), "root": root})
                if resp.status_code == 200:
                    hits = (resp.json() or {}).get("hits")
                    if not isinstance(hits, list):
                        return None
                    ANSWERS["shared"] += 1
                    return [_relative(h, root) for h in hits if isinstance(h, dict)]
                body = _json(resp)
                if resp.status_code == 503 and (body.get("deferred")
                                                or body.get("error") == "deferred"):
                    return Deferred(str(body.get("reason") or ""))
                if resp.status_code == 503 and body.get("error") == "loading":
                    if time.monotonic() < deadline:
                        sleep(2.0)
                        continue
                    # Still warming past the wait: the server is spending the memory
                    # right now, so a second local load is the worst answer.
                    return Deferred("the shared graph is still loading after %.0fs"
                                    % _wait_s())
                return None
    except Exception:  # noqa: BLE001 - any transport failure means "answer locally"
        return None


def _json(resp) -> dict:
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _relative(hit: dict, root: str) -> dict:
    path = str(hit.get("path") or "")
    if path and os.path.isabs(path):
        try:
            rel = os.path.relpath(path, root)
        except ValueError:  # another drive on Windows: keep the absolute path
            rel = path
        if not rel.startswith(".."):
            path = rel
    return {"name": str(hit.get("name") or ""), "type": str(hit.get("type") or ""),
            "path": path, "line": int(hit.get("line") or 0),
            "signature": str(hit.get("signature") or "")}
