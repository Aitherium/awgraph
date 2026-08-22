"""`python -m awgraph` — the invocation that still works when the console shim does not.

Smart App Control is enforcing on the primary dev host
(`VerifiedAndReputablePolicyState = 1`) and judges each unsigned binary on its own
reputation, so a pip-generated console script can be blocked while its siblings run.
Measured 2026-08-21: `awgit.exe` was blocked ("An Application Control policy has blocked
this file") while `awgraph.exe` and `awrelay.exe`, installed the same way minutes apart,
ran fine. `pip install -e` does not repair it -- it regenerates another unsigned exe that
is judged the same way.

From bash the symptom is `exit 126` / "Permission denied", which reads as a file-mode or
antivirus problem, while `pip show`, `import awgraph` and `shutil.which('awgraph')` all report
the tool healthy. That combination cost real time before anyone asked PowerShell.

This entry point needs no new executable and no security exception: it runs through the
signed, trusted `python.exe`. It exists BEFORE the shim is blocked rather than after,
because the alternative is discovering the outage from the symptom. ATI007 asserts every
agent-tooling package carries one.

A delegation only -- the CLI lives in `cli.py`, and two entry points that can disagree
about argument handling is its own defect.
"""
from __future__ import annotations

import sys

from awgraph.cli import main

if __name__ == "__main__":
    sys.exit(main())
