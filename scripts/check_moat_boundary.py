#!/usr/bin/env python3
"""Publish-time moat guard — inspect the BUILT wheel and sdist before PyPI.

This is the last gate before bytes leave the building, and it asks its question
of the artifact rather than the tree. That distinction is the whole point: the
source can be spotless while the build is not. A stray `hatch` include, a file
left in the working tree at build time, or an sdist that ships what the wheel
excludes are all invisible to a source scan and all reach the user.

`python -m build` publishes BOTH a wheel and an sdist, so both are inspected.
The sdist is the usual offender, because exclusion rules are commonly written
for the wheel and quietly do not apply to it.

Rules:
  * **MOAT001** no monorepo import (`lib.`, `services.`, `from AitherOS`). awgraph
    is standalone by contract; one of these is a `ModuleNotFoundError` on a
    stranger's machine, and the plugin seam exists precisely so fleet integrations
    attach from OUTSIDE the published package.
  * **MOAT002** no internal identifier — debt-row ids, checker rule ids, absolute
    monorepo paths. No secret scanner fires on these because none is a credential;
    what leaks is the SHAPE of the platform, under a permissive licence.
  * **MOAT003** the keystone modules are PRESENT. A guard that only looks for bad
    things passes an EMPTY artifact perfectly, which is the most dangerous thing
    it could do — an empty wheel installs fine and every import fails at runtime.
    This is the positive assertion that makes the other two meaningful.

Exit: 0 clean, 1 a rule failed, 2 could not judge (no artifact, unreadable
archive) — never 0 for "I could not look".

    python scripts/check_moat_boundary.py [dist/awgraph-*.whl dist/*.tar.gz ...]

With no argument it picks the newest wheel AND the newest sdist in `dist/`.
"""

from __future__ import annotations

import ast
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, List, Tuple

#: A monorepo import that cannot resolve once installed.
_MONOREPO_IMPORT = re.compile(
    rb"^\s*(?:from|import)\s+(?:lib|services)(?:\.|\s|$)"
    rb"|^\s*from\s+AitherOS(?:\.|\s|$)",
    re.MULTILINE,
)

#: Internal shapes. Deliberately narrow — a rule that floods gets switched off,
#: which is how this repo's per-file-ignores came to exist.
_INTERNAL = (
    (re.compile(rb"\bD-\d{3,4}\b"), "debt-ledger row id"),
    (re.compile(rb"\b(?:AWG|HYG|PQ|ADK|MCP|NAV|TP|DC|MOAT)\d{3}\b"), "internal checker rule id"),
    (re.compile(rb"[A-Za-z]:[\\/]AitherOS-Fresh"), "absolute monorepo path"),
    (re.compile(rb"aitheros-|aither-vllm|aither-worker"), "internal hostname"),
)

#: Modules whose ABSENCE means the artifact is broken regardless of how clean it
#: scans. __init__ is the entry point; cli is the command-line interface.
_KEYSTONES = ("awgraph/__init__.py", "awgraph/cli.py", "awgraph/base.py", "awgraph/degradation.py", "awgraph/graph.py", "awgraph/logging.py", "awgraph/mcp_server.py", "awgraph/multilang.py", "awgraph/plugins.py", "awgraph/registry.py", "awgraph/store.py")


class CouldNotJudgeError(Exception):
    """Exit 2. Never 0 — silence is not a pass."""


def _newest(pattern: str) -> Path | None:
    dist = Path(__file__).resolve().parent.parent / "dist"
    hits = sorted(dist.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def _entries(path: Path) -> List[Tuple[str, Callable[[], bytes]]]:
    """[(member name, lazy reader)] for a wheel or an sdist."""
    try:
        if path.suffix == ".whl" or path.suffix == ".zip":
            zf = zipfile.ZipFile(path)
            return [(n, (lambda n=n: zf.read(n))) for n in zf.namelist()
                    if not n.endswith("/")]
        if "".join(path.suffixes[-2:]) in (".tar.gz", ".tar.bz2") or path.suffix == ".tgz":
            tf = tarfile.open(path)
            out: List[Tuple[str, Callable[[], bytes]]] = []
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                # An sdist nests everything under `<name>-<version>/`; strip it so
                # member names line up with the wheel's.
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name

                def _read(m=m) -> bytes:
                    f = tf.extractfile(m)
                    return f.read() if f else b""

                out.append((rel, _read))
            return out
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        raise CouldNotJudgeError(f"{path.name}: cannot read ({exc})") from exc
    raise CouldNotJudgeError(f"{path.name}: not a wheel or sdist")


#: Roots that do not exist once the package is installed from PyPI.
_MONOREPO_ROOTS = ("lib", "services", "AitherOS")


def unguarded_monorepo_imports(blob: bytes) -> List[str]:
    """Monorepo imports that are NOT inside an ImportError-handling `try`.

    🚨 WHY THIS IS AST AND NOT A REGEX. The regex this replaced was
    a MULTILINE regex on 'from lib/services', whose leading-whitespace class matched the
    indentation of an import sitting inside a `try:` -- and a GUARDED import cannot
    ModuleNotFoundError, which is the entire failure MOAT001 exists to prevent.

    Measured 2026-08-25: it blocked awgraph 1.4.1 with 6 findings across graph.py,
    registry.py and store.py. All 9 monorepo imports in those files are inside
    `try: ... except ImportError:` with a `# Not available in public package` fallback --
    the correct optional-enhancement idiom, and the package installs and runs without
    them. Zero were unguarded. So the guard was refusing to publish a package that works.

    That is the expensive direction for this kind of rule: a false NEGATIVE ships one bad
    release, but a false POSITIVE that nobody can satisfy gets the whole gate deleted, and
    then the real defect ships unnoticed. Narrowing it to the decidable question keeps the
    protection and removes the flood.

    An unparseable file is FLAGGED, never skipped: it cannot be shown safe, and silently
    passing what you could not read is how a guard becomes decorative.
    """
    try:
        tree = ast.parse(blob.decode("utf-8", "replace"))
    except SyntaxError as exc:
        return [f"unparseable ({exc.msg}) — cannot be shown safe"]

    bad: List[str] = []

    def _handles_import_error(node: ast.Try) -> bool:
        for h in node.handlers:
            if h.type is None:  # bare except
                return True
            names = ast.unparse(h.type)
            if "ImportError" in names or "ModuleNotFoundError" in names or "Exception" in names:
                return True
        return False

    def walk(node: ast.AST, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_guarded = guarded
            if isinstance(node, ast.Try) and child in node.body:
                child_guarded = guarded or _handles_import_error(node)
            root = None
            if isinstance(child, ast.ImportFrom) and child.module:
                root = child.module.split(".")[0]
            elif isinstance(child, ast.Import):
                for a in child.names:
                    r = a.name.split(".")[0]
                    if r in _MONOREPO_ROOTS and not child_guarded:
                        bad.append(f"line {child.lineno}: import {a.name}")
            if root in _MONOREPO_ROOTS and not child_guarded:
                bad.append(f"line {child.lineno}: from {child.module}")
            walk(child, child_guarded)

    walk(tree, False)
    return bad

def inspect(path: Path) -> List[str]:
    entries = _entries(path)
    if not entries:
        raise CouldNotJudgeError(f"{path.name}: archive is empty")

    findings: List[str] = []
    seen = {name for name, _ in entries}

    for name, read in entries:
        if not name.endswith(".py"):
            continue
        try:
            blob = read()
        except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
            raise CouldNotJudgeError(f"{path.name}:{name}: unreadable ({exc})") from exc
        for where in unguarded_monorepo_imports(blob):
            findings.append(f"MOAT001 {path.name}:{name} imports the monorepo "
                            f"UNGUARDED ({where}) — ModuleNotFoundError once installed. "
                            f"Wrap it in try/except ImportError with a fallback.")
        for pattern, label in _INTERNAL:
            for hit in set(pattern.findall(blob)):
                findings.append(f"MOAT002 {path.name}:{name} leaks an {label}: "
                                f"{hit.decode('utf-8', 'replace')}")

    for keystone in _KEYSTONES:
        if keystone not in seen:
            findings.append(f"MOAT003 {path.name} is MISSING {keystone} — the artifact "
                            f"is broken; an empty/partial wheel installs fine and fails "
                            f"at import")
    return findings


def _targets(argv: List[str]) -> List[Path]:
    if argv:
        paths = [Path(a) for a in argv]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise CouldNotJudgeError("named artifact(s) not found: "
                                + ", ".join(str(m) for m in missing))
        return paths
    found = [p for p in (_newest("*.whl"), _newest("*.tar.gz")) if p]
    if not found:
        raise CouldNotJudgeError("no wheel or sdist in dist/ — nothing was built, so "
                            "there is nothing to clear for publication")
    return found


def self_test() -> int:
    """Prove every rule can still fail, and that a clean artifact passes."""
    import tempfile

    bad = 0

    def check(label: str, got: bool, want: bool) -> None:
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    # MOAT001 guarded-vs-unguarded. The regex this replaced could not tell them apart and
    # blocked awgraph 1.4.1 with 6 findings, all of them correctly-guarded optional
    # imports. Both directions are pinned: a false positive gets the gate deleted, a false
    # negative ships a broken package.
    for src, want, why in (
        (b"from lib.faculties.CodeGraph import x", True, "top-level unguarded"),
        (b"import services.foo", True, "plain import unguarded"),
        (b"from AitherOS.thing import y", True, "AitherOS root unguarded"),
        (b"def f():\n    from lib.x import y\n", True, "function-level unguarded"),
        (b"def f(:\n  bad syntax", True, "unparseable is flagged, never skipped"),
        (b"try:\n    from lib.x import y\nexcept ImportError:\n    y = None\n",
         False, "guarded by ImportError"),
        (b"try:\n    from lib.x import y\nexcept ModuleNotFoundError:\n    y = None\n",
         False, "guarded by ModuleNotFoundError"),
        (b"def f():\n    try:\n        from lib.x import y\n    except ImportError:\n"
         b"        pass\n", False, "guarded inside a function"),
        (b"from collections import defaultdict", False, "stdlib is not a monorepo import"),
    ):
        check(f"MOAT001 {why}", bool(unguarded_monorepo_imports(src)), want)

    def wheel(members: dict) -> Path:
        p = Path(tempfile.mkdtemp()) / "awgraph-0.0.0-py3-none-any.whl"
        with zipfile.ZipFile(p, "w") as zf:
            for name, body in members.items():
                zf.writestr(name, body)
        return p

    clean = {k: "print('hi')\n" for k in _KEYSTONES}

    check("a clean wheel passes", inspect(wheel(clean)) == [], True)

    check("MOAT001 catches a monorepo import",
          any(f.startswith("MOAT001") for f in
              inspect(wheel({**clean, "awgraph/x.py": "from lib.core import X\n"}))), True)

    check("MOAT002 catches a debt id",
          any("debt-ledger row id" in f for f in
              inspect(wheel({**clean, "awgraph/x.py": "# see D-0000\n"}))), True)

    check("MOAT002 catches a checker rule id",
          any("checker rule id" in f for f in
              inspect(wheel({**clean, "awgraph/x.py": "# MOAT001 says so\n"}))), True)

    # The one that matters most: an EMPTY-of-keystones artifact must NOT pass.
    check("MOAT003 refuses an artifact missing a keystone",
          any(f.startswith("MOAT003") for f in
              inspect(wheel({"awgraph/cli.py": "x=1\n"}))), True)

    # The package's own imports should not be flagged.
    check("does NOT flag the package's own imports",
          inspect(wheel({**clean, "awgraph/y.py": "from awgraph.client import call\n"})) == [],
          True)

    try:
        _targets(["definitely-not-here.whl"])
        check("a missing artifact cannot judge (exit 2)", False, True)
    except CouldNotJudgeError:
        check("a missing artifact cannot judge (exit 2)", True, True)

    try:
        broken = Path(tempfile.mkdtemp()) / "awgraph-0.0.0.tar.gz"
        broken.write_bytes(b"not a tarball")
        _entries(broken)
        check("an unreadable archive cannot judge (exit 2)", False, True)
    except CouldNotJudgeError:
        check("an unreadable archive cannot judge (exit 2)", True, True)

    print("check_moat_boundary self-test:", "OK" if not bad else f"{bad} BROKEN")
    return 0 if not bad else 1


def main(argv: List[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    try:
        targets = _targets(argv)
    except CouldNotJudgeError as exc:
        print(f"NOT VERIFIED: {exc}", file=sys.stderr)
        return 2

    findings: List[str] = []
    for path in targets:
        print(f"inspecting {path.name}")
        try:
            findings.extend(inspect(path))
        except CouldNotJudgeError as exc:
            print(f"NOT VERIFIED: {exc}", file=sys.stderr)
            return 2

    if findings:
        print(f"\nmoat guard: {len(findings)} violation(s) — NOT publishing",
              file=sys.stderr)
        for f in findings:
            print(f"    {f}", file=sys.stderr)
        return 1

    names = ", ".join(p.name for p in targets)
    print(f"moat guard: clean — {names} carry no monorepo import, no internal "
          f"identifier, and all keystone modules are present")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
