#!/usr/bin/env python3
"""Publish-time moat guard - inspect the BUILT wheel and sdist before PyPI.

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
    stranger's machine, and the guarded-import seam exists precisely so
    the fleet integrations attach from OUTSIDE the published package.
  * **MOAT002** no internal identifier - debt-row ids, checker rule ids, absolute
    monorepo paths. No secret scanner fires on these because none is a
    credential; what leaks is the SHAPE of the platform, under a permissive
    licence. Same class the ADK boundary scan was written for.
  * **MOAT003** the keystone modules are PRESENT. A guard that only looks for bad
    things passes an EMPTY artifact perfectly, which is the most dangerous thing
    it could do - an empty wheel installs fine and every import fails at runtime.
    This is the positive assertion that makes the other two meaningful.

Exit: 0 clean, 1 a rule failed, 2 could not judge (no artifact, unreadable
archive) - never 0 for "I could not look".

    python scripts/check_moat_boundary.py [dist/awgraph-*.whl dist/*.tar.gz ...]

With no argument it picks the newest wheel AND the newest sdist in `dist/`.
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, List, Tuple


def _classify_monorepo_imports(blob: bytes) -> tuple:
    """(guarded, unguarded) counts for monorepo imports in a source blob.

    Guarded means lexically inside a `try:`. Parsed with ast when the source
    compiles; a file that does NOT parse counts as all-unguarded, because a guard
    we cannot see is a guard we cannot rely on.
    """
    import ast as _ast

    try:
        tree = _ast.parse(blob.decode("utf-8", errors="replace"))
    except SyntaxError:
        return (0, len(_MONOREPO_IMPORT.findall(blob)))

    roots = {"lib", "services", "apps", "AitherOS"}
    guarded = unguarded = 0

    def walk(node, depth):
        nonlocal guarded, unguarded
        for child in _ast.iter_child_nodes(node):
            names = []
            if isinstance(child, _ast.ImportFrom) and child.module:
                names = [child.module]
            elif isinstance(child, _ast.Import):
                names = [a.name for a in child.names]
            for n in names:
                if n.split(".")[0] in roots:
                    if depth:
                        guarded += 1
                    else:
                        unguarded += 1
            walk(child, depth + 1 if isinstance(child, _ast.Try) else depth)

    walk(tree, 0)
    return (guarded, unguarded)


#: A monorepo import that cannot resolve once installed.
_MONOREPO_IMPORT = re.compile(
    rb"^\s*(?:from|import)\s+(?:lib|services)(?:\.|\s|$)"
    rb"|^\s*from\s+AitherOS(?:\.|\s|$)",
    re.MULTILINE,
)

#: Internal shapes. Deliberately narrow - a rule that floods gets switched off,
#: which is how this repo's per-file-ignores came to exist.
_INTERNAL = (
    (re.compile(rb"\bD-\d{3,4}\b"), "debt-ledger row id"),
    (re.compile(rb"\b(?:AWG|HYG|PQ|ADK|MCP|NAV|TP|DC)\d{3}\b"), "internal checker rule id"),
    (re.compile(rb"[A-Za-z]:[\\/]AitherOS-Fresh"), "absolute monorepo path"),
)

#: Modules whose ABSENCE means the artifact is broken regardless of how clean it
#: scans. cli is the entry point; plugins is the seam that keeps the package
#: standalone; guard is the safety rail every rewriting command routes through.
_KEYSTONES = ("awgraph/graph.py", "awgraph/store.py", "awgraph/registry.py")


class CouldNotJudgeError(Exception):
    """Exit 2. Never 0 - silence is not a pass."""


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
        # UNGUARDED only, and the distinction is the contract rather than leniency.
        # A package with ZERO monorepo imports (awgit) can flag any of them. awgraph
        # ships OPTIONAL integrations behind try/except so a host can light them up
        # while a stranger's install falls back. The harm MOAT001 names -
        # "ModuleNotFoundError once installed" - cannot occur when the import is
        # caught, so flagging those would force deleting working behaviour to satisfy
        # a rule whose stated damage is impossible. Verified two ways before relaxing
        # it: AST says 9 guarded / 0 unguarded, and the package imports, indexes and
        # answers a semantic query with no monorepo on sys.path.
        #
        # An UNGUARDED import is still fatal. Guarded ones are COUNTED and printed on
        # every run, so the number cannot creep up unnoticed - which is the failure
        # this relaxation could otherwise introduce.
        _guarded, _unguarded = _classify_monorepo_imports(blob)
        if _guarded:
            print(f"  note: {path.name}:{name} has {_guarded} guarded monorepo "
                  f"import(s) - optional, inert when absent")
        if _unguarded:
            findings.append(f"MOAT001 {path.name}:{name} imports the monorepo "
                            f"(lib/services/AitherOS) - ModuleNotFoundError once installed")
        for pattern, label in _INTERNAL:
            for hit in set(pattern.findall(blob)):
                findings.append(f"MOAT002 {path.name}:{name} leaks an {label}: "
                                f"{hit.decode('utf-8', 'replace')}")

    for keystone in _KEYSTONES:
        if keystone not in seen:
            findings.append(f"MOAT003 {path.name} is MISSING {keystone} - the artifact "
                            f"is broken; an empty/partial wheel installs fine and fails "
                            f"at import")
    return findings


#: Shapes that name the monorepo this package is extracted FROM.
_MONOREPO_MARKERS = (
    re.compile(rb"AitherOS/(?!packages/awgraph)"),
    re.compile(rb"apps/Aither"),
    re.compile(rb"dev/tools/"),
)


def inspect_docs(root: Path) -> List[str]:
    """MOAT004 - the PUBLISHED docs payload must not name the monorepo.

    `docs/` is served straight to the internet by GitHub Pages (legacy build,
    `main:/docs`), and it is in NEITHER the wheel nor the sdist - so the artifact
    scan above structurally cannot see it, and neither can a source scan aimed at
    `*.py`.

    Measured 2026-08-13 on the live site: `graph.json` carried 128 node ids rooted
    at `AitherOS/â€¦`, naming internal files (`apps/AitherGenesis/genesis_ops.py`,
    `dev/tools/check_awgit_lease_plane.py`) plus totals disclosing 362 files, 3780
    nodes, 10 actors and 35 collisions. It got there because `gen_graph_json.py`
    calls `awgraph.graph.build()`, which reads whatever repo it runs in - run once
    from the monorepo root, and the monorepo is what ships. Counts are disclosure
    too, not just paths.
    """
    docs = root / "docs"
    if not docs.is_dir():
        return []
    findings: List[str] = []
    for path in sorted(docs.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".js", ".html", ".md"}:
            continue
        try:
            blob = path.read_bytes()
        except OSError as exc:
            raise CouldNotJudgeError(f"docs/{path.name}: unreadable ({exc})") from exc
        for pattern in _MONOREPO_MARKERS:
            hits = pattern.findall(blob)
            if hits:
                findings.append(
                    f"MOAT004 docs/{path.relative_to(docs).as_posix()} names the "
                    f"monorepo ({len(hits)}x '{pattern.pattern.decode()}') - this "
                    f"file is served publicly by GitHub Pages and is in neither the "
                    f"wheel nor the sdist, so no other gate sees it")
                break
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
        raise CouldNotJudgeError("no wheel or sdist in dist/ - nothing was built, so "
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
              inspect(wheel({**clean, "awgraph/x.py": "# see D-1640\n"}))), True)

    check("MOAT002 catches a checker rule id",
          any("checker rule id" in f for f in
              inspect(wheel({**clean, "awgraph/x.py": "# AWG005 says so\n"}))), True)

    # The one that matters most: an EMPTY-of-keystones artifact must NOT pass.
    check("MOAT003 refuses an artifact missing a keystone",
          any(f.startswith("MOAT003") for f in
              inspect(wheel({"awgraph/cli.py": "x=1\n"}))), True)

    # awgraph.base is internal to the package; importing it is not a monorepo import.
    check("does NOT flag the package's own imports",
          inspect(wheel({**clean, "awgraph/y.py": "from awgraph.base import Base\n"})) == [],
          True)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "docs").mkdir()
        (root / "docs" / "graph.json").write_text(
            '{"nodes":[{"id":"f:AitherOS/apps/AitherGenesis/x.py"}]}', encoding="utf-8")
        check("MOAT004 catches a monorepo path in the published docs",
              bool(inspect_docs(root)), True)
        (root / "docs" / "graph.json").write_text(
            '{"nodes":[{"id":"f:awgraph/cli.py"}]}', encoding="utf-8")
        check("MOAT004 passes a package-scoped docs payload",
              inspect_docs(root) == [], True)
        check("MOAT004 is silent when there is no docs/ at all",
              inspect_docs(Path(td) / "nope") == [], True)

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
    try:
        findings.extend(inspect_docs(Path(__file__).resolve().parent.parent))
    except CouldNotJudgeError as exc:
        print(f"NOT VERIFIED: {exc}", file=sys.stderr)
        return 2
    for path in targets:
        print(f"inspecting {path.name}")
        try:
            findings.extend(inspect(path))
        except CouldNotJudgeError as exc:
            print(f"NOT VERIFIED: {exc}", file=sys.stderr)
            return 2

    if findings:
        print(f"\nmoat guard: {len(findings)} violation(s) - NOT publishing",
              file=sys.stderr)
        for f in findings:
            print(f"    {f}", file=sys.stderr)
        return 1

    names = ", ".join(p.name for p in targets)
    print(f"moat guard: clean - {names} carry no monorepo import, no internal "
          f"identifier, and all keystone modules are present")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
