# awgraph for agents

Read this if you are an agent (or a human) editing this package. Short on
purpose: the commands, the traps that cost a session, and where the rest lives.
Nothing here is read at runtime — it is for you.

## What this is

PyPI distribution **`awgraph`** (version in `pyproject.toml`), import package
`awgraph`, Python >= 3.10. A call-graph-aware code index: it parses a
repository into chunks (functions, classes, routes) with a real call graph, so
an agent can ask *which symbols* and *who calls what* instead of grepping.

This repository is a **synced mirror** of the AitherOS monorepo (lane
`.github/workflows/sync-awgraph.yml`). Hand edits made here are overwritten on
the next sync — change the source and let the lane publish.

## Build, test, verify

```bash
python -m pytest tests -q        # the suite: 87 tests, green at v1.4.14
pip install -e .                 # editable install for developing against it
```

Both were run from a source checkout with no prior install. The suite resolves
the package from the workdir, so it never asserts an artifact — that is the
publish lane's job: `publish-brick.yml` builds the wheel, installs it and
imports it, because a tree that tests green can still ship a broken wheel.

## Rules that keep this useful

- **Name new tests for the invariant they defend.** The suite already reads
  that way — `test_backend_guard.py`, `test_cache_isolation.py`,
  `test_chunks_snapshot_iteration.py` — and an index is exactly the kind of
  tool whose failures are silent (stale chunks, a shared cache leaking between
  runs, a snapshot mutated mid-iteration). The filename is the contract.
- **The registry drives the public surface.** This repo's README header,
  `llms.txt` and `aither-manifest.json` are generated from the ecosystem
  registry (one yaml in the AitherOS monorepo) and rewritten on every sync.
  Change the registry; do not hand-edit the generated blocks.
- **The install line is a measured claim.** `check_ecosystem_install_lines`
  asserts the advertised `pip install` channel is real and ours. A rename or
  a move lands with the registry entry in the same change.
- **Releases are lane-driven.** PyPI publishes run from CI after the version
  in `pyproject.toml` changes; nothing is hand-uploaded.

## Read next

- `llms.txt` — the install/use card written for an agent to execute
- `README.md` — the human front door
- `docs/` — the generated docs site source
