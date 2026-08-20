# awgraph — the code graph your agent reads instead of grepping

An agent asked to fix a bug does not know which files matter, so it greps, opens
whatever matched, and spends most of its context window on code it will not
change. awgraph indexes the repository into a graph of symbols — functions,
methods, classes, their calls and callers — and answers a natural-language task
with the handful of chunks that task actually needs.

```bash
pip install awgraph
```

Python 3.10+. Nothing else is required to index and query.

## What it costs, measured

The interesting question is not "is a graph better than grep" — it is *what does
each cost to reach the same answer*. Measured on 33 real commits, where the task
is a commit message with the answer filenames stripped out and the truth is the
set of files that commit actually modified. `k` is the result budget, and it is
swept for **both** retrievers, because `k` bounds grep's output too — sweeping it
for only one arm manufactures a win:

| k | awgraph recall | awgraph tokens | grep recall | grep tokens | awgraph cheaper by |
|---|---|---|---|---|---|
| 10 | 0.803 | 1,311 | 0.924 | 351,427 | 268x |
| 25 | 0.939 | 3,132 | 0.985 | 504,640 | 161x |
| 50 | 0.939 | 6,158 | **1.000** | 668,299 | 109x |
| 100 | 0.939 | 12,059 | 1.000 | 735,727 | 61x |
| 200 | 0.939 | 23,386 | 1.000 | 735,727 | 31x |
| 400 | **1.000** | 45,269 | 1.000 | 735,727 | **16x** |

**awgraph reaches the same ceiling as exhaustive grep — recall 1.000 — for 16x
less context. At every budget in between it costs 16-268x fewer tokens.**

Read it honestly, because the shape matters:

- **grep is the better finder at any matched `k`.** It reaches 1.000 at k=50
  while awgraph is still at 0.939. awgraph is not more accurate; it is
  dramatically cheaper for the same eventual answer, and on a long agent loop
  the context budget is what runs out first.
- **grep's cost is not a rounding error.** Reaching 1.000 costs it 668k tokens
  per task — more than most models will accept in one window at all. That is the
  real argument: not that grep is worse, but that at full recall it does not fit.
- **awgraph's token count is for previews**, not whole function bodies —
  signature + docstring + a body preview per chunk. An agent that then reads the
  full body of its top hits pays more than the number above. grep's figure is
  whole files, which is what an agent actually has to read. The comparison is
  fair at the *retrieval* step and generous to awgraph after it.

**Do you actually need the embeddings?** Ablated on the same 33 tasks, same
index, semantic half off:

| k | keyword only | with embeddings | gain |
|---|---|---|---|
| 10 | 0.682 | 0.803 | +0.121 |
| 25 | 0.818 | 0.939 | +0.121 |
| 50 | 0.909 | 0.939 | +0.030 |

So yes at small `k`, and less so as the budget grows — which is the regime that
matters, since the whole point is a small `k`. Embedding on CPU is the slow part
of setup, and this is what it buys.

Caveats, because a benchmark without them is marketing: n=33, one repository,
Python only, and `k` is a knob a caller chooses rather than something the tool
tunes for itself.

Two things measured and **not** confirmed, recorded because a benchmark that
only reports its wins is an advertisement:

- **Embedding coverage was not the gap.** Going from 33.3% of chunks carrying
  vectors to 100% moved recall@10 from 0.800 to 0.803. The earlier claim that
  partial coverage understated the result is refuted.
- **A naive fusion did not work.** Run the graph, fall back to grep when it
  returns few files: 0.894 recall at 348,389 tokens — worse recall than grep AND
  nearly grep's full cost, because the fallback fires on almost every task and
  pays both bills. A trigger keyed on result *count* cannot help; it fires when
  the graph is confidently wrong and stays quiet when the graph is confidently
  right. Keying it on score instead is untested future work.

## Setup: index once, embed lazily

Two costs, and only one of them scales with repo size.

| step | 2,400 chunks | 43,730 chunks |
|---|---|---|
| parse + index | 49.8s | 75.5s |
| embed (CPU) | — | ~97 min at ~450 vectors/min |

**Indexing is close to size-insensitive** — 27x the files for 1.5x the time,
because parsing runs across workers. Embedding is the part that hurts on CPU, so
it is optional, cached and incremental: re-indexing reuses stored vectors and
only embeds what changed.

Without any embedding backend, queries fall back to keyword scoring and still
work. **That fallback is silent by design and dangerous by nature** — a graph
with no vectors looks like a working graph that is merely worse. Check coverage
rather than assuming it:

```python
embedded = sum(1 for c in graph.chunks.values() if c.embedding is not None)
print(f"{embedded}/{len(graph.chunks)} chunks carry vectors")
```

## Use it from the terminal

```
pip install awgraph

awgraph index .                          # parse + persist an index for this repo
awgraph query "retry with exponential backoff"
awgraph callers send_request             # who calls this
awgraph calls send_request               # what does this call
awgraph stats                            # what is in the index
awgraph selftest                         # prove the install works
```

`query` prints `path:line  [type] name` and the signature, so results paste
straight into an editor. `--json` on any read command gives machine-readable
output for wiring into a tool loop.

Exit codes are meaningful: **0** success, **1** a real negative answer (no
match), **2** the command could not run at all — so a script can tell "nothing
matched" from "there is no index yet", which are different problems with
different fixes.

The index is cached **outside your repository** — under `AWGRAPH_CACHE_DIR` if
set, otherwise the platform user-cache directory, keyed by a digest of the
absolute repo path. Nothing is written into the tree you point it at.

`awgraph stats` always prints embedding coverage, including `0.0%`. Without an
embedding backend `hybrid_query` silently falls back to keyword scoring and
still returns ten confident-looking results, so "is the semantic half actually
on?" is a question you should never have to answer by reading the source.

## Use it from a coding agent (MCP)

```
pip install "awgraph[mcp]"
```

then one line in your client's MCP config — Claude Code, Cursor, Windsurf, Zed:

```json
{"mcpServers": {"awgraph": {"command": "awgraph", "args": ["mcp"]}}}
```

Your agent gains `code_index`, `code_search`, `code_callers`, `code_calls` and
`code_stats`. It searches by meaning and gets back symbols with file, line,
signature, calls and callers — rather than pasting file text into its own
context, which is the cost this package exists to remove.

Index once per repository (`code_index`); it is cached on disk **outside** the
repo. Indexing is never implicit: a search against an unindexed repo tells the
agent to index rather than pausing for minutes, because a long silent call reads
as a hang and usually gets killed.

## Use it from Python

```python
import asyncio
from awgraph import CodeGraph

async def main():
    graph = CodeGraph(root_path="/abs/path/to/repo", auto_index=False)
    await graph.index_codebase("/abs/path/to/repo")   # absolute path required

    for chunk in await graph.hybrid_query("retry with exponential backoff", max_results=5):
        print(chunk.name, chunk.source_path, chunk.start_line)

asyncio.run(main())
```

`index_codebase` needs an **absolute** path. Given a relative one it walks
nothing, indexes zero chunks, and returns successfully — so assert on
`len(graph.chunks)` rather than on the absence of an exception.

The query does not need to contain the symbol name. Asking for "backoff policy
for flaky calls" against a class documented as *"Backoff policy for flaky
calls"* returns it by meaning, not by string match.

## Where it sits

Three packages, three different questions about the same repository:

- **[awgit](https://github.com/Aitherium/awgit)** — semantic version control.
  Stable node ids, semantic edit-ops, leases so concurrent agents do not
  overwrite each other, stacked commits with one PR each. It knows **what
  changed and who is editing it**.
- **awgraph** — code intelligence. Symbols, call paths, dependencies, blast
  radius. It knows **what the code is and what depends on what**.
- **[awdk](https://github.com/Aitherium/awdk)** — the agent runtime
  that consumes both.

The seam is the useful part: awgit tells you a commit touched
`RetryPolicy.next_delay`; awgraph tells you what calls it and which tests cover
it; the agent reads that instead of the repository.

## Related work

[GitNexus](https://github.com/abhigyanpatwari/GitNexus) is the closest analogue
and worth reading. Its recommended mode augments grep with graph context rather
than replacing grep — a conclusion these measurements independently reach. Note
its licence is PolyForm Noncommercial (source-available, commercial use
forbidden), where awgraph is Apache 2.0. Its published figures measure SWE-bench
task resolution; the numbers above measure retrieval recall. Those are different
axes and should not be compared directly.

## Licence

Apache 2.0.

<!-- aither-ecosystem:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

## The aw family

Standalone tools that share one idea: **replace something you would otherwise have to _trust_ with something you can _check_.**

Each installs on its own, works offline, and needs no account.

| | instead of trusting | you check |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | a framework's idea of how your agents should run | one loop you can read, pointed at a backend you already pay for |
| [awskills](https://github.com/Aitherium/awskills) | that an agent knows your procedure | the procedure written down, versioned, and loadable by any agent |
| [awm](https://github.com/Aitherium/awm) | that memory stayed in its lane | tenant:user:project scopes, so a write cannot cross a boundary |
| [awnode](https://github.com/Aitherium/awnode) | a vendor's cloud with every prompt | a local gateway routing to backends you chose |
| **awgraph** _(you are here)_ | that grep found everything | an AST + tree-sitter call graph an agent can traverse |
| [awgit](https://github.com/Aitherium/awgit) | that no one else is editing this file | a lease, refused at commit time if you do not hold it |
| [awseal](https://github.com/Aitherium/awseal) | that the artifact came from who you think | an Ed25519 seal — the key that verifies is not the key that forges |
| [awshare](https://github.com/Aitherium/awshare) | that the download is intact | content-addressed bundles, verified on fetch |
| [awnest](https://github.com/Aitherium/awnest) | that there is a person on the other end | a verdict with evidence, where "we could not tell" is not "yes" |
| [awnboard](https://github.com/Aitherium/awnboard) | a share link anyone who sees it can use | an invitation addressed to one person, for one gate, revocable |
| [awnix](https://github.com/Aitherium/awnix) | that the box is what you left it as | an immutable image you built, with atomic rollback |
| [awrecover](https://github.com/Aitherium/awrecover) | that the restore worked | a restore that fully lands or does not land at all |
| [awrelay](https://github.com/Aitherium/awrelay) | a SaaS in the middle of your agents | findings, alerts and coordination over your own transport |
| [awmail](https://github.com/Aitherium/awmail) | a mailbox somebody else can read | mail your agents send and receive over your own server |
| [awfind](https://github.com/Aitherium/awfind) | one vendor's idea of the web | results from whichever providers you configured |
| [awbrowse](https://github.com/Aitherium/awbrowse) | that the page said what you were told | the render, the DOM and the requests it made |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | a vendor's quantisation defaults | sub-byte KV cache kernels you can benchmark yourself |
| [AitherZero](https://github.com/Aitherium/AitherZero) | a pile of scripts nobody has numbered | numbered, discoverable automation with declarative playbooks |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | what a page tells your browser to do | a federated search and desktop bridge you host |

[**awnix**](https://github.com/Aitherium/awnix) is the ground floor — A Linux you can hand to an agent — immutable base, capabilities included.

## The Aitherium ecosystem

Every repository here is public. Each publishes an `aither-manifest.json` beside its page, so any surface can read every sibling's — the network is browsable from any node in it.

| repo | what it is | pages |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | Build AI agent fleets — 3 lines, any backend, local or cloud | [docs](https://aitherium.github.io/awdk/) |
| [awskills](https://github.com/Aitherium/awskills) | Portable agent skills — self-contained procedures an agent loads on demand | [docs](https://aitherium.github.io/awskills/) |
| [awm](https://github.com/Aitherium/awm) | A portable, scoped agent memory | [docs](https://aitherium.github.io/awm/) |
| [awnode](https://github.com/Aitherium/awnode) | A lightweight local gateway — bridges your apps to the AI backends you chose | [docs](https://aitherium.github.io/awnode/) |
| [awrun](https://github.com/Aitherium/awrun) | A priority-aware queue and dispatcher for agentic runs and ad-hoc CI builds | [docs](https://aitherium.github.io/awrun/) |
| **awgraph** _(you are here)_ | A semantic code graph for agents — AST + tree-sitter, call graphs | [docs](https://aitherium.github.io/awgraph/) |
| [awgit](https://github.com/Aitherium/awgit) | Semantic version control on top of git — edit-ops and leases | [docs](https://aitherium.github.io/awgit/) |
| [awseal](https://github.com/Aitherium/awseal) | Sign an artifact so a stranger can verify it | [docs](https://aitherium.github.io/awseal/) |
| [awshare](https://github.com/Aitherium/awshare) | Publish an artifact and fetch it back verified | [docs](https://aitherium.github.io/awshare/) |
| [awnest](https://github.com/Aitherium/awnest) | Prove there is a human before you let them into the nest | — |
| [awnboard](https://github.com/Aitherium/awnboard) | A front gate you can put in front of anything, and hand someone the key to | — |
| [awnix](https://github.com/Aitherium/awnix) | A Linux you can hand to an agent — immutable base, capabilities included | [docs](https://aitherium.github.io/awnix/) |
| [awrecover](https://github.com/Aitherium/awrecover) | Labelled snapshots with an all-or-nothing restore | [docs](https://aitherium.github.io/awrecover/) |
| [awrelay](https://github.com/Aitherium/awrelay) | Portable agent messaging — findings, alerts, coordination | [docs](https://aitherium.github.io/awrelay/) |
| [awmail](https://github.com/Aitherium/awmail) | Give an agent an email address — send, and actually receive | — |
| [awfind](https://github.com/Aitherium/awfind) | A portable search client — query, results, ranking | — |
| [awbrowse](https://github.com/Aitherium/awbrowse) | A portable browser client — navigate, console, network, DOM, screenshot | — |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | Near-optimal KV cache quantization for LLM inference — sub-byte compression | [docs](https://aitherium.github.io/aitherkvcache/) |
| [AitherZero](https://github.com/Aitherium/AitherZero) | PowerShell 7+ automation framework — numbered, self-describing scripts | [docs](https://aitherium.github.io/AitherZero/) |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | Browser extension — federated AI search, page context, and the Living OS overlay | — |

<!-- aither-ecosystem:end -->
