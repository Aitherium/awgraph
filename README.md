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
each cost*. Measured on a corpus of real commits, where the task is a commit
message with the answer filenames stripped out and the truth is the set of files
that commit actually modified:

| retriever | recall@10 | context tokens per task |
|---|---|---|
| ranked multi-term grep | 0.933 | 351,369 |
| **awgraph** | **0.800** | **1,503** |

**86% of grep's recall for 0.43% of the context — a 234x reduction.**

Read that honestly: **grep still wins on recall.** awgraph is not a better
*finder*; it is a dramatically cheaper one, and on a long agent loop the context
budget is usually what runs out first. If you need maximum recall and cost is no
object, grep. If you are paying per token across thousands of turns, this is a
different order of magnitude.

Caveats, because a benchmark without them is marketing: n=15 tasks, and only
33.3% of chunks carried embeddings on that run, so the semantic half was working
at partial strength — the recall number understates it.

A naive fusion (run the graph, fall back to grep when it returns few files) was
also measured: it reaches grep's 0.933 recall at 352,872 tokens — *more* than
grep alone, because the fallback fires on nearly every task and pays both bills.
Reported because it did not work; a smarter trigger is future work.

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

## Use it

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
- **[aither-adk](https://github.com/Aitherium/aither-adk)** — the agent runtime
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
