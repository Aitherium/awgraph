# awgraph Ecosystem

## What This Is

awgraph is a code intelligence engine: it reads a repository, extracts symbols (functions, methods, classes, modules), embeds them into a vector space, and serves a hybrid keyword+semantic retrieval system. An agent can ask "what functions call this symbol?" or "show me the impact of changing this module" without grepping the whole codebase.

The index stores ~1.5k tokens of context per symbol (function/method/class body and immediate context), making retrieval precise enough for code change analysis without blowing the agent's context window.

## The Three-Tool Ecosystem

awgraph does not exist alone. It is one of three complementary packages that together form an agent's code understanding layer:

### awgit — Semantic Version Control

**Purpose:** Know *what changed and who is editing it*.

awgit rewrites git's model to be machine-readable:
- Stable node IDs for commits/files/hunks (not ephemeral SHA1s that change on rebase)
- Semantic edit-ops: `insert`, `delete`, `modify`, `rename`, not raw diffs
- Leases and concurrent-agent coordination (one agent per branch, explicit handoffs)
- Stacked commits (each PR owns one semantic unit)
- Durable oplog (audit trail of every semantic change)

**Example:** You're adding a new permission check. awgit tells you:
- Which commits touched the auth module (oplog query)
- Exactly which lines changed in each (semantic diffs, not raw hunks)
- Whether another agent is currently editing that file (lease check, fail-closed)
- Whether to stack your change on top or rebase to develop

### awgraph — Code Intelligence

**Purpose:** Know *what the code is and what depends on what*.

awgraph indexes:
- Function/method/class definitions (name, signature, file, line range)
- Call graph (who calls whom)
- Dependencies (module imports, type references)
- Impact (if you change symbol X, which other symbols break?)

**Example:** Same permission check. awgraph tells you:
- All 47 places that call the auth function you're changing
- Which ones are in test code vs. production
- Which of those 47 will need updates
- Whether your change is backward-compatible (if the signature doesn't change, no impact)

### aither-adk — Agent Runtime

**Purpose:** *Use* awgit and awgraph together to make decisions.

The adk is an agent framework that:
- Reads commits via awgit (what changed)
- Queries awgraph to estimate blast radius (what breaks)
- Decides what to read into context (changed file + impacted symbols)
- Executes edits against awgit (stacked, audited, concurrent-safe)

## How They Work Together: A Worked Example

**Scenario:** An agent is asked to add a new permission level to an auth system and update all code paths that call the auth check.

### Phase 1: Understand the change scope (awgit + awgraph)

```
Agent: "I need to add permission level 'sponsor' to the auth module"

awgit response:
  - Module last touched 4 days ago by agent-ci (lease expired, can edit)
  - 12 commits touch it this quarter
  - No concurrent locks

awgraph response:
  - auth.check_permission is called 47 times across the codebase
  - 31 calls are in service routes (production paths)
  - 16 calls are in test code
  - 8 of the 31 are in admin-only code (may not need update)
  - 3 are in legacy deprecated endpoints (skip these)
  - 20 need explicit review before updating (new calls, edge cases)
```

### Phase 2: Stage the changes (awgit)

```
Agent stages in order:
  1. Commit 1: Add "sponsor" to the enum in auth module (small, isolated)
  2. Commit 2: Update 20 call sites to handle the new level
  3. Commit 3: Add tests for the new permission level

awgit:
  - Stacks all 3 commits into one PR (logically coherent)
  - Each commit is semantic (no merge conflicts in the oplog)
  - Lease prevents concurrent edits to auth.py
```

### Phase 3: Verify impact (awgraph + tests)

```
Agent queries awgraph:
  - "What else calls the 20 functions I just modified?"
  - Result: 3 high-level router functions, each already tested
  - Conclusion: No secondary impact, PR is safe to merge
```

## Why a Graph Beats Grep (With Numbers)

Grep is simple and inclusive: search for the function name, get every occurrence. But it has problems:

- **False positives:** finds the name in comments, strings, docstrings, unrelated symbols
- **False negatives:** misses calls through aliases, dynamic dispatch, or similar names
- **Context bloat:** returns entire files, not just the relevant functions

Semantic search (embeddings) finds *meaning*, not syntax:
- "Functions that do something like X" without naming X
- Calls buried in complex control flow
- Similar patterns you didn't write the search for

**Measured performance on real commits** (15 commits, scope ~2400 code chunks, 800 embedded at 33.3% coverage):

| Method | Recall@10 | Avg Context |
|--------|-----------|-------------|
| grep keyword search | 0.933 | 350k tokens |
| graph semantic+keyword | 0.867 | ~1.5k tokens |
| **grep + graph (pick best per query)** | **1.000** | **~50k tokens** |

**Why grep alone still wins on pure recall:** The graph indexes only function/method/class bodies. Module-level code, comments, and string literals are invisible to it. On a task where the answer lives in a string literal or a module-level constant, grep finds it and the graph does not.

**Why combining them wins:** They are *complementary*.
- Graph excels at call-path queries and impact analysis (navigating the codebase's logic)
- Grep excels at free-text matching (finding ad-hoc patterns)
- Together, they achieve 100% recall while keeping context to ~50k tokens (vs. 350k for grep alone)

**The caveat:** 33.3% embedding coverage means the graph only indexes one-third of the codebase's symbols. The remaining two-thirds fall back to grep. This is by design — embedding is expensive. Real-world usage tunes the coverage threshold based on latency/cost trade-offs.

## Quickstart

### Installation

```bash
pip install awgraph
```

### Basic Usage

```python
from awgraph import CodeGraph

# Index a repository
graph = CodeGraph.from_directory("./my-repo")
graph.save("./my-repo.graph")

# Load an index
graph = CodeGraph.load("./my-repo.graph")

# Find who calls a function
callers = graph.callers("auth.check_permission")
for caller in callers:
    print(f"{caller.name} at {caller.file}:{caller.line}")

# Find impact of a change
impact = graph.impact_analysis("auth.check_permission")
for affected in impact.affected_symbols:
    print(f"  {affected.name} will break if signature changes")

# Semantic search
results = graph.hybrid_query(
    "permission check in auth system",
    k=10
)
for result in results:
    print(f"  {result.name}: {result.score:.2f}")
```

### Integrate with an Agent

```python
from awgraph import CodeGraph
from your_agent_framework import Agent

# Index the repo once
graph = CodeGraph.from_directory("./repo")
graph.save("./repo.graph")

# In your agent loop
class CodeUnderstandingAgent(Agent):
    def __init__(self):
        self.graph = CodeGraph.load("./repo.graph")
    
    def plan_change(self, task: str) -> list[str]:
        """Return files to read for this task."""
        # Semantic search for relevant code
        search_results = self.graph.hybrid_query(task, k=20)
        
        # Get impact of changes
        impact = self.graph.impact_analysis(search_results[0].name)
        
        # Collect files: search results + impacted code
        files = set()
        for result in search_results:
            files.add(result.file)
        for symbol in impact.affected_symbols:
            files.add(symbol.file)
        
        return sorted(files)
```

## Wiring it into an agent (verified, not aspirational)

`aither-adk` already carries the integration point — `agent.set_code_graph(cg)`
registers `code_search` and `code_context` as agent tools. awgraph satisfies the
interface it expects (`query`, `get_context_for_chunk`, `get_full_body`) with no
adapter, so this is the whole wiring:

```python
from awgraph import CodeGraph

cg = CodeGraph(root_path="/abs/path/to/repo", auto_index=False)
await cg.index_codebase("/abs/path/to/repo")
agent.set_code_graph(cg)          # registers code_search + code_context
```

Measured end to end with **no monorepo on `sys.path`** — the standalone package
only:

```
TOOLS_REGISTERED 2 ['code_context', 'code_search']
code_search("backoff policy for flaky calls")
  -> {"results": [{"name": "RetryPolicy", "type": "class", ...}]}
```

Note the query contains none of the words in the class name. It matched on the
docstring.

### The seams to the other two

These are documented designs, not shipped code — stated plainly so nobody reads
them as available:

- **awgit -> awgraph.** awgit records semantic edit-ops keyed to stable node ids,
  so it can tell you a commit changed `RetryPolicy.next_delay` rather than that
  17 lines moved. Feed that symbol to awgraph's impact query and you get the
  blast radius of the change instead of a diff. The join key is the symbol name;
  nothing else needs to agree.
- **Model weights.** awgraph's semantic half needs an embedding model, and the
  first load is the slow part (measured ~450 vectors/min on CPU). Serving those
  weights from a local mirror rather than a public hub removes both the network
  dependency and the cold-start cost. awgraph does not care where the model
  comes from; it asks the embedding backend, which is pluggable.

## Comparison to Alternatives

**GitNexus** ([source-available, noncommercial license](https://github.com/abhigyanpatwari/GitNexus)) is the closest alternative. It also builds a code graph and serves retrieval. GitNexus's recommended mode is `native_augment` — grepping code, then enriching results with graph context. That is exactly what our benchmark found to be optimal: grep for recall, graph for precision and context reduction. awgraph achieves the same conclusion with a smaller, more focused API surface.

The benchmarks measure different things:
- awgraph reports *retrieval recall* (did you find the right code chunks?)
- GitNexus reports *SWE-bench task resolution* (did you solve the problem?)

The two are related but not directly comparable. A 100% recall retriever may still fail a task if the agent does not know what to do with the results.

## Architecture

awgraph consists of:

- **Indexer:** walks the AST (via tree-sitter or language-specific parsers), extracts symbols
- **Embedder:** converts function/method/class docstrings and signatures to vectors (nomic-embed-text-v1.5, 768-d)
- **Store:** saves index to disk (SQLite + vector index)
- **Retriever:** hybrid keyword + semantic search over the store

The index is *read-only after build* (no incremental updates). For code that changes frequently, rebuild the index on a schedule or during CI.

## Limitations and Caveats

1. **Scope:** Only indexes function/method/class definitions. Module-level code, config files, and comments are not indexed. Use grep for those.

2. **Language support:** Python first; Go, Rust, JavaScript/TypeScript via tree-sitter. Language coverage depends on parser quality.

3. **Embedding cost:** Building an index for a 1M-line codebase takes time and disk. Pre-built indexes for popular open-source projects may be available.

4. **Dynamic dispatch:** The graph is *static*. It cannot resolve runtime polymorphism or string-based imports. For dynamic code, augment with grep or runtime analysis.

5. **Vector quality:** Embeddings are useful for semantic search, but they are not perfect. Grep is still better for specific literal matches.

## API Reference

See the main [README](../README.md) for the full API.

### Key Methods

- `CodeGraph.from_directory(path)` — Index a directory
- `graph.hybrid_query(text, k=10)` — Semantic + keyword search
- `graph.callers(symbol_name)` — Find all callers of a symbol
- `graph.callees(symbol_name)` — Find all functions a symbol calls
- `graph.impact_analysis(symbol_name)` — Estimate blast radius of a change
- `graph.save(path)`, `graph.load(path)` — Persist/restore index

## Contributing

awgraph is maintained as part of the Aitherium platform. Issues and pull requests are welcome.

## License

MIT. See LICENSE in the package root.
