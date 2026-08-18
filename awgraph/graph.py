#!/usr/bin/env python3
"""
CodeGraph - Python AST Indexer with Call Graph
===============================================

A focused, single-purpose indexer that does ONE thing well:
Parse Python code into semantic chunks with full call graph.

WHAT IT DOES:
1. Real AST parsing (not regex)
2. Extracts functions, classes, methods
3. Builds call graph: what calls what, what is called by what
4. Outputs chunks ready for embedding

WHAT IT DOESN'T DO:
- PDFs, web pages, "universal" anything
- That's for other neurons to handle

USAGE:
------
    from awgraph import CodeGraph

    graph = CodeGraph()

    # Index a codebase
    await graph.index_codebase("/path/to/code")

    # Query with call graph awareness
    chunks = await graph.query("rate limiter")

    for chunk in chunks:
        print(f"{chunk.name} calls: {chunk.calls}")
        print(f"{chunk.name} called by: {chunk.called_by}")

Author: AitherOS
Version: 1.0.0
"""

import ast
import asyncio
import hashlib
import json
from array import array
from awgraph.logging import get_logger
import math
import os
import pickle
import re
import subprocess
import threading
import time
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from awgraph.base import BaseFacultyGraph, GraphSyncConfig
import httpx
AsyncClient = httpx.AsyncClient
from awgraph.degradation import get_registry, SubsystemTier
_reg = get_registry()  # noqa: E402

# Optional: numpy for fast vector operations
try:
    import numpy as np
    _HAS_NUMPY = True
    _reg.register_ok("np", "numpy", SubsystemTier.COGNITIVE)
except ImportError as _import_err:
    _reg.register_failed("np", "numpy", _import_err, SubsystemTier.COGNITIVE)
    np = None
    _HAS_NUMPY = False


def _as_f32(vec):
    """Compact a raw embedding into a float32 row.

    A 768-dim vector as a boxed Python float list costs ~25KB; as a float32
    ndarray (or stdlib array('f') without numpy) it costs ~3KB — at 101K
    chunks that is ~0.9GB vs ~110MB (memory audit 2026-07). Both forms
    iterate, len() and zip() like the list they replace.
    """
    if vec is None:
        return None
    if _HAS_NUMPY:
        if isinstance(vec, np.ndarray):
            return vec.astype(np.float32, copy=False)
        return np.asarray(vec, dtype=np.float32)
    return vec if isinstance(vec, array) else array("f", vec)

# Embedding engine — routes to sentence-transformers or vLLM (no Ollama)
# In the public package, embeddings must be provided externally via plugin hooks
def get_embedding_engine():
    """No-op: embeddings not available in public package."""
    return None

_HAS_EMBEDDING_ENGINE = False

# Optional: vLLM backend (OpenAI-compatible API for generation)
_vllm_url: Optional[str] = None
_vllm_model: Optional[str] = None  # actual model name served by vLLM
_vllm_checked = False

def _detect_vllm() -> Optional[str]:
    """Detect running vLLM instance. Returns base URL or None."""
    global _vllm_url, _vllm_model, _vllm_checked
    if _vllm_checked:
        return _vllm_url
    _vllm_checked = True
    # Env var takes priority
    url = os.environ.get("VLLM_URL") or os.environ.get("NVIDIA_NIM_URL")
    if url:
        _vllm_url = url.rstrip("/")
    # Probe common ports
    if not _vllm_url:
        try:
            import httpx
            for port in (8120, 8116):
                try:
                    r = httpx.get(f"http://localhost:{port}/v1/models", timeout=0.3)
                    if r.status_code == 200:
                        _vllm_url = f"http://localhost:{port}"
                        # Extract the actual model name from vLLM
                        data = r.json()
                        if data.get("data"):
                            _vllm_model = data["data"][0]["id"]
                        break
                except Exception:
                    continue
        except ImportError:
            pass
    return _vllm_url

def _vllm_model_name() -> str:
    """Get the actual model name served by vLLM."""
    return _vllm_model or "deepseek-r1:14b"


# Model mapping — vLLM serves deepseek-r1:14b
ELASTIC_REFLEX = "llama3.2:latest"       # Fast: neurons, rerank, embeddings
ELASTIC_AGENT = "mistral-nemo:latest"    # Balanced: agent tasks, tool calling
ELASTIC_REASON = "deepseek-r1:14b"       # Deep: analysis, complex reasoning


# ── Code-specialized embedding (CodeRankEmbed) ───────────────────────────────
# When configured, CodeGraph embeds CODE with a code-specialized model instead of
# the general-purpose nomic-embed-text. Measured 2026-07-21 on the 200-query
# corpus: CodeRankEmbed lifts recall@10 0.895->0.980 / MRR 0.740->0.816.
# It is ASYMMETRIC — documents embed plain, QUERIES need the instruction prefix
# below; omitting the prefix on queries collapses the query/document vector-space
# match. Same 768-dim as nomic, so embeddings.npy shape is unchanged. Config is
# bind-mounted (toggle without rebuild); the is_query call-site wiring is BAKED.
_CODE_EMBED_URL = os.environ.get("AITHER_CODEGRAPH_EMBED_URL", "").strip()
_CODE_EMBED_MODEL = os.environ.get("AITHER_CODEGRAPH_EMBED_MODEL", "").strip()
_CODE_QUERY_PREFIX = os.environ.get(
    "AITHER_CODEGRAPH_QUERY_PREFIX",
    "Represent this query for searching relevant code: ",
)


def _code_embed_enabled() -> bool:
    return bool(_CODE_EMBED_URL and _CODE_EMBED_MODEL)


async def _embed_via_code_service(
    texts: List[str], is_query: bool
) -> List[Optional[list]]:
    """Embed via the code-specialized embedder (CodeRankEmbed).

    Queries get the instruction prefix; documents are sent plain. Trusts the
    internal CA via AitherHttp.AsyncClient (never verify=False). Any failure or
    shape mismatch returns all-None so the caller degrades exactly as it does on
    an EmbeddingEngine miss (never a silent wrong-length result).
    """
    payload_texts = (
        [_CODE_QUERY_PREFIX + t for t in texts] if is_query else list(texts)
    )
    try:
        async with AsyncClient(timeout=120.0) as client:
            r = await client.post(
                _CODE_EMBED_URL,
                json={"model": _CODE_EMBED_MODEL, "input": payload_texts},
            )
            if r.status_code != 200:
                logger.warning(
                    f"[CODE-EMBED] {_CODE_EMBED_MODEL} returned {r.status_code}; "
                    f"batch of {len(texts)} treated as failed"
                )
                return [None] * len(texts)
            data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))
            vecs = [d.get("embedding") for d in data]
            if len(vecs) != len(texts):
                logger.warning(
                    f"[CODE-EMBED] expected {len(texts)} vectors, got {len(vecs)} "
                    f"(model={_CODE_EMBED_MODEL}) — treating batch as failed"
                )
                return [None] * len(texts)
            return vecs
    except Exception as e:
        logger.warning(f"[CODE-EMBED] {_CODE_EMBED_MODEL} failed: {e}")
        return [None] * len(texts)


async def _embed_texts(
    texts: List[str], model: str = "nomic-embed-text", is_query: bool = False
) -> List[Optional[list]]:
    """Embed texts via EmbeddingEngine (sentence-transformers -> vLLM).

    When the code-specialized embedder is configured, CODE embeddings route there
    instead, with the query instruction prefix applied only when ``is_query=True``.
    Document callers keep the default (False); QUERY callers MUST pass
    ``is_query=True`` — otherwise CodeRankEmbed embeds the query without its prefix
    and silently mismatches the document space, collapsing retrieval.
    """
    if _code_embed_enabled():
        return await _embed_via_code_service(texts, is_query)
    if _HAS_EMBEDDING_ENGINE:
        try:
            engine = get_embedding_engine()
            out = await engine.embed_batch(texts, model=model)
            # A backend that returns all-None WITHOUT raising produces exactly the
            # same result as "nothing to do", and says nothing about why.
            # That is how 4,337 chunks failed to embed while the API answered
            # {"embedded": true}: the raise-path logs, this path did not. Never
            # let a total miss pass silently.
            if out and all(v is None for v in out):
                logger.warning(
                    f"EmbeddingEngine returned no vectors for all {len(texts)} "
                    f"texts (model={model}) — backend reachable but produced "
                    f"nothing; treating batch as failed"
                )
            return out
        except Exception as e:
            logger.warning(f"EmbeddingEngine failed: {e}")
    return [None] * len(texts)


async def _llm_generate(prompt: str, model: str = ELASTIC_REFLEX,
                         temperature: float = 0.0, max_tokens: int = 200) -> str:
    """Generate text via vLLM."""
    vllm = _detect_vllm()
    if vllm:
        try:
            import httpx
            async with AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"{vllm}/v1/chat/completions",
                    json={
                        "model": _vllm_model_name(),
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug(f"vLLM generate failed: {e}")

    return ""

_HAS_LLM = True  # At least one backend check succeeded at call time

logger = get_logger("CodeGraph")


# ============================================================================
# RENAME-SAFE STABLE IDS (gated by AITHER_CODEGRAPH_ID_VERSION=2, default 1)
# ============================================================================

_V1_ID_PREFIXES = ("func_", "class_", "method_")


def _codegraph_id_v2() -> bool:
    """True when the rename-safe stable-id scheme is enabled."""
    return str(os.environ.get("AITHER_CODEGRAPH_ID_VERSION", "1")).strip() == "2"


def _is_v1_id(cid: str) -> bool:
    """A legacy name-baked id (``{type}_{name}_{path_sha8}``)."""
    return isinstance(cid, str) and cid.startswith(_V1_ID_PREFIXES)


def _finalize_stable_ids(cg: "CodeGraph") -> Dict[str, str]:
    """Re-key any v1-format chunk ids to rename-safe stable ids IN PLACE and
    rebuild the call edges + indices.  Idempotent and reindex-preserving (the
    persisted id manager maps each ``(name, path)`` to a stable id, so the same
    symbol keeps the same id across reindexes).

    Returns ``old_id -> new_id`` (empty when nothing changed).  No-op unless v2.
    """
    if not _codegraph_id_v2():
        return {}
    if not any(_is_v1_id(cid) for cid in cg.chunks):
        return {}  # already finalized

    from collections import defaultdict
    try:
        from lib.faculties.CodeGraphIDMigration import migrate_chunks
    except ImportError:
        # Not available in public package — skip v1 migration
        return cg.chunks

    try:
        manager = getattr(cg, "_id_manager", None)
        new_chunks, old_to_new, manager = migrate_chunks(dict(cg.chunks), manager)
        cg._id_manager = manager
        cg.chunks = new_chunks
    except Exception as e:
        # Migration failed — keep original chunks
        logger.debug(f"ID migration failed: {e}")
        return cg.chunks

    # Rebuild the secondary indices against the new stable ids.
    cg.by_name = defaultdict(list)
    cg.by_file = defaultdict(list)
    cg.by_class = defaultdict(list)
    cg.routes = {}
    for sid, ch in cg.chunks.items():
        cg.by_name[ch.name].append(sid)
        cg.by_file[ch.source_path].append(sid)
        if ch.parent_class:
            cg.by_class[ch.parent_class].append(sid)
        if getattr(ch, "route_path", None) and getattr(ch, "route_method", None):
            cg.routes[f"{ch.route_method} {ch.route_path}"] = sid
    return old_to_new


# ============================================================================
# PICKLE HMAC INTEGRITY HELPERS
# ============================================================================

def _pickle_hmac_key() -> bytes:
    """Get the HMAC key for pickle integrity validation."""
    secret = os.environ.get("AITHER_INTERNAL_SECRET", "")
    if not secret:
        logger.warning(
            "AITHER_INTERNAL_SECRET not set — using default pickle HMAC key. "
            "Set this env var in production!"
        )
        secret = "aither-pickle-hmac-default"
    return secret.encode()


def _compute_file_hmac(filepath: str) -> str:
    """Compute HMAC-SHA256 of a file's contents."""
    import hmac as _hmac_mod
    import hashlib as _hashlib_mod
    with open(filepath, "rb") as f:
        data = f.read()
    return _hmac_mod.new(_pickle_hmac_key(), data, _hashlib_mod.sha256).hexdigest()


def _verify_pickle_hmac(filepath: str) -> bool:
    """Verify HMAC sidecar for a pickle file. Returns True if valid or legacy (no sidecar)."""
    import hmac as _hmac_mod
    hmac_path = filepath + ".hmac"
    if not os.path.exists(hmac_path):
        logger.warning("[CodeGraph] No HMAC sidecar for %s — refusing to load (rebuild required)", filepath)
        return False
    try:
        with open(hmac_path, "r", encoding="utf-8") as f:
            stored = f.read().strip()
        computed = _compute_file_hmac(filepath)
        if not _hmac_mod.compare_digest(stored, computed):
            logger.error(f"[CodeGraph] HMAC mismatch for {filepath} — cache tampered, deleting")
            os.unlink(filepath)
            os.unlink(hmac_path)
            return False
        return True
    except Exception as e:
        logger.error("[CodeGraph] HMAC verification error for %s: %s — refusing to load", filepath, e)
        return False


def _write_pickle_hmac(filepath: str) -> None:
    """Write HMAC sidecar after saving a pickle file."""
    try:
        hmac_val = _compute_file_hmac(filepath)
        with open(filepath + ".hmac", "w", encoding="utf-8") as f:
            f.write(hmac_val)
    except Exception as e:
        logger.warning(f"[CodeGraph] Failed to write HMAC sidecar: {e}")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ChunkType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


@dataclass
class CodeChunk:
    """
    A semantic chunk of Python code with full relationship data.
    
    This is NOT just "text found at line X" - it's a node in the
    code graph with edges to what it calls and what calls it.
    """
    id: str
    name: str
    chunk_type: ChunkType
    source_path: str
    
    # Location
    start_line: int
    end_line: int
    
    # Semantic content
    signature: str = ""  # Full signature: "async def foo(x: int) -> str"
    docstring: str = ""
    body_preview: str = ""  # First N chars of body
    
    # Imports used by this chunk
    imports: List[str] = field(default_factory=list)
    import_map: Dict[str, str] = field(default_factory=dict)  # local_name -> full_module.name
    
    # CALL GRAPH - the key insight
    calls: List[str] = field(default_factory=list)  # What functions/methods this code calls
    called_by: List[str] = field(default_factory=list)  # What calls this (backfilled)
    
    # For classes
    base_classes: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    
    # For methods
    parent_class: Optional[str] = None
    
    # Quality metrics
    complexity: int = 0  # Cyclomatic complexity estimate
    line_count: int = 0

    # Git stats (populated by enrich_with_git_stats)
    git_commits: int = 0          # Total commits touching this file
    git_contributors: int = 0     # Unique authors on this file
    git_last_modified: str = ""   # ISO date of last commit
    git_churn_rate: float = 0.0   # Commits per month (higher = less stable)

    # Graph centrality (computed after backfill)
    centrality: float = 0.0       # In-degree centrality: called_by count / max_possible
    fan_in: int = 0               # Raw count of unique callers
    fan_out: int = 0              # Raw count of unique callees

    # Route metadata (populated by RouteExtractor)
    route_path: Optional[str] = None
    route_method: Optional[str] = None

    # Scope (for platform/tenant/workspace/user/agent isolation). Empty sub-scope
    # = broader (workspace-wide / tenant-wide). Code is workspace-shared by
    # default (user_id/agent_id ""); set them only for user/agent-private code.
    tenant_id: str = "platform"
    workspace_id: str = ""
    user_id: str = ""
    agent_id: str = ""

    # Embedding (populated by Mind service later). NOTE: overridden by a
    # property below the class — the vector lives on `_embedding` only until
    # _ensure_embedding_matrix() folds it into the shared float32 matrix,
    # after which reads rehydrate the matrix row on demand. Keeps 101K chunks
    # from each pinning a boxed float list (~0.9GB fleet-wide, audit 2026-07).
    embedding: Optional[List[float]] = None

    # Rename-safe stable id (set when AITHER_CODEGRAPH_ID_VERSION=2 finalizes the
    # graph; `id` then equals this and name/path become mutable properties).
    stable_id: Optional[str] = None

    # Relevance score from hybrid_query() — recovered from discarded hybrid_score
    # before return. Used to scale chunk injection priority by intent type.
    relevance_score: float = 0.7

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.chunk_type.value}_{self.name}_{hashlib.sha256(self.source_path.encode()).hexdigest()[:8]}"

    def concept_card(self, max_edges: int = 10) -> Dict[str, Any]:
        """Compact, deterministic concept card — signature + call graph only.

        No body_preview, no docstring body, no embedding: ~30-40 tokens vs
        ~130 for the full serialized chunk. Edge lists are sorted so the same
        index state always yields byte-identical output (vLLM prompt-cache
        friendly, and snapshot diffs stay stable across runs).
        """
        chunk_type_val = (
            self.chunk_type.value if hasattr(self.chunk_type, "value")
            else str(self.chunk_type)
        )
        return {
            "id": self.id,
            "name": self.name,
            "type": chunk_type_val,
            "file": self.source_path,
            "line": self.start_line,
            "signature": self.signature,
            "calls": sorted(set(self.calls))[:max_edges],
            "called_by": sorted(set(self.called_by))[:max_edges],
            "fan_in": self.fan_in,
            "fan_out": self.fan_out,
            "centrality": round(self.centrality, 3),
            "complexity": self.complexity,
        }


# ── CodeChunk.embedding property (memory audit 2026-07) ──────────────────────
# Attached AFTER the dataclass is built so the generated __init__ signature is
# untouched: `self.embedding = ...` in __init__ now routes through the setter.
# The raw vector lives in __dict__["_embedding"]; once the graph folds it into
# the shared float32 matrix (_bind_matrix_row), reads rehydrate the matrix row
# on demand — every existing `chunk.embedding is None` check keeps working,
# in this file and in external readers (codegraph_api stats, service harvest).

def _codechunk_embedding_get(self):
    vec = self.__dict__.get("_embedding")
    if vec is not None:
        return vec
    src = self.__dict__.get("_emb_source")
    if src is not None:
        _mat, _row = src
        try:
            return _mat[_row]
        except Exception:
            return None
    return None


def _codechunk_embedding_set(self, value) -> None:
    self.__dict__["_embedding"] = value
    # A direct assignment supersedes any matrix-row binding — without this a
    # stale row would shadow `embedding = None` (e.g. embed_chunks force=True).
    self.__dict__.pop("_emb_source", None)


def _codechunk_bind_matrix_row(self, mat, row: int) -> None:
    """Release the boxed vector; reads now rehydrate from the matrix row."""
    self.__dict__["_embedding"] = None
    self.__dict__["_emb_source"] = (mat, row)


CodeChunk.embedding = property(_codechunk_embedding_get, _codechunk_embedding_set)
CodeChunk._bind_matrix_row = _codechunk_bind_matrix_row


@dataclass
class FileGraph:
    """Graph of all chunks in a file."""
    source_path: str
    chunks: List[CodeChunk] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    import_map: Dict[str, str] = field(default_factory=dict)
    module_docstring: str = ""
    parse_errors: List[str] = field(default_factory=list)
    processing_ms: float = 0.0


# ============================================================================
# AST VISITOR - Extracts calls from function bodies
# ============================================================================

class CallExtractor(ast.NodeVisitor):
    """Extracts all function/method calls from an AST node."""
    
    def __init__(self):
        self.calls: List[str] = []
    
    def visit_Call(self, node: ast.Call):
        """Extract the name of what's being called or passed as reference."""
        if isinstance(node.func, ast.Name):
            # Simple call: foo()
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.foo()
            if isinstance(node.func.value, ast.Name):
                self.calls.append(f"{node.func.value.id}.{node.func.attr}")
            else:
                self.calls.append(node.func.attr)

        # Also extract any variables/functions passed as arguments
        # (e.g., mcp.tool()(my_target_function) or map(my_func, items))
        for arg in node.args:
            if isinstance(arg, ast.Name):
                self.calls.append(arg.id)
            elif isinstance(arg, ast.Attribute):
                if isinstance(arg.value, ast.Name):
                    self.calls.append(f"{arg.value.id}.{arg.attr}")
                else:
                    self.calls.append(arg.attr)
        
        # Continue visiting children
        self.generic_visit(node)


class RouteExtractor(ast.NodeVisitor):
    """Extracts FastAPI/Starlette route decorators from function definitions."""
    _ROUTE_METHODS = frozenset({"get", "post", "put", "delete", "patch", "options", "head", "api_route", "route"})

    def __init__(self):
        self.routes: List[Dict[str, str]] = []

    def visit_FunctionDef(self, node):
        self._check_decorators(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._check_decorators(node)
        self.generic_visit(node)

    def _check_decorators(self, node):
        for deco in node.decorator_list:
            if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute):
                method = deco.func.attr.lower()
                if method in self._ROUTE_METHODS:
                    path = ""
                    if deco.args and isinstance(deco.args[0], ast.Constant) and isinstance(deco.args[0].value, str):
                        path = deco.args[0].value
                    http_method = method.upper()
                    if method in ("api_route", "route"):
                        for kw in deco.keywords:
                            if kw.arg == "methods" and isinstance(kw.value, ast.List):
                                http_method = ",".join(
                                    el.value.upper() for el in kw.value.elts
                                    if isinstance(el, ast.Constant) and isinstance(el.value, str)
                                )
                        if http_method in ("API_ROUTE", "ROUTE"):
                            http_method = "GET"
                    self.routes.append({"func_name": node.name, "route_path": path, "http_method": http_method})


def extract_routes(tree: ast.AST) -> List[Dict[str, str]]:
    """Extract all FastAPI route decorators from an AST tree."""
    extractor = RouteExtractor()
    extractor.visit(tree)
    return extractor.routes


def extract_calls(node: ast.AST) -> List[str]:
    """Extract all function/method calls from an AST node."""
    extractor = CallExtractor()
    extractor.visit(node)
    return list(set(extractor.calls))  # Deduplicate


def get_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build full function signature from AST node."""
    args = []
    
    # Regular args
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception as e:
                logger.debug(f"[CallExtractor.get_signature] Operation failed: {e}")
        args.append(arg_str)
    
    # Return type
    returns = ""
    if node.returns:
        try:
            returns = f" -> {ast.unparse(node.returns)}"
        except Exception as e:
            logger.debug(f"[CallExtractor.get_signature] Operation failed: {e}")
    
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)}){returns}"


def estimate_complexity(node: ast.AST) -> int:
    """Estimate cyclomatic complexity (simplified)."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


# ============================================================================
# PARSER - Single file parsing
# ============================================================================

def parse_file_sync(file_path: str, content: Optional[str] = None) -> FileGraph:
    """
    Parse a single Python file into a FileGraph.

    This is the CPU-bound work that runs in a process pool.

    `content`, when given, is parsed INSTEAD of reading `file_path` from disk —
    that is how the semantic-VCS capture layer parses a git blob at a specific
    commit. Default None preserves the original read-from-disk behaviour.
    """
    start = time.perf_counter()
    graph = FileGraph(source_path=file_path)

    try:
        if content is None:
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")
        graph.imports = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            graph.parse_errors.append(str(e))
            graph.processing_ms = (time.perf_counter() - start) * 1000
            return graph
        
        # Module-level docstring
        graph.module_docstring = ast.get_docstring(tree) or ""
        
        # Extract imports first
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    graph.imports.append(alias.name)
                    graph.import_map[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    graph.imports.append(node.module)
                    for alias in node.names:
                        graph.import_map[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        
        # Module-level chunk: docstring + top-level constants.
        #
        # Without this, ONLY functions/classes/methods are indexed, so facts that
        # live at module scope are invisible to every query. Verified on the live
        # index: 59,128 chunks, zero module-level entries, and "What port does
        # AitherCodeGraph run on?" missed the top-200 of BOTH keyword and semantic
        # search — the answer (8194) exists only in the module docstring and in
        # module-level get_port() calls. That whole question class ("what port",
        # "default timeout", "which URL", "which model") failed SILENTLY, returning
        # ten confident wrong results rather than nothing.
        module_chunk = _extract_module(tree, file_path, graph.module_docstring)
        if module_chunk is not None:
            graph.chunks.append(module_chunk)

        # First pass: extract all functions and classes
        class_methods: Dict[str, List[str]] = defaultdict(list)  # class_name -> [method_names]
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunk = _extract_class(node, lines, file_path, graph.imports, graph.import_map)
                graph.chunks.append(chunk)
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_chunk = _extract_method(item, lines, file_path, node.name, graph.imports, graph.import_map)
                        graph.chunks.append(method_chunk)
                        class_methods[node.name].append(item.name)
                
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = _extract_function(node, lines, file_path, graph.imports, graph.import_map)
                graph.chunks.append(chunk)
        
        # Update class chunks with their methods
        for chunk in graph.chunks:
            if chunk.chunk_type == ChunkType.CLASS:
                chunk.methods = class_methods.get(chunk.name, [])

        # Extract routes and attach to function/method chunks
        routes = extract_routes(tree)
        route_by_func = {r["func_name"]: r for r in routes}
        for chunk in graph.chunks:
            short_name = chunk.name.split(".")[-1] if "." in chunk.name else chunk.name
            if short_name in route_by_func:
                r = route_by_func[short_name]
                chunk.route_path = r["route_path"]
                chunk.route_method = r["http_method"]

    except Exception as e:
        graph.parse_errors.append(str(e))

    graph.processing_ms = (time.perf_counter() - start) * 1000
    return graph


def parse_source_bytes(content: bytes, path: str) -> FileGraph:
    """Parse Python source bytes into a FileGraph WITHOUT touching disk.

    Used by the semantic-VCS capture layer (``lib/awgit``) to parse a git blob
    (``git show <sha>:<path>``) at a specific commit — node identity comes from
    the live graph, body hashes from the exact commit's blobs, so an edit-op
    stays self-consistent even if the working tree has moved on.

    Args:
        content: The file's source bytes (UTF-8; other encodings degrade to
                 'ignore', matching ``parse_file_sync``).
        path:    Repo-relative path — becomes the ``FileGraph.source_path``.
    """
    return parse_file_sync(path, content=content.decode("utf-8", errors="ignore"))


_MODULE_CONST_LIMIT = 60
_MODULE_VALUE_CHARS = 120


def _extract_module(
    tree: ast.Module, source_path: str, docstring: str
) -> Optional[CodeChunk]:
    """One searchable chunk per file carrying module scope.

    Holds the module docstring plus top-level constant assignments — where ports,
    default timeouts, URLs and model names actually live. These are not reachable
    from any function/class/method chunk, so before this they were unindexed and
    the corresponding questions were unanswerable rather than merely mis-ranked.

    Returns None when there is nothing at module scope worth indexing, so we do
    not add ~2.4k empty chunks that would dilute every ranking.
    """
    consts: List[str] = []
    for node in ast.iter_child_nodes(tree):
        targets: List[str] = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        if not targets or value is None:
            continue
        try:
            rendered = ast.unparse(value)
        except Exception:
            continue
        if len(rendered) > _MODULE_VALUE_CHARS:
            rendered = rendered[:_MODULE_VALUE_CHARS] + "..."
        for t in targets:
            # Skip private/dunder scaffolding; keep real configuration.
            if t.startswith("__"):
                continue
            consts.append(f"{t} = {rendered}")
        if len(consts) >= _MODULE_CONST_LIMIT:
            break

    if not docstring and not consts:
        return None

    name = Path(source_path).stem
    body = "\n".join(consts)
    return CodeChunk(
        id=f"module_{name}_{hashlib.sha256(source_path.encode()).hexdigest()[:8]}",
        name=name,
        chunk_type=ChunkType.MODULE,
        source_path=source_path,
        start_line=1,
        end_line=1,
        signature=f"module {name}",
        docstring=docstring or "",
        body_preview=body,
        line_count=len(consts),
    )


def _extract_class(node: ast.ClassDef, lines: List[str], source_path: str, imports: List[str], import_map: Dict[str, str]) -> CodeChunk:
    """Extract a class definition."""
    start_line = node.lineno
    end_line = node.end_lineno or start_line
    
    # Get base classes
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(base.attr)
    
    # Get docstring
    docstring = ast.get_docstring(node) or ""
    
    # Build signature
    signature = f"class {node.name}"
    if bases:
        signature += f"({', '.join(bases)})"
    
    # Extract calls from class body (decorators, default values, etc.)
    calls = extract_calls(node)
    
    return CodeChunk(
        id=f"class_{node.name}_{hashlib.sha256(source_path.encode()).hexdigest()[:8]}",
        name=node.name,
        chunk_type=ChunkType.CLASS,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring[:500],
        body_preview="\n".join(lines[start_line-1:min(start_line+10, end_line)])[:500],
        imports=[i for i in imports if i in bases],  # Relevant imports
        import_map=import_map,
        calls=calls,
        base_classes=bases,
        complexity=estimate_complexity(node),
        line_count=end_line - start_line + 1,
    )


def _extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str], source_path: str, imports: List[str], import_map: Dict[str, str]) -> CodeChunk:
    """Extract a function definition."""
    start_line = node.lineno
    end_line = node.end_lineno or start_line
    
    docstring = ast.get_docstring(node) or ""
    signature = get_signature(node)
    calls = extract_calls(node)
    
    return CodeChunk(
        id=f"func_{node.name}_{hashlib.sha256(source_path.encode()).hexdigest()[:8]}",
        name=node.name,
        chunk_type=ChunkType.FUNCTION,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring[:500],
        body_preview="\n".join(lines[start_line-1:min(start_line+10, end_line)])[:500],
        imports=imports,
        import_map=import_map,
        calls=calls,
        complexity=estimate_complexity(node),
        line_count=end_line - start_line + 1,
    )


def _extract_method(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str], source_path: str, class_name: str, imports: List[str], import_map: Dict[str, str]) -> CodeChunk:
    """Extract a method definition."""
    start_line = node.lineno
    end_line = node.end_lineno or start_line
    
    docstring = ast.get_docstring(node) or ""
    signature = get_signature(node)
    calls = extract_calls(node)
    
    return CodeChunk(
        id=f"method_{class_name}_{node.name}_{hashlib.sha256(source_path.encode()).hexdigest()[:8]}",
        name=f"{class_name}.{node.name}",
        chunk_type=ChunkType.METHOD,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring[:500],
        body_preview="\n".join(lines[start_line-1:min(start_line+5, end_line)])[:300],
        imports=imports,
        import_map=import_map,
        calls=calls,
        parent_class=class_name,
        complexity=estimate_complexity(node),
        line_count=end_line - start_line + 1,
    )


# ============================================================================
# FAST FILE DISCOVERY
# ============================================================================

# Persist the embedding cache every N batches. The end-of-run-only save meant
# any failure lost the entire run (measured: died at ~6min having reached 62%
# coverage, next attempt restarted from 0). Env-tunable; 0 disables.
_CHECKPOINT_EVERY = int(os.getenv("AITHER_CODEGRAPH_EMBED_CHECKPOINT_BATCHES", "50"))

# LRU cap for lazy-loaded chunk body cache (in-memory full source text).
# Prevents unbounded growth when accessing large codebases. Each entry stores
# full source lines for a chunk; 256 entries ≈ ~2.5MB assuming 10KB avg per chunk.
_BODY_CACHE_MAX = int(os.getenv("AITHER_CODEGRAPH_BODY_CACHE_MAX", "256"))


def _drop_mirrored_duplicates(files: List[Path], root: Path) -> List[Path]:
    """Drop files from a mirrored copy of the tree that is already indexed.

    The service image COPYs `AitherOS/lib/` to BOTH `/app/lib` and
    `/app/AitherOS/lib` so either import path resolves. Indexing `/app` then
    picks up every library file TWICE (measured: 2,833 files duplicated), which
    both inflates the index and — far worse — makes each symbol appear twice in
    results, halving the effective diversity of any top-N retrieval.

    Dedupe by path SHAPE rather than by excluding a directory name: an
    `--exclude AitherOS` would be catastrophic on the host checkout, where the
    real code lives under `AitherOS/` and that pattern would exclude the entire
    codebase. Here we only drop `<root>/AitherOS/<tail>` when `<root>/<tail>`
    was also discovered, so a tree that exists ONLY under `AitherOS/` (i.e. the
    host layout) is left completely untouched.
    """
    try:
        rels = {}
        for f in files:
            try:
                rels[f.relative_to(root).as_posix()] = f
            except ValueError:
                continue
        if not rels:
            return files
        dropped = set()
        for rel in rels:
            if rel.startswith("AitherOS/") and rel[len("AitherOS/"):] in rels:
                dropped.add(rel)
        if not dropped:
            return files
        logger.info(
            f"[CodeGraph] Dropped {len(dropped)} mirrored duplicate files "
            f"(AitherOS/ copy of an already-discovered path)"
        )
        return [f for rel, f in rels.items() if rel not in dropped]
    except Exception as e:
        # Never let a dedupe optimisation break indexing.
        logger.warning(f"[CodeGraph] mirror-dedupe skipped: {e}")
        return files


async def discover_python_files(root: Path) -> Tuple[List[Path], float]:
    """Use ripgrep/fd for fast file discovery."""
    start = time.perf_counter()
    files: List[Path] = []
    
    # Try fd first (note: fd syntax is `fd <PATTERN> <PATH>`, use "." to match all)
    try:
        result = await asyncio.create_subprocess_exec(
            "fd", ".", str(root),
            "-e", "py", "--type", "f",
            "--exclude", ".git",
            "--exclude", "node_modules",
            "--exclude", "__pycache__",
            "--exclude", ".venv",
            "--exclude", "venv",
            "--exclude", "Worktrees",
            "--exclude", ".worktrees",
            "--exclude", "site-packages",
            "--exclude", "runtime",
            "--exclude", "training-data",
            "--exclude", "test_artifacts",
            "--exclude", "_archive",
            "--exclude", "external",
            # Bind-mounted data tree: vendored repos under Library/repos are
            # NOT this codebase (register them as separate registry roots) —
            # indexing them quintupled the index and poisoned locate()'s
            # suffix matching with duplicate paths.
            "--exclude", "Library",
            "--exclude", "dormant_llm_puzzle*",
            "--exclude", "DormantPuzzle*",
            "--exclude", "PromptDrivenStrategy*",
            # Build/tooling artifacts — not source. `dist/` alone was ~2,291
            # compiled/packaged .py files (≈30% of discovery), inflating the
            # index with duplicates and driving needless reconcile work.
            "--exclude", "dist",
            "--exclude", "build",
            "--exclude", ".next",
            "--exclude", ".turbo",
            "--exclude", ".pytest_cache",
            "--exclude", ".mypy_cache",
            "--exclude", ".ruff_cache",
            "--exclude", ".tox",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()

        if result.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                if line:
                    files.append(Path(line))
            return files, (time.perf_counter() - start) * 1000
    except FileNotFoundError as e:
        logger.debug(f"[CallExtractor.discover_python_files] Operation failed: {e}")
    
    # Fallback to ripgrep
    try:
        result = await asyncio.create_subprocess_exec(
            "rg", "--files", "-g", "*.py",
            "--glob", "!.git/**",
            "--glob", "!node_modules/**",
            "--glob", "!__pycache__/**",
            "--glob", "!.venv/**",
            "--glob", "!venv/**",
            "--glob", "!Worktrees/**",
            "--glob", "!.worktrees/**",
            "--glob", "!site-packages/**",
            "--glob", "!runtime/**",
            "--glob", "!training-data/**",
            "--glob", "!test_artifacts/**",
            "--glob", "!_archive/**",
            "--glob", "!external/**",
            "--glob", "!Library/**",
            # Keep the rg fallback in sync with the fd backend so discovery is
            # deterministic across backends (divergence caused reconcile churn).
            "--glob", "!dist/**",
            "--glob", "!build/**",
            "--glob", "!.next/**",
            "--glob", "!.turbo/**",
            "--glob", "!.pytest_cache/**",
            "--glob", "!.mypy_cache/**",
            "--glob", "!.ruff_cache/**",
            "--glob", "!.tox/**",
            "--glob", "!dormant_llm_puzzle*/**",
            "--glob", "!DormantPuzzle*/**",
            "--glob", "!PromptDrivenStrategy*/**",
            str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        
        if result.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                if line:
                    p = Path(line)
                    files.append(p if p.is_absolute() else root / p)
            return files, (time.perf_counter() - start) * 1000
    except FileNotFoundError as e:
        logger.debug(f"[CallExtractor.discover_python_files] Operation failed: {e}")
    
    # Final fallback — rglob is sync I/O; offload to thread so we don't
    # block the event loop (especially dangerous on Docker bind mounts).
    _RGLOB_EXCLUDE_PARTS = frozenset({
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "Worktrees", "worktrees", ".worktrees",
        "site-packages", "dist-packages",
        "runtime", "training-data", "test_artifacts",
        "_archive", "Library",
        "Canvas-Studio", "external", "AEON_PORTRAITS",
        "affect_gallery", "simulator-temp",
    })
    _RGLOB_MAX_FILES = 2000  # Safety cap — prevents runaway crawl
    _RGLOB_TIMEOUT_S = 30    # Max seconds for rglob before giving up

    def _sync_rglob():
        found = []
        deadline = time.perf_counter() + _RGLOB_TIMEOUT_S
        for f in root.rglob("*.py"):
            if time.perf_counter() > deadline:
                logger.warning(
                    f"[CodeGraph] rglob timeout after {_RGLOB_TIMEOUT_S}s "
                    f"({len(found)} files found so far) — indexing partial set"
                )
                break
            if _RGLOB_EXCLUDE_PARTS.isdisjoint(f.parts):
                found.append(f)
                if len(found) >= _RGLOB_MAX_FILES:
                    logger.info(
                        f"[CodeGraph] rglob hit {_RGLOB_MAX_FILES} file cap — "
                        f"install fd-find for faster/complete discovery"
                    )
                    break
        return found

    files.extend(await asyncio.to_thread(_sync_rglob))
    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > 5000:
        logger.warning(
            f"[CodeGraph] Slow file discovery: {elapsed_ms:.0f}ms for {len(files)} files. "
            f"Install fd-find in Docker image to fix this."
        )
    return files, elapsed_ms


# ============================================================================
# MAIN CODE GRAPH CLASS
# ============================================================================

class CodeGraph(BaseFacultyGraph):
    """
    Python code indexer with full call graph.

    Indexes Python code and builds a graph of:
    - What each function/method calls
    - What calls each function/method (backfilled)
    """

    _scope_level = "tenant"  # Chunks carry tenant_id for multi-tenant isolation
    _QUERY_CACHE_MAX = 512  # Max cached query embeddings (~1.5MB at 768-dim)

    def __init__(
        self,
        max_workers: int = 8,
        root_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        auto_index: bool = False,
    ):
        super().__init__()
        # Tenant partition tracking: tenant_id -> set of chunk_ids
        self._tenant_chunks: Dict[str, set] = defaultdict(set)
        self._sync_config = GraphSyncConfig(
            enabled=True,
            domain="code",
            source_graph="CodeGraph",
            batch_size=50,
            provenance=True,
        )
        self.max_workers = max_workers

        # The index
        self.chunks: Dict[str, CodeChunk] = {}  # id -> chunk
        self.by_name: Dict[str, List[str]] = defaultdict(list)  # name -> [chunk_ids]
        self.by_file: Dict[str, List[str]] = defaultdict(list)  # file -> [chunk_ids]
        self.by_class: Dict[str, List[str]] = defaultdict(list)  # parent_class -> [chunk_ids]
        self.routes: Dict[str, str] = {}  # "GET /api/v1/chat" -> chunk_id

        # Query embedding cache: query_text → embedding vector
        self._query_embed_cache: Dict[str, list] = {}
        self._query_cache_order: List[str] = []  # LRU eviction order

        # Keyword query result cache: "query:max_results" → List[CodeChunk]
        # Brute-force scan of 28K chunks takes ~60ms; cache turns repeats to ~0ms
        self._keyword_result_cache: Dict[str, List] = {}
        self._keyword_cache_order: List[str] = []  # LRU eviction order
        self._KEYWORD_CACHE_MAX = 64

        # Pre-computed embedding matrix for fast cosine similarity
        # Building np.array from 28K chunk embeddings takes ~500ms — do it ONCE
        self._embedding_matrix: Optional[Any] = None  # np.ndarray or None
        self._embedding_ids: Optional[List[str]] = None  # chunk IDs aligned with matrix rows
        self._embedding_norms: Optional[Any] = None  # pre-computed row norms
        self._embedding_penalties: Optional[Any] = None  # pre-computed test-file penalty vector
        self._embedding_row_map: Optional[Dict[str, int]] = None  # chunk id → matrix row

        # Stats
        self.total_files = 0
        self.discovery_ms = 0.0
        self.parsing_ms = 0.0
        self.backfill_ms = 0.0

        # Full body cache: chunk_id → full source text (lazy loaded)
        # OrderedDict with LRU eviction: move_to_end on hit, popitem(last=False) on overflow
        self._body_cache: OrderedDict[str, str] = OrderedDict()
        self._root_path: Optional[str] = root_path
        self._cache_dir: Optional[str] = cache_dir
    
    async def index_codebase(
        self,
        root_path: str,
        on_progress: Optional[callable] = None,
        tenant_id: str = "platform",
    ) -> Dict[str, Any]:
        """
        Index a codebase with full call graph analysis.
        
        Returns stats about the indexing operation.
        """
        root = Path(root_path)
        total_start = time.perf_counter()
        
        # Phase 1: Discovery
        if on_progress:
            on_progress(0.0, "Discovering files...")
        
        files, self.discovery_ms = await discover_python_files(root)
        files = _drop_mirrored_duplicates(files, root)
        self.total_files = len(files)
        
        logger.info(f"Discovered {len(files)} files in {self.discovery_ms:.0f}ms")
        
        if on_progress:
            on_progress(0.1, f"Found {len(files)} files")
        
        # Phase 2: Parse in parallel
        if on_progress:
            on_progress(0.15, "Parsing files...")

        parse_start = time.perf_counter()
        loop = asyncio.get_event_loop()

        # In Docker, ProcessPoolExecutor reliably crashes due to /dev/shm limits.
        # Use ThreadPoolExecutor directly in Docker to avoid BrokenProcessPool spam.
        _in_docker = os.path.exists("/.dockerenv")

        results: List[FileGraph] = []
        pool_type = "thread" if _in_docker else "process"
        
        if pool_type == "process":
            try:
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                    tasks = [
                        loop.run_in_executor(pool, parse_file_sync, str(f))
                        for f in files
                    ]
                    for i, coro in enumerate(asyncio.as_completed(tasks)):
                        result = await coro
                        results.append(result)
                        if on_progress and (i + 1) % 50 == 0:
                            progress = 0.15 + 0.6 * (i + 1) / len(files)
                            on_progress(progress, f"Parsed {i+1}/{len(files)}")
            except (BrokenProcessPool, BrokenPipeError, OSError, RuntimeError) as e:
                # ProcessPool failed (Docker /dev/shm, fork issues, etc.)
                logger.warning(f"ProcessPoolExecutor failed ({e}), falling back to ThreadPoolExecutor")
                results.clear()
                pool_type = "thread"

        if pool_type == "thread":
            if _in_docker:
                logger.info("Using ThreadPoolExecutor (Docker mode)")
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                # Process in batches to avoid GIL starvation and keep event loop responsive
                # BATCH_SIZE = self.max_workers * 2 keeps the pipeline full without flooding
                batch_size = max(10, self.max_workers * 2)
                total_files = len(files)
                
                for i in range(0, total_files, batch_size):
                    batch = files[i : i + batch_size]
                    tasks = [
                        loop.run_in_executor(pool, parse_file_sync, str(f))
                        for f in batch
                    ]
                    # Process batch results as they complete
                    for coro in asyncio.as_completed(tasks):
                        try:
                            result = await coro
                            results.append(result)
                        except Exception as e:
                            logger.warning(f"Failed to parse file in batch: {e}")
                    
                    # Update progress
                    processed_count = min(i + batch_size, total_files)
                    if on_progress and processed_count % 50 == 0:
                        progress = 0.15 + 0.6 * processed_count / total_files
                        on_progress(progress, f"Parsed {processed_count}/{total_files} (thread)")
                    
                    # Yield to event loop to prevent timeouts
                    await asyncio.sleep(0.05)
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    result = await coro
                    results.append(result)
                    if on_progress and (i + 1) % 50 == 0:
                        progress = 0.15 + 0.6 * (i + 1) / len(files)
                        on_progress(progress, f"Parsed {i+1}/{len(files)}")
        
        self.parsing_ms = (time.perf_counter() - parse_start) * 1000
        
        # Phase 3: Build index
        if on_progress:
            on_progress(0.75, "Building index...")
        
        for graph in results:
            for chunk in graph.chunks:
                chunk.tenant_id = tenant_id
                self.chunks[chunk.id] = chunk
                self._tenant_chunks[tenant_id].add(chunk.id)
                self.by_name[chunk.name].append(chunk.id)
                self.by_file[graph.source_path].append(chunk.id)
                if chunk.parent_class:
                    self.by_class[chunk.parent_class].append(chunk.id)
                if chunk.route_path is not None:
                    route_key = f"{chunk.route_method or 'GET'} {chunk.route_path}"
                    self.routes[route_key] = chunk.id
                # Sync to AitherKnowledgeGraph (fire-and-forget)
                self._queue_sync({
                    "id": chunk.id,
                    "name": chunk.name,
                    "type": chunk.chunk_type.value,
                    "properties": {
                        "source_path": chunk.source_path,
                        "signature": chunk.signature,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "parent_class": chunk.parent_class or "",
                        "complexity": chunk.complexity,
                        "centrality": chunk.centrality,
                        "fan_in": chunk.fan_in,
                        "fan_out": chunk.fan_out,
                        "git_commits": chunk.git_commits,
                        "git_churn_rate": chunk.git_churn_rate,
                    },
                }, tenant_id=tenant_id)
        # Flush remaining sync queue
        self._flush_to_bus()

        # Phase 4: Backfill called_by relationships
        if on_progress:
            on_progress(0.85, "Building call graph...")
        
        backfill_start = time.perf_counter()
        self._backfill_called_by()
        # v2: re-key freshly-parsed v1 ids to rename-safe stable ids (no-op on v1)
        _finalize_stable_ids(self)
        self._compute_centrality()
        self._invalidate_keyword_cache()
        self.backfill_ms = (time.perf_counter() - backfill_start) * 1000

        # Phase 5: Git stats enrichment (non-blocking, best-effort)
        git_stats_result = {}
        if on_progress:
            on_progress(0.90, "Enriching with git stats...")
        try:
            git_stats_result = await self.enrich_with_git_stats(root_path)
        except Exception as e:
            logger.debug(f"Git stats enrichment failed (non-fatal): {e}")
            git_stats_result = {"error": str(e)[:200]}

        if on_progress:
            on_progress(1.0, "Complete!")

        total_ms = (time.perf_counter() - total_start) * 1000

        # Count high-risk nodes
        high_risk = [c for c in self.chunks.values() if c.centrality >= 0.5]

        stats = {
            "total_files": self.total_files,
            "total_chunks": len(self.chunks),
            "functions": sum(1 for c in self.chunks.values() if c.chunk_type == ChunkType.FUNCTION),
            "methods": sum(1 for c in self.chunks.values() if c.chunk_type == ChunkType.METHOD),
            "classes": sum(1 for c in self.chunks.values() if c.chunk_type == ChunkType.CLASS),
            "high_centrality_nodes": len(high_risk),
            "git_enrichment": git_stats_result,
            "discovery_ms": self.discovery_ms,
            "parsing_ms": self.parsing_ms,
            "backfill_ms": self.backfill_ms,
            "total_ms": total_ms,
        }
        
        logger.info(f"Indexing complete: {stats}")
        return stats

    async def index_tenant_code(
        self,
        root_path: str,
        tenant_id: str,
        workspace_id: str = "",
        on_progress: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Index a tenant's workspace code separately from platform code.

        Tenant chunks coexist in the same index but are filtered by tenant_id
        in queries. Each tenant's chunks are tracked for targeted purging.
        """
        if not tenant_id or tenant_id == "platform":
            raise ValueError("tenant_id must be a non-platform tenant")

        # Purge stale tenant chunks before re-indexing
        self.purge_tenant(tenant_id)

        stats = await self.index_codebase(
            root_path=root_path,
            on_progress=on_progress,
            tenant_id=tenant_id,
        )

        # Set workspace_id on all newly-indexed chunks
        if workspace_id:
            for cid in self._tenant_chunks.get(tenant_id, set()):
                if cid in self.chunks:
                    self.chunks[cid].workspace_id = workspace_id

        stats["tenant_id"] = tenant_id
        stats["workspace_id"] = workspace_id
        return stats

    def purge_tenant(self, tenant_id: str) -> int:
        """Remove all chunks belonging to a tenant. Returns count removed."""
        chunk_ids = self._tenant_chunks.pop(tenant_id, set())
        removed = 0
        for cid in chunk_ids:
            chunk = self.chunks.pop(cid, None)
            if not chunk:
                continue
            removed += 1
            # Clean secondary indices
            if chunk.name in self.by_name:
                try:
                    self.by_name[chunk.name].remove(cid)
                except ValueError:
                    pass
            if chunk.source_path in self.by_file:
                try:
                    self.by_file[chunk.source_path].remove(cid)
                except ValueError:
                    pass
            if chunk.parent_class and chunk.parent_class in self.by_class:
                try:
                    self.by_class[chunk.parent_class].remove(cid)
                except ValueError:
                    pass
        if removed:
            self._invalidate_embedding_matrix()
            self._invalidate_keyword_cache()
            logger.info(f"[CodeGraph] Purged {removed} chunks for tenant {tenant_id}")
        return removed

    def tenant_stats(self) -> Dict[str, Dict[str, int]]:
        """Return chunk counts per tenant for monitoring."""
        stats: Dict[str, Dict[str, int]] = {}
        for tid, cids in self._tenant_chunks.items():
            stats[tid] = {"chunks": len(cids)}
        # Count platform chunks not explicitly tracked
        tracked = set()
        for cids in self._tenant_chunks.values():
            tracked |= cids
        platform_count = len(self.chunks) - len(tracked)
        if platform_count > 0:
            stats.setdefault("platform", {})["chunks"] = platform_count
        return stats

    async def enrich_with_langextract(
        self,
        root_path: str,
        max_files: int = 50,
        on_progress: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Enrich indexed chunks with LangExtract concept extraction.

        Runs structured extraction on docstrings/comments to identify
        concepts, algorithms, relationships, and entities. Results are
        stored as metadata on each CodeChunk for downstream graph ingest.

        Requires: pip install langextract

        Args:
            root_path: Codebase root (used for relative paths).
            max_files: Max files to run extraction on (most documented first).
            on_progress: Progress callback.

        Returns:
            Stats dict with enrichment counts.
        """
        try:
            from lib.faculties.LangExtractFaculty import (
                LangExtractFaculty, LANGEXTRACT_AVAILABLE,
            )
        except ImportError:
            logger.warning("LangExtractFaculty not available — skipping enrichment")
            return {"enriched_chunks": 0, "extractions": 0, "skipped": "not_installed"}

        if not LANGEXTRACT_AVAILABLE:
            return {"enriched_chunks": 0, "extractions": 0, "skipped": "not_installed"}

        faculty = LangExtractFaculty()
        if not await faculty.initialize():
            return {"enriched_chunks": 0, "extractions": 0, "skipped": "init_failed"}

        if on_progress:
            on_progress(0.0, "Starting LangExtract enrichment...")

        # Prioritize chunks with docstrings (most documentation = most value)
        documented_chunks = [
            c for c in self.chunks.values()
            if c.docstring and len(c.docstring.strip()) > 20
        ]
        # Sort by docstring length descending (richest docs first)
        documented_chunks.sort(key=lambda c: len(c.docstring), reverse=True)
        documented_chunks = documented_chunks[:max_files]

        enriched = 0
        total_extractions = 0

        for i, chunk in enumerate(documented_chunks):
            try:
                text = f"{chunk.name}\n{chunk.signature}\n{chunk.docstring}"
                results = await faculty.extract_from_text(
                    text=text,
                    task="code_concepts",
                    source_file=chunk.source_path,
                )
                if results:
                    # Store extractions as metadata on the chunk
                    chunk.imports = list(set(
                        chunk.imports + [
                            r.text for r in results
                            if r.extraction_class in ("concept", "algorithm")
                        ]
                    ))
                    # Store full extraction data for graph ingest
                    if not hasattr(chunk, 'extractions'):
                        object.__setattr__(chunk, 'extractions', [])
                    for r in results:
                        chunk.extractions.append(r.to_dict())  # type: ignore[attr-defined]
                    enriched += 1
                    total_extractions += len(results)
            except Exception as e:
                logger.debug(f"LangExtract enrichment failed for {chunk.name}: {e}")

            if on_progress and (i + 1) % 10 == 0:
                on_progress(
                    (i + 1) / len(documented_chunks),
                    f"Enriched {enriched}/{i + 1} chunks",
                )

        stats = {
            "enriched_chunks": enriched,
            "extractions": total_extractions,
            "candidates": len(documented_chunks),
        }
        logger.info(f"LangExtract enrichment complete: {stats}")
        return stats

    def _path_to_module_prefix(self, path: str) -> str:
        """Convert a file path to a possible module prefix (e.g. lib.faculties.CodeGraph)."""
        p = path.replace("\\", "/")
        if p.endswith(".py"):
            p = p[:-3]
        if p.endswith("/__init__"):
            p = p[:-9]
        return p.replace("/", ".")

    def _backfill_called_by(self):
        """
        Build the called_by relationships by inverting the calls graph.
        
        This is the key step that enables "what calls this function?"
        """
        # Build a map of name -> chunk_ids
        name_map: Dict[str, List[str]] = defaultdict(list)
        for chunk_id, chunk in self.chunks.items():
            # Map both full name and short name
            name_map[chunk.name].append(chunk_id)
            if "." in chunk.name:
                short_name = chunk.name.split(".")[-1]
                name_map[short_name].append(chunk_id)
        
        # For each chunk, add it to the called_by list of what it calls
        for caller_id, caller in self.chunks.items():
            for called_name in caller.calls:
                # Determine expected module if it comes from an import
                expected_module = None
                if getattr(caller, 'import_map', None):
                    if "." in called_name:
                        base_name = called_name.split(".")[0]
                        if base_name in caller.import_map:
                            expected_module = caller.import_map[base_name]
                    elif called_name in caller.import_map:
                        expected_module = caller.import_map[called_name]
                
                # Find chunks matching this name
                potential_callees = name_map.get(called_name, [])
                for callee_id in potential_callees:
                    callee = self.chunks[callee_id]
                    
                    # Same file calls are always valid
                    if caller.source_path == callee.source_path:
                        if caller.name not in callee.called_by:
                            callee.called_by.append(caller.name)
                        continue
                        
                    # If we expect a specific module origin, enforce it
                    if expected_module:
                        callee_mod = self._path_to_module_prefix(callee.source_path)
                        expected_base = expected_module
                        if "." in expected_module:
                            # Strip off the imported functionality name if it equals the chunk name
                            mod_parts = expected_module.split(".")
                            short_callee_name = callee.name.split(".")[-1] if "." in callee.name else callee.name
                            if mod_parts[-1] == short_callee_name or callee.name.startswith(f"{mod_parts[-1]}."):
                                expected_base = ".".join(mod_parts[:-1])

                        if not callee_mod.endswith(expected_base):
                            continue # Skip: false positive match from a different file

                    if caller.name not in callee.called_by:
                        callee.called_by.append(caller.name)

    # ── graph-integrity hooks (implicit call edges) ──────────────────────

    def _integrity_node_exists(self, ref: str) -> bool:
        # a call target resolves if it's a known chunk id OR a known symbol name
        return ref in self.chunks or bool(self.by_name.get(ref))

    def _integrity_iter_refs(self):
        for cid, chunk in list(self.chunks.items()):
            for callee in list(getattr(chunk, "calls", []) or []):
                yield (cid, callee, "calls")

    def _integrity_remove_ref(self, source_id: str, target_ref: str, kind: str) -> bool:
        chunk = self.chunks.get(source_id)
        if chunk and target_ref in getattr(chunk, "calls", []):
            chunk.calls = [c for c in chunk.calls if c != target_ref]
            return True
        return False

    def _compute_centrality(self):
        """
        Compute graph centrality metrics for all chunks after backfill.

        Sets fan_in (unique callers), fan_out (unique callees), and
        centrality (normalized in-degree: fan_in / max_fan_in).
        """
        if not self.chunks:
            return

        max_fan_in = 1  # avoid division by zero

        for chunk in self.chunks.values():
            chunk.fan_in = len(chunk.called_by)
            chunk.fan_out = len(chunk.calls)
            if chunk.fan_in > max_fan_in:
                max_fan_in = chunk.fan_in

        for chunk in self.chunks.values():
            chunk.centrality = chunk.fan_in / max_fan_in

    async def enrich_with_git_stats(
        self,
        root_path: str,
        max_commits: int = 500,
    ) -> Dict[str, Any]:
        """
        Enrich indexed chunks with git history data.

        Runs `git log --numstat` once and distributes commit counts,
        contributor counts, and churn rates to each file's chunks.

        Args:
            root_path: Repo root (must be a git repo).
            max_commits: Max commits to scan (default 500).

        Returns:
            Stats dict with enrichment counts.
        """
        import subprocess as _sp
        from datetime import datetime as _dt

        root = Path(root_path)
        if not (root / ".git").exists():
            return {"enriched_files": 0, "error": "not_a_git_repo"}

        # Run git log once: hash, author, date, then --numstat for changed files
        try:
            # utf-8/replace, not text=True: locale-codec (cp1252) decode errors
            # in the pipe-reader thread silently null out stdout on Windows
            result = _sp.run(
                [
                    "git", "log", f"--max-count={max_commits}",
                    "--format=COMMIT|%H|%aN|%aI", "--numstat",
                ],
                capture_output=True, encoding="utf-8", errors="replace", cwd=str(root),
                # 30s was a MARGINAL cap, not a safe one. Measured
                # in-container on an idle box: this exact command takes 27.7s for
                # 500 commits — a ~8% margin — because .git is bind-mounted from a
                # Windows host over 9P. During an actual index run (parsing ~67s
                # and backfill ~57s of CPU alongside it) it lost that race EVERY
                # time, so `enriched_files` was always 0 while the index still
                # reported success. Overridable for large repos.
                timeout=float(os.getenv("AITHER_CODEGRAPH_GIT_TIMEOUT", "180")),
            )
            if result.returncode != 0:
                return {"enriched_files": 0, "error": result.stderr[:200]}
        except Exception as e:
            return {"enriched_files": 0, "error": str(e)[:200]}

        # Parse git log output into per-file stats
        file_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "commits": 0, "authors": set(), "last_date": "",
        })

        current_author = ""
        current_date = ""
        earliest_date = ""
        latest_date = ""

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("COMMIT|"):
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    current_author = parts[2]
                    current_date = parts[3][:10]  # YYYY-MM-DD
                    if not latest_date or current_date > latest_date:
                        latest_date = current_date
                    if not earliest_date or current_date < earliest_date:
                        earliest_date = current_date
                continue
            # numstat line: added\tremoved\tfilename
            parts = line.split("\t", 2)
            if len(parts) == 3 and parts[2]:
                fpath = parts[2]
                if not fpath.endswith(".py"):
                    continue
                stats = file_stats[fpath]
                stats["commits"] += 1
                stats["authors"].add(current_author)
                if not stats["last_date"] or current_date > stats["last_date"]:
                    stats["last_date"] = current_date

        # Compute time span for churn rate
        months_span = 1.0
        if earliest_date and latest_date:
            try:
                d0 = _dt.strptime(earliest_date, "%Y-%m-%d")
                d1 = _dt.strptime(latest_date, "%Y-%m-%d")
                months_span = max(1.0, (d1 - d0).days / 30.0)
            except ValueError:
                pass

        # Index the chunk files by their trailing path segments ONCE, so matching
        # is a dict lookup instead of an O(git_paths x indexed_files) scan.
        #
        # The old `normalized.endswith(rel_path) or rel_path.endswith(normalized)`
        # test matched NOTHING in the container. git log emits repo-relative
        # paths ("AitherOS/lib/faculties/CodeGraph.py") while the image COPYs
        # AitherOS/lib to /app/lib, so the indexed path is
        # "/app/lib/faculties/CodeGraph.py" — the "AitherOS" segment is present on
        # one side and absent on the other, so NEITHER string is a suffix of the
        # other and every file silently failed to match. Same path-shape mismatch
        # that orphaned every embedding in an earlier incident.
        #
        # Keying on the last 3 segments makes it layout-independent while staying
        # specific enough that same-named files in different packages
        # (e.g. */models.py) do not collide.
        def _tail(p: str, n: int = 3) -> str:
            return "/".join(p.replace("\\", "/").rstrip("/").split("/")[-n:])

        by_tail: Dict[str, List[str]] = defaultdict(list)
        for indexed_path, chunk_ids in self.by_file.items():
            by_tail[_tail(indexed_path)].extend(chunk_ids)

        # Distribute stats to chunks
        enriched_files = 0
        for rel_path, stats in file_stats.items():
            matching_chunk_ids = by_tail.get(_tail(rel_path), [])

            if not matching_chunk_ids:
                continue

            enriched_files += 1
            commits = stats["commits"]
            contributors = len(stats["authors"])
            last_mod = stats["last_date"]
            churn = commits / months_span

            for chunk_id in matching_chunk_ids:
                chunk = self.chunks.get(chunk_id)
                if chunk:
                    chunk.git_commits = commits
                    chunk.git_contributors = contributors
                    chunk.git_last_modified = last_mod
                    chunk.git_churn_rate = round(churn, 2)

        logger.info(
            f"Git stats enrichment: {enriched_files} files, "
            f"{len(file_stats)} git paths, {months_span:.1f} month span"
        )
        return {
            "enriched_files": enriched_files,
            "git_paths": len(file_stats),
            "months_span": round(months_span, 1),
            "max_commits": max_commits,
        }

    def get_high_risk_chunks(
        self,
        min_centrality: float = 0.5,
        min_churn: float = 3.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find chunks that are both high-centrality AND high-churn.

        These are the riskiest nodes to modify: many callers + frequent changes.

        Args:
            min_centrality: Minimum centrality score (0-1).
            min_churn: Minimum churn rate (commits/month).
            limit: Max results.

        Returns:
            List of risk dicts sorted by combined risk score.
        """
        risky = []
        for chunk in self.chunks.values():
            if chunk.chunk_type == ChunkType.MODULE:
                continue
            risk_score = (chunk.centrality * 0.6) + (
                min(1.0, chunk.git_churn_rate / 10.0) * 0.4
            )
            if chunk.centrality >= min_centrality or chunk.git_churn_rate >= min_churn:
                risky.append({
                    "name": chunk.name,
                    "file": chunk.source_path,
                    "centrality": round(chunk.centrality, 3),
                    "fan_in": chunk.fan_in,
                    "fan_out": chunk.fan_out,
                    "git_commits": chunk.git_commits,
                    "git_churn_rate": chunk.git_churn_rate,
                    "git_contributors": chunk.git_contributors,
                    "complexity": chunk.complexity,
                    "risk_score": round(risk_score, 3),
                })
        risky.sort(key=lambda r: r["risk_score"], reverse=True)
        return risky[:limit]

    def find_orphans(
        self,
        *,
        exclude_tests: bool = True,
        min_lines: int = 5,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find code chunks with zero callers (potential dead code candidates).

        These are CANDIDATES only — dynamic calls (getattr, decorators, framework
        entry points) cannot be tracked by static AST analysis.

        Args:
            exclude_tests: Skip chunks in test files.
            min_lines: Minimum line count to report (filters trivial helpers).
            exclude_patterns: Glob-style patterns for function names to skip.

        Returns:
            List of orphan dicts sorted by line_count descending.
        """
        import fnmatch

        # Names that are always entry points or framework hooks
        _ENTRY_NAMES = frozenset({
            "__init__", "__main__", "main", "app", "router",
            "lifespan", "startup", "shutdown", "on_startup", "on_shutdown",
            "setup", "teardown", "conftest", "pytest_configure",
        })

        exclude_patterns = exclude_patterns or []
        orphans: List[Dict[str, Any]] = []

        for chunk_id, chunk in self.chunks.items():
            # Skip modules — top-level modules always have 0 callers
            if chunk.chunk_type == ChunkType.MODULE:
                continue

            # Skip if has callers
            if chunk.called_by:
                continue

            # Skip entry points and dunder methods
            short_name = chunk.name.split(".")[-1] if "." in chunk.name else chunk.name
            if short_name in _ENTRY_NAMES or short_name.startswith("__"):
                continue

            # Skip test files
            if exclude_tests:
                path_lower = chunk.source_path.lower().replace("\\", "/")
                if "/tests/" in path_lower or "/test_" in path_lower or path_lower.endswith("_test.py"):
                    continue

            # Skip small chunks
            if chunk.line_count < min_lines:
                continue

            # Skip user-specified patterns
            if any(fnmatch.fnmatch(short_name, pat) for pat in exclude_patterns):
                continue

            orphans.append({
                "name": chunk.name,
                "file": chunk.source_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type.value,
                "line_count": chunk.line_count,
                "complexity": chunk.complexity,
                "calls_out": len(chunk.calls),
            })

        orphans.sort(key=lambda o: o["line_count"], reverse=True)
        return orphans

    def _invalidate_keyword_cache(self):
        """Clear keyword result cache (called after index changes)."""
        self._keyword_result_cache.clear()
        self._keyword_cache_order.clear()

    _QUERY_STOP_WORDS = frozenset({
        "how", "does", "the", "what", "is", "a", "an", "and", "or", "to",
        "in", "for", "of", "with", "from", "are", "do", "its", "it", "by",
        "on", "that", "this", "be", "can", "has", "have", "i", "my", "all",
    })

    _TEST_PENALTY = 0.25  # Score multiplier for chunks from test files

    # Reciprocal Rank Fusion damping constant (standard value from the RRF
    # literature). Larger k flattens the curve and weights deep ranks more;
    # 60 keeps the top handful dominant without collapsing to winner-take-all.
    _RRF_K = 60

    @staticmethod
    def _is_test_path(source_path: str) -> bool:
        """Check if a source path belongs to a test file."""
        p = source_path.replace("\\", "/").lower()
        basename = p.rsplit("/", 1)[-1]
        return (
            basename.startswith("test_")
            or basename.endswith("_test.py")
            or "/tests/" in p
            or "/test/" in p
            or basename == "conftest.py"
        )

    def _score_chunks_against_tokens(
        self,
        tokens: List[str],
        scope_paths: Optional[List[str]],
        tenant_id: Optional[str],
    ) -> List[Tuple[float, CodeChunk]]:
        """Score all chunks against query tokens. CPU-bound work, run via offload()."""
        results: List[Tuple[float, CodeChunk]] = []

        for chunk in self.chunks.values():
            # ── Scope filtering ──
            if tenant_id and chunk.tenant_id not in (tenant_id, "platform"):
                continue
            if scope_paths and not any(
                chunk.source_path.startswith(sp) for sp in scope_paths
            ):
                continue

            score = 0.0
            name_lower = chunk.name.lower()
            sig_lower = chunk.signature.lower()
            doc_lower = chunk.docstring.lower() if chunk.docstring else ""
            calls_lower = " ".join(chunk.calls).lower()
            body_lower = chunk.body_preview.lower() if chunk.body_preview else ""

            for token in tokens:
                # Name match (strongest signal)
                if token in name_lower:
                    score += 10.0
                    if name_lower == token:
                        score += 5.0  # Exact match bonus
                # Signature match
                if token in sig_lower:
                    score += 3.0
                # Docstring match
                if token in doc_lower:
                    score += 2.0
                # Call graph match (this chunk calls or references the token)
                if token in calls_lower:
                    score += 1.5
                # Body preview match (weakest but still relevant)
                if token in body_lower:
                    score += 0.5

            # Bonus: fraction of keywords matched (rewards broader coverage)
            if score > 0 and len(tokens) > 1:
                matched_count = sum(
                    1 for t in tokens
                    if t in name_lower or t in sig_lower or t in doc_lower
                )
                coverage = matched_count / len(tokens)
                score *= (1.0 + coverage)  # up to 2x boost

            # Deprioritize test files — they match keywords but aren't useful context
            if score > 0 and self._is_test_path(chunk.source_path):
                score *= self._TEST_PENALTY

            if score > 0:
                results.append((score, chunk))

        return results

    async def query(
        self,
        query: str,
        max_results: int = 10,
        include_callers: bool = True,
        include_callees: bool = True,
        min_score: float = 0.0,
        scope_paths: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[CodeChunk]:
        """
        Query the code graph.

        Supports both single-keyword queries (e.g. "inject") and natural
        language queries (e.g. "How does ContextEngine inject items?").
        Tokenizes multi-word queries into keywords and scores chunks by
        how many keywords match across name, signature, docstring, and calls.

        Scope filtering:
            scope_paths: Only return chunks from files under these directories.
            tenant_id: Only return chunks belonging to this tenant (or platform).
        """
        # Check keyword result cache first (saves ~60ms per repeated query)
        _scope_suffix = ""
        if scope_paths:
            _scope_suffix += ":" + "|".join(sorted(scope_paths))
        if tenant_id:
            _scope_suffix += f":t={tenant_id}"
        cache_key = f"{query}:{max_results}:{min_score}{_scope_suffix}"
        if cache_key in self._keyword_result_cache:
            # Move to end of LRU order
            try:
                self._keyword_cache_order.remove(cache_key)
            except ValueError as e:
                logger.debug(f"[CodeGraph.query] Operation failed: {e}")
            self._keyword_cache_order.append(cache_key)
            return self._keyword_result_cache[cache_key]

        query_lower = query.lower()
        # Tokenize into meaningful keywords
        tokens = [
            w for w in query_lower.replace("?", "").replace(".", "").replace(",", "").split()
            if w not in self._QUERY_STOP_WORDS and len(w) > 1
        ]
        if not tokens:
            tokens = [query_lower]

        # Offload CPU-bound scoring loop to thread pool (event-loop offload)
        try:
            from lib.core.EventLoopMonitor import offload
            results = await offload(self._score_chunks_against_tokens, tokens, scope_paths, tenant_id)
        except ImportError:
            # Not available in public package — use standard asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._score_chunks_against_tokens, tokens, scope_paths, tenant_id)

        # Sort by score descending
        results.sort(key=lambda x: -x[0])

        final = [chunk for score, chunk in results[:max_results] if score >= min_score]

        # Store in keyword result cache (LRU, max 64)
        if len(self._keyword_result_cache) >= self._KEYWORD_CACHE_MAX:
            evict_key = self._keyword_cache_order.pop(0)
            self._keyword_result_cache.pop(evict_key, None)
        self._keyword_result_cache[cache_key] = final
        self._keyword_cache_order.append(cache_key)

        return final
    
    def get_context_for_chunk(self, chunk_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get the full context for a chunk, including its callers and callees.
        
        This is what you inject into the LLM prompt for surgical context.
        """
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return {}
        
        context = {
            "chunk": chunk,
            "callers": [],
            "callees": [],
        }
        
        # Get callers
        for caller_name in chunk.called_by[:10]:  # Limit to 10
            caller_ids = self.by_name.get(caller_name, [])
            for cid in caller_ids[:2]:  # Max 2 per name
                if cid in self.chunks:
                    context["callers"].append(self.chunks[cid])
        
        # Get callees
        for callee_name in chunk.calls[:10]:
            callee_ids = self.by_name.get(callee_name, [])
            for cid in callee_ids[:2]:
                if cid in self.chunks:
                    context["callees"].append(self.chunks[cid])

        # `ordered` is the list to actually inject, in DEPENDENCY order: a chunk's
        # callees before the chunk, the chunk before its callers. DeepSeek-Coder
        # (arXiv:2401.14196, Algorithm 1) measured that ordering repository context this
        # way improves project-level performance — "the context each file relies on is
        # placed before that file in the input sequence".
        #
        # `callers` and `callees` are left exactly as they were, so nothing that reads
        # them changes. This is additive on purpose: it hands the consumer the right
        # order instead of silently re-ranking a field someone may already depend on.
        #
        # Measured on this repo's own import graph (2,300 modules, 6,437 edges), the
        # ordering the tool produced before was 62.9% dependency-correct and this is
        # 78.6%. The gap is not cosmetic: reading a dependent before its dependency is
        # what produced four separate defects in one day of K3 engine work.
        try:
            from lib.cognitive.dependency_order import topo_order

            members = {}
            for c in context["callees"] + [chunk] + context["callers"]:
                members[c.id] = c
            name_to_ids = {}
            for cid, c in members.items():
                name_to_ids.setdefault(getattr(c, "name", ""), []).append(cid)
            deps = {}
            for cid, c in members.items():
                d = set()
                for callee_name in getattr(c, "calls", ()) or ():
                    for other in name_to_ids.get(callee_name, ()):
                        if other != cid:
                            d.add(other)
                deps[cid] = d
            # Rank keeps the subject chunk ahead of unrelated peers when nothing
            # constrains them; it never overrides a dependency edge.
            rank = {cid: (1.0 if cid == chunk.id else 0.0) for cid in members}
            context["ordered"] = [members[i]
                                  for i in topo_order(list(members), deps, rank)]
        except Exception as exc:  # noqa: BLE001 - ordering is an enhancement, not a gate
            # Never let ordering break context assembly: a caller that loses `ordered`
            # still has callers/callees and is exactly as well off as before this
            # change. Logged rather than swallowed so a persistent failure is visible.
            logger.debug("dependency ordering unavailable for %s: %s", chunk_id, exc)
            context["ordered"] = context["callees"] + [chunk] + context["callers"]

        return context

    def locate(self, file: str, line: int) -> List[Any]:
        """Resolve a file:line (e.g. an exception crash frame) to the chunks
        containing that line — innermost (smallest-span) chunk first, i.e. the
        exact function/method, then its enclosing class.

        Path matching is suffix-based both ways so repo-relative crash-frame
        paths (lib/core/X.py) join against absolute indexed source paths.
        """
        rel = file.replace("\\", "/").lstrip("/")
        if not rel:
            return []
        hits = []
        for indexed_path, chunk_ids in self.by_file.items():
            normalized = indexed_path.replace("\\", "/")
            if not (normalized.endswith(rel) or rel.endswith(normalized)):
                continue
            for cid in chunk_ids:
                chunk = self.chunks.get(cid)
                if chunk and chunk.start_line <= line <= chunk.end_line:
                    hits.append(chunk)
        hits.sort(key=lambda c: c.end_line - c.start_line)
        return hits

    # ====================================================================
    # TRACE / IMPACT / EXPLORE / CONTEXT-FOR-TASK / ROUTES / AFFECTED TESTS
    # ====================================================================

    def _resolve_symbol(self, name: str) -> List[str]:
        """Resolve a symbol name to chunk IDs, handling partial matches."""
        if name in self.by_name:
            return self.by_name[name]
        name_lower = name.lower()
        for key, ids in self.by_name.items():
            if key.lower() == name_lower:
                return ids
        matches = []
        for key, ids in self.by_name.items():
            if key.lower().endswith(f".{name_lower}") or key.lower() == name_lower:
                matches.extend(ids)
        return matches

    async def trace_path(self, source: str, target: str, max_depth: int = 8, max_paths: int = 3) -> List[List[Dict]]:
        """Trace call path from source to target via bidirectional BFS."""
        source_ids = self._resolve_symbol(source)
        target_ids = self._resolve_symbol(target)
        if not source_ids or not target_ids:
            return []
        target_id_set = set(target_ids)
        paths: List[List[Dict]] = []
        for src_id in source_ids[:3]:
            queue = [(src_id, [src_id])]
            visited = {src_id}
            while queue and len(paths) < max_paths:
                current_id, path = queue.pop(0)
                if len(path) > max_depth:
                    continue
                current = self.chunks.get(current_id)
                if not current:
                    continue
                if current_id in target_id_set and len(path) > 1:
                    hop_list = []
                    for i, cid in enumerate(path):
                        c = self.chunks.get(cid)
                        if c:
                            body = self.get_full_body(cid)
                            snippet = (body or c.body_preview or "")[:500]
                            hop_list.append({"chunk_id": cid, "name": c.name, "file": c.source_path,
                                             "line": c.start_line, "signature": c.signature,
                                             "code_snippet": snippet, "direction": "forward", "hop": i})
                    paths.append(hop_list)
                    continue
                for call_name in current.calls[:15]:
                    for cid in self.by_name.get(call_name, [])[:2]:
                        if cid not in visited:
                            visited.add(cid)
                            queue.append((cid, path + [cid]))
            if paths:
                break
        if not paths:
            for tgt_id in target_ids[:3]:
                queue = [(tgt_id, [tgt_id])]
                visited = {tgt_id}
                source_id_set = set(source_ids)
                while queue and len(paths) < max_paths:
                    current_id, path = queue.pop(0)
                    if len(path) > max_depth:
                        continue
                    current = self.chunks.get(current_id)
                    if not current:
                        continue
                    if current_id in source_id_set and len(path) > 1:
                        reversed_path = list(reversed(path))
                        hop_list = []
                        for i, cid in enumerate(reversed_path):
                            c = self.chunks.get(cid)
                            if c:
                                body = self.get_full_body(cid)
                                snippet = (body or c.body_preview or "")[:500]
                                hop_list.append({"chunk_id": cid, "name": c.name, "file": c.source_path,
                                                 "line": c.start_line, "signature": c.signature,
                                                 "code_snippet": snippet, "direction": "reverse", "hop": i})
                        paths.append(hop_list)
                        continue
                    for caller_name in current.called_by[:15]:
                        for cid in self.by_name.get(caller_name, [])[:2]:
                            if cid not in visited:
                                visited.add(cid)
                                queue.append((cid, path + [cid]))
        return paths[:max_paths]

    def impact_analysis(self, symbol: str, max_depth: int = 4) -> Dict[str, Any]:
        """Transitive blast radius via reverse call graph BFS."""
        seed_ids = self._resolve_symbol(symbol)
        if not seed_ids:
            return {"symbol": symbol, "error": "Symbol not found", "total_affected": 0}
        by_depth: Dict[int, List[Dict]] = defaultdict(list)
        visited = set(seed_ids)
        queue = [(sid, 0) for sid in seed_ids]
        affected_files: Set[str] = set()
        affected_tests: List[Dict] = []
        centrality_sum = 0.0
        while queue:
            current_id, depth = queue.pop(0)
            if depth > max_depth:
                continue
            current = self.chunks.get(current_id)
            if not current:
                continue
            if depth > 0:
                info = {"name": current.name, "file": current.source_path, "line": current.start_line,
                        "chunk_type": current.chunk_type.value, "signature": current.signature,
                        "centrality": round(current.centrality, 3)}
                by_depth[depth].append(info)
                affected_files.add(current.source_path)
                centrality_sum += current.centrality
                if self._is_test_path(current.source_path):
                    affected_tests.append({"name": current.name, "file": current.source_path, "distance": depth})
            for caller_name in current.called_by:
                for cid in self.by_name.get(caller_name, [])[:5]:
                    if cid not in visited:
                        visited.add(cid)
                        queue.append((cid, depth + 1))
        total_affected = sum(len(v) for v in by_depth.values())
        return {"symbol": symbol, "seed_chunks": len(seed_ids), "direct_callers": by_depth.get(1, []),
                "depth_2": by_depth.get(2, []), "depth_3": by_depth.get(3, []), "depth_4": by_depth.get(4, []),
                "total_affected": total_affected, "affected_files": sorted(affected_files),
                "affected_tests": affected_tests, "risk_score": round(centrality_sum, 3)}

    def impact_of_commit(self, commit_sha: str, max_depth: int = 2) -> Dict[str, Any]:
        """Commit → changed symbols → blast radius, via the reverse call graph.

        Maps the commit's diff hunks onto indexed chunks (line-range overlap,
        suffix path matching like locate()), then BFS over called_by for the
        transitive blast radius. The index reflects the CURRENT tree, so hunks
        in files that were since deleted or heavily moved may not resolve.
        """
        import subprocess as _sp

        root = Path(self._root_path or ".")
        if not (root / ".git").exists():
            return {"commit": commit_sha, "error": "not_a_git_repo"}

        # git show handles root commits (no parent) where `sha^` would fail
        try:
            # encoding+errors is load-bearing: text=True alone uses the locale
            # codec (cp1252 on Windows) and a non-decodable byte in a big diff
            # kills the pipe-reader thread, silently leaving stdout=None.
            result = _sp.run(
                ["git", "show", "--unified=0", "--format=", commit_sha],
                capture_output=True, encoding="utf-8", errors="replace",
                cwd=str(root), timeout=60,
            )
            if result.returncode != 0:
                return {"commit": commit_sha, "error": (result.stderr or "")[:200]}
        except Exception as e:
            return {"commit": commit_sha, "error": str(e)[:200]}

        # Parse unified-0 diff: "+++ b/<path>" then "@@ -a,b +c,d @@" hunks
        hunks: List[Tuple[str, int, int]] = []  # (file, new_start, new_end)
        current_file = ""
        for line in (result.stdout or "").splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
            elif line.startswith("@@") and current_file:
                m = re.search(r"\+(\d+)(?:,(\d+))?", line)
                if m:
                    start = int(m.group(1))
                    count = int(m.group(2)) if m.group(2) is not None else 1
                    # Pure deletion (count 0): probe the line the hunk lands on
                    hunks.append((current_file, start, start + max(count, 1) - 1))

        # Map hunks → overlapping chunks (suffix path matching, like locate())
        changed_ids: Dict[str, None] = {}
        touched_files = sorted({f for f, _, _ in hunks})
        for hunk_file, h_start, h_end in hunks:
            rel = hunk_file.replace("\\", "/").lstrip("/")
            for indexed_path, chunk_ids in self.by_file.items():
                normalized = indexed_path.replace("\\", "/")
                if not (normalized.endswith(rel) or rel.endswith(normalized)):
                    continue
                for cid in chunk_ids:
                    chunk = self.chunks.get(cid)
                    if chunk and chunk.start_line <= h_end and chunk.end_line >= h_start:
                        changed_ids[cid] = None

        changed_symbols = [
            self.chunks[cid].concept_card(max_edges=5) for cid in changed_ids
        ]

        # Blast radius: BFS over reverse call edges from every changed chunk
        visited = set(changed_ids)
        queue = [(cid, 0) for cid in changed_ids]
        blast: List[Dict[str, Any]] = []
        affected_files: Set[str] = set()
        while queue:
            current_id, depth = queue.pop(0)
            if depth > max_depth:
                continue
            current = self.chunks.get(current_id)
            if not current:
                continue
            if depth > 0:
                blast.append({"name": current.name, "file": current.source_path,
                              "line": current.start_line, "distance": depth})
                affected_files.add(current.source_path)
            for caller_name in current.called_by:
                for cid in self.by_name.get(caller_name, [])[:5]:
                    if cid not in visited:
                        visited.add(cid)
                        queue.append((cid, depth + 1))

        return {
            "commit": commit_sha,
            "files_in_diff": touched_files,
            "changed_symbols": changed_symbols,
            "blast_radius_count": len(blast),
            "blast_radius": blast[:100],
            "affected_files": sorted(affected_files),
        }

    def explore(self, symbols: List[str], include_code: bool = True) -> Dict[str, Any]:
        """Multi-symbol exploration grouped by file with relationship map."""
        files: Dict[str, Dict] = defaultdict(lambda: {"chunks": [], "line_ranges": []})
        relationships: List[Dict] = []
        all_chunk_ids: List[str] = []
        for sym in symbols:
            ids = self._resolve_symbol(sym)
            for cid in ids:
                chunk = self.chunks.get(cid)
                if not chunk:
                    continue
                all_chunk_ids.append(cid)
                entry = {"id": cid, "name": chunk.name, "chunk_type": chunk.chunk_type.value,
                         "start_line": chunk.start_line, "end_line": chunk.end_line,
                         "signature": chunk.signature, "docstring": (chunk.docstring or "")[:300]}
                if include_code:
                    body = self.get_full_body(cid)
                    entry["code"] = (body or chunk.body_preview or "")[:2000]
                files[chunk.source_path]["chunks"].append(entry)
                files[chunk.source_path]["line_ranges"].append([chunk.start_line, chunk.end_line])
        for i, cid_a in enumerate(all_chunk_ids):
            ca = self.chunks.get(cid_a)
            if not ca:
                continue
            for cid_b in all_chunk_ids[i + 1:]:
                cb = self.chunks.get(cid_b)
                if not cb:
                    continue
                if cb.name in ca.calls or cb.name.split(".")[-1] in ca.calls:
                    relationships.append({"from": ca.name, "to": cb.name, "type": "calls"})
                if ca.name in cb.calls or ca.name.split(".")[-1] in cb.calls:
                    relationships.append({"from": cb.name, "to": ca.name, "type": "calls"})
                if ca.parent_class and ca.parent_class == cb.name:
                    relationships.append({"from": ca.name, "to": cb.name, "type": "member_of"})
                if cb.parent_class and cb.parent_class == ca.name:
                    relationships.append({"from": cb.name, "to": ca.name, "type": "member_of"})
        return {"files": dict(files), "relationships": relationships, "total_chunks": len(all_chunk_ids)}

    async def context_for_task(self, task: str, max_results: int = 10) -> Dict[str, Any]:
        """Given a natural language task, return entry points + related symbols."""
        results = await self.hybrid_query(task, max_results=max_results)
        if not results:
            return {"entry_points": [], "related_symbols": [], "files_involved": []}
        entry_points, related_symbols, call_chains = [], [], []
        files_involved: Set[str] = set()
        for chunk in results[:3]:
            ctx = self.get_context_for_chunk(chunk.id, depth=1)
            if ctx:
                for caller in ctx.get("callers", []):
                    related_symbols.append({"name": caller.name, "file": caller.source_path, "relation": "caller"})
                for callee in ctx.get("callees", []):
                    related_symbols.append({"name": callee.name, "file": callee.source_path, "relation": "callee"})
        if results:
            impact = self.impact_analysis(results[0].name, max_depth=2)
            if impact.get("total_affected", 0) > 0:
                call_chains.append({"root": results[0].name, "total_affected": impact["total_affected"],
                                    "direct_callers": [c["name"] for c in impact.get("direct_callers", [])[:5]]})
        for chunk in results:
            files_involved.add(chunk.source_path)
            is_entry = chunk.fan_in >= 3 or chunk.route_path is not None or chunk.name in ("main", "app", "lifespan")
            entry_points.append({"name": chunk.name, "file": chunk.source_path, "line": chunk.start_line,
                                 "signature": chunk.signature, "is_entry_point": is_entry, "fan_in": chunk.fan_in,
                                 "route": f"{chunk.route_method} {chunk.route_path}" if chunk.route_path else None})
        seen_names: Set[str] = set()
        deduped_related = []
        for r in related_symbols:
            if r["name"] not in seen_names:
                seen_names.add(r["name"])
                deduped_related.append(r)
        return {"entry_points": entry_points, "related_symbols": deduped_related[:20],
                "files_involved": sorted(files_involved), "call_chains": call_chains,
                "suggested_starting_point": entry_points[0]["name"] if entry_points else None}

    def search_routes(self, pattern: str = "") -> List[Dict[str, Any]]:
        """Search registered HTTP routes by URL pattern."""
        results = []
        pattern_lower = pattern.lower()
        for route_key, chunk_id in self.routes.items():
            if not pattern or pattern_lower in route_key.lower():
                chunk = self.chunks.get(chunk_id)
                if chunk:
                    results.append({"route": route_key, "method": chunk.route_method, "path": chunk.route_path,
                                    "handler": chunk.name, "file": chunk.source_path, "line": chunk.start_line,
                                    "signature": chunk.signature})
        results.sort(key=lambda r: r["route"])
        return results

    def find_affected_tests(self, symbol: str, max_depth: int = 4) -> List[Dict]:
        """Given a changed symbol, find which test files need re-running."""
        impact = self.impact_analysis(symbol, max_depth=max_depth)
        tests = list(impact.get("affected_tests", []))
        affected_modules: Set[str] = set()
        for depth_key in ("direct_callers", "depth_2", "depth_3", "depth_4"):
            for item in impact.get(depth_key, []):
                fpath = item.get("file", "")
                if fpath:
                    affected_modules.add(fpath.replace("/", ".").replace("\\", ".").rstrip(".py"))
        seed_ids = self._resolve_symbol(symbol)
        for sid in seed_ids:
            sc = self.chunks.get(sid)
            if sc:
                affected_modules.add(sc.source_path.replace("/", ".").replace("\\", ".").rstrip(".py"))
        seen_test_files: Set[str] = {t["file"] for t in tests}
        for fpath, chunk_ids in self.by_file.items():
            if not self._is_test_path(fpath) or fpath in seen_test_files:
                continue
            for cid in chunk_ids:
                chunk = self.chunks.get(cid)
                if not chunk:
                    continue
                for imp in (chunk.imports or []):
                    for mod in affected_modules:
                        if mod in imp or imp in mod:
                            tests.append({"name": chunk.name, "file": fpath, "distance": max_depth + 1, "via": "import"})
                            seen_test_files.add(fpath)
                            break
                    if fpath in seen_test_files:
                        break
        return tests

    def export_for_embedding(self) -> List[Dict[str, Any]]:
        """
        Export chunks in a format ready for embedding with MindClient.
        
        Each chunk becomes a document with metadata for filtering.
        """
        documents = []
        
        for chunk in self.chunks.values():
            # Build the text to embed
            text_parts = [chunk.signature]
            if chunk.docstring:
                text_parts.append(chunk.docstring)
            text_parts.append(chunk.body_preview)
            
            doc = {
                "id": chunk.id,
                "text": "\n".join(text_parts),
                "metadata": {
                    "name": chunk.name,
                    "type": chunk.chunk_type.value,
                    "file": chunk.source_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "complexity": chunk.complexity,
                    "calls_count": len(chunk.calls),
                    "called_by_count": len(chunk.called_by),
                },
            }
            documents.append(doc)

        return documents

    def export_concept_snapshot(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export the whole index as a committable JSONL concept-card snapshot.

        One JSON line per chunk (compact concept card + content_hash of the
        chunk's source line-range), sorted by (file, line, name) with sorted
        keys — re-running on an unchanged tree produces a byte-identical file,
        so git diffs of the snapshot show exactly which symbols changed.
        Timestamps live in the returned metadata, never in the file.
        """
        if not self.chunks:
            return {"exported": 0, "error": "index_empty"}

        base = self._cache_dir or os.path.join(
            self._root_path or ".", "AitherOS", "Library", "Data", "codegraph"
        )
        if output_path is None:
            output_path = os.path.join(base, "concept-cards.jsonl")
        elif not os.path.isabs(output_path):
            # Relative paths land under the codegraph data dir (the HTTP layer
            # only permits relative paths, so callers can't write elsewhere)
            output_path = os.path.join(base, output_path)

        # Hash each chunk's source line-range, reading every file only once
        content_hashes: Dict[str, Optional[str]] = {}
        for file_key, chunk_ids in self.by_file.items():
            file_lines: Optional[List[str]] = None
            try:
                fpath = Path(file_key)
                if not fpath.is_absolute() and self._root_path:
                    fpath = Path(self._root_path) / file_key
                file_lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                file_lines = None
            for cid in chunk_ids:
                chunk = self.chunks.get(cid)
                if not chunk:
                    continue
                if file_lines is None:
                    content_hashes[cid] = None
                else:
                    span = "\n".join(file_lines[chunk.start_line - 1:chunk.end_line])
                    content_hashes[cid] = hashlib.sha256(span.encode("utf-8")).hexdigest()[:16]

        ordered = sorted(
            self.chunks.values(), key=lambda c: (c.source_path, c.start_line, c.name)
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_path = f"{output_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            for chunk in ordered:
                card = chunk.concept_card()
                card["content_hash"] = content_hashes.get(chunk.id)
                f.write(json.dumps(card, sort_keys=True, ensure_ascii=True) + "\n")
        os.replace(tmp_path, output_path)

        return {
            "exported": len(ordered),
            "output_path": str(output_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ====================================================================
    # EMBEDDING METHODS
    # ====================================================================

    async def _save_embedding_cache(self, cached: Dict[str, list], cache_path: str) -> float:
        """Write the embedding cache to disk. Returns size in MB.

        Shared by the periodic checkpoint and the end-of-run save so both go
        through exactly one code path (and one HMAC write).
        """
        def _write() -> float:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp = f"{cache_path}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Atomic swap: a checkpoint interrupted mid-write must never leave a
            # truncated cache behind — that would fail the HMAC on next load and
            # silently discard every embedding computed so far.
            os.replace(tmp, cache_path)
            _write_pickle_hmac(cache_path)
            return os.path.getsize(cache_path) / (1024 * 1024)

        return await asyncio.to_thread(_write)

    async def embed_chunks(
        self,
        model: str = "nomic-embed-text",
        batch_size: int = 64,
        cache_path: Optional[str] = None,
        on_progress: Optional[callable] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all indexed chunks via EmbeddingEngine.

        Embeddings are stored on CodeChunk.embedding and persisted to disk.
        Incremental: only embeds chunks that don't have embeddings yet.

        Args:
            model: Embedding model (default: nomic-embed-text, 768-dim)
            batch_size: Texts per API call
            cache_path: Where to persist embeddings (default: Library/Data/codegraph_embeddings.pkl)
            on_progress: callback(fraction, message)
            force: Re-embed all chunks even if cached

        Returns:
            Stats dict with total, cached, new, failed, embed_ms
        """
        if not _HAS_EMBEDDING_ENGINE and not _detect_vllm():
            raise RuntimeError("No embedding backend available (need EmbeddingEngine or vLLM)")

        if cache_path is None:
            # MUST go through _get_data_path (same helper _save_chunk_cache uses).
            # The old hard-coded `Path(__file__).parent.parent.parent / "Library"`
            # resolves to /app/Library/Data/ inside the container — a directory
            # that DOES NOT EXIST and is NOT the mount. The library volume is
            # mounted at /app/AitherOS/Library. os.makedirs() silently created
            # the wrong path in the container's ephemeral layer, so every embed
            # run's cache was written outside the volume and destroyed on the
            # next recreate — and on load os.path.exists() was False, so `cached`
            # came back EMPTY every time and nothing was ever reused.
            cache_path = _get_data_path(
                str(Path(__file__).parent.parent.parent), "codegraph_embeddings.pkl"
            )

        # Load cached embeddings from disk (offloaded to thread for 9P safety)
        cached: Dict[str, list] = {}
        if not force and os.path.exists(cache_path):
            if not _verify_pickle_hmac(cache_path):
                logger.warning("Embedding cache HMAC invalid — starting fresh")
            else:
                def _load_pickle():
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
                try:
                    cached = await asyncio.to_thread(_load_pickle)
                    logger.info(f"Loaded {len(cached)} cached embeddings")
                except Exception as e:
                    logger.warning(f"Failed to load embedding cache: {e}")

        # Apply cached embeddings to current chunks — compacted to float32 rows
        # on the way in, and the compact row is shared back into `cached` so the
        # checkpoint pickle sheds the boxed float64 lists too.
        applied = 0
        for chunk_id, chunk in self.chunks.items():
            if chunk_id in cached and chunk.embedding is None:
                chunk.embedding = _as_f32(cached[chunk_id])
                cached[chunk_id] = chunk.embedding
                applied += 1

        # Prune ORPHANED cache entries — embeddings whose chunk id no longer
        # exists. Two reasons, both load-bearing:
        #  1) Correctness: a reindex mints new chunk ids, so every prior vector
        #     is orphaned. The cache otherwise grows forever with dead entries
        #     (observed: 128,740 cached embeddings against 59,128 live chunks,
        #     embedding_coverage 0.0 — every one of them orphaned).
        #  2) Memory: embeddings are Python float lists (~18KB each), so those
        #     128,740 dead entries held ~2.3GB for the whole run. Combined with
        #     `results` and the flattened copy below, the process walked into
        #     its 12GiB cap and died mid-run — losing 100% of the work, because
        #     the only save happens at the very end.
        _orphans = [cid for cid in cached if cid not in self.chunks]
        if _orphans:
            for cid in _orphans:
                del cached[cid]
            logger.info(
                f"Pruned {len(_orphans)} orphaned embedding cache entries "
                f"(chunk ids no longer in the index)"
            )
            del _orphans

        # force=True means RE-embed everything, not just fill the gaps. Skipping
        # the pickle load (above) is not enough on its own: after a reindex the
        # chunks already carry in-memory vectors (hydrated from the durable cache
        # on boot), so `chunk.embedding is None` was False for them and force
        # silently became incremental. Observed 2026-07-21 during the CodeRankEmbed
        # cutover: a force re-embed left 61,758 chunks on their OLD nomic
        # vectors and only embedded the 39,579 new ones with the code model — a
        # MIXED vector space where queries match half the corpus. Clear first so
        # force actually forces; an interruption then leaves coderank+None (a
        # coherent partial), never coderank+nomic.
        if force:
            for _c in self.chunks.values():
                _c.embedding = None

        # Find chunks needing embeddings
        need_embedding = [
            (cid, chunk) for cid, chunk in self.chunks.items()
            if chunk.embedding is None
        ]

        if not need_embedding:
            logger.info(f"All {len(self.chunks)} chunks already have embeddings")
            return {"total": len(self.chunks), "cached": applied, "new": 0, "embed_ms": 0}

        logger.info(
            f"Embedding {len(need_embedding)} chunks "
            f"({applied} from cache, {len(self.chunks) - applied - len(need_embedding)} in memory)"
        )

        # Build texts to embed — rich representation for semantic matching
        texts = []
        chunk_ids = []
        for cid, chunk in need_embedding:
            parts = [chunk.signature]
            if chunk.docstring:
                parts.append(chunk.docstring[:300])
            if chunk.body_preview:
                parts.append(chunk.body_preview[:300])
            if chunk.calls:
                parts.append(f"calls: {', '.join(chunk.calls[:10])}")
            if chunk.called_by:
                parts.append(f"called by: {', '.join(chunk.called_by[:10])}")
            if chunk.parent_class:
                parts.append(f"class: {chunk.parent_class}")
            texts.append("\n".join(parts))
            chunk_ids.append(cid)

        start = time.perf_counter()
        backend = "vLLM" if _detect_vllm() else "EmbeddingEngine"
        logger.info(f"Embedding {len(need_embedding)} chunks via {backend}")
        total_batches = (len(texts) + batch_size - 1) // batch_size

        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append(texts[i : i + batch_size])

        # Batches are applied to chunks inline and released, so this only tracks
        # WHICH batches completed — never the embedding payloads themselves.
        _APPLIED = "applied"
        results: List[Optional[str]] = [None] * len(batches)
        consecutive_failures = 0
        new_count = 0

        # Warm the embedding backend ONCE with a generous timeout. The first
        # call may spend >30s loading (or downloading, then failing) the local
        # sentence-transformers model before falling back to vLLM — under the
        # 30s per-batch timeout that counted as 3 consecutive "failures" and
        # aborted the entire run while the backend was actually fine
        # (vLLM: 128 texts in 0.4s once settled).
        try:
            await asyncio.wait_for(_embed_texts(["warmup"], model=model), timeout=120.0)
        except (asyncio.TimeoutError, Exception) as _warm_err:  # noqa: BLE001
            logger.warning(f"Embedding backend warmup did not settle: {_warm_err!r}")

        for idx, batch in enumerate(batches):
            if consecutive_failures >= 3:
                logger.warning(
                    f"Embedding backend down ({consecutive_failures} consecutive failures) "
                    f"— aborting remaining {total_batches - idx} batches"
                )
                break
            try:
                embeddings = await asyncio.wait_for(
                    # 30s was a MARGINAL cap, the same mistake as the git
                    # enrichment's 30s in an earlier incident. A warm 64-text batch measures
                    # ~1.2s, so 30s looks generous — but the run that failed all
                    # 4,337 chunks started ~53s after boot, competing with the
                    # 1.8s matrix build and DocGraph warming, and lost 3 batches
                    # in a row to the cap, which tripped the abort guard and
                    # killed the whole run. Headroom here is nearly free: a batch
                    # that is genuinely wedged still aborts after 3 of these.
                    _embed_texts(batch, model=model),
                    timeout=float(os.getenv("AITHER_CODEGRAPH_EMBED_BATCH_TIMEOUT", "120")),
                )
                # Apply IMMEDIATELY and drop the batch. Buffering every batch in
                # `results` for the whole run held the entire embedding set in
                # RAM on top of the copy already living on the chunks, which is
                # what walked this process into its 12GiB cap (~1.1GB/min climb,
                # dead at ~6min, losing the whole run because the only save is
                # at the end). Streaming keeps just one batch resident.
                _base = idx * batch_size
                for _off, _emb in enumerate(embeddings):
                    _i = _base + _off
                    if _i >= len(chunk_ids) or _emb is None:
                        continue
                    _cid = chunk_ids[_i]
                    # float32 row, not the raw list — one shared object on the
                    # chunk AND in the checkpoint dict (~3KB vs ~25KB each)
                    self.chunks[_cid].embedding = _as_f32(_emb)
                    cached[_cid] = self.chunks[_cid].embedding
                    new_count += 1
                del embeddings
                results[idx] = _APPLIED
                consecutive_failures = 0
            except asyncio.TimeoutError:
                logger.warning(f"Embedding batch {idx+1}/{total_batches} timed out (30s)")
                results[idx] = "failed"
                consecutive_failures += 1
            except Exception as e:
                logger.error(f"Embedding batch {idx+1}/{total_batches} failed: {e}")
                results[idx] = "failed"
                consecutive_failures += 1
            if on_progress:
                on_progress(
                    (idx + 1) / total_batches,
                    f"Embedded {min((idx + 1) * batch_size, len(texts))}/{len(texts)}",
                )

            # CHECKPOINT. Without this the ONLY save is at the very end, so any
            # failure during a ~15min run loses 100% of the work and the next
            # attempt starts from zero — observed repeatedly: the process walked
            # into its memory cap at ~6min, restarted, and embedding_coverage
            # went straight back to 0.0 having reached 0.622. Saving
            # periodically makes the run RESUMABLE: each attempt loads what
            # landed last time (only chunks with embedding None are re-embedded),
            # so progress is monotonic and the job converges across restarts
            # instead of looping forever.
            if _CHECKPOINT_EVERY and (idx + 1) % _CHECKPOINT_EVERY == 0 and new_count:
                try:
                    await self._save_embedding_cache(cached, cache_path)
                    logger.info(
                        f"[EMBED] checkpoint at batch {idx + 1}/{total_batches} "
                        f"— {len(cached)} embeddings persisted"
                    )
                except Exception as _ck_err:  # never let a checkpoint kill the run
                    logger.warning(f"[EMBED] checkpoint failed: {_ck_err}")

            await asyncio.sleep(0.05)

        embed_ms = (time.perf_counter() - start) * 1000

        # Embeddings were applied inline per batch (streaming), so there is no
        # buffered result set left to walk here. `texts`/`batches` are dead too.
        del texts, batches, results
        if new_count > 0:
            self._has_embeddings_cached = True  # Invalidate hybrid_query fast-path cache
            self._invalidate_embedding_matrix()  # Force matrix rebuild on next query

        # ── CodeIndex Qdrant fold (leak path #1) ──────────────────────────
        # Persist embeddings to Qdrant for cross-process semantic search.
        # Guarded behind AITHER_CODEGRAPH_QDRANT_PERSIST (default ON) — if
        # CodeIndex unavailable or Qdrant down, silently degrades (never raises).
        flag_value = os.environ.get("AITHER_CODEGRAPH_QDRANT_PERSIST", "1").strip()
        if flag_value.lower() not in ("0", "false", "no"):
            await self._persist_to_qdrant(batch_size=batch_size)

        # Persist to disk (offloaded to thread for 9P safety)
        try:
            sz = await self._save_embedding_cache(cached, cache_path)
            logger.info(f"Saved {len(cached)} embeddings ({sz:.1f}MB)")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

        stats = {
            "total": len(self.chunks),
            "cached": applied,
            "new": new_count,
            "failed": len(need_embedding) - new_count,
            "embed_ms": embed_ms,
            "batches": total_batches,
            "model": model,
        }
        logger.info(f"Embedding complete: {stats}")
        return stats

    async def _persist_to_qdrant(self, batch_size: int = 64) -> None:
        """Persist embedded chunks to Qdrant via CodeIndex (best-effort).

        Groups chunks by (tenant_id, workspace_id) and upserts to the
        corresponding collection. Only chunks with embeddings are upserted.
        Never raises — any failure is logged at debug level.
        """
        try:
            from lib.clients.code_index import CodeIndex, IndexScope
        except ImportError:
            logger.debug("[CodeGraph] CodeIndex unavailable for Qdrant persist")
            return

        # Group chunks by the FULL scope tuple (tenant, workspace, user, agent)
        # so user/agent-private code is never merged into a broader scope — a
        # chunk carrying a user_id must not be upserted with scope_user="" (which
        # would make it visible to the whole workspace). "" stays workspace-wide.
        by_scope: dict = {}
        for chunk in self.chunks.values():
            if chunk.embedding is None:
                continue
            scope_key = (chunk.tenant_id or "platform",
                         chunk.workspace_id or "",
                         getattr(chunk, "user_id", "") or "",
                         getattr(chunk, "agent_id", "") or "")
            by_scope.setdefault(scope_key, []).append(chunk)

        # Upsert each scope's chunks
        index = CodeIndex()
        total_upserted = 0
        for (tenant_id, workspace_id, user_id, agent_id), chunks in by_scope.items():
            try:
                scope = IndexScope(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    agent_id=agent_id,
                )
                # Build chunk dicts for upsert
                chunk_dicts = []
                for chunk in chunks:
                    _emb = chunk.embedding
                    if _emb is None:
                        continue
                    chunk_dicts.append({
                        "file_path": chunk.source_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "symbol": chunk.name,
                        "chunk_type": chunk.chunk_type.value,
                        "content": "\n".join([
                            chunk.signature,
                            chunk.docstring or "",
                            chunk.body_preview or "",
                        ]).strip(),
                        # JSON boundary: embeddings are float32 rows (ndarray /
                        # array('f')) in memory now — convert to plain floats here
                        "vector": [float(v) for v in _emb],
                        "commit_stamp": "",  # CodeGraph doesn't track git stamps
                    })
                if chunk_dicts:
                    upserted = await index.upsert_chunks(chunk_dicts, scope=scope)
                    total_upserted += upserted
            except Exception as exc:  # noqa: BLE001 — Qdrant persist is best-effort
                logger.debug(
                    "[CodeGraph] Qdrant upsert failed for scope %s/%s: %s",
                    tenant_id, workspace_id, exc,
                )
        if total_upserted > 0:
            logger.debug(
                "[CodeGraph] Persisted %d chunks to Qdrant", total_upserted,
            )

    @property
    def embedding_coverage(self) -> float:
        """Fraction of chunks with embeddings (0.0–1.0)."""
        if not self.chunks:
            return 0.0
        return sum(1 for c in self.chunks.values() if c.embedding is not None) / len(self.chunks)

    async def semantic_query(
        self,
        query: str,
        max_results: int = 10,
        model: str = "nomic-embed-text",
    ) -> List[Tuple[float, "CodeChunk"]]:
        """
        Semantic search using embedding cosine similarity.

        Returns list of (similarity_score, chunk) sorted descending.
        Requires embed_chunks() to have been called first.
        """
        if not _HAS_EMBEDDING_ENGINE and not _detect_vllm():
            return []

        # Presence check only — do NOT materialize a list over every chunk here.
        # This ran a ~95K-item scan on every query, synchronously, before any
        # await point, which is part of why the caller's asyncio.wait_for cap
        # could not bound this call (see the numpy branch below).
        if not any(c.embedding is not None for c in self.chunks.values()):
            logger.warning("No embeddings available — call embed_chunks() first")
            return []

        query_vec = self._query_embed_cache.get(query)
        if query_vec is None:
            try:
                vecs = await _embed_texts([query], model=model, is_query=True)
                query_vec = vecs[0] if vecs else None
                if not query_vec:
                    return []
                # Cache for next time
                self._cache_query_embedding(query, query_vec)
            except Exception as e:
                logger.error(f"Failed to embed query: {e}")
                return []
        else:
            logger.debug(f"[SEMANTIC] Query embedding cache HIT for: {query[:40]}")

        if _HAS_NUMPY:
            # Use pre-computed matrix (avoids ~2500ms np.array construction per query)
            await self._ensure_embedding_matrix_async()
            if self._embedding_matrix is None:
                return []

            qv = np.array(query_vec, dtype=np.float32)
            q_norm = np.linalg.norm(qv)
            if q_norm == 0:
                return []

            # The whole scoring pass is synchronous CPU work over ~95K rows.
            # Run it in the executor so the caller's asyncio.wait_for timeout
            # has an await point to cancel at — inline, the cap was silently
            # unenforceable and a single cold query could block for 14s.
            def _score() -> List[Tuple[float, str]]:
                valid = (self._embedding_norms > 0)
                sims = np.zeros(len(self._embedding_ids), dtype=np.float32)
                sims[valid] = (
                    (self._embedding_matrix[valid] @ qv)
                    / (self._embedding_norms[valid] * q_norm)
                )
                # Vectorized test-file penalty. Previously a per-query Python
                # loop over every embedded chunk; now precomputed once with the
                # matrix (see _ensure_embedding_matrix).
                sims *= self._embedding_penalties
                top = np.argsort(sims)[::-1][:max_results]
                return [(float(sims[i]), self._embedding_ids[i]) for i in top if sims[i] > 0]

            loop = asyncio.get_running_loop()
            scored = await loop.run_in_executor(None, _score)
            return [(sim, self.chunks[cid]) for sim, cid in scored if cid in self.chunks]
        else:
            # Pure-Python fallback
            embedded = [
                (cid, c) for cid, c in self.chunks.items() if c.embedding is not None
            ]
            q_norm = math.sqrt(sum(x * x for x in query_vec))
            if q_norm == 0:
                return []
            results = []
            for cid, chunk in embedded:
                emb = chunk.embedding
                dot = sum(a * b for a, b in zip(emb, query_vec))
                e_norm = math.sqrt(sum(x * x for x in emb))
                if e_norm == 0:
                    continue
                sim = dot / (e_norm * q_norm)
                if sim > 0:
                    if self._is_test_path(chunk.source_path):
                        sim *= self._TEST_PENALTY
                    results.append((sim, chunk))
            results.sort(key=lambda x: -x[0])
            return results[:max_results]

    # ── Embedding Matrix Cache ────────────────────────────────────────

    def _ensure_embedding_matrix(self) -> None:
        """Build pre-computed numpy matrix from chunk embeddings (once).

        The matrix construction (np.array from 61K Python lists) takes ~2500ms.
        Cache it so every semantic_query pays only ~50ms for cosine similarity.
        Invalidated when embeddings change (embed_chunks, reindex_files).
        """
        if self._embedding_matrix is not None:
            return
        if not _HAS_NUMPY:
            return
        embedded = [(cid, c) for cid, c in self.chunks.items() if c.embedding is not None]
        if not embedded:
            return
        t0 = time.time()
        self._embedding_ids = [cid for cid, _ in embedded]
        self._embedding_matrix = np.array(
            [c.embedding for _, c in embedded], dtype=np.float32
        )
        self._embedding_norms = np.linalg.norm(self._embedding_matrix, axis=1)
        # Test-file penalty as a vector, computed once here rather than as a
        # per-query Python loop over every embedded chunk in semantic_query().
        self._embedding_penalties = np.array(
            [
                self._TEST_PENALTY if self._is_test_path(c.source_path) else 1.0
                for _, c in embedded
            ],
            dtype=np.float32,
        )
        # id → row map so a chunk's vector can be rehydrated from the matrix
        self._embedding_row_map = {cid: i for i, cid in enumerate(self._embedding_ids)}
        # Fold (memory audit 2026-07): the matrix is now the canonical vector
        # store. Release each covered chunk's per-chunk vector (boxed lists
        # measured ~0.9GB at 101K chunks) and bind it to its matrix row —
        # reads through CodeChunk.embedding rehydrate the float32 row on
        # demand, and the next rebuild re-collects vectors THROUGH those
        # bindings, so _invalidate_embedding_matrix() loses nothing.
        for i, (cid, c) in enumerate(embedded):
            c._bind_matrix_row(self._embedding_matrix, i)
        elapsed = (time.time() - t0) * 1000
        logger.info(
            f"[MATRIX] Pre-computed embedding matrix: {self._embedding_matrix.shape} in {elapsed:.0f}ms"
        )

    async def _ensure_embedding_matrix_async(self) -> None:
        """Non-blocking version — runs matrix computation in thread executor."""
        if self._embedding_matrix is not None:
            return
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_embedding_matrix)

    def _invalidate_embedding_matrix(self) -> None:
        """Clear cached matrix — call after embeddings change.

        Folded chunks keep their (matrix, row) binding, so the old matrix stays
        reachable as each chunk's vector source until the rebuild rebinds them
        to the new one — invalidation never drops vector data.
        """
        self._embedding_matrix = None
        self._embedding_ids = None
        self._embedding_norms = None
        self._embedding_penalties = None
        self._embedding_row_map = None

    # ── Query Embedding Cache ─────────────────────────────────────────

    def _cache_query_embedding(self, query: str, vec: list) -> None:
        """Store query embedding in LRU cache, evicting oldest if full."""
        if query in self._query_embed_cache:
            # Move to end (most recent)
            self._query_cache_order.remove(query)
            self._query_cache_order.append(query)
            return
        if len(self._query_embed_cache) >= self._QUERY_CACHE_MAX:
            oldest = self._query_cache_order.pop(0)
            self._query_embed_cache.pop(oldest, None)
        self._query_embed_cache[query] = vec
        self._query_cache_order.append(query)

    def _background_cache_query(self, query: str) -> None:
        """Fire-and-forget: embed query in background and cache result."""
        async def _do():
            try:
                vecs = await _embed_texts([query], is_query=True)
                if vecs and vecs[0]:
                    self._cache_query_embedding(query, vecs[0])
                    logger.debug(f"[CACHE] Background-cached embedding for: {query[:40]}")
            except Exception as e:
                logger.debug(f"[CodeGraph._do] Operation failed: {e}")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do())
        except RuntimeError as e:
            logger.debug(f"[CodeGraph._do] Graph operation failed: {e}")

    async def pre_embed_query(self, query: str) -> None:
        """Pre-compute and cache query embedding for future hybrid_query() calls.

        Called during typing speculation. When the actual hybrid_query() fires,
        it finds the embedding already cached and skips the embed step.
        """
        if query in self._query_embed_cache:
            return  # Already cached
        try:
            vecs = await _embed_texts([query], is_query=True)
            if vecs and vecs[0]:
                self._cache_query_embedding(query, vecs[0])
                logger.debug(f"[PREFETCH] Pre-embedded query: {query[:40]}")
        except Exception as e:
            logger.debug(f"[PREFETCH] Query pre-embed failed: {e}")

    async def warm_query_cache(self, queries: List[str]) -> int:
        """
        Pre-embed multiple queries in parallel.  Batches into a single API
        call so N queries cost ~1× round-trip, not N×.

        Returns number of newly cached embeddings.
        """
        uncached = [q for q in queries if q not in self._query_embed_cache]
        if not uncached:
            return 0
        try:
            vecs = await _embed_texts(uncached, is_query=True)
            cached = 0
            for q, v in zip(uncached, vecs):
                if v:
                    self._cache_query_embedding(q, v)
                    cached += 1
            logger.info(f"[CACHE] Warmed {cached}/{len(uncached)} query embeddings in batch")
            return cached
        except Exception as e:
            logger.warning(f"[CACHE] Batch warmup failed: {e}")
            return 0

    async def _expand_context(
        self,
        chunk_ids: Set[str],
        query: str = "",
        max_expand: int = 15,
    ) -> List["CodeChunk"]:
        """
        Expand a set of hit chunk IDs with structurally and semantically related chunks.

        Phase 1 — Structural: parent class, sibling methods, file neighbors, call graph.
        Phase 2 — Semantic diversity: embedding-similar chunks from DIFFERENT files
                  than the initial hits.  This is what lifts hard/architectural queries.
        """
        expanded: List["CodeChunk"] = []
        seen = set(chunk_ids)
        hit_files = {self.chunks[cid].source_path for cid in chunk_ids if cid in self.chunks}

        # --- Phase 1: structural expansion (same as before) ---
        structural_budget = max_expand // 2 or max_expand

        for cid in list(chunk_ids):
            chunk = self.chunks.get(cid)
            if not chunk or len(expanded) >= structural_budget:
                break

            # 1a. Parent class: if this is a method, pull the class chunk
            if chunk.parent_class:
                for other_id in self.by_name.get(chunk.parent_class, []):
                    if other_id not in seen and other_id in self.chunks:
                        seen.add(other_id)
                        expanded.append(self.chunks[other_id])

                # Sibling methods from same class (O(1) via by_class index)
                for other_id in self.by_class.get(chunk.parent_class, []):
                    if other_id not in seen:
                        other = self.chunks.get(other_id)
                        if other and other.chunk_type == ChunkType.METHOD:
                            seen.add(other_id)
                            expanded.append(other)
                            if len(expanded) >= structural_budget:
                                break

            # 1b. Same-file neighbors (functions/classes in the same module)
            file_siblings = self.by_file.get(chunk.source_path, [])
            for sib_id in file_siblings[:8]:
                if sib_id not in seen and sib_id in self.chunks:
                    seen.add(sib_id)
                    expanded.append(self.chunks[sib_id])
                    if len(expanded) >= structural_budget:
                        break

            # 1c. Call-graph expansion (1 level)
            ctx = self.get_context_for_chunk(cid)
            for related in ctx.get("callers", []) + ctx.get("callees", []):
                if related.id not in seen:
                    seen.add(related.id)
                    expanded.append(related)
                    if len(expanded) >= structural_budget:
                        break

        # --- Phase 2: semantic diversity (cross-file) ---
        diversity_budget = max_expand - len(expanded)
        if diversity_budget > 0 and query and _HAS_EMBEDDING_ENGINE:
            sem_results = await self.semantic_query(query, max_results=diversity_budget * 3)
            for _sim, chunk in sem_results:
                if chunk.id not in seen and chunk.source_path not in hit_files:
                    seen.add(chunk.id)
                    expanded.append(chunk)
                    if len(expanded) >= max_expand:
                        break

        return expanded

    # ========================================================================
    # MULTI-HOP CHAIN EXPANSION — Architectural query support
    # ========================================================================

    def _multi_hop_expand(
        self,
        seed_chunk_ids: List[str],
        query: str,
        max_depth: int = 3,
        max_chains: int = 5,
    ) -> List[List["CodeChunk"]]:
        """
        BFS on call graph to build multi-hop responsibility chains.

        For architectural queries, follows calls/called_by relationships
        up to max_depth levels, pruning irrelevant branches.

        Returns list of chains (each chain is a list of CodeChunks).
        """
        query_tokens = set(query.lower().split())
        chains: List[List["CodeChunk"]] = []
        visited_chains: set = set()  # Avoid duplicate chain fingerprints

        for seed_id in seed_chunk_ids:
            seed = self.chunks.get(seed_id)
            if not seed:
                continue

            # BFS with path tracking
            queue: List[Tuple[str, List[str], int]] = [(seed_id, [seed_id], 0)]
            seen_in_search: set = {seed_id}

            while queue and len(chains) < max_chains * 3:
                current_id, path, depth = queue.pop(0)
                current = self.chunks.get(current_id)
                if not current or depth >= max_depth:
                    continue

                # Get neighbors: forward calls + reverse callers
                neighbors: List[str] = []
                for call_name in current.calls[:8]:
                    for cid in self.by_name.get(call_name, [])[:2]:
                        if cid not in seen_in_search:
                            neighbors.append(cid)
                for caller_name in current.called_by[:8]:
                    for cid in self.by_name.get(caller_name, [])[:2]:
                        if cid not in seen_in_search:
                            neighbors.append(cid)

                for neighbor_id in neighbors:
                    neighbor = self.chunks.get(neighbor_id)
                    if not neighbor:
                        continue

                    # Relevance check: does this hop relate to the query?
                    hop_text = (neighbor.name + " " + (neighbor.docstring or "")).lower()
                    hop_tokens = set(hop_text.split())
                    overlap = len(query_tokens & hop_tokens)

                    # Prune: zero relevance AND same file = dead branch
                    if overlap == 0 and neighbor.source_path == current.source_path:
                        continue

                    new_path = path + [neighbor_id]
                    seen_in_search.add(neighbor_id)

                    # Record chain if it spans 2+ nodes
                    if len(new_path) >= 2:
                        chain_key = tuple(sorted(new_path))
                        if chain_key not in visited_chains:
                            visited_chains.add(chain_key)
                            chain = [self.chunks[cid] for cid in new_path if cid in self.chunks]
                            if len(chain) >= 2:
                                chains.append(chain)

                    # Continue BFS
                    if depth + 1 < max_depth:
                        queue.append((neighbor_id, new_path, depth + 1))

        return chains

    def _score_chains(
        self,
        chains: List[List["CodeChunk"]],
        query: str,
    ) -> List[Tuple[float, List["CodeChunk"]]]:
        """
        Score and rank chains by relevance to query.

        Score = sum(hop_relevance * 0.7^depth) * cross_file_bonus * sqrt(len)
        """
        query_tokens = set(query.lower().split())
        scored: List[Tuple[float, List["CodeChunk"]]] = []

        for chain in chains:
            hop_score = 0.0
            unique_files: set = set()

            for depth, chunk in enumerate(chain):
                # Relevance of this hop
                hop_text = (chunk.name + " " + (chunk.docstring or "") + " " + chunk.signature).lower()
                hop_tokens = set(hop_text.split())
                overlap = len(query_tokens & hop_tokens)
                relevance = min(1.0, overlap / max(1, len(query_tokens)))

                # Decay by depth
                hop_score += relevance * (0.7 ** depth)
                unique_files.add(chunk.source_path)

            # Cross-file bonus: chains spanning multiple files are more valuable
            cross_file_bonus = 1.0 + 0.1 * len(unique_files)

            # Length bonus: longer chains (that stayed relevant) are richer
            length_bonus = math.sqrt(len(chain))

            total_score = hop_score * cross_file_bonus * length_bonus
            scored.append((total_score, chain))

        scored.sort(key=lambda x: -x[0])
        return scored

    async def _rerank(
        self,
        query: str,
        chunks: List["CodeChunk"],
        top_k: int = 10,
        mode: str = "embedding",
    ) -> List["CodeChunk"]:
        """
        Re-rank candidates by relevance to query.

        Modes:
            'embedding' — Fast: cosine similarity of query embedding vs chunk embeddings.
                          Zero LLM calls. ~5ms for 20 candidates. Default.
            'llm'       — Accurate: parallel chunked LLM scoring via nemotron-mini.
                          Splits candidates into groups of 5, scores in parallel.
            'hybrid'    — Embedding pre-filter → LLM re-score top candidates.
        """
        if len(chunks) <= top_k:
            return chunks[:top_k]

        if mode == "embedding":
            return await self._rerank_by_embedding(query, chunks, top_k)
        elif mode == "llm":
            return await self._rerank_by_llm(query, chunks, top_k)
        elif mode == "hybrid":
            # Embedding narrows to 2x top_k, then LLM picks final top_k
            narrowed = await self._rerank_by_embedding(query, chunks, top_k * 2)
            return await self._rerank_by_llm(query, narrowed, top_k)
        else:
            return chunks[:top_k]

    async def _rerank_by_embedding(
        self,
        query: str,
        chunks: List["CodeChunk"],
        top_k: int = 10,
    ) -> List["CodeChunk"]:
        """
        Re-rank using embedding cosine similarity. Zero LLM calls.

        Embeds the query once, computes cosine sim against each candidate's
        existing embedding. Falls back to original order for un-embedded chunks.
        """
        if not _HAS_EMBEDDING_ENGINE and not _detect_vllm():
            return chunks[:top_k]

        # Embed the query via EmbeddingEngine
        try:
            vecs = await _embed_texts([query], model="nomic-embed-text", is_query=True)
            query_vec = vecs[0] if vecs else None
            if not query_vec:
                return chunks[:top_k]
        except Exception as e:
            logger.debug(f"Embedding re-rank failed (query embed): {e}")
            return chunks[:top_k]

        if _HAS_NUMPY:
            qv = np.array(query_vec, dtype=np.float32)
            q_norm = np.linalg.norm(qv)
            if q_norm == 0:
                return chunks[:top_k]

            scored = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    cv = np.array(chunk.embedding, dtype=np.float32)
                    c_norm = np.linalg.norm(cv)
                    sim = float(np.dot(qv, cv) / (q_norm * c_norm)) if c_norm > 0 else 0.0
                else:
                    sim = 0.0
                scored.append((sim, chunk))
        else:
            q_norm = math.sqrt(sum(x * x for x in query_vec))
            if q_norm == 0:
                return chunks[:top_k]
            scored = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    dot = sum(a * b for a, b in zip(chunk.embedding, query_vec))
                    c_norm = math.sqrt(sum(x * x for x in chunk.embedding))
                    sim = dot / (q_norm * c_norm) if c_norm > 0 else 0.0
                else:
                    sim = 0.0
                scored.append((sim, chunk))

        scored.sort(key=lambda x: -x[0])
        return [chunk for _, chunk in scored[:top_k]]

    async def _rerank_by_llm(
        self,
        query: str,
        chunks: List["CodeChunk"],
        top_k: int = 10,
        group_size: int = 5,
    ) -> List["CodeChunk"]:
        """
        Parallel chunked LLM re-ranking via nemotron-mini.

        Splits candidates into groups of `group_size`, scores each group in
        parallel via asyncio.gather. ~4x faster than sequential for 20 candidates.
        """
        if (not _HAS_EMBEDDING_ENGINE and not _detect_vllm()) or len(chunks) <= top_k:
            return chunks[:top_k]

        candidates = chunks[:20]  # Cap at 20

        # Split into groups
        groups = [
            candidates[i : i + group_size]
            for i in range(0, len(candidates), group_size)
        ]

        async def _score_group(group: List["CodeChunk"], offset: int) -> List[Tuple[int, int]]:
            """Score a group of candidates. Returns list of (score, global_index)."""
            items = []
            for j, c in enumerate(group):
                sig = c.signature[:120]
                doc = (c.docstring[:80] if c.docstring else "")
                calls = ", ".join(c.calls[:5]) if c.calls else "none"
                items.append(f"{j}. {c.name} | {sig} | calls: {calls} | {doc}")

            prompt = (
                f"Rate each code function's relevance to the query on a scale 0-9.\n"
                f"Query: {query}\n\n"
                f"Candidates:\n" + "\n".join(items) + "\n\n"
                f"Reply ONLY with one number (0-9) per line, one line per candidate. "
                f"No explanations."
            )
            try:
                text = await _llm_generate(prompt, model=ELASTIC_REFLEX, max_tokens=50)
                lines = text.strip().split("\n")
                results = []
                for j, line in enumerate(lines):
                    digits = [ch for ch in line.strip() if ch.isdigit()]
                    score = int(digits[0]) if digits else 5
                    results.append((score, offset + j))
                # Pad if model returned fewer lines
                while len(results) < len(group):
                    results.append((5, offset + len(results)))
                return results
            except Exception as e:
                logger.debug(f"LLM re-rank group failed: {e}")
                return [(5, offset + j) for j in range(len(group))]

        # Fire all groups in parallel
        group_tasks = [
            _score_group(group, i * group_size)
            for i, group in enumerate(groups)
        ]
        group_results = await asyncio.gather(*group_tasks)

        # Flatten and sort by score descending
        all_scored = []
        for results in group_results:
            all_scored.extend(results)
        all_scored.sort(key=lambda x: -x[0])

        return [candidates[idx] for _, idx in all_scored if idx < len(candidates)][:top_k]

    # Precompiled patterns for query classification
    _RE_CAMELCASE = re.compile(r'[A-Z][a-z]+[A-Z]')       # CamelCase identifiers
    _RE_SNAKE = re.compile(r'[a-z]+_[a-z]+')               # snake_case identifiers
    _RE_DOTPATH = re.compile(r'\w+\.\w+\.\w+')             # dotted paths (a.b.c)
    _RE_FILE_EXT = re.compile(r'\.\w{1,4}(?:\s|$|[,)])')   # file extensions (.py, .yaml)
    _RE_ARCHITECTURAL = re.compile(
        r'\b(?:trace|lifecycle|pipeline|flow|architecture|end.to.end|'
        r'full\s+path|how\s+does\s+\w+\s+handle|what\s+happens\s+when)\b',
        re.IGNORECASE,
    )
    # NOTE: "port" was removed from this set on 2026-07-20. It is a
    # single-service ATTRIBUTE, not a cross-file concern — "What port does X run on?" is
    # answered by one service file, yet the bare term routed the query to cross_domain
    # (kw=0.9/sem=0.1) via the first-match-wins tree below, before it could reach the
    # `focused` branch (kw=0.3/sem=0.7) that its single symbol earns it. Verified live:
    # that exact query returned AitherStrata/registry/turboquant while the correct file
    # was indexed and ranked #1 on a direct search. This is a correct-by-construction
    # fix, not a tuned one: no corpus can make "port" mean cross-file.
    _RE_CROSS_DOMAIN = re.compile(
        r'\b(?:config|yaml|services\.yaml|\.env|how\s+do\s+I\s+add|'
        r'set\s*up|deploy|install|register)\b',
        re.IGNORECASE,
    )
    # Relationship patterns: "X interacts with Y", "X links to Y", "across"
    _RE_RELATIONSHIP = re.compile(
        r'\b(?:interact|route\s+\w+\s+to|link|connect|between|across|'
        r'communicate|integrate|layers|and\s+how|query\s+and)\b',
        re.IGNORECASE,
    )

    @staticmethod
    def classify_query(query: str) -> Tuple[float, float, str]:
        """
        Classify a query and return optimal (keyword_weight, semantic_weight, reason).

        Categories (from grid search data, validated by benchmark):
            relationship   → kw=0.8, sem=0.2  (cross-entity interactions, need exact file matching)
            conceptual     → kw=0.2, sem=0.8  (natural language, conceptual reasoning)
            architectural  → kw=0.0, sem=1.0  (cross-abstraction, multi-file tracing)
            cross_domain   → kw=0.9, sem=0.1  (code+config, literal identifiers across domains)
            focused        → kw=0.3, sem=0.7  (single entity behavior, embeddings map cleanly)
        """
        q = query.strip()

        # Count signal types
        camel_hits = len(CodeGraph._RE_CAMELCASE.findall(q))
        snake_hits = len(CodeGraph._RE_SNAKE.findall(q))
        dot_hits = len(CodeGraph._RE_DOTPATH.findall(q))
        file_hits = len(CodeGraph._RE_FILE_EXT.findall(q))
        symbol_count = camel_hits + snake_hits + dot_hits + file_hits

        is_architectural = bool(CodeGraph._RE_ARCHITECTURAL.search(q))
        is_cross_domain = bool(CodeGraph._RE_CROSS_DOMAIN.search(q))
        is_relationship = bool(CodeGraph._RE_RELATIONSHIP.search(q))

        words = q.split()
        word_count = len(words)

        # STRONG literal evidence: real identifiers/paths a user pastes when they
        # mean a specific symbol — snake_case, dotted paths, file extensions.
        # CamelCase is DELIBERATELY excluded here: it pervades natural-language
        # prose about a codebase ("the ContextEngine injects...", "AitherCapture —
        # Universal Coding-Session Capture SDK") and, when it was allowed to route
        # on its own, sent 12% of realistic prose queries into keyword-heavy
        # weights. Measured on the 200-query docstring corpus: the
        # adaptive path scored recall@10=0.885/MRR=0.727 while a flat kw=0.2/sem=0.8
        # scored 0.905/0.750 — i.e. the classifier's keyword-heavy branches were a
        # net LOSS. The sweep also proved keyword-only is the worst config
        # (recall 0.72 vs 0.90). So keyword-heavy now requires a strong literal
        # token, not prose; everything else defaults semantic-heavy. Genuine
        # literal queries (with .py/.yaml/dotpaths/snake_case) route exactly as
        # before — this fixes false-positive routing, it does not retune weights
        # to the corpus.
        strong_literal = (snake_hits + dot_hits + file_hits) >= 1

        # Decision tree (ordered by specificity)

        # 1. Architectural: "trace the full lifecycle of..." — pure semantic.
        if is_architectural and word_count > 6:
            return (0.0, 1.0, "architectural")

        # 2. Cross-domain: config/yaml/deploy — keyword-leaning ONLY when a real
        #    file/path token is present, not on the bare prose word.
        if is_cross_domain and (file_hits or dot_hits):
            return (0.9, 0.1, "cross_domain")

        # 3. Relationship: "X links Y to Z" — keyword-heavy needs exact symbol
        #    matching, so require a strong literal token, not bare "link/across/
        #    between/layers" prose that _RE_RELATIONSHIP also matches.
        if is_relationship and strong_literal:
            return (0.8, 0.2, "relationship")

        # 4. Symbol-dense: 2+ REAL literal identifiers (not incidental CamelCase).
        if strong_literal and symbol_count >= 2:
            return (0.7, 0.3, "multi_symbol")

        # 5. Single strong literal identifier → focused, still semantic-leaning.
        if strong_literal:
            return (0.3, 0.7, "focused")

        # 6. Everything else is natural language → semantic default. This is where
        #    CamelCase-in-prose now lands (was misrouted keyword-heavy above).
        if word_count > 6:
            return (0.2, 0.8, "conceptual")

        # 7. Short/ambiguous — lean semantic (the corpus-best default).
        return (0.3, 0.7, "balanced")

    async def hybrid_query(
        self,
        query: str,
        max_results: int = 10,
        keyword_weight: float | None = None,
        semantic_weight: float | None = None,
        expand_context: bool = True,
        rerank: bool | str = False,
        **kwargs,
    ) -> List["CodeChunk"]:
        """
        Hybrid search combining keyword scoring and semantic similarity.

        Falls back to keyword-only if no embeddings are available.

        Pipeline: classify → keyword + semantic → merge → context expansion → (optional) re-rank

        Args:
            query: Natural language or keyword query
            max_results: Maximum results
            keyword_weight: Weight for keyword score (0–1). None = auto-classify.
            semantic_weight: Weight for semantic score (0–1). None = auto-classify.
            expand_context: Pull parent class, sibling methods, file neighbors
            rerank: False=off, True/'embedding'=fast embedding rerank (~5ms),
                    'llm'=parallel LLM scoring, 'hybrid'=embedding then LLM
        """
        # Adaptive weight classification
        query_type = "balanced"  # default if weights provided explicitly
        if keyword_weight is None or semantic_weight is None:
            keyword_weight, semantic_weight, query_type = self.classify_query(query)
            logger.debug(f"Query classified as '{query_type}': kw={keyword_weight}, sem={semantic_weight}")

        # Always run keyword search (pass through scope filters from kwargs)
        _scope_paths = kwargs.get("scope_paths")
        _scope_tenant = kwargs.get("tenant_id")
        keyword_results = await self.query(
            query, max_results=max_results * 3,
            scope_paths=_scope_paths, tenant_id=_scope_tenant,
        )

        # Check if semantic search is available (cached flag avoids 28K-item scan)
        if not hasattr(self, '_has_embeddings_cached'):
            self._has_embeddings_cached = any(
                c.embedding is not None for c in self.chunks.values()
            )
        if not self._has_embeddings_cached or not _HAS_EMBEDDING_ENGINE:
            return keyword_results[:max_results]

        # Run semantic search.
        # If query embedding is cached: cosine math only (~1ms) — always succeeds.
        # If uncached: embed round-trip + cosine over the full matrix, then fire a
        # background cache warmup for next time.
        #
        # The cold cap was 0.15s, chosen when the embed was fast and the matrix
        # small. It is now too tight and was silently gutting retrieval on
        # first-time queries: the code embedder round-trip is ~21ms on a
        # warm connection but ~780ms on a fresh TLS connection (a new AsyncClient
        # per call in _embed_via_code_service), and the cosine grew with the index
        # (61k→101k chunks). Measured live 2026-07-21: cold queries scored
        # recall@10=0.720 (== keyword-ONLY — the semantic arm timed out every
        # time and fell back) while the SAME queries warm-cached scored 0.960. The
        # cap is a ceiling, not the latency: warm-connection queries still return
        # in ~120ms; raising it only lets the genuinely-slow first hit COMPLETE
        # (and background-cache) instead of throwing away the embeddings. A slow/
        # hung backend is still bounded (1.0s) and still degrades to keyword-only.
        # (Follow-up: pool the code-embedder connection so cold embeds are ~21ms.)
        cache_hit = query in self._query_embed_cache
        timeout = 1.0  # ceiling only (was 0.15 cold); see note above

        # Build the embedding matrix BEFORE entering the timed region.
        #
        # It was previously built lazily inside semantic_query, i.e. INSIDE the
        # wait_for below. Measured 2026-07-20: the build takes ~1900ms for
        # (57413, 768) while the cache-hit budget is 1000ms — so the FIRST
        # semantic query after every boot, embed, or reindex (anything that
        # calls _invalidate_embedding_matrix) blew the cap, logged at debug, and
        # silently returned keyword-only. The comment above claiming a cache hit
        # is "cosine math only (~1ms) — always succeeds" was true only once the
        # matrix already existed. Warm queries measure 44ms, so the budget was
        # never the problem; paying a one-time 1.9s setup inside a 1.0s cap was.
        if _HAS_NUMPY and self._embedding_matrix is None:
            try:
                await self._ensure_embedding_matrix_async()
            except Exception as e:
                logger.warning(f"[HYBRID] embedding matrix build failed: {e}")

        try:
            semantic_results = await asyncio.wait_for(
                self.semantic_query(query, max_results=max_results * 3),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.debug("[HYBRID] Semantic timed out — keyword-only + background cache warmup")
            self._background_cache_query(query)  # Next time will be a cache hit
            semantic_results = None
        if not semantic_results:
            if not cache_hit:
                self._background_cache_query(query)  # Pre-cache for next call
            return keyword_results[:max_results]

        # Build fused score maps via Reciprocal Rank Fusion (RRF).
        #
        # Both signals MUST live on the same scale. The previous scheme mixed
        # reciprocal rank for keyword (1.0, 0.5, 0.33 — steep) with
        # `sim / max_sim` for semantic (1.0, 0.98, 0.96 — flat, because cosine
        # similarities on code embeddings cluster tightly). That made the
        # semantic term a near-constant *membership bonus* rather than a ranking
        # signal: at sem=0.7 every one of the 30 semantic candidates scored
        # ~0.66 while keyword rank 4 scored 0.075, so semantically-adjacent
        # chunks displaced exact lexical matches out of the top-N. Measured:
        # overall F1 0.293 (keyword-only) -> 0.229 (with embeddings), and
        # architectural queries (kw=0.0/sem=1.0 — pure flat signal) fell to 0.
        #
        # RRF puts both on 1/(k + rank), so rank position discriminates on both
        # sides and the classify_query() weights apply to a real gradient.
        kw_scores: Dict[str, float] = {}
        for rank, chunk in enumerate(keyword_results):
            kw_scores[chunk.id] = 1.0 / (self._RRF_K + rank + 1)

        sem_scores: Dict[str, float] = {}
        for rank, (_sim, chunk) in enumerate(semantic_results):
            sem_scores[chunk.id] = 1.0 / (self._RRF_K + rank + 1)

        # Merge and score (with scope filtering)
        all_ids = set(kw_scores.keys()) | set(sem_scores.keys())
        combined = []
        for cid in all_ids:
            chunk = self.chunks.get(cid)
            if not chunk:
                continue
            # Scope filter: tenant
            if _scope_tenant and chunk.tenant_id not in (_scope_tenant, "platform"):
                continue
            # Scope filter: paths
            if _scope_paths and not any(chunk.source_path.startswith(sp) for sp in _scope_paths):
                continue
            kw = kw_scores.get(cid, 0.0)
            sem = sem_scores.get(cid, 0.0)
            score = keyword_weight * kw + semantic_weight * sem
            # Deprioritize test files in hybrid results too
            if self._is_test_path(chunk.source_path):
                score *= self._TEST_PENALTY
            combined.append((score, cid))

        combined.sort(key=lambda x: -x[0])
        top_ids = [cid for _, cid in combined[:max_results] if cid in self.chunks]
        # Build a map of chunk id -> score for later attachment
        score_map = {cid: score for score, cid in combined}
        top_chunks = [self.chunks[cid] for cid in top_ids]
        # Attach computed scores to top chunks (fail-open: default 0.7 if missing)
        for chunk in top_chunks:
            chunk.relevance_score = score_map.get(chunk.id, 0.7)

        # Context expansion: pull structurally + semantically related chunks
        if expand_context and top_ids:
            expanded = await self._expand_context(
                set(top_ids[:5]), query=query, max_expand=max_results,
            )
            # Append expanded chunks after the scored results
            top_chunks = top_chunks + expanded

            # Multi-hop chain expansion for architectural queries
            if query_type == "architectural":
                try:
                    chains = self._multi_hop_expand(top_ids[:5], query)
                    if chains:
                        scored_chains = self._score_chains(chains, query)
                        for _score, chain in scored_chains[:3]:
                            top_chunks.extend(chain)
                        logger.debug(
                            f"[MULTI_HOP] {len(chains)} chains found, "
                            f"top 3 injected ({sum(len(c) for _, c in scored_chains[:3])} chunks)"
                        )
                except Exception as e:
                    logger.debug(f"[MULTI_HOP] Chain expansion failed: {e}")

            # Deduplicate while preserving order
            seen: Set[str] = set()
            deduped = []
            for c in top_chunks:
                if c.id not in seen:
                    seen.add(c.id)
                    deduped.append(c)
            top_chunks = deduped[:max_results * 2]

        # Re-ranking: explicit mode or auto-apply embedding rerank when
        # expand_context added extra chunks that need to compete for slots
        if rerank:
            mode = rerank if isinstance(rerank, str) else "embedding"
            top_chunks = await self._rerank(query, top_chunks, top_k=max_results, mode=mode)
        elif expand_context and len(top_chunks) > max_results:
            # Expanded chunks need scoring to compete with originals
            top_chunks = await self._rerank_by_embedding(query, top_chunks, top_k=max_results)

        return top_chunks[:max_results]

    # ── Full body retrieval ──────────────────────────────────────────────

    def get_full_body(self, chunk_id: str) -> Optional[str]:
        """
        Get the full source text of a chunk by reading the source file.

        Lazy-loads and caches the result. Falls back to body_preview
        if the source file is missing. LRU eviction maintains the cache
        at _BODY_CACHE_MAX entries.

        Args:
            chunk_id: The chunk ID to look up.

        Returns:
            Full source text, or body_preview fallback, or None if unknown.
        """
        if chunk_id in self._body_cache:
            # LRU hit: move to end (most recently used)
            self._body_cache.move_to_end(chunk_id)
            return self._body_cache[chunk_id]

        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return None

        # Try reading from source file
        root = self._root_path or ""
        source_path = os.path.join(root, chunk.source_path) if root else chunk.source_path

        try:
            with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            # Extract the chunk's line range (1-indexed start_line/end_line)
            start = max(0, chunk.start_line - 1)
            end = min(len(lines), chunk.end_line)
            body = "".join(lines[start:end])
            # LRU eviction: ensure cache doesn't exceed _BODY_CACHE_MAX entries
            if len(self._body_cache) >= _BODY_CACHE_MAX:
                self._body_cache.popitem(last=False)  # Remove oldest (first) entry
            self._body_cache[chunk_id] = body
            return body
        except (FileNotFoundError, OSError):
            # Fall back to body_preview
            preview = getattr(chunk, "body_preview", None)
            if preview:
                # LRU eviction on preview storage as well
                if len(self._body_cache) >= _BODY_CACHE_MAX:
                    self._body_cache.popitem(last=False)
                self._body_cache[chunk_id] = preview
            return preview

    @property
    def memory_usage_mb(self) -> float:
        """Estimate memory usage of the CodeGraph in megabytes."""
        total_bytes = 0
        # Chunks
        for chunk in self.chunks.values():
            total_bytes += len(getattr(chunk, "body_preview", "") or "")
            total_bytes += len(getattr(chunk, "signature", "") or "")
            total_bytes += len(getattr(chunk, "docstring", "") or "")
            # Only vectors still boxed ON the chunk; matrix-folded chunks are
            # covered by the matrix nbytes below (and an ndarray would be
            # ambiguous under plain truthiness anyway)
            _emb = chunk.__dict__.get("_embedding")
            if _emb is not None:
                total_bytes += len(_emb) * 4  # float32 rows
        # Body cache
        for body in self._body_cache.values():
            total_bytes += len(body)
        # Query caches
        for emb in self._query_embed_cache.values():
            total_bytes += len(emb) * 8
        # Embedding matrix
        if self._embedding_matrix is not None and _HAS_NUMPY:
            total_bytes += self._embedding_matrix.nbytes
        return total_bytes / (1024 * 1024)

    # ========================================================================
    # PYTHON METRICS — instant from in-memory index
    # ========================================================================

    _AREA_PREFIXES = [
        ("lib/", "lib"), ("services/", "services"),
        ("apps/AitherVeil/", "frontend"), ("apps/AitherGenesis/", "genesis"),
        ("apps/AitherNode/", "node"), ("apps/", "apps"),
        ("dev/tests/", "tests"), ("boot/", "boot"),
        ("config/", "config"), ("scripts/", "scripts"),
    ]

    def _classify_area(self, path: str) -> str:
        """Classify a Python file into a project area."""
        normalized = path.replace("\\", "/")
        for prefix, area in self._AREA_PREFIXES:
            if prefix in normalized:
                idx = normalized.find(prefix)
                if idx >= 0 and normalized[idx:].startswith(prefix):
                    return area
        return "other"

    def get_python_metrics(self) -> Dict[str, Any]:
        """
        Instant Python-specific metrics from in-memory CodeGraph index.

        O(n) over chunks, typically <10ms for ~5000 chunks.

        Returns dict with:
            total_py_files, total_chunks, total_py_lines, functions, classes,
            methods, avg_complexity, top_complex_files, by_area, test_functions,
            test_lines
        """
        if not self.chunks:
            return {
                "total_py_files": 0, "total_chunks": 0, "total_py_lines": 0,
                "functions": 0, "classes": 0, "methods": 0,
                "avg_complexity": 0.0, "top_complex_files": [],
                "by_area": {}, "test_functions": 0, "test_lines": 0,
            }

        functions = 0
        classes = 0
        methods = 0
        complexities = []
        file_complexity: Dict[str, List[int]] = {}
        area_lines: Dict[str, int] = {}
        test_functions = 0
        test_lines = 0

        for chunk in self.chunks.values():
            ct = chunk.chunk_type
            if ct == ChunkType.FUNCTION:
                functions += 1
                if chunk.name.startswith("test_"):
                    test_functions += 1
            elif ct == ChunkType.METHOD:
                methods += 1
                if chunk.name.startswith("test_"):
                    test_functions += 1
            elif ct == ChunkType.CLASS:
                classes += 1

            if chunk.complexity:
                complexities.append(chunk.complexity)
                fp = chunk.source_path
                if fp not in file_complexity:
                    file_complexity[fp] = []
                file_complexity[fp].append(chunk.complexity)

            lc = chunk.line_count or (chunk.end_line - chunk.start_line + 1)
            area = self._classify_area(chunk.source_path)
            area_lines[area] = area_lines.get(area, 0) + lc

            # Test lines
            path_lower = chunk.source_path.lower().replace("\\", "/")
            if ("/tests/" in path_lower or "/test/" in path_lower
                    or "/test_" in path_lower):
                test_lines += lc

        # Top complex files
        file_avg: List[Tuple[str, float]] = []
        for fp, cxs in file_complexity.items():
            file_avg.append((fp, sum(cxs) / len(cxs)))
        file_avg.sort(key=lambda x: x[1], reverse=True)

        avg_cx = sum(complexities) / len(complexities) if complexities else 0.0

        return {
            "total_py_files": len(self.by_file),
            "total_chunks": len(self.chunks),
            "total_py_lines": sum(area_lines.values()),
            "functions": functions,
            "classes": classes,
            "methods": methods,
            "avg_complexity": round(avg_cx, 2),
            "top_complex_files": [
                (Path(fp).name, round(cx, 2)) for fp, cx in file_avg[:10]
            ],
            "by_area": area_lines,
            "test_functions": test_functions,
            "test_lines": test_lines,
        }


# ============================================================================
# CLI
# ============================================================================

async def main():
    """CLI entry point."""
    import sys
    
    root_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent.parent)
    
    print(f"\n{'='*60}")
    print("CODE GRAPH - Python AST Indexer")
    print(f"{'='*60}")
    print(f"Root: {root_path}")
    
    graph = CodeGraph(max_workers=8)
    
    def on_progress(progress: float, message: str):
        print(f"[{progress*100:5.1f}%] {message}")
    
    stats = await graph.index_codebase(root_path, on_progress=on_progress)
    
    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Files:      {stats['total_files']:,}")
    print(f"Chunks:     {stats['total_chunks']:,}")
    print(f"  Functions: {stats['functions']:,}")
    print(f"  Methods:   {stats['methods']:,}")
    print(f"  Classes:   {stats['classes']:,}")
    print(f"\nTiming:")
    print(f"  Discovery:  {stats['discovery_ms']:.0f}ms")
    print(f"  Parsing:    {stats['parsing_ms']:.0f}ms")
    print(f"  Call graph: {stats['backfill_ms']:.0f}ms")
    print(f"  Total:      {stats['total_ms']:.0f}ms")
    
    # Test query
    print(f"\n{'='*60}")
    print("QUERY TEST: 'IRCEngine'")
    print(f"{'='*60}")
    
    chunks = await graph.query("IRCEngine", max_results=5)
    
    for i, chunk in enumerate(chunks):
        print(f"\n{i+1}. {chunk.chunk_type.value}: {chunk.name}")
        print(f"   File: {Path(chunk.source_path).name}:{chunk.start_line}")
        print(f"   Calls: {chunk.calls[:5]}{'...' if len(chunk.calls) > 5 else ''}")
        print(f"   Called by: {chunk.called_by[:5]}{'...' if len(chunk.called_by) > 5 else ''}")
    
    # Show context for first result
    if chunks:
        print(f"\n{'='*60}")
        print(f"FULL CONTEXT FOR: {chunks[0].name}")
        print(f"{'='*60}")
        
        ctx = graph.get_context_for_chunk(chunks[0].id)
        print(f"\nCallers ({len(ctx['callers'])}):")
        for c in ctx["callers"][:3]:
            print(f"  - {c.name} ({c.chunk_type.value})")
        
        print(f"\nCallees ({len(ctx['callees'])}):")
        for c in ctx["callees"][:3]:
            print(f"  - {c.name} ({c.chunk_type.value})")


# ============================================================================
# SINGLETON + PERSISTENT INDEX
# ============================================================================

_codegraph_instance: Optional["CodeGraph"] = None
_codegraph_lock = threading.Lock()
_codegraph_indexing = False  # Guard against concurrent index attempts


def get_codegraph(
    root_path: Optional[str] = None,
    auto_index: bool = True,
    max_workers: int = 8,
) -> "CodeGraph":
    """
    Get or create the CodeGraph singleton.

    First call indexes the codebase and loads embeddings.
    Subsequent calls return the same instance with warm cache.

    Thread-safe: concurrent callers during indexing get the instance
    immediately (possibly with empty chunks) rather than blocking.

    Args:
        root_path: Codebase root to index. Default: AitherOS root.
                   Can be ANY project path for onboarding external codebases.
        auto_index: If True, index + load embeddings on first call.
        max_workers: Number of threads/processes for indexing.
    """
    global _codegraph_instance, _codegraph_indexing

    if os.getenv("AITHER_TESTING") == "1":
        auto_index = False

    if _codegraph_instance is not None and _codegraph_instance.chunks:
        return _codegraph_instance

    with _codegraph_lock:
        if _codegraph_instance is not None and _codegraph_instance.chunks:
            return _codegraph_instance

        # Only create if not exists - preserve instance if warming up (empty chunks)
        if _codegraph_instance is None:
            cg = CodeGraph(max_workers=max_workers)
            _codegraph_instance = cg
        else:
            cg = _codegraph_instance

        # If another thread/task is already indexing, return immediately
        # to avoid stacking concurrent index operations
        if _codegraph_indexing:
            logger.debug("[CodeGraph] Index already in progress — returning instance as-is")
            return cg

        if auto_index:
            _codegraph_indexing = True
            try:
                if root_path is None:
                    root_path = os.environ.get(
                        "AITHEROS_ROOT", str(Path(__file__).parent.parent.parent)
                    )

                # Try loading persistent chunk cache first
                cache_loaded = _load_chunk_cache(cg, root_path)

                if not cache_loaded:
                    # Full index required
                    # Detect if we're inside an already-running event loop (e.g. uvicorn)
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        running_loop = None

                    if running_loop is not None:
                        # We're inside an async context — schedule indexing as a background task
                        # Return the un-indexed instance; callers handle empty .chunks gracefully
                        logger.info("[CodeGraph] Async context detected — scheduling background index")

                        async def _bg_index(cg_ref, rp):
                            global _codegraph_indexing
                            try:
                                await cg_ref.index_codebase(rp)
                                _save_chunk_cache(cg_ref, rp)
                                logger.info(f"[CodeGraph] Background index complete: {len(cg_ref.chunks)} chunks")
                            except Exception as exc:
                                logger.warning(f"[CodeGraph] Background index failed: {exc}")
                            finally:
                                _codegraph_indexing = False

                        running_loop.create_task(_bg_index(cg, root_path))
                    else:
                        # Sync context — safe to create a new loop
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(cg.index_codebase(root_path))
                            _save_chunk_cache(cg, root_path)
                        finally:
                            loop.close()
                            _codegraph_indexing = False
                else:
                    _codegraph_indexing = False
                    # Cache loaded — heal it against the current filesystem
                    # (deleted files evicted, new/changed files reindexed).
                    _schedule_reconcile(cg, root_path)
            except Exception:
                _codegraph_indexing = False
                raise

            # Load embedding cache. Apply-and-release: the pickle held ~101K
            # entries (~2.5GB of boxed float lists once unpickled) while only
            # ~37K matched live chunks — iterate popping each entry, keep the
            # matches as compact float32 rows, and drop the orphans (chunk ids
            # from prior layouts, which embed_chunks prunes but boot never did)
            # so the transient dict shrinks as the loop runs instead of
            # surviving it whole (memory audit 2026-07: this load was the
            # largest single spike in the aithergraph container's 5.3GB heap).
            embed_path = _get_data_path(root_path, "codegraph_embeddings.pkl")
            if os.path.exists(embed_path):
                if not _verify_pickle_hmac(embed_path):
                    logger.warning("[CodeGraph] Embedding cache HMAC invalid — skipping")
                else:
                    try:
                        with open(embed_path, "rb") as f:
                            cached = pickle.load(f)
                        applied = 0
                        orphaned = 0
                        for cid in list(cached.keys()):
                            emb = cached.pop(cid)
                            chunk = cg.chunks.get(cid)
                            if chunk is None:
                                orphaned += 1
                                continue
                            if emb is not None:
                                chunk.embedding = _as_f32(emb)
                                applied += 1
                        del cached
                        if orphaned:
                            logger.info(
                                f"[CodeGraph] Skipped {orphaned} orphaned embedding "
                                f"cache entries at boot (chunk ids no longer in the index)"
                            )
                        logger.info(f"[CodeGraph] Loaded {applied} embeddings from cache")
                    except Exception as e:
                        logger.warning(f"[CodeGraph] Embedding cache load failed: {e}")

        return cg


def _get_data_path(root_path: str, filename: str) -> str:
    """Get path in the Library/Data/codegraph dir, using Paths module for Docker awareness."""
    try:
        from paths import Paths
        data_dir = str(Paths.DATA / "codegraph")
    except Exception:
        data_dir = os.path.join(root_path, "Library", "Data", "codegraph")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)


def _save_chunk_cache(cg: "CodeGraph", root_path: str):
    """Persist parsed chunks + file mtimes for instant reload."""
    cache = {
        "chunks": {},
        "mtimes": {},
        "stats": {
            "total_files": cg.total_files,
            "total_chunks": len(cg.chunks),
        },
    }
    # Persist the stable-id manager so reindex preserves ids across restarts.
    _mgr = getattr(cg, "_id_manager", None)
    if _mgr is not None:
        try:
            cache["id_manager"] = _mgr.to_dict()
        except Exception:
            pass
    for cid, chunk in cg.chunks.items():
        cache["chunks"][cid] = {
            "id": chunk.id,
            "name": chunk.name,
            "chunk_type": chunk.chunk_type.value,
            "source_path": chunk.source_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "signature": chunk.signature,
            "docstring": chunk.docstring,
            "body_preview": chunk.body_preview,
            "calls": chunk.calls,
            "called_by": chunk.called_by,
            "parent_class": chunk.parent_class,
            "base_classes": chunk.base_classes,
            "imports": chunk.imports,
            "tenant_id": chunk.tenant_id,
            "workspace_id": chunk.workspace_id,
            "stable_id": chunk.stable_id,
        }

    # Record file mtimes for incremental detection
    for fpath in cg.by_file:
        try:
            full_path = os.path.join(root_path, fpath) if not os.path.isabs(fpath) else fpath
            if os.path.exists(full_path):
                cache["mtimes"][fpath] = os.path.getmtime(full_path)
        except OSError as e:
            logger.debug(f"[CodeGraph._save_chunk_cache] Operation failed: {e}")

    cache_path = _get_data_path(root_path, "codegraph_chunks.pkl")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        _write_pickle_hmac(cache_path)
        logger.info(f"[CodeGraph] Saved {len(cache['chunks'])} chunks to cache")
    except Exception as e:
        logger.warning(f"[CodeGraph] Failed to save chunk cache: {e}")


def _load_chunk_cache(cg: "CodeGraph", root_path: str) -> bool:
    """Load persisted chunks from disk. Returns True if cache was valid."""
    cache_path = _get_data_path(root_path, "codegraph_chunks.pkl")
    if not os.path.exists(cache_path):
        return False

    if not _verify_pickle_hmac(cache_path):
        logger.warning("[CodeGraph] Chunk cache HMAC invalid — treating as missing")
        return False

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except Exception as e:
        logger.warning(f"[CodeGraph] Chunk cache corrupted: {e}")
        return False

    chunks_data = cache.get("chunks", {})
    if not chunks_data:
        return False

    # Restore the persisted stable-id manager (if any) BEFORE finalize so a
    # reindex/migration reuses existing (name, path) → id mappings.
    if cache.get("id_manager"):
        try:
            from lib.faculties.StableNodeIDLayer import StableNodeIDManager
            cg._id_manager = StableNodeIDManager.from_dict(cache["id_manager"])
        except Exception:
            pass

    # Reconstruct CodeChunk objects
    for cid, data in chunks_data.items():
        try:
            chunk = CodeChunk(
                id=data["id"],
                name=data["name"],
                chunk_type=ChunkType(data["chunk_type"]),
                source_path=data["source_path"],
                start_line=data["start_line"],
                end_line=data["end_line"],
                signature=data["signature"],
                docstring=data.get("docstring"),
                body_preview=data.get("body_preview", data.get("body", "")),
                calls=data.get("calls", []),
                called_by=data.get("called_by", []),
                parent_class=data.get("parent_class"),
                base_classes=data.get("base_classes", []),
                imports=data.get("imports", []),
            )
            chunk.tenant_id = data.get("tenant_id", "platform")
            chunk.workspace_id = data.get("workspace_id", "")
            chunk.stable_id = data.get("stable_id")
            cg.chunks[cid] = chunk
            cg._tenant_chunks[chunk.tenant_id].add(cid)
            cg.by_name[chunk.name].append(cid)
            cg.by_file[chunk.source_path].append(cid)
            if chunk.parent_class:
                cg.by_class[chunk.parent_class].append(cid)
        except Exception:
            continue

    cg.total_files = cache.get("stats", {}).get("total_files", len(cg.by_file))

    # v2: migrate a v1 cache to rename-safe stable ids (backup + embeddings
    # remap + HMAC regen).  Failure stays in-memory; never loads an unverified
    # pickle.  No-op when AITHER_CODEGRAPH_ID_VERSION != 2.
    if _codegraph_id_v2():
        _migrate_loaded_cache_to_v2(cg, root_path, cache_path)

    # Stash the persisted per-file mtimes — reconcile_index() diffs them
    # against the filesystem. (They were always SAVED "for incremental
    # detection" but nothing ever read them back, so a warm cache was
    # trusted forever: deleted files never evicted, off-process changes
    # never reindexed.)
    cg._cached_mtimes = cache.get("mtimes", {}) or {}

    logger.info(
        f"[CodeGraph] Loaded {len(cg.chunks)} chunks from cache "
        f"({cg.total_files} files)"
    )
    return True


def _remap_embeddings_cache(root_path: str, old_to_new: Dict[str, str]) -> None:
    """Re-key the HMAC-verified embeddings pickle from old ids to new ids."""
    emb_path = _get_data_path(root_path, "codegraph_embeddings.pkl")
    if not os.path.exists(emb_path):
        return
    if not _verify_pickle_hmac(emb_path):
        logger.warning("[CodeGraph] embeddings cache HMAC invalid — skipping remap")
        return
    try:
        with open(emb_path, "rb") as f:
            cached = pickle.load(f)
        remapped = {old_to_new.get(k, k): v for k, v in cached.items()}
        with open(emb_path, "wb") as f:
            pickle.dump(remapped, f, protocol=pickle.HIGHEST_PROTOCOL)
        _write_pickle_hmac(emb_path)
    except Exception as e:
        logger.warning("[CodeGraph] embeddings remap failed: %s", e)


def _migrate_loaded_cache_to_v2(cg: "CodeGraph", root_path: str, cache_path: str) -> None:
    """One-time v1→v2 migration of a just-loaded cache: finalize stable ids,
    back up the v1 pickle, remap embeddings, and re-persist + regen HMAC."""
    try:
        old_to_new = _finalize_stable_ids(cg)
        if not old_to_new:
            return
        import shutil
        for suffix in ("", ".hmac"):
            src = cache_path + suffix
            if os.path.exists(src):
                try:
                    shutil.copy2(src, cache_path + ".v1.bak" + suffix)
                except Exception:
                    pass
        _remap_embeddings_cache(root_path, old_to_new)
        _save_chunk_cache(cg, root_path)
        logger.info(
            "[CodeGraph] migrated %d chunks to stable ids (v2); v1 backed up to .v1.bak",
            len(old_to_new),
        )
    except Exception as e:
        logger.warning("[CodeGraph] v2 migration failed (in-memory unaffected): %s", e)


async def reindex_files(changed_files: List[str], root_path: Optional[str] = None, tenant_id: str = "platform", persist: bool = True):
    """
    Incrementally re-index changed files in the live singleton.

    This is the correct way to update the index when files change.
    Uses parse_file_sync() (the same parser as full index), then
    re-embeds only the changed chunks.

    ``persist=False`` skips the per-call full-cache write — for batched callers
    (the reconcile loop) that re-index hundreds of files and save ONCE at the
    end. Saving the entire growing cache after every 50-file batch was a disk
    storm (~40 full-cache writes per reconcile pass) that starved the event loop
    after every genesis recreate.
    """
    # auto_index=False: if the live index is mid-repair (chunks temporarily
    # empty), auto_index would reload the full cache and re-fire the
    # reconcile scheduler — the feedback loop that thrashed 8154.
    cg = get_codegraph(root_path=root_path, auto_index=False)

    if root_path is None:
        root_path = str(Path(__file__).parent.parent.parent)

    py_files = [f for f in changed_files if f.endswith(".py")]
    if not py_files:
        return 0

    reindexed = 0
    new_chunk_ids = []

    for filepath in py_files[:50]:
        try:
            # Remove old chunks for this file
            rel_path = filepath
            if os.path.isabs(filepath):
                try:
                    rel_path = os.path.relpath(filepath, root_path)
                except ValueError:
                    rel_path = filepath

            old_ids = cg.by_file.get(rel_path, [])
            for oid in old_ids:
                cg.chunks.pop(oid, None)
                for name_list in cg.by_name.values():
                    if oid in name_list:
                        name_list.remove(oid)
            cg.by_file[rel_path] = []

            # Re-parse the file
            abs_path = filepath if os.path.isabs(filepath) else os.path.join(root_path, filepath)
            if not os.path.exists(abs_path):
                continue

            # read_text + ast.parse + visit is CPU/IO-heavy; the incremental
            # reindex ran it inline on the event loop for all 50 files per batch
            # (the watchdog caught this as the dominant residual stall). Offload
            # the parse; the fast index mutations below stay on-loop so shared
            # cg state isn't raced by a worker thread. The await also yields
            # between files, keeping uvicorn responsive.
            file_graph = await asyncio.to_thread(parse_file_sync, abs_path)
            for chunk in file_graph.chunks:
                cg.chunks[chunk.id] = chunk
                cg.by_name[chunk.name].append(chunk.id)
                cg.by_file[file_graph.source_path].append(chunk.id)
                if chunk.parent_class:
                    cg.by_class[chunk.parent_class].append(chunk.id)
                if chunk.route_path is not None:
                    route_key = f"{chunk.route_method or 'GET'} {chunk.route_path}"
                    cg.routes[route_key] = chunk.id
                new_chunk_ids.append(chunk.id)
                # Sync to AitherKnowledgeGraph
                cg._queue_sync({
                    "id": chunk.id,
                    "name": chunk.name,
                    "type": chunk.chunk_type.value,
                    "properties": {
                        "source_path": chunk.source_path,
                        "signature": chunk.signature,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                }, tenant_id=tenant_id)

            reindexed += 1
        except Exception as e:
            logger.debug(f"[CodeGraph] Failed to re-index {filepath}: {e}")

    # Backfill call graph for new chunks. This walks ~all chunks resolving
    # call edges (O(N·edges), 1-5s on a 28K-chunk graph) — offload it so the
    # every-reindex backfill can't stall the event loop / uvicorn. It only
    # reads chunks and appends to called_by lists (GIL-atomic), so a thread is
    # safe here.
    if new_chunk_ids:
        await asyncio.to_thread(cg._backfill_called_by)
        # v2: preserve existing stable ids for unchanged symbols, mint for new
        # ones — reindex no longer re-orphans edges (no-op on v1). Kept on-loop:
        # it reassigns cg.chunks/by_file (see _finalize_stable_ids) so offloading
        # would race concurrent queries; its steady-state path is a cheap no-op.
        # The one-time v1->v2 migration offload is handled in Stage 1's reconcile
        # rework (with snapshot locking).
        old_to_new = _finalize_stable_ids(cg)
        # migrate_chunks re-keys cg.chunks; without this remap the embed block
        # below would miss the migrated chunks (cid not in cg.chunks) and
        # silently skip their embeddings.
        if old_to_new:
            new_chunk_ids = [old_to_new.get(cid, cid) for cid in new_chunk_ids]

    # Re-embed new chunks (incremental — only the changed ones)
    if new_chunk_ids and cg.embedding_coverage > 0:
        need_embed = [
            (cid, cg.chunks[cid]) for cid in new_chunk_ids
            if cid in cg.chunks and cg.chunks[cid].embedding is None
        ]
        if need_embed:
            texts = [c.signature + "\n" + (c.docstring or "") + "\n" + (c.body_preview or "")
                     for _, c in need_embed]
            embeddings = await _embed_texts(texts)
            for (cid, chunk), emb in zip(need_embed, embeddings):
                if emb:
                    chunk.embedding = _as_f32(emb)  # compact float32 row

    # Persist updated cache (batched reconcile callers pass persist=False and
    # save ONCE at the end — see the docstring; per-batch full saves were a storm).
    if persist:
        _save_chunk_cache(cg, root_path)

    if reindexed > 0:
        cg._invalidate_embedding_matrix()  # Force matrix rebuild on next query
        cg._invalidate_keyword_cache()  # Clear keyword result cache
        logger.info(f"[CodeGraph] Incrementally re-indexed {reindexed} files "
                     f"({len(new_chunk_ids)} chunks)")

    return reindexed


def _evict_file_chunks(cg: "CodeGraph", file_key: str) -> int:
    """Remove every chunk indexed under by_file[file_key], cleaning the
    secondary indices. file_key must be the EXACT by_file key (reindex_files
    evicts by relpath, which misses on absolute-keyed indexes and duplicates
    chunks — evict here first)."""
    ids = cg.by_file.pop(file_key, [])
    for cid in ids:
        chunk = cg.chunks.pop(cid, None)
        if not chunk:
            continue
        lst = cg.by_name.get(chunk.name)
        if lst and cid in lst:
            lst.remove(cid)
        if chunk.parent_class:
            lst = cg.by_class.get(chunk.parent_class)
            if lst and cid in lst:
                lst.remove(cid)
    return len(ids)


# Reentrancy guard: reconcile triggers reindex_files, whose internal
# get_codegraph cache load must NOT schedule another reconcile — without this
# an emptied index re-loads the cache and re-fires reconcile in a tight loop.
_reconcile_running = False


async def reconcile_index(
    root_path: Optional[str] = None,
    batch: int = 50,
    max_files: int = 2000,
    cg: Optional["CodeGraph"] = None,
    force: bool = False,
) -> Dict[str, int]:
    """Diff the live index against the filesystem and incrementally repair it.

    Evicts chunks of deleted files and re-indexes new/changed files (mtime vs
    the cache's persisted mtimes). This is the read side of the mtime data the
    chunk cache always persisted; without it a warm cache was trusted forever.
    Runs automatically in the background after a cache load (gate with
    AITHER_CODEGRAPH_RECONCILE=0) and from the scheduler's idle task.

    SAFETY: aborts (unless force=True) when the apparent deletion set is a
    large fraction of the index — that signature means file DISCOVERY was
    incomplete (fd/rg missing → rglob's 2000-file cap), not that the files
    are gone. Trusting it once mass-evicted a 67k-chunk index.
    """
    global _reconcile_running
    if _reconcile_running:
        return {"skipped": 1}
    _reconcile_running = True
    try:
        return await _reconcile_index_inner(root_path, batch, max_files, cg, force)
    finally:
        _reconcile_running = False


async def _reconcile_index_inner(
    root_path: Optional[str],
    batch: int,
    max_files: int,
    cg: Optional["CodeGraph"],
    force: bool,
) -> Dict[str, int]:
    if root_path is None:
        root_path = os.environ.get(
            "AITHEROS_ROOT", str(Path(__file__).parent.parent.parent)
        )
    if cg is None:
        cg = get_codegraph(root_path=root_path, auto_index=False)
    # The reconcile must diff the filesystem against the PERSISTED index, not an
    # empty in-memory one. In the worker the CodeGraph singleton is never warmed
    # by queries (those hit the CodeGraph service :8153), so without this the
    # in-memory index is empty → every one of the ~7.6k repo files looks "new"
    # → reconcile re-indexes 50 files EVERY cycle forever, ignoring the ~34k-chunk
    # cache already on disk (the churn that pegged the worker). Load it once when
    # empty so the diff converges (backfill shrinks to zero) instead of looping.
    if not cg.by_file:
        try:
            # Offload the pickle load (~34k chunks) off the event loop.
            loaded = await asyncio.to_thread(_load_chunk_cache, cg, root_path)
            if loaded:
                logger.info(
                    f"[CodeGraph] reconcile: loaded persisted index "
                    f"({len(cg.by_file)} files, {len(cg.chunks)} chunks) before diff"
                )
        except Exception as e:
            logger.warning(f"[CodeGraph] reconcile: chunk cache load failed: {e}")
    cached_mtimes: Dict[str, float] = getattr(cg, "_cached_mtimes", {}) or {}

    disk_paths, _ms = await discover_python_files(Path(root_path))

    def _stat_mtimes(paths) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p in paths:
            try:
                out[str(p)] = os.path.getmtime(p)
            except OSError:
                continue
        return out

    # os.path.getmtime() per file, over ~7.6k repo files, is a tight
    # synchronous loop with no yield point. On this repo's Windows/WSL2
    # bind-mounted filesystem each stat syscall is slow enough that the
    # whole scan blocked the event loop for 5-10s straight (LOOP-WATCHDOG,
    # CodeGraph.py:3778). Offload the whole scan to a thread.
    disk: Dict[str, float] = await asyncio.to_thread(_stat_mtimes, disk_paths)

    indexed = set(cg.by_file.keys())
    disk_set = set(disk)
    deleted = indexed - disk_set
    new = disk_set - indexed

    # View-aware deletions: this cache lives on the bind-mounted Library
    # volume, SHARED between processes with different filesystem views — the
    # host venv sees the whole repo, while service containers only COPY
    # services/ + lib/ (Dockerfile.Services). A file whose top-level directory
    # doesn't even exist in this process's view (e.g. dev/ inside aithergraph)
    # wasn't deleted — it's invisible from here. Evicting it would drop valid
    # chunks a full-view process indexed and set up cache ping-pong. Leave
    # those to a process that can actually see that subtree.
    # Signal: the file's PARENT directory is absent too. A genuinely deleted
    # file normally leaves its package directory behind, so parent-exists +
    # file-missing → evict. Parent missing → the whole subtree is invisible
    # here (not COPYed into this image) → skip. Trade-off: a real whole-
    # directory deletion is only evicted by a full-view process (or force) —
    # acceptable, since the >10% valve below still backstops mass eviction.
    _out_of_view = set()
    _parent_seen: Dict[str, bool] = {}
    for _f in deleted:
        _parent = os.path.dirname(_f)
        _ok = _parent_seen.get(_parent)
        if _ok is None:
            _ok = os.path.isdir(_parent)
            _parent_seen[_parent] = _ok
        if not _ok:
            _out_of_view.add(_f)
    if _out_of_view:
        deleted -= _out_of_view
        logger.info(
            f"[CodeGraph] reconcile: ignoring {len(_out_of_view)} indexed files "
            f"outside this process's view (parent dir absent — shared cache, "
            f"partial container image)"
        )
    changed = {
        f for f in indexed & disk_set
        if disk[f] > cached_mtimes.get(f, 0.0) + 1e-6
    }

    if not disk_set or (
        not force and len(deleted) > max(25, len(indexed) // 10)
    ):
        logger.warning(
            f"[CodeGraph] reconcile ABORTED: {len(deleted)}/{len(indexed)} indexed "
            f"files look deleted but discovery returned only {len(disk_set)} files "
            f"— discovery is likely incomplete (install fd-find/ripgrep, or pass "
            f"force=True if the deletions are real)"
        )
        return {"aborted": 1, "deleted": len(deleted), "discovered": len(disk_set)}

    evicted = 0
    for f in deleted | changed:
        evicted += _evict_file_chunks(cg, f)

    to_reindex = sorted(new | changed)
    deferred = max(0, len(to_reindex) - max_files)
    if deferred:
        logger.warning(
            f"[CodeGraph] reconcile capped at {max_files} files — "
            f"{deferred} deferred to the next pass"
        )
        to_reindex = to_reindex[:max_files]

    reindexed = 0
    # Throttle between batches so the reconcile (AST parse + the single end-save)
    # releases the GIL and never starves the main event loop — the [LAG] stalls
    # that made every chat turn slow after a genesis recreate. persist=False:
    # save the full cache ONCE below, not after each 50-file batch.
    try:
        _throttle = float(os.environ.get("AITHER_CODEGRAPH_RECONCILE_THROTTLE_S", "0.15"))
    except (TypeError, ValueError):
        _throttle = 0.15
    for i in range(0, len(to_reindex), batch):
        reindexed += await reindex_files(to_reindex[i:i + batch], root_path, persist=False)
        if _throttle > 0:
            await asyncio.sleep(_throttle)

    if evicted or reindexed:
        if hasattr(cg, "_invalidate_embedding_matrix"):
            cg._invalidate_embedding_matrix()
        if hasattr(cg, "_invalidate_keyword_cache"):
            cg._invalidate_keyword_cache()
        _save_chunk_cache(cg, root_path)
        cg._cached_mtimes = {f: disk.get(f, 0.0) for f in cg.by_file}

    return {
        "deleted": len(deleted), "new": len(new), "changed": len(changed),
        "evicted": evicted, "reindexed": reindexed, "deferred": deferred,
    }


def _schedule_reconcile(cg: "CodeGraph", root_path: str) -> None:
    """Run reconcile_index in a background daemon thread (post cache load)."""
    if os.getenv("AITHER_TESTING") == "1":
        return
    if os.environ.get("AITHER_CODEGRAPH_RECONCILE", "1").lower() in ("0", "false", "off"):
        return

    if _reconcile_running:
        return

    def _run() -> None:
        try:
            result = asyncio.run(reconcile_index(root_path))
            if any(result.values()):
                logger.info(f"[CodeGraph] cache reconcile: {result}")
        except Exception as e:
            logger.warning(f"[CodeGraph] cache reconcile failed: {e!r}")

    threading.Thread(target=_run, daemon=True, name="codegraph-reconcile").start()


if __name__ == "__main__":
    asyncio.run(main())
