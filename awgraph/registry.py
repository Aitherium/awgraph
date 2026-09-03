"""
CodeGraphRegistry — Multi-Root CodeGraph Manager with Tenant Isolation
========================================================================

Wraps multiple CodeGraph instances so AitherOS can index external repos.
Each root gets an isolated cache directory.  ``query_all()`` searches
across all roots and merges results.

Tenant Isolation:
    - The default AitherOS root is owned by "platform"
    - External roots are tagged with a tenant_id
    - ``query_for_tenant()`` returns results only from roots owned by that tenant
    - ``query_all()`` still searches everything (platform-only usage)
    - PLATFORM callers see all roots; TENANT callers see only their own

Does NOT replace ``get_codegraph()`` — wraps it.  The default AitherOS
root reuses the existing singleton.

Usage:
    registry = get_codegraph_registry()
    registry.register_root("/path/to/repo", label="my-lib", tenant_id="tnt_abc123")
    results = await registry.query_for_tenant("tnt_abc123", "circuit breaker")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from awgraph.logging import get_logger

logger = get_logger("CodeGraphRegistry")

# Cache base — use standard ~/.awgraph or /tmp
try:
    from pathlib import Path
    _CACHE_BASE = Path.home() / ".awgraph" / "cache"
except Exception:
    _CACHE_BASE = Path("/tmp/awgraph/cache")

def _aither_verify():
    """SSL verification — use True for public package."""
    return True


_CONFIG_PATH = None  # No config file in public package
_MAX_ROOTS = 10
_MAX_ROOTS_PER_TENANT = 5

PLATFORM_TENANT = "platform"

# Strata is not available in public package — registry is in-memory only
_STRATA_ROOTS_KEY = None


# =============================================================================
# REGISTRY
# =============================================================================

class CodeGraphRegistry:
    """Manages multiple CodeGraph instances, one per repository root.

    Each root is tagged with a tenant_id for access control.
    The default AitherOS root is owned by "platform".
    """

    def __init__(self):
        self._instances: Dict[str, Any] = {}  # normalized_path -> CodeGraph
        self._labels: Dict[str, str] = {}     # label -> normalized_path
        self._owners: Dict[str, str] = {}     # normalized_path -> tenant_id
        self._default_root: Optional[str] = None
        self._init_default()
        self._load_persisted_roots()

    def _init_default(self):
        """Set the default root to the AitherOS repo root."""
        try:
            root = Path(__file__).parent.parent.parent.resolve()
            self._default_root = str(root)
            self._labels["default"] = str(root)
            self._owners[str(root)] = PLATFORM_TENANT
        except Exception as exc:
            logger.debug(f"Default root path setup failed: {exc}")

    def _normalize_path(self, root_path: str) -> str:
        """Normalize a path for use as a registry key.

        Handles:
        - aither:// virtual paths (kept as-is, no filesystem resolve)
        - Absolute local paths (resolved via Path.resolve())
        - Relative paths (resolved relative to cwd)
        """
        if root_path.startswith("aither://"):
            # Virtual Strata path — don't try to resolve on local filesystem
            return root_path
        return str(Path(root_path).resolve())

    def _is_accessible_root(self, path: str) -> bool:
        """Check if a root path is accessible.

        In distributed container setups, source code is:
        - Baked into container images (e.g. /app/AitherOS)
        - Accessible via aither:// Strata paths
        - Mounted as Docker volumes

        We accept the path if:
        1. It's a local directory that exists (classic single-machine)
        2. It's an aither:// virtual path (always accepted, Strata resolves it)
        3. The default root (/app in container) exists
        """
        if path.startswith("aither://"):
            return True  # Virtual paths are resolved by Strata at read time
        return Path(path).is_dir()

    def _cache_dir_for(self, root_path: str, tenant_id: str = PLATFORM_TENANT) -> Path:
        """Get isolated cache directory for a root, namespaced by tenant."""
        norm = self._normalize_path(root_path)
        if norm == self._default_root:
            return _CACHE_BASE / "default"
        # Tenant-isolated: cache_base / tenant_id / repo_hash
        hash_prefix = hashlib.sha256(norm.encode()).hexdigest()[:16]
        return _CACHE_BASE / tenant_id / hash_prefix

    # --------------------------------------------------------------------- #
    # REGISTRATION
    # --------------------------------------------------------------------- #

    def register_root(
        self,
        root_path: str,
        label: Optional[str] = None,
        auto_index: bool = False,
        tenant_id: str = PLATFORM_TENANT,
    ) -> Any:
        """Register a new root and create/return its CodeGraph instance.

        Args:
            root_path: Path to the repository root. Accepts:
                       - Local absolute paths (e.g. /app/AitherOS)
                       - aither:// virtual paths (resolved via Strata)
                       - Container-relative paths
            label: Optional human-friendly label (e.g., "my-lib")
            auto_index: If True, index the root immediately
            tenant_id: Tenant that owns this root (default: "platform")

        Returns:
            CodeGraph instance for this root
        """
        norm = self._normalize_path(root_path)

        if not self._is_accessible_root(norm):
            raise ValueError(f"Root path does not exist or is not a directory: {root_path}")

        if len(self._instances) >= _MAX_ROOTS and norm not in self._instances:
            raise ValueError(f"Maximum number of roots ({_MAX_ROOTS}) reached")

        # Enforce per-tenant limit for non-platform tenants
        if tenant_id != PLATFORM_TENANT and norm not in self._instances:
            tenant_count = sum(
                1 for t in self._owners.values() if t == tenant_id
            )
            if tenant_count >= _MAX_ROOTS_PER_TENANT:
                raise ValueError(
                    f"Tenant '{tenant_id}' has reached the maximum "
                    f"of {_MAX_ROOTS_PER_TENANT} roots"
                )

        # Return existing instance if already registered
        if norm in self._instances:
            if label and label not in self._labels:
                self._labels[label] = norm
            return self._instances[norm]

        # For default root, reuse the existing singleton
        if norm == self._default_root:
            try:
                # awgraph's OWN CodeGraph. This read `lib.faculties.CodeGraph`,
                # which is the monorepo module this package was extracted FROM --
                # a leftover from before awgraph existed, and unreachable once
                # installed. `awgraph.graph` exports both names.
                from awgraph.graph import get_codegraph
                cg = get_codegraph(auto_index=auto_index)
                self._instances[norm] = cg
                self._owners[norm] = PLATFORM_TENANT  # Always platform
                if label:
                    self._labels[label] = norm
                logger.info(f"Registered default root: {norm}")
                self._persist_roots()
                return cg
            except Exception as e:
                logger.warning(f"Failed to get default CodeGraph: {e}")
                return None

        # External root: create a new CodeGraph instance with isolated cache
        try:
            from awgraph.graph import CodeGraph
            cache_dir = self._cache_dir_for(norm, tenant_id)
            cache_dir.mkdir(parents=True, exist_ok=True)

            cg = CodeGraph(
                root_path=norm,
                cache_dir=str(cache_dir),
                auto_index=auto_index,
            )
            self._instances[norm] = cg
            self._owners[norm] = tenant_id
            if label:
                self._labels[label] = norm
            logger.info(
                f"Registered external root: {norm} "
                f"(label={label}, tenant={tenant_id}, cache={cache_dir})"
            )
            self._persist_roots()
            return cg
        except Exception as e:
            logger.error(f"Failed to create CodeGraph for {norm}: {e}")
            raise

    def unregister_root(self, root_path: str, tenant_id: Optional[str] = None) -> bool:
        """Unregister a root. Cannot unregister the default root.

        Args:
            root_path: Path to unregister
            tenant_id: If provided, only allow unregister if tenant matches

        Returns:
            True if successfully unregistered
        """
        norm = self._normalize_path(root_path)
        if norm == self._default_root:
            logger.warning("Cannot unregister default root")
            return False
        if norm not in self._instances:
            return False

        # Enforce tenant ownership
        if tenant_id and self._owners.get(norm) != tenant_id:
            logger.warning(
                f"Tenant '{tenant_id}' cannot unregister root owned by "
                f"'{self._owners.get(norm)}'"
            )
            return False

        del self._instances[norm]
        self._owners.pop(norm, None)
        # Remove label mappings
        for lbl, path in list(self._labels.items()):
            if path == norm:
                del self._labels[lbl]

        logger.info(f"Unregistered root: {norm}")
        self._persist_roots()
        return True

    # --------------------------------------------------------------------- #
    # ACCESSORS
    # --------------------------------------------------------------------- #

    def get(self, root_path: str) -> Optional[Any]:
        """Get CodeGraph instance by root path."""
        norm = self._normalize_path(root_path)
        return self._instances.get(norm)

    def get_by_label(self, label: str) -> Optional[Any]:
        """Get CodeGraph instance by label."""
        norm = self._labels.get(label)
        if norm:
            return self._instances.get(norm)
        return None

    def get_default(self) -> Optional[Any]:
        """Get the default (AitherOS) CodeGraph instance."""
        if self._default_root:
            return self._instances.get(self._default_root)
        return None

    def get_owner(self, root_path: str) -> str:
        """Get the tenant that owns a root."""
        norm = self._normalize_path(root_path)
        return self._owners.get(norm, PLATFORM_TENANT)

    def get_tenant_roots(self, tenant_id: str) -> List[str]:
        """Get all root paths owned by a tenant."""
        return [
            path for path, owner in self._owners.items()
            if owner == tenant_id
        ]

    # --------------------------------------------------------------------- #
    # TENANT-SCOPED QUERY
    # --------------------------------------------------------------------- #

    async def query_for_tenant(
        self,
        tenant_id: str,
        query: str,
        max_results_per_root: int = 5,
    ) -> List[Tuple[str, List[Any]]]:
        """Search only roots owned by a specific tenant.

        This is the method TENANT callers should use. It never returns
        results from roots owned by other tenants or platform.

        Args:
            tenant_id: Tenant to scope results to
            query: Search query
            max_results_per_root: Max results per root

        Returns:
            List of (label_or_path, results) tuples
        """
        results = []
        path_to_label = {}
        for lbl, path in self._labels.items():
            path_to_label[path] = lbl

        for norm, cg in self._instances.items():
            # Only include roots owned by this tenant
            if self._owners.get(norm) != tenant_id:
                continue

            label = path_to_label.get(norm, norm)
            try:
                if hasattr(cg, 'hybrid_query'):
                    hits = await cg.hybrid_query(query, max_results=max_results_per_root)
                elif hasattr(cg, 'search'):
                    hits = cg.search(query, top_k=max_results_per_root)
                else:
                    hits = []
                if hits:
                    results.append((label, hits))
            except Exception as e:
                logger.debug(f"Query failed for root {label}: {e}")

        return results

    # --------------------------------------------------------------------- #
    # CROSS-ROOT QUERY (platform-only)
    # --------------------------------------------------------------------- #

    async def query_all(
        self,
        query: str,
        max_results_per_root: int = 5,
    ) -> List[Tuple[str, List[Any]]]:
        """Search across ALL registered roots (platform-only usage).

        Returns:
            List of (label_or_path, results) tuples
        """
        results = []
        path_to_label = {}
        for lbl, path in self._labels.items():
            path_to_label[path] = lbl

        for norm, cg in self._instances.items():
            label = path_to_label.get(norm, norm)
            try:
                if hasattr(cg, 'hybrid_query'):
                    hits = await cg.hybrid_query(query, max_results=max_results_per_root)
                elif hasattr(cg, 'search'):
                    hits = cg.search(query, top_k=max_results_per_root)
                else:
                    hits = []
                if hits:
                    results.append((label, hits))
            except Exception as e:
                logger.debug(f"Query failed for root {label}: {e}")

        return results

    # --------------------------------------------------------------------- #
    # STATUS
    # --------------------------------------------------------------------- #

    def list_roots(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered roots with metadata.

        Args:
            tenant_id: If provided, only list roots owned by this tenant.
                       If None, list all roots.
        """
        path_to_label = {}
        for lbl, path in self._labels.items():
            path_to_label.setdefault(path, []).append(lbl)

        roots = []
        for norm, cg in self._instances.items():
            owner = self._owners.get(norm, PLATFORM_TENANT)
            if tenant_id and owner != tenant_id:
                continue

            labels = path_to_label.get(norm, [])
            roots.append({
                "path": norm,
                "labels": labels,
                "is_default": norm == self._default_root,
                "tenant_id": owner,
                "chunks": len(getattr(cg, 'chunks', {})),
                "coverage": getattr(cg, 'embedding_coverage', 0.0),
                "cache_dir": str(self._cache_dir_for(norm, owner)),
            })
        return roots

    def get_status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get registry status, optionally filtered by tenant."""
        roots = self.list_roots(tenant_id=tenant_id)
        return {
            "total_roots": len(roots),
            "max_roots": _MAX_ROOTS,
            "max_roots_per_tenant": _MAX_ROOTS_PER_TENANT,
            "default_root": self._default_root,
            "labels": {
                lbl: path for lbl, path in self._labels.items()
                if not tenant_id or self._owners.get(path) == tenant_id
            },
            "roots": roots,
        }

    # --------------------------------------------------------------------- #
    # CONFIG LOADING
    # --------------------------------------------------------------------- #

    def load_config(self):
        """Load external roots from config/codegraph.yaml."""
        try:
            import yaml
            if not _CONFIG_PATH.exists():
                return
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            roots = config.get("roots", [])

            for root_entry in roots:
                if isinstance(root_entry, str):
                    path = root_entry
                    label = None
                    auto_index = False
                    tenant_id = PLATFORM_TENANT
                elif isinstance(root_entry, dict):
                    path = root_entry.get("path", "")
                    label = root_entry.get("label")
                    auto_index = root_entry.get("auto_index", False)
                    tenant_id = root_entry.get("tenant_id", PLATFORM_TENANT)
                else:
                    continue

                if path and self._is_accessible_root(path):
                    try:
                        self.register_root(
                            path, label=label,
                            auto_index=auto_index, tenant_id=tenant_id,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to register root from config: {path}: {e}")

            logger.info(f"Loaded {len(roots)} roots from codegraph.yaml")
        except Exception as e:
            logger.warning(f"Failed to load codegraph.yaml: {e}")

    # --------------------------------------------------------------------- #
    # SCOPE ROOTS — Load from scope_roots.yaml for full monorepo indexing
    # --------------------------------------------------------------------- #

    def load_scope_roots(self):
        """Load subproject roots from config/scope_roots.yaml.

        This is the preferred config for AitherScope — it defines every
        subproject in the monorepo for full-codebase visualization.
        """
        try:
            import yaml
            scope_config = Path(__file__).parent.parent.parent / "config" / "scope_roots.yaml"
            if not scope_config.exists():
                logger.debug("scope_roots.yaml not found, skipping")
                return

            with open(scope_config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            roots = config.get("roots", [])
            aitheros_root = Path(__file__).parent.parent.parent.resolve()
            registered = 0

            for entry in roots:
                if not isinstance(entry, dict):
                    continue
                rel_path = entry.get("path", "")
                label = entry.get("label", "")
                auto_index = entry.get("auto_index", False)

                # Resolve relative to AitherOS root
                full_path = (aitheros_root / rel_path).resolve()
                if not full_path.is_dir():
                    logger.debug(f"Scope root not found: {full_path}")
                    continue

                try:
                    self.register_root(
                        str(full_path),
                        label=label or rel_path,
                        auto_index=auto_index,
                        tenant_id=PLATFORM_TENANT,
                    )
                    registered += 1
                except Exception as e:
                    logger.debug(f"Failed to register scope root {rel_path}: {e}")

            logger.info(f"Loaded {registered}/{len(roots)} scope roots")
        except Exception as e:
            logger.warning(f"Failed to load scope_roots.yaml: {e}")

    async def index_all_roots(self, force: bool = False):
        """Index all registered roots (called on Genesis startup).

        Runs indexing in background tasks for each root.
        Only indexes roots that have auto_index=True or force=True.
        """
        tasks = []
        for norm, cg in self._instances.items():
            if cg is None:
                continue
            try:
                if hasattr(cg, 'index') and callable(cg.index):
                    chunks_before = len(getattr(cg, 'chunks', {}))
                    if force or chunks_before == 0:
                        tasks.append((norm, cg))
            except Exception as e:
                logger.debug(f"Cannot check index state for {norm}: {e}")

        if not tasks:
            logger.info("All roots already indexed or no roots registered")
            return

        logger.info(f"Indexing {len(tasks)} roots...")
        for norm, cg in tasks:
            try:
                label = next(
                    (l for l, p in self._labels.items() if p == norm),
                    norm,
                )
                cg.index()
                chunks = len(getattr(cg, 'chunks', {}))
                logger.info(f"  ✓ {label}: {chunks} chunks indexed")
            except Exception as e:
                logger.warning(f"  ✗ Index failed for {norm}: {e}")

    # --------------------------------------------------------------------- #
    # FULL GRAPH — Unified node/edge structure for AitherScope visualization
    # --------------------------------------------------------------------- #

    def get_full_graph(self) -> Dict[str, Any]:
        """Walk all registered roots and return a unified graph.

        Returns:
            {
                "nodes": [{id, label, type, path, group, lines, ...}],
                "edges": [{source, target, type, weight}],
                "subprojects": [str],
                "generated_at": str,
            }
        """
        from datetime import datetime
        nodes = []
        edges = []
        subprojects = []
        seen_ids: Set[str] = set()

        path_to_label = {}
        for lbl, path in self._labels.items():
            path_to_label.setdefault(path, lbl)

        for norm, cg in self._instances.items():
            if cg is None:
                continue

            label = path_to_label.get(norm, Path(norm).name)
            subprojects.append(label)

            chunks = getattr(cg, 'chunks', {})
            by_file = getattr(cg, 'by_file', {})

            # Add file-level nodes
            for file_path, chunk_ids in by_file.items():
                file_id = f"{label}:{file_path}"
                if file_id in seen_ids:
                    continue
                seen_ids.add(file_id)

                # Aggregate metrics from chunks in this file
                total_lines = 0
                func_count = 0
                class_count = 0
                for cid in chunk_ids:
                    chunk = chunks.get(cid)
                    if not chunk:
                        continue
                    lines = getattr(chunk, 'line_count', 0) or 0
                    total_lines += lines
                    ct = getattr(chunk, 'chunk_type', None)
                    if ct is not None:
                        ct_val = ct.value if hasattr(ct, 'value') else str(ct)
                        if ct_val == 'function':
                            func_count += 1
                        elif ct_val == 'class':
                            class_count += 1

                nodes.append({
                    "id": file_id,
                    "label": Path(file_path).name,
                    "type": "module",
                    "path": file_path,
                    "group": label,
                    "lines": total_lines,
                    "functions": func_count,
                    "classes": class_count,
                })

            # Add function/class-level nodes and call edges
            for cid, chunk in chunks.items():
                ct = getattr(chunk, 'chunk_type', None)
                if ct is None:
                    continue
                ct_val = ct.value if hasattr(ct, 'value') else str(ct)

                if ct_val in ('function', 'method', 'class'):
                    node_id = f"{label}:{cid}"
                    if node_id in seen_ids:
                        continue
                    seen_ids.add(node_id)

                    nodes.append({
                        "id": node_id,
                        "label": getattr(chunk, 'name', cid),
                        "type": ct_val,
                        "path": getattr(chunk, 'source_path', ''),
                        "group": label,
                        "lines": getattr(chunk, 'line_count', 0) or 0,
                        "parent": f"{label}:{getattr(chunk, 'source_path', '')}",
                    })

                    # Call edges
                    calls = getattr(chunk, 'calls', []) or []
                    for called_name in calls:
                        by_name = getattr(cg, 'by_name', {})
                        called_ids = by_name.get(called_name, [])
                        for target_cid in called_ids:
                            target_id = f"{label}:{target_cid}"
                            edges.append({
                                "source": node_id,
                                "target": target_id,
                                "type": "call",
                                "weight": 1,
                            })

                    # Import edges (from file to file)
                    imports = getattr(chunk, 'imports', []) or []
                    src_file_id = f"{label}:{getattr(chunk, 'source_path', '')}"
                    for imp in imports:
                        # Try to resolve import to a file in this root
                        imp_parts = imp.replace(".", "/")
                        for fp in by_file.keys():
                            if imp_parts in fp or fp.endswith(f"{imp_parts}.py"):
                                target_file_id = f"{label}:{fp}"
                                if src_file_id != target_file_id:
                                    edges.append({
                                        "source": src_file_id,
                                        "target": target_file_id,
                                        "type": "import",
                                        "weight": 1,
                                    })
                                break

        return {
            "nodes": nodes,
            "edges": edges,
            "subprojects": subprojects,
            "generated_at": datetime.now().isoformat(),
        }

    def get_dead_code(self) -> list:
        """Find dead code across all registered roots.

        Returns list of {name, type, file, lines, reason, group}.
        """
        dead = []
        path_to_label = {}
        for lbl, path in self._labels.items():
            path_to_label.setdefault(path, lbl)

        for norm, cg in self._instances.items():
            if cg is None:
                continue
            label = path_to_label.get(norm, Path(norm).name)
            chunks = getattr(cg, 'chunks', {})

            for cid, chunk in chunks.items():
                ct = getattr(chunk, 'chunk_type', None)
                if ct is None:
                    continue
                ct_val = ct.value if hasattr(ct, 'value') else str(ct)
                if ct_val not in ('function', 'method'):
                    continue

                called_by = getattr(chunk, 'called_by', None)
                if called_by and len(called_by) > 0:
                    continue

                name = getattr(chunk, 'name', cid)
                # Skip entry points, dunders, tests
                name_lower = name.lower()
                if (name_lower.startswith("__") and name_lower.endswith("__")):
                    continue
                if any(p in name_lower for p in ("main", "setup", "test_", "conftest")):
                    continue

                source_path = getattr(chunk, 'source_path', '')
                if "/tests/" in source_path or "\\tests\\" in source_path:
                    continue

                # Check for framework decorators in body
                body = getattr(chunk, 'body_preview', '') or ''
                if any(d in body for d in ("@app.", "@router.", "@click.", "@pytest.")):
                    continue

                dead.append({
                    "name": name,
                    "type": ct_val,
                    "file": source_path,
                    "lines": getattr(chunk, 'line_count', 0) or 0,
                    "start_line": getattr(chunk, 'start_line', 0),
                    "end_line": getattr(chunk, 'end_line', 0),
                    "reason": "No callers found in codebase",
                    "group": label,
                    "confidence": 0.7,
                })

        dead.sort(key=lambda d: d.get("lines", 0), reverse=True)
        return dead

    # --------------------------------------------------------------------- #
    # STRATA PERSISTENCE — survive container restarts
    # --------------------------------------------------------------------- #

    def _persist_roots(self):
        """Save root metadata (no-op in public package).

        The public package doesn't have Strata access, so persistence is
        not available. The registry is in-memory only.
        """
        pass

    def _load_persisted_roots(self):
        """Load root metadata on startup (no-op in public package).

        The public package doesn't have Strata access, so the registry is
        in-memory only and does not persist across restarts.
        """
        pass


# =============================================================================
# SINGLETON
# =============================================================================

_registry: Optional[CodeGraphRegistry] = None


def get_codegraph_registry() -> CodeGraphRegistry:
    global _registry
    if _registry is None:
        _registry = CodeGraphRegistry()
    return _registry


def reset_codegraph_registry():
    """Reset singleton (for tests)."""
    global _registry
    _registry = None
