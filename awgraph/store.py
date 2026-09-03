"""
SQLite+FTS5 storage backend for CodeGraph.

Provides CRUD operations, full-text search, embedding storage, and file mtime
tracking for the code graph data structure.
"""

import json
import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["CodeGraphStore"]


def get_logger(name: str):
    """Lazy import to avoid circular deps."""
    from awgraph.logging import get_logger as _get_logger
    return _get_logger(name)


logger = get_logger(__name__)


class CodeGraphStore:
    """
    SQLite+FTS5 storage for CodeGraph chunks, embeddings, and metadata.

    Features:
    - WAL mode, PRAGMA synchronous=NORMAL, 64MB cache
    - FTS5 full-text search on name, signature, docstring, body_preview
    - Embedding storage with struct.pack/unpack for float arrays
    - File mtime tracking for change detection
    - Bulk operations in single transactions
    - Lazy imports to avoid circular dependencies
    """

    def __init__(self, db_path: str):
        """Initialize CodeGraphStore and create tables if needed.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection with WAL mode and pragmas."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            # WAL mode for concurrent read access
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL: sync after each transaction (safe + faster than FULL)
            self._conn.execute("PRAGMA synchronous=NORMAL")
            # 64MB cache (16000 pages * 4KB default page size)
            self._conn.execute("PRAGMA cache_size=16000")
            # Foreign key constraints
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self.conn:
            # Main chunks table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    signature TEXT DEFAULT '',
                    docstring TEXT DEFAULT '',
                    body_preview TEXT DEFAULT '',
                    imports TEXT DEFAULT '[]',
                    import_map TEXT DEFAULT '{}',
                    calls TEXT DEFAULT '[]',
                    called_by TEXT DEFAULT '[]',
                    base_classes TEXT DEFAULT '[]',
                    methods TEXT DEFAULT '[]',
                    parent_class TEXT,
                    complexity INTEGER DEFAULT 0,
                    line_count INTEGER DEFAULT 0,
                    git_commits INTEGER DEFAULT 0,
                    git_contributors INTEGER DEFAULT 0,
                    git_last_modified TEXT DEFAULT '',
                    git_churn_rate REAL DEFAULT 0.0,
                    centrality REAL DEFAULT 0.0,
                    fan_in INTEGER DEFAULT 0,
                    fan_out INTEGER DEFAULT 0,
                    tenant_id TEXT DEFAULT 'platform',
                    workspace_id TEXT DEFAULT '',
                    embedding BLOB,
                    created_at REAL,
                    updated_at REAL
                )
            """)

            # Create index on (source_path, chunk_type, name) for efficient lookups
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_source_path
                ON chunks(source_path)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_tenant_workspace
                ON chunks(tenant_id, workspace_id)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_parent_class
                ON chunks(parent_class)
            """)

            # FTS5 virtual table for full-text search
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    name, signature, docstring, body_preview,
                    content='chunks', content_rowid='rowid'
                )
            """)

            # Triggers to keep FTS5 in sync with chunks table
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, name, signature, docstring, body_preview)
                    VALUES (new.rowid, new.name, new.signature, new.docstring, new.body_preview);
                END
            """)
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE rowid = old.rowid;
                END
            """)
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE rowid = old.rowid;
                    INSERT INTO chunks_fts(rowid, name, signature, docstring, body_preview)
                    VALUES (new.rowid, new.name, new.signature, new.docstring, new.body_preview);
                END
            """)

            # Embeddings table (separate to allow NULL for chunks without embeddings)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    created_at REAL,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                )
            """)

            # File mtime tracking for change detection
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS file_mtimes (
                    file_path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    updated_at REAL
                )
            """)

            # Metadata key-value store
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL
                )
            """)

    def _codechunk_from_row(self, row: sqlite3.Row) -> "CodeChunk":
        """Convert a database row to CodeChunk dataclass."""
        # Lazy import to avoid circular deps
        try:
            from awgraph.graph import CodeChunk, ChunkType
        except ImportError:
            # The nested `lib.faculties.CodeGraph` fallback that used to sit here
            # was unreachable by construction: `awgraph.graph` is a sibling module
            # of this one, so if it cannot be imported neither can this file. It
            # only ever made the wheel look like it depended on the monorepo.
            return dict(row)

        return CodeChunk(
            id=row["id"],
            name=row["name"],
            chunk_type=ChunkType(row["chunk_type"]),
            source_path=row["source_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            signature=row["signature"],
            docstring=row["docstring"],
            body_preview=row["body_preview"],
            imports=json.loads(row["imports"]),
            import_map=json.loads(row["import_map"]),
            calls=json.loads(row["calls"]),
            called_by=json.loads(row["called_by"]),
            base_classes=json.loads(row["base_classes"]),
            methods=json.loads(row["methods"]),
            parent_class=row["parent_class"],
            complexity=row["complexity"],
            line_count=row["line_count"],
            git_commits=row["git_commits"],
            git_contributors=row["git_contributors"],
            git_last_modified=row["git_last_modified"],
            git_churn_rate=row["git_churn_rate"],
            centrality=row["centrality"],
            fan_in=row["fan_in"],
            fan_out=row["fan_out"],
            tenant_id=row["tenant_id"],
            workspace_id=row["workspace_id"],
            embedding=None,  # Embeddings stored separately
        )

    def _codechunk_to_dict(self, chunk) -> Dict:
        """Convert CodeChunk to dict for insertion."""
        import time
        now = time.time()

        return {
            "id": chunk.id,
            "name": chunk.name,
            "chunk_type": chunk.chunk_type.value,
            "source_path": chunk.source_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "signature": chunk.signature,
            "docstring": chunk.docstring,
            "body_preview": chunk.body_preview,
            "imports": json.dumps(chunk.imports),
            "import_map": json.dumps(chunk.import_map),
            "calls": json.dumps(chunk.calls),
            "called_by": json.dumps(chunk.called_by),
            "base_classes": json.dumps(chunk.base_classes),
            "methods": json.dumps(chunk.methods),
            "parent_class": chunk.parent_class,
            "complexity": chunk.complexity,
            "line_count": chunk.line_count,
            "git_commits": chunk.git_commits,
            "git_contributors": chunk.git_contributors,
            "git_last_modified": chunk.git_last_modified,
            "git_churn_rate": chunk.git_churn_rate,
            "centrality": chunk.centrality,
            "fan_in": chunk.fan_in,
            "fan_out": chunk.fan_out,
            "tenant_id": chunk.tenant_id,
            "workspace_id": chunk.workspace_id,
            "created_at": now,
            "updated_at": now,
        }

    def put(self, chunk) -> None:
        """Insert or update a chunk.

        Args:
            chunk: CodeChunk dataclass instance
        """
        data = self._codechunk_to_dict(chunk)
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))

        with self.conn:
            self.conn.execute(
                f"INSERT OR REPLACE INTO chunks ({cols}) VALUES ({placeholders})",
                tuple(data.values())
            )
        logger.debug(f"Stored chunk: {chunk.id}")

    def get(self, chunk_id: str) -> Optional:
        """Retrieve a chunk by ID.

        Args:
            chunk_id: The chunk ID

        Returns:
            CodeChunk or None if not found
        """
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?",
            (chunk_id,)
        ).fetchone()
        return self._codechunk_from_row(row) if row else None

    def delete(self, chunk_id: str) -> bool:
        """Delete a chunk by ID.

        Args:
            chunk_id: The chunk ID

        Returns:
            True if deleted, False if not found
        """
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM chunks WHERE id = ?",
                (chunk_id,)
            )
            deleted = cursor.rowcount > 0
        if deleted:
            logger.debug(f"Deleted chunk: {chunk_id}")
        return deleted

    def has(self, chunk_id: str) -> bool:
        """Check if a chunk exists.

        Args:
            chunk_id: The chunk ID

        Returns:
            True if chunk exists
        """
        row = self.conn.execute(
            "SELECT 1 FROM chunks WHERE id = ? LIMIT 1",
            (chunk_id,)
        ).fetchone()
        return row is not None

    def count(self) -> int:
        """Get total chunk count.

        Returns:
            Number of chunks in store
        """
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"] if row else 0

    def all_chunks(self) -> Dict[str, Any]:
        """Retrieve all chunks as dict.

        Returns:
            Dict mapping chunk_id -> CodeChunk
        """
        rows = self.conn.execute("SELECT * FROM chunks").fetchall()
        return {row["id"]: self._codechunk_from_row(row) for row in rows}

    def by_name(self, name: str) -> List[str]:
        """Find all chunk IDs with matching name.

        Args:
            name: The chunk name

        Returns:
            List of chunk IDs
        """
        rows = self.conn.execute(
            "SELECT id FROM chunks WHERE name = ?",
            (name,)
        ).fetchall()
        return [row["id"] for row in rows]

    def by_file(self, file_path: str) -> List[str]:
        """Find all chunk IDs from a source file.

        Args:
            file_path: The source file path

        Returns:
            List of chunk IDs
        """
        rows = self.conn.execute(
            "SELECT id FROM chunks WHERE source_path = ? ORDER BY start_line",
            (file_path,)
        ).fetchall()
        return [row["id"] for row in rows]

    def by_class(self, class_name: str) -> List[str]:
        """Find all chunk IDs for a class (class def + its methods).

        Args:
            class_name: The class name

        Returns:
            List of chunk IDs (class first, then methods)
        """
        rows = self.conn.execute("""
            SELECT id FROM chunks
            WHERE name = ? OR parent_class = ?
            ORDER BY (chunk_type = 'class') DESC, start_line
        """, (class_name, class_name)).fetchall()
        return [row["id"] for row in rows]

    def routes(self) -> Dict[str, str]:
        """Get all FastAPI/Flask routes.

        Returns:
            Dict mapping route_key (e.g., "GET:/path/to/route") -> chunk_id
        """
        # Look for @app.get, @app.post, @router.get, etc. decorators in docstrings
        rows = self.conn.execute("""
            SELECT id, docstring FROM chunks
            WHERE chunk_type IN ('function', 'method')
            AND (docstring LIKE '%@%' OR docstring LIKE '%route%' OR docstring LIKE '%endpoint%')
        """).fetchall()

        result = {}
        for row in rows:
            docstring = row["docstring"] or ""
            # Very simple route detection - look for @app.method or @router.method
            for line in docstring.split("\n"):
                line = line.strip()
                if "@" in line and ("app." in line or "router." in line):
                    # Extract method (get, post, etc.) and path
                    # e.g., "@app.get('/path')" -> "GET:/path"
                    parts = line.split("(")
                    if len(parts) > 1:
                        method_part = parts[0]
                        if "get" in method_part.lower():
                            method = "GET"
                        elif "post" in method_part.lower():
                            method = "POST"
                        elif "put" in method_part.lower():
                            method = "PUT"
                        elif "delete" in method_part.lower():
                            method = "DELETE"
                        else:
                            continue

                        path_part = parts[1].rstrip(")").strip("'\"")
                        route_key = f"{method}:{path_part}"
                        result[route_key] = row["id"]

        return result

    def fts_search(self, query: str, limit: int = 50) -> List[Tuple[str, float]]:
        """Full-text search chunks.

        Args:
            query: FTS5 query string (supports AND, OR, NOT, prefix matching)
            limit: Max results to return

        Returns:
            List of (chunk_id, rank) tuples, sorted by relevance (best first)
        """
        rows = self.conn.execute("""
            SELECT chunks.id, chunks_fts.rank
            FROM chunks_fts
            JOIN chunks ON chunks_fts.rowid = chunks.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY chunks_fts.rank
            LIMIT ?
        """, (query, limit)).fetchall()
        return [(row["id"], row["rank"]) for row in rows]

    def bulk_put(self, chunks: List) -> None:
        """Insert multiple chunks in a single transaction.

        Args:
            chunks: List of CodeChunk instances
        """
        with self.conn:
            for chunk in chunks:
                data = self._codechunk_to_dict(chunk)
                cols = ", ".join(data.keys())
                placeholders = ", ".join("?" * len(data))
                self.conn.execute(
                    f"INSERT OR REPLACE INTO chunks ({cols}) VALUES ({placeholders})",
                    tuple(data.values())
                )
        logger.info(f"Bulk stored {len(chunks)} chunks")

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a source file.

        Args:
            file_path: The source file path

        Returns:
            Number of chunks deleted
        """
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM chunks WHERE source_path = ?",
                (file_path,)
            )
            count = cursor.rowcount
        logger.info(f"Deleted {count} chunks from {file_path}")
        return count

    def put_embedding(self, chunk_id: str, embedding: List[float]) -> None:
        """Store embedding for a chunk.

        Args:
            chunk_id: The chunk ID
            embedding: List of floats (embedding vector)
        """
        import time

        # Pack floats as binary blob (struct.pack single-precision)
        vector_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO embeddings (chunk_id, vector, created_at)
                VALUES (?, ?, ?)
            """, (chunk_id, vector_bytes, time.time()))
        logger.debug(f"Stored embedding for chunk: {chunk_id}")

    def get_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Retrieve embedding for a chunk.

        Args:
            chunk_id: The chunk ID

        Returns:
            List of floats or None if not found
        """
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE chunk_id = ?",
            (chunk_id,)
        ).fetchone()

        if not row:
            return None

        # Unpack binary blob back to floats
        vector_bytes = row["vector"]
        num_floats = len(vector_bytes) // 4
        embedding = struct.unpack(f"{num_floats}f", vector_bytes)
        return list(embedding)

    def all_embeddings(self) -> Dict[str, List[float]]:
        """Retrieve all embeddings.

        Returns:
            Dict mapping chunk_id -> embedding vector
        """
        rows = self.conn.execute(
            "SELECT chunk_id, vector FROM embeddings"
        ).fetchall()

        result = {}
        for row in rows:
            vector_bytes = row["vector"]
            num_floats = len(vector_bytes) // 4
            embedding = struct.unpack(f"{num_floats}f", vector_bytes)
            result[row["chunk_id"]] = list(embedding)
        return result

    def get_mtime(self, file_path: str) -> Optional[float]:
        """Get stored mtime for a file.

        Args:
            file_path: The file path

        Returns:
            Modification time (seconds since epoch) or None if not tracked
        """
        row = self.conn.execute(
            "SELECT mtime FROM file_mtimes WHERE file_path = ?",
            (file_path,)
        ).fetchone()
        return row["mtime"] if row else None

    def put_mtime(self, file_path: str, mtime: float) -> None:
        """Store mtime for a file.

        Args:
            file_path: The file path
            mtime: Modification time (seconds since epoch)
        """
        import time
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO file_mtimes (file_path, mtime, updated_at)
                VALUES (?, ?, ?)
            """, (file_path, mtime, time.time()))

    def get_changed_files(self, current_mtimes: Dict[str, float]) -> List[str]:
        """Find files that have changed since last scan.

        Args:
            current_mtimes: Dict mapping file_path -> current mtime

        Returns:
            List of changed file paths
        """
        changed = []
        for file_path, current_mtime in current_mtimes.items():
            stored_mtime = self.get_mtime(file_path)
            if stored_mtime is None or current_mtime > stored_mtime:
                changed.append(file_path)
        return changed

    def get_meta(self, key: str) -> Optional[str]:
        """Get metadata value.

        Args:
            key: Metadata key

        Returns:
            Value or None if not found
        """
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,)
        ).fetchone()
        return row["value"] if row else None

    def put_meta(self, key: str, value: str) -> None:
        """Store metadata value.

        Args:
            key: Metadata key
            value: Metadata value
        """
        import time
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, time.time()))

    @staticmethod
    def migrate_from_pickle(pickle_path: str, db_path: str) -> None:
        """Migrate chunks from legacy pickle format to SQLite.

        Args:
            pickle_path: Path to pickle file with chunks dict
            db_path: Path to new SQLite database
        """
        import pickle

        pickle_file = Path(pickle_path)
        if not pickle_file.exists():
            logger.warning(f"Pickle file not found: {pickle_path}")
            return

        try:
            with open(pickle_file, "rb") as f:
                chunks_dict = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load pickle: {e}")
            return

        store = CodeGraphStore(db_path)
        chunks = list(chunks_dict.values())
        store.bulk_put(chunks)
        logger.info(f"Migrated {len(chunks)} chunks from pickle to SQLite")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Closed CodeGraphStore connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
