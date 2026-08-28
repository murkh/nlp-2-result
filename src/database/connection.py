"""
Database Connection and Repository Manager for Multi-Agent Knowledge Base Q&A Platform.
Provides dual support for PostgreSQL + pgvector and an in-memory SQLite/vector fallback engine.
"""

import json
import math
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.config import Settings, get_settings
from src.database.models import ColumnMetadata, Dataset, DocumentChunk, QueryLog, TableMetadata


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vector representations."""
    if not vec_a or not vec_b:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class DatabaseManager:
    """
    Unified database manager providing metadata storage, vector similarity search,
    hybrid document search, and dedicated dynamic table operations.
    """

    def __init__(self, settings: Optional[Settings] = None, in_memory: bool = False):
        self.settings = settings or get_settings()
        self.in_memory = in_memory or (
            os.getenv("USE_IN_MEMORY_DB", "false").lower() in ("true", "1")
        )
        self._pg_pool = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._sqlite_lock = threading.RLock()

        # Try connecting to PostgreSQL if not explicitly in_memory
        if not self.in_memory:
            self._init_postgres()

        if self._pg_pool is None:
            # Fallback to in-memory SQLite store
            self.in_memory = True
            self._init_sqlite()

    def _init_postgres(self):
        """Initialize PostgreSQL connection pool if available."""
        try:
            import psycopg
            from psycopg_pool import ConnectionPool

            # Test quick connection
            with psycopg.connect(self.settings.database_url, connect_timeout=2) as test_conn:
                test_conn.execute("SELECT 1")
            self._pg_pool = ConnectionPool(
                conninfo=self.settings.database_url, min_size=1, max_size=10
            )
        except Exception:
            self._pg_pool = None

    def _init_sqlite(self):
        """Initialize SQLite in-memory database with required schema."""
        self._sqlite_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._sqlite_conn.row_factory = sqlite3.Row
        cur = self._sqlite_conn.cursor()

        # Create base tables
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            file_type TEXT NOT NULL,
            category TEXT NOT NULL,
            blob_path TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            row_count INTEGER,
            page_count INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS table_metadata (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            table_name TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            description TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            column_count INTEGER NOT NULL DEFAULT 0,
            embedding TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS column_metadata (
            id TEXT PRIMARY KEY,
            table_id TEXT NOT NULL REFERENCES table_metadata(id) ON DELETE CASCADE,
            column_name TEXT NOT NULL,
            data_type TEXT NOT NULL,
            is_primary_key INTEGER NOT NULL DEFAULT 0,
            is_foreign_key INTEGER NOT NULL DEFAULT 0,
            foreign_target_table TEXT,
            foreign_target_column TEXT,
            null_percentage REAL DEFAULT 0.0,
            distinct_values_count INTEGER DEFAULT 0,
            sample_values TEXT NOT NULL DEFAULT '[]',
            description TEXT NOT NULL,
            embedding TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(table_id, column_name)
        );

        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER,
            section_title TEXT,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            char_count INTEGER NOT NULL,
            embedding TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS query_logs (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            query_text TEXT NOT NULL,
            engine TEXT NOT NULL,
            status TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0.0,
            generated_code TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL
        );
        """)
        self._sqlite_conn.commit()

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Context manager for acquiring a database connection."""
        if self._pg_pool:
            with self._pg_pool.connection() as conn:
                yield conn
        else:
            with self._sqlite_lock:
                yield self._sqlite_conn

    # -------------------------------------------------------------------------
    # Dataset Operations
    # -------------------------------------------------------------------------
    def save_dataset(self, dataset: Dataset) -> str:
        """Insert or update a dataset record."""
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO datasets (id, name, description, file_type, category, blob_path,
                                              file_size_bytes, content_hash, row_count, page_count, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (content_hash) DO UPDATE SET
                            name = EXCLUDED.name,
                            blob_path = EXCLUDED.blob_path,
                            row_count = EXCLUDED.row_count,
                            page_count = EXCLUDED.page_count,
                            updated_at = EXCLUDED.updated_at
                        RETURNING id;
                        """,
                        (
                            dataset.id,
                            dataset.name,
                            dataset.description,
                            dataset.file_type,
                            dataset.category,
                            dataset.blob_path,
                            dataset.file_size_bytes,
                            dataset.content_hash,
                            dataset.row_count,
                            dataset.page_count,
                            dataset.created_at,
                            dataset.updated_at,
                        ),
                    )
                    row = cur.fetchone()
                    return str(row[0]) if row else dataset.id
            else:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO datasets (id, name, description, file_type, category, blob_path,
                                          file_size_bytes, content_hash, row_count, page_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(content_hash) DO UPDATE SET
                        name=excluded.name,
                        blob_path=excluded.blob_path,
                        row_count=excluded.row_count,
                        page_count=excluded.page_count,
                        updated_at=excluded.updated_at
                    """,
                    (
                        dataset.id,
                        dataset.name,
                        dataset.description,
                        dataset.file_type,
                        dataset.category,
                        dataset.blob_path,
                        dataset.file_size_bytes,
                        dataset.content_hash,
                        dataset.row_count,
                        dataset.page_count,
                        dataset.created_at.isoformat(),
                        dataset.updated_at.isoformat(),
                    ),
                )
                conn.commit()
                return dataset.id

    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Fetch a dataset by its ID."""
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM datasets WHERE id = %s;", (dataset_id,))
                    row = cur.fetchone()
                    if not row:
                        return None
                    return Dataset(
                        id=str(row[0]),
                        name=row[1],
                        description=row[2],
                        file_type=row[3],
                        category=row[4],
                        blob_path=row[5],
                        file_size_bytes=row[6],
                        content_hash=row[7],
                        row_count=row[8],
                        page_count=row[9],
                        created_at=row[10],
                        updated_at=row[11],
                    )
            else:
                cur = conn.cursor()
                cur.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return Dataset(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    file_type=row["file_type"],
                    category=row["category"],
                    blob_path=row["blob_path"],
                    file_size_bytes=row["file_size_bytes"],
                    content_hash=row["content_hash"],
                    row_count=row["row_count"],
                    page_count=row["page_count"],
                )

    def list_datasets(self, category: Optional[str] = None) -> List[Dataset]:
        """List all datasets, optionally filtered by category."""
        results = []
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    if category:
                        cur.execute(
                            "SELECT * FROM datasets WHERE category = %s ORDER BY created_at DESC;",
                            (category,),
                        )
                    else:
                        cur.execute("SELECT * FROM datasets ORDER BY created_at DESC;")
                    for row in cur.fetchall():
                        results.append(
                            Dataset(
                                id=str(row[0]),
                                name=row[1],
                                description=row[2],
                                file_type=row[3],
                                category=row[4],
                                blob_path=row[5],
                                file_size_bytes=row[6],
                                content_hash=row[7],
                                row_count=row[8],
                                page_count=row[9],
                                created_at=row[10],
                                updated_at=row[11],
                            )
                        )
            else:
                cur = conn.cursor()
                if category:
                    cur.execute(
                        "SELECT * FROM datasets WHERE category = ? ORDER BY created_at DESC",
                        (category,),
                    )
                else:
                    cur.execute("SELECT * FROM datasets ORDER BY created_at DESC")
                for row in cur.fetchall():
                    results.append(
                        Dataset(
                            id=row["id"],
                            name=row["name"],
                            description=row["description"],
                            file_type=row["file_type"],
                            category=row["category"],
                            blob_path=row["blob_path"],
                            file_size_bytes=row["file_size_bytes"],
                            content_hash=row["content_hash"],
                            row_count=row["row_count"],
                            page_count=row["page_count"],
                        )
                    )
        return results

    # -------------------------------------------------------------------------
    # Structured Metadata Operations
    # -------------------------------------------------------------------------
    def save_table_metadata(self, table_meta: TableMetadata) -> str:
        """Save structured table metadata with vector embedding."""
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO table_metadata (id, dataset_id, table_name, display_name,
                                                   description, row_count, column_count, embedding, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (table_name) DO UPDATE SET
                            display_name = EXCLUDED.display_name,
                            description = EXCLUDED.description,
                            row_count = EXCLUDED.row_count,
                            column_count = EXCLUDED.column_count,
                            embedding = EXCLUDED.embedding
                        RETURNING id;
                        """,
                        (
                            table_meta.id,
                            table_meta.dataset_id,
                            table_meta.table_name,
                            table_meta.display_name,
                            table_meta.description,
                            table_meta.row_count,
                            table_meta.column_count,
                            table_meta.embedding,
                            table_meta.created_at,
                        ),
                    )
                    row = cur.fetchone()
                    return str(row[0]) if row else table_meta.id
            else:
                cur = conn.cursor()
                emb_json = json.dumps(table_meta.embedding) if table_meta.embedding else None
                cur.execute(
                    """
                    INSERT INTO table_metadata (id, dataset_id, table_name, display_name,
                                               description, row_count, column_count, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(table_name) DO UPDATE SET
                        display_name=excluded.display_name,
                        description=excluded.description,
                        row_count=excluded.row_count,
                        column_count=excluded.column_count,
                        embedding=excluded.embedding
                    """,
                    (
                        table_meta.id,
                        table_meta.dataset_id,
                        table_meta.table_name,
                        table_meta.display_name,
                        table_meta.description,
                        table_meta.row_count,
                        table_meta.column_count,
                        emb_json,
                        table_meta.created_at.isoformat(),
                    ),
                )
                conn.commit()
                return table_meta.id

    def save_column_metadata(self, col_meta: ColumnMetadata) -> str:
        """Save column metadata record with sample values and embedding."""
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO column_metadata (
                            id, table_id, column_name, data_type, is_primary_key, is_foreign_key,
                            foreign_target_table, foreign_target_column, null_percentage,
                            distinct_values_count, sample_values, description, embedding, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (table_id, column_name) DO UPDATE SET
                            data_type = EXCLUDED.data_type,
                            is_primary_key = EXCLUDED.is_primary_key,
                            is_foreign_key = EXCLUDED.is_foreign_key,
                            foreign_target_table = EXCLUDED.foreign_target_table,
                            foreign_target_column = EXCLUDED.foreign_target_column,
                            null_percentage = EXCLUDED.null_percentage,
                            distinct_values_count = EXCLUDED.distinct_values_count,
                            sample_values = EXCLUDED.sample_values,
                            description = EXCLUDED.description,
                            embedding = EXCLUDED.embedding
                        RETURNING id;
                        """,
                        (
                            col_meta.id,
                            col_meta.table_id,
                            col_meta.column_name,
                            col_meta.data_type,
                            col_meta.is_primary_key,
                            col_meta.is_foreign_key,
                            col_meta.foreign_target_table,
                            col_meta.foreign_target_column,
                            col_meta.null_percentage,
                            col_meta.distinct_values_count,
                            json.dumps(col_meta.sample_values),
                            col_meta.description,
                            col_meta.embedding,
                            col_meta.created_at,
                        ),
                    )
                    row = cur.fetchone()
                    return str(row[0]) if row else col_meta.id
            else:
                cur = conn.cursor()
                emb_json = json.dumps(col_meta.embedding) if col_meta.embedding else None
                cur.execute(
                    """
                    INSERT INTO column_metadata (
                        id, table_id, column_name, data_type, is_primary_key, is_foreign_key,
                        foreign_target_table, foreign_target_column, null_percentage,
                        distinct_values_count, sample_values, description, embedding, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(table_id, column_name) DO UPDATE SET
                        data_type=excluded.data_type,
                        is_primary_key=excluded.is_primary_key,
                        is_foreign_key=excluded.is_foreign_key,
                        foreign_target_table=excluded.foreign_target_table,
                        foreign_target_column=excluded.foreign_target_column,
                        null_percentage=excluded.null_percentage,
                        distinct_values_count=excluded.distinct_values_count,
                        sample_values=excluded.sample_values,
                        description=excluded.description,
                        embedding=excluded.embedding
                    """,
                    (
                        col_meta.id,
                        col_meta.table_id,
                        col_meta.column_name,
                        col_meta.data_type,
                        1 if col_meta.is_primary_key else 0,
                        1 if col_meta.is_foreign_key else 0,
                        col_meta.foreign_target_table,
                        col_meta.foreign_target_column,
                        col_meta.null_percentage,
                        col_meta.distinct_values_count,
                        json.dumps(col_meta.sample_values),
                        col_meta.description,
                        emb_json,
                        col_meta.created_at.isoformat(),
                    ),
                )
                conn.commit()
                return col_meta.id

    def list_tables(self) -> List[TableMetadata]:
        """List all table metadata records."""
        tables = []
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, dataset_id, table_name, display_name, description, row_count, column_count, created_at FROM table_metadata;"
                    )
                    for row in cur.fetchall():
                        tables.append(
                            TableMetadata(
                                id=str(row[0]),
                                dataset_id=str(row[1]),
                                table_name=row[2],
                                display_name=row[3],
                                description=row[4],
                                row_count=row[5],
                                column_count=row[6],
                                created_at=row[7],
                            )
                        )
            else:
                cur = conn.cursor()
                cur.execute("SELECT * FROM table_metadata")
                for row in cur.fetchall():
                    tables.append(
                        TableMetadata(
                            id=row["id"],
                            dataset_id=row["dataset_id"],
                            table_name=row["table_name"],
                            display_name=row["display_name"],
                            description=row["description"],
                            row_count=row["row_count"],
                            column_count=row["column_count"],
                        )
                    )
        return tables

    def get_columns_for_table(self, table_id: str) -> List[ColumnMetadata]:
        """Fetch all column metadata records for a given table."""
        columns = []
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, table_id, column_name, data_type, is_primary_key, is_foreign_key,
                               foreign_target_table, foreign_target_column, null_percentage,
                               distinct_values_count, sample_values, description, created_at
                        FROM column_metadata WHERE table_id = %s;
                        """,
                        (table_id,),
                    )
                    for row in cur.fetchall():
                        samples = (
                            json.loads(row[10]) if isinstance(row[10], str) else (row[10] or [])
                        )
                        columns.append(
                            ColumnMetadata(
                                id=str(row[0]),
                                table_id=str(row[1]),
                                column_name=row[2],
                                data_type=row[3],
                                is_primary_key=bool(row[4]),
                                is_foreign_key=bool(row[5]),
                                foreign_target_table=row[6],
                                foreign_target_column=row[7],
                                null_percentage=float(row[8] or 0.0),
                                distinct_values_count=int(row[9] or 0),
                                sample_values=samples,
                                description=row[11],
                                created_at=row[12],
                            )
                        )
            else:
                cur = conn.cursor()
                cur.execute("SELECT * FROM column_metadata WHERE table_id = ?", (table_id,))
                for row in cur.fetchall():
                    samples = json.loads(row["sample_values"]) if row["sample_values"] else []
                    columns.append(
                        ColumnMetadata(
                            id=row["id"],
                            table_id=row["table_id"],
                            column_name=row["column_name"],
                            data_type=row["data_type"],
                            is_primary_key=bool(row["is_primary_key"]),
                            is_foreign_key=bool(row["is_foreign_key"]),
                            foreign_target_table=row["foreign_target_table"],
                            foreign_target_column=row["foreign_target_column"],
                            null_percentage=float(row["null_percentage"] or 0.0),
                            distinct_values_count=int(row["distinct_values_count"] or 0),
                            sample_values=samples,
                            description=row["description"],
                        )
                    )
        return columns

    # -------------------------------------------------------------------------
    # Two-Stage Vector Search for Schema Pruning
    # -------------------------------------------------------------------------
    def search_table_metadata_vectors(
        self, query_embedding: List[float], query_text: str = "", top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Stage 1: Coarse retrieval of top-K relevant tables via vector similarity."""
        results = []
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, dataset_id, table_name, display_name, description, row_count, column_count,
                               (1.0 - (embedding <=> %s::vector)) AS similarity
                        FROM table_metadata
                        WHERE embedding IS NOT NULL
                        ORDER BY similarity DESC
                        LIMIT %s;
                        """,
                        (query_embedding, top_k),
                    )
                    for row in cur.fetchall():
                        results.append(
                            {
                                "id": str(row[0]),
                                "dataset_id": str(row[1]),
                                "table_name": row[2],
                                "display_name": row[3],
                                "description": row[4],
                                "row_count": row[5],
                                "column_count": row[6],
                                "similarity": float(row[7]) if row[7] is not None else 0.0,
                            }
                        )
            else:
                cur = conn.cursor()
                cur.execute("SELECT * FROM table_metadata WHERE embedding IS NOT NULL")
                rows = cur.fetchall()
                scored_rows = []
                keywords = [k.lower() for k in re.findall(r"\w+", query_text)] if query_text else []
                for row in rows:
                    emb = json.loads(row["embedding"])
                    sim = cosine_similarity(query_embedding, emb)
                    # Keyword boost if table name or description contains query tokens
                    for kw in keywords:
                        if len(kw) > 2 and (
                            kw in row["table_name"].lower() or kw in row["description"].lower()
                        ):
                            sim += 0.15
                    scored_rows.append((sim, row))
                scored_rows.sort(key=lambda x: x[0], reverse=True)
                for sim, row in scored_rows[:top_k]:
                    results.append(
                        {
                            "id": row["id"],
                            "dataset_id": row["dataset_id"],
                            "table_name": row["table_name"],
                            "display_name": row["display_name"],
                            "description": row["description"],
                            "row_count": row["row_count"],
                            "column_count": row["column_count"],
                            "similarity": float(sim),
                        }
                    )
        return results

    def search_column_metadata_vectors(
        self, query_embedding: List[float], table_ids: List[str], query_text: str = ""
    ) -> List[Dict[str, Any]]:
        """Stage 2: Fine retrieval of columns for specified tables ordered by relevance & key status."""
        results = []
        if not table_ids:
            return results

        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, table_id, column_name, data_type, is_primary_key, is_foreign_key,
                               foreign_target_table, foreign_target_column, null_percentage,
                               distinct_values_count, sample_values, description,
                               (1.0 - (embedding <=> %s::vector)) AS similarity
                        FROM column_metadata
                        WHERE table_id = ANY(%s::uuid[])
                        ORDER BY is_primary_key DESC, is_foreign_key DESC, similarity DESC;
                        """,
                        (query_embedding, table_ids),
                    )
                    for row in cur.fetchall():
                        samples = (
                            json.loads(row[10]) if isinstance(row[10], str) else (row[10] or [])
                        )
                        results.append(
                            {
                                "id": str(row[0]),
                                "table_id": str(row[1]),
                                "column_name": row[2],
                                "data_type": row[3],
                                "is_primary_key": bool(row[4]),
                                "is_foreign_key": bool(row[5]),
                                "foreign_target_table": row[6],
                                "foreign_target_column": row[7],
                                "null_percentage": float(row[8] or 0.0),
                                "distinct_values_count": int(row[9] or 0),
                                "sample_values": samples,
                                "description": row[11],
                                "similarity": float(row[12]) if row[12] is not None else 0.0,
                            }
                        )
            else:
                cur = conn.cursor()
                placeholders = ",".join(["?"] * len(table_ids))
                cur.execute(
                    f"SELECT * FROM column_metadata WHERE table_id IN ({placeholders})", table_ids
                )
                rows = cur.fetchall()
                scored_cols = []
                keywords = [k.lower() for k in re.findall(r"\w+", query_text)] if query_text else []
                for row in rows:
                    emb = json.loads(row["embedding"]) if row["embedding"] else []
                    sim = cosine_similarity(query_embedding, emb)
                    for kw in keywords:
                        if len(kw) > 2 and (
                            kw in row["column_name"].lower() or kw in row["description"].lower()
                        ):
                            sim += 0.15
                    scored_cols.append(
                        (bool(row["is_primary_key"]), bool(row["is_foreign_key"]), sim, row)
                    )
                # Sort: PKs first, FKs second, then by similarity descending
                scored_cols.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
                for is_pk, is_fk, sim, row in scored_cols:
                    samples = json.loads(row["sample_values"]) if row["sample_values"] else []
                    results.append(
                        {
                            "id": row["id"],
                            "table_id": row["table_id"],
                            "column_name": row["column_name"],
                            "data_type": row["data_type"],
                            "is_primary_key": is_pk,
                            "is_foreign_key": is_fk,
                            "foreign_target_table": row["foreign_target_table"],
                            "foreign_target_column": row["foreign_target_column"],
                            "null_percentage": float(row["null_percentage"] or 0.0),
                            "distinct_values_count": int(row["distinct_values_count"] or 0),
                            "sample_values": samples,
                            "description": row["description"],
                            "similarity": float(sim),
                        }
                    )
        return results

    # -------------------------------------------------------------------------
    # Unstructured Document Chunks & Hybrid Search (RRF)
    # -------------------------------------------------------------------------
    def save_document_chunks(self, chunks: List[DocumentChunk]):
        """Batch save document chunks."""
        if not chunks:
            return
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    for chunk in chunks:
                        cur.execute(
                            """
                            INSERT INTO document_chunks (
                                id, dataset_id, chunk_index, page_number, section_title,
                                content, token_count, char_count, embedding, created_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                            """,
                            (
                                chunk.id,
                                chunk.dataset_id,
                                chunk.chunk_index,
                                chunk.page_number,
                                chunk.section_title,
                                chunk.content,
                                chunk.token_count,
                                chunk.char_count,
                                chunk.embedding,
                                chunk.created_at,
                            ),
                        )
            else:
                cur = conn.cursor()
                for chunk in chunks:
                    emb_json = json.dumps(chunk.embedding) if chunk.embedding else None
                    cur.execute(
                        """
                        INSERT INTO document_chunks (
                            id, dataset_id, chunk_index, page_number, section_title,
                            content, token_count, char_count, embedding, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.id,
                            chunk.dataset_id,
                            chunk.chunk_index,
                            chunk.page_number,
                            chunk.section_title,
                            chunk.content,
                            chunk.token_count,
                            chunk.char_count,
                            emb_json,
                            chunk.created_at.isoformat(),
                        ),
                    )
                conn.commit()

    def hybrid_search_document_chunks(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        dataset_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid dense vector + sparse keyword search using Reciprocal Rank Fusion (RRF).
        Formula: RRF = (0.6 / (60 + dense_rank)) + (0.4 / (60 + sparse_rank))
        """
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    # Execute SQL RRF query
                    sql = """
                    WITH dense_search AS (
                        SELECT id, dataset_id, content, page_number, section_title,
                               ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS dense_rank
                        FROM document_chunks
                        WHERE embedding IS NOT NULL
                          AND (%s::uuid IS NULL OR dataset_id = %s::uuid)
                        LIMIT 50
                    ),
                    sparse_search AS (
                        SELECT id, dataset_id, content, page_number, section_title,
                               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(content_tsvector, plainto_tsquery('english', %s)) DESC) AS sparse_rank
                        FROM document_chunks
                        WHERE content_tsvector @@ plainto_tsquery('english', %s)
                          AND (%s::uuid IS NULL OR dataset_id = %s::uuid)
                        LIMIT 50
                    )
                    SELECT
                        COALESCE(d.id, s.id) AS chunk_id,
                        COALESCE(d.dataset_id, s.dataset_id) AS dataset_id,
                        COALESCE(d.content, s.content) AS content,
                        COALESCE(d.page_number, s.page_number) AS page_number,
                        COALESCE(d.section_title, s.section_title) AS section_title,
                        (COALESCE(1.0 / (60.0 + d.dense_rank), 0.0) * 0.6 +
                         COALESCE(1.0 / (60.0 + s.sparse_rank), 0.0) * 0.4) AS rrf_score
                    FROM dense_search d
                    FULL OUTER JOIN sparse_search s ON d.id = s.id
                    ORDER BY rrf_score DESC
                    LIMIT %s;
                    """
                    cur.execute(
                        sql,
                        (
                            query_embedding,
                            dataset_id,
                            dataset_id,
                            query_text,
                            query_text,
                            dataset_id,
                            dataset_id,
                            top_k,
                        ),
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            "chunk_id": str(r[0]),
                            "dataset_id": str(r[1]),
                            "content": r[2],
                            "page_number": r[3],
                            "section_title": r[4],
                            "rrf_score": float(r[5]),
                        }
                        for r in rows
                    ]
            else:
                # In-memory implementation of RRF hybrid search
                cur = conn.cursor()
                query = "SELECT * FROM document_chunks"
                params = ()
                if dataset_id:
                    query += " WHERE dataset_id = ?"
                    params = (dataset_id,)
                cur.execute(query, params)
                all_chunks = cur.fetchall()

                if not all_chunks:
                    return []

                # 1. Dense search ranking
                dense_scored = []
                for chunk in all_chunks:
                    emb = json.loads(chunk["embedding"]) if chunk["embedding"] else []
                    sim = cosine_similarity(query_embedding, emb)
                    dense_scored.append((chunk["id"], sim, chunk))
                dense_scored.sort(key=lambda x: x[1], reverse=True)
                dense_ranks = {cid: rank + 1 for rank, (cid, _, _) in enumerate(dense_scored[:50])}

                # 2. Sparse search ranking (keyword match / BM25 proxy)
                terms = [w.lower() for w in re.findall(r"\w+", query_text)]
                sparse_scored = []
                for chunk in all_chunks:
                    text = chunk["content"].lower()
                    term_hits = sum(text.count(t) for t in terms)
                    if term_hits > 0:
                        sparse_scored.append((chunk["id"], term_hits, chunk))
                sparse_scored.sort(key=lambda x: x[1], reverse=True)
                sparse_ranks = {
                    cid: rank + 1 for rank, (cid, _, _) in enumerate(sparse_scored[:50])
                }

                # 3. Calculate RRF scores
                all_candidate_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
                chunk_map = {c["id"]: c for c in all_chunks}
                rrf_results = []

                for cid in all_candidate_ids:
                    d_rank = dense_ranks.get(cid)
                    s_rank = sparse_ranks.get(cid)
                    d_score = (1.0 / (60.0 + d_rank)) if d_rank else 0.0
                    s_score = (1.0 / (60.0 + s_rank)) if s_rank else 0.0
                    total_rrf = (d_score * 0.6) + (s_score * 0.4)
                    c = chunk_map[cid]
                    rrf_results.append(
                        {
                            "chunk_id": c["id"],
                            "dataset_id": c["dataset_id"],
                            "content": c["content"],
                            "page_number": c["page_number"],
                            "section_title": c["section_title"],
                            "rrf_score": float(total_rrf),
                        }
                    )

                rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
                return rrf_results[:top_k]

    # -------------------------------------------------------------------------
    # Dedicated Dynamic Table Management (Strategy A)
    # -------------------------------------------------------------------------
    def create_dedicated_table(
        self, table_name: str, column_defs: List[Tuple[str, str]], pkey_col: Optional[str] = None
    ):
        """Create a dedicated dynamic table for structured ingestion."""
        col_sqls = []
        if not pkey_col:
            col_sqls.append(
                "_row_id INTEGER PRIMARY KEY AUTOINCREMENT"
                if self.in_memory
                else "_row_id BIGSERIAL PRIMARY KEY"
            )

        for col_name, col_type in column_defs:
            is_pk = col_name == pkey_col
            type_str = col_type
            if is_pk:
                type_str += " PRIMARY KEY"
            col_sqls.append(f'"{col_name}" {type_str}')

        sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  ' + ",\n  ".join(col_sqls) + "\n);"

        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(sql)
            else:
                conn.execute(sql)
                conn.commit()

    def insert_table_rows(self, table_name: str, columns: List[str], rows: List[List[Any]]):
        """Batch insert rows into a dedicated table."""
        if not rows:
            return
        cols_str = ", ".join([f'"{c}"' for c in columns])
        placeholders = ", ".join(["%s" if self._pg_pool else "?" for _ in columns])
        sql = f'INSERT INTO "{table_name}" ({cols_str}) VALUES ({placeholders})'

        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.executemany(sql, rows)
            else:
                conn.executemany(sql, rows)
                conn.commit()

    def execute_sql_query(
        self, sql_query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> Tuple[List[str], List[Tuple[Any, ...]]]:
        """Execute a read-only SQL query and return column names and row tuples."""
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(sql_query, params or ())
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchall() if cur.description else []
                    return cols, rows
            else:
                cur = conn.cursor()
                cur.execute(sql_query, params or ())
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall() if cur.description else []
                return cols, [tuple(r) for r in rows]

    # -------------------------------------------------------------------------
    # Query Logs
    # -------------------------------------------------------------------------
    def log_query(self, log_entry: QueryLog):
        """Insert execution log entry."""
        with self.get_connection() as conn:
            if self._pg_pool:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO query_logs (id, session_id, query_text, engine, status,
                                               prompt_tokens, completion_tokens, latency_ms,
                                               generated_code, error_message, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            log_entry.id,
                            log_entry.session_id,
                            log_entry.query_text,
                            log_entry.engine,
                            log_entry.status,
                            log_entry.prompt_tokens,
                            log_entry.completion_tokens,
                            log_entry.latency_ms,
                            log_entry.generated_code,
                            log_entry.error_message,
                            log_entry.created_at,
                        ),
                    )
            else:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO query_logs (id, session_id, query_text, engine, status,
                                           prompt_tokens, completion_tokens, latency_ms,
                                           generated_code, error_message, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_entry.id,
                        log_entry.session_id,
                        log_entry.query_text,
                        log_entry.engine,
                        log_entry.status,
                        log_entry.prompt_tokens,
                        log_entry.completion_tokens,
                        log_entry.latency_ms,
                        log_entry.generated_code,
                        log_entry.error_message,
                        log_entry.created_at.isoformat(),
                    ),
                )
                conn.commit()


# Singleton instance helper
_db_manager_instance: Optional[DatabaseManager] = None


def get_db_manager(settings: Optional[Settings] = None, in_memory: bool = False) -> DatabaseManager:
    """Returns singleton or configured DatabaseManager instance."""
    global _db_manager_instance
    if _db_manager_instance is None or in_memory:
        _db_manager_instance = DatabaseManager(settings=settings, in_memory=in_memory)
    return _db_manager_instance
