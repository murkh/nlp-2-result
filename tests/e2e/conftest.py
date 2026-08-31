"""
E2E Test Fixtures for Tiers 1-4 (tests/e2e/)
Multi-Agent Knowledge Base Q&A Platform
"""

import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.config import get_settings

# -----------------------------------------------------------------------------
# Mock Vector Embedding Generator (EMBEDDING_DIM-wide deterministic vectors)
# -----------------------------------------------------------------------------
EMBEDDING_DIM = get_settings().embedding_dim


class MockEmbeddingProvider:
    """Generates deterministic float vectors of EMBEDDING_DIM width from a text hash."""

    dim = EMBEDDING_DIM

    @staticmethod
    def embed_text(text: str) -> List[float]:
        seed = sum(ord(c) for c in text) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @staticmethod
    def embed_batch(texts: List[str]) -> List[List[float]]:
        return [MockEmbeddingProvider.embed_text(t) for t in texts]

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# -----------------------------------------------------------------------------
# Mock LLM & Intent Classifier
# -----------------------------------------------------------------------------
class MockLLMProvider:
    """Deterministic LLM responses for SQL, Python DataFrame, Intent, and Synthesis."""

    @staticmethod
    def classify_intent(query: str) -> Dict[str, Any]:
        q_lower = query.lower().strip()
        greetings = ["hello", "hi", "hey", "what can you do", "help", "who are you", "good morning"]
        if any(g in q_lower for g in greetings) and len(q_lower.split()) <= 6:
            return {
                "intent": "GREETING_OR_CHITCHAT",
                "suggested_strategy": "direct",
                "response": "Hello! I am your Multi-Agent Knowledge Base assistant. How can I help you today?",
            }

        ambiguous = [
            "data",
            "show me stats",
            "analyze",
            "report",
            "insights",
            "what do you have",
            "stats",
            "revenue",
            "show sales",
        ]
        if q_lower in ambiguous or len(q_lower.split()) <= 1 or not q_lower:
            return {
                "intent": "AMBIGUOUS_QUERY",
                "suggested_strategy": "clarify",
                "candidate_datasets": ["sales_q3", "customer_churn", "financial_records"],
                "response": "Your query is ambiguous. Which dataset would you like to analyze? Options: sales_q3, customer_churn, financial_records.",
            }

        unstructured_keywords = [
            "policy",
            "document",
            "contract",
            "section",
            "paragraph",
            "article",
            "clause",
            "guidelines",
            "manual",
            "handbook",
        ]
        if any(w in q_lower for w in unstructured_keywords):
            return {
                "intent": "UNSTRUCTURED_QUERY",
                "suggested_strategy": "unstructured_rag",
                "response": "Executing hybrid dense + sparse RAG search across indexed documents.",
            }

        return {
            "intent": "STRUCTURED_QUERY",
            "suggested_strategy": "dedicated_db",
            "response": "Executing structured query against dedicated database/DuckDB/sandbox.",
        }

    @staticmethod
    def generate_sql(query: str, schema_ddl: str, dialect: str = "postgres") -> str:
        q_lower = query.lower()
        if "top" in q_lower and "sales" in q_lower:
            return "SELECT region, SUM(amount) AS total_sales FROM tbl_sales GROUP BY region ORDER BY total_sales DESC LIMIT 20;"
        if "count" in q_lower or "how many" in q_lower:
            return "SELECT COUNT(*) AS total_count FROM tbl_sales LIMIT 20;"
        if "average" in q_lower or "avg" in q_lower:
            return "SELECT AVG(amount) AS average_amount FROM tbl_sales LIMIT 20;"
        return "SELECT * FROM tbl_sales LIMIT 20;"

    @staticmethod
    def generate_pandas_code(query: str) -> str:
        return (
            "import pandas as pd\n"
            "df = pd.read_parquet(file_path)\n"
            "result_df = df.groupby('region')['amount'].sum().reset_index().sort_values('amount', ascending=False).head(20)\n"
        )

    @staticmethod
    def synthesize_answer(
        query: str, evidence: List[Dict[str, Any]], citations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return {
            "answer": f"Based on the dataset, total records analyzed: {len(evidence)}. The top region by sales is North.",
            "evidence_table": evidence[:5],
            "citations": (
                citations if citations is not None else ["[Doc1, Page 2]", "[Doc2, Section 3.1]"]
            ),
            "telemetry": {
                "prompt_tokens": 340,
                "completion_tokens": 120,
                "total_tokens": 460,
                "latency_ms": 45.2,
            },
        }


# -----------------------------------------------------------------------------
# In-Memory Database / SQLite Storage for Testing
# -----------------------------------------------------------------------------
class InMemoryTestDB:
    """Provides an isolated SQLite database mimicking the Postgres metadata catalog."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE datasets (
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE table_metadata (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                table_name TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                description TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                column_count INTEGER NOT NULL DEFAULT 0,
                embedding TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );

            CREATE TABLE column_metadata (
                id TEXT PRIMARY KEY,
                table_id TEXT NOT NULL,
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (table_id) REFERENCES table_metadata(id) ON DELETE CASCADE,
                UNIQUE (table_id, column_name)
            );

            CREATE TABLE document_chunks (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER,
                section_title TEXT,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                embedding TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );

            CREATE TABLE query_logs (
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def execute(self, query: str, params: tuple = ()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        self.conn.commit()
        return cur

    def fetchall(self, query: str, params: tuple = ()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]

    def fetchone(self, query: str, params: tuple = ()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None


# -----------------------------------------------------------------------------
# Pytest Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mock_embeddings():
    return MockEmbeddingProvider()


@pytest.fixture(scope="session")
def mock_llm():
    return MockLLMProvider()


@pytest.fixture(scope="function")
def test_db():
    db = InMemoryTestDB()
    return db


@pytest.fixture(scope="function")
def sample_data_dir(tmp_path):
    """Creates synthetic CSV, Parquet, Excel, PDF, TXT, DOCX, and MD test files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample CSV
    sales_df = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "customer_id": [1, 2, 1, 3, 2, 4, 5, 3],
            "region": ["North", "South", "East", "West", "North", "South", "East", "West"],
            "amount": [250.0, 450.5, 120.0, 980.0, 310.0, 620.0, 150.0, 890.0],
            "quantity": [2, 5, 1, 10, 3, 6, 2, 8],
            "order_date": [
                "2026-01-15",
                "2026-01-18",
                "2026-02-01",
                "2026-02-10",
                "2026-02-15",
                "2026-03-01",
                "2026-03-10",
                "2026-03-15",
            ],
        }
    )
    csv_path = data_dir / "sales_data.csv"
    sales_df.to_csv(csv_path, index=False)

    # 2. Sample Parquet
    customers_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince", "Evan Wright"],
            "tier": ["Gold", "Silver", "Platinum", "Silver", "Gold"],
            "signup_year": [2022, 2023, 2021, 2024, 2023],
            "active": [True, True, False, True, True],
        }
    )
    parquet_path = data_dir / "customers.parquet"
    customers_df.to_parquet(parquet_path, index=False)

    # 3. Sample Excel
    excel_path = data_dir / "inventory.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "product_id": ["P100", "P200", "P300"],
                "stock": [45, 120, 8],
                "unit_cost": [12.5, 34.0, 150.0],
            }
        ).to_excel(writer, sheet_name="Stock", index=False)
        pd.DataFrame(
            {
                "product_id": ["P100", "P200", "P300"],
                "warehouse": ["WH-East", "WH-West", "WH-Central"],
            }
        ).to_excel(writer, sheet_name="Locations", index=False)

    # 4. Sample TXT Document
    txt_path = data_dir / "security_policy.txt"
    txt_content = (
        "Multi-Agent Knowledge Base Platform Security Guidelines.\n\n"
        "Section 1: Authentication & Access Control\n"
        "All API access requires valid bearer tokens. Subprocess execution in Strategy C "
        "must enforce AST whitelisting and resource limits.\n\n"
        "Section 2: Data Encryption & Storage\n"
        "All blob storage files must be encrypted at rest. Vector embeddings in PostgreSQL "
        "use HNSW indexing with cosine distance.\n"
    )
    txt_path.write_text(txt_content, encoding="utf-8")

    # 5. Sample Markdown Document
    md_path = data_dir / "architecture.md"
    md_content = (
        "# Multi-Agent Architecture\n\n"
        "## Ingestion Subsystem\n"
        "Parses structured CSV/Parquet and unstructured PDF/TXT into blob store.\n\n"
        "## Execution Engines\n"
        "Strategy A: PostgreSQL Text2SQL.\n"
        "Strategy B: DuckDB in-memory SQL over Parquet.\n"
        "Strategy C: Sandboxed Python DataFrame execution.\n"
    )
    md_path.write_text(md_content, encoding="utf-8")

    sales_parquet_path = data_dir / "sales.parquet"
    sales_df.to_parquet(sales_parquet_path, index=False)

    return {
        "dir": data_dir,
        "csv": csv_path,
        "parquet": parquet_path,
        "sales_parquet": sales_parquet_path,
        "excel": excel_path,
        "txt": txt_path,
        "md": md_path,
        "sales_df": sales_df,
        "customers_df": customers_df,
    }


@pytest.fixture(scope="function")
def blob_storage_dir(tmp_path):
    storage_path = tmp_path / "blobs"
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path
