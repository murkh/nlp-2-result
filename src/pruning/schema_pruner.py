"""
Two-Stage Vector Schema Pruner for Token-Efficient Multi-Agent Q&A.
Reduces prompt token consumption by >85% via Stage 1 table vector retrieval
and Stage 2 column vector retrieval with compulsory PK/FK preservation and LIMIT 20 enforcement.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Settings, get_settings
from src.database.connection import DatabaseManager, get_db_manager
from src.ingestion.metadata_extractor import EmbeddingService
from src.storage.blob_store import BlobStorageManager, get_blob_manager

# Column role heuristics. An id alone is not an answer an analyst can act on, so
# the human-readable identifier and the measures are marked and preserved.
DISPLAY_NAME_PATTERN = re.compile(
    r"(^|_)(no|num|number|code|name|title|label|desc|description|status|type)(_|$)"
)
MEASURE_NAME_PATTERN = re.compile(
    r"(^|_)(amount|amt|value|val|total|sum|qty|quantity|price|cost|rate|balance|discount|tax)(_|$)"
)
DATE_NAME_PATTERN = re.compile(r"(^|_)(date|time|timestamp|dt|day|month|year)(_|$)")

_TEXT_TYPES = ("char", "text", "string", "object", "uuid")
_NUMERIC_TYPES = ("int", "float", "double", "decimal", "numeric", "real", "number", "money")
_DATE_TYPES = ("date", "time", "timestamp")

# Bounded, unlike the unconditional PK/FK bypass: a wide fact table must not be
# able to blow the whole token budget on measures.
MAX_DISPLAY_BYPASS_PER_TABLE = 2
MAX_MEASURE_BYPASS_PER_TABLE = 3


def classify_column_role(column_name: str, data_type: str) -> Optional[str]:
    """
    Classify a column as 'display', 'measure', or 'date' from its name and type.

    Generalizes the key-column heuristic in MetadataExtractor.profile_table, which
    computed the same intuition (name/amount/date) only to discard it after
    building a prose description.
    """
    name = (column_name or "").lower()
    dtype = (data_type or "").lower()

    is_text = any(t in dtype for t in _TEXT_TYPES)
    is_numeric = any(t in dtype for t in _NUMERIC_TYPES)
    is_date = any(t in dtype for t in _DATE_TYPES)

    if is_date or (DATE_NAME_PATTERN.search(name) and not is_numeric):
        return "date"
    if is_numeric and MEASURE_NAME_PATTERN.search(name):
        return "measure"
    # Untyped/unknown columns fall back to the name alone rather than being skipped.
    if DISPLAY_NAME_PATTERN.search(name) and (is_text or not is_numeric):
        return "display"
    return None


@dataclass
class PrunedColumn:
    """Retained column in the pruned schema."""

    column_name: str
    data_type: str
    is_primary_key: bool
    is_foreign_key: bool
    sample_values: List[Any]
    description: str
    role: Optional[str] = None


@dataclass
class PrunedTable:
    """Retained table and its pruned columns in the pruned schema."""

    table_id: str
    table_name: str
    display_name: str
    description: str
    blob_path: Optional[str] = None
    columns: List[PrunedColumn] = field(default_factory=list)


@dataclass
class PrunedSchemaContext:
    """Output context produced by the Two-Stage Schema Pruner."""

    table_names: List[str]
    table_ids: List[str]
    ddl_prompt_snippet: str
    file_paths: Dict[str, str]  # table_name -> absolute blob path
    retained_columns: Dict[str, List[str]]
    token_count_pruned: int
    token_count_full: int
    token_savings_percent: float
    column_roles: Dict[str, str] = field(default_factory=dict)  # column_name -> role


def estimate_token_count(text: str) -> int:
    """Estimate token count for a text string (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class TwoStageSchemaPruner:
    """
    Two-Stage Vector Schema Pruner.
    Stage 1: Coarse retrieval of top-K tables using cosine similarity on table embeddings.
    Stage 2: Fine retrieval of relevant columns per table with compulsory PK/FK retention.
    Formats minimal DDL prompt snippet with sample values and LIMIT 20 guardrails.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        blob_manager: Optional[BlobStorageManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        settings: Optional[Settings] = None,
    ):
        self.db_manager = db_manager or get_db_manager()
        self.blob_manager = blob_manager or get_blob_manager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.settings = settings or get_settings()

    def prune_schema(
        self,
        query: str,
        dataset_ids: Optional[List[str]] = None,
        top_k_tables: Optional[int] = None,
        max_cols_per_table: Optional[int] = None,
        total_max_cols: Optional[int] = None,
    ) -> PrunedSchemaContext:
        """
        Execute two-stage schema pruning for a given natural language query.
        """
        k_tables = top_k_tables or self.settings.default_top_k_tables
        cols_per_table = max_cols_per_table or self.settings.default_max_cols_per_table
        max_cols = total_max_cols or self.settings.default_total_max_cols

        # 1. Embed query
        query_embedding = self.embedding_service.embed_text(query)

        # 2. Stage 1: Retrieve candidate tables
        candidate_tables_raw = self.db_manager.search_table_metadata_vectors(
            query_embedding=query_embedding, query_text=query, top_k=k_tables
        )

        if not candidate_tables_raw:
            # Fallback: get any available tables if no embeddings matched
            all_tables = self.db_manager.list_tables()
            for t in all_tables[:k_tables]:
                candidate_tables_raw.append(
                    {
                        "id": t.id,
                        "dataset_id": t.dataset_id,
                        "table_name": t.table_name,
                        "display_name": t.display_name,
                        "description": t.description,
                        "row_count": t.row_count,
                        "column_count": t.column_count,
                        "similarity": 0.5,
                    }
                )

        if not candidate_tables_raw:
            empty_ddl = "-- No structured database tables found.\n"
            return PrunedSchemaContext(
                table_names=[],
                table_ids=[],
                ddl_prompt_snippet=empty_ddl,
                file_paths={},
                retained_columns={},
                token_count_pruned=estimate_token_count(empty_ddl),
                token_count_full=estimate_token_count(empty_ddl),
                token_savings_percent=0.0,
            )

        selected_table_ids = [t["id"] for t in candidate_tables_raw]
        table_map: Dict[str, PrunedTable] = {}

        # Resolve dataset file paths for DuckDB/Pandas engines
        for t in candidate_tables_raw:
            dataset_rec = self.db_manager.get_dataset(t["dataset_id"])
            abs_blob = None
            if dataset_rec:
                try:
                    abs_blob = str(self.blob_manager.get_absolute_path(dataset_rec.blob_path))
                except Exception:
                    abs_blob = dataset_rec.blob_path

            table_map[t["id"]] = PrunedTable(
                table_id=t["id"],
                table_name=t["table_name"],
                display_name=t["display_name"],
                description=t["description"],
                blob_path=abs_blob,
                columns=[],
            )

        # 3. Stage 2: Retrieve candidate columns for selected tables
        col_rows = self.db_manager.search_column_metadata_vectors(
            query_embedding=query_embedding, table_ids=selected_table_ids, query_text=query
        )

        # Group and prune columns per table
        total_retained_cols = 0
        cols_by_table: Dict[str, List[PrunedColumn]] = {tid: [] for tid in selected_table_ids}
        role_bypass_used: Dict[str, Dict[str, int]] = {
            tid: {"display": 0, "measure": 0} for tid in selected_table_ids
        }

        for crow in col_rows:
            tid = crow["table_id"]
            if tid not in cols_by_table:
                continue

            is_key = crow["is_primary_key"] or crow["is_foreign_key"]
            role = classify_column_role(crow["column_name"], crow["data_type"])
            current_table_col_count = len(cols_by_table[tid])
            within_budget = current_table_col_count < cols_per_table and total_retained_cols < max_cols

            # A PK/FK bypasses the budget unconditionally. A display/measure column
            # bypasses it too, but only up to a per-table bound: without this, the id
            # survives pruning while the column that makes it readable does not.
            role_bypass = False
            if not within_budget and not is_key and role in ("display", "measure"):
                limit = (
                    MAX_DISPLAY_BYPASS_PER_TABLE
                    if role == "display"
                    else MAX_MEASURE_BYPASS_PER_TABLE
                )
                role_bypass = role_bypass_used[tid][role] < limit

            if within_budget or is_key or role_bypass:
                p_col = PrunedColumn(
                    column_name=crow["column_name"],
                    data_type=crow["data_type"],
                    is_primary_key=crow["is_primary_key"],
                    is_foreign_key=crow["is_foreign_key"],
                    sample_values=crow["sample_values"][:3] if crow["sample_values"] else [],
                    description=crow["description"],
                    role=role,
                )
                cols_by_table[tid].append(p_col)
                total_retained_cols += 1
                if role_bypass:
                    role_bypass_used[tid][role] += 1

        for tid, cols in cols_by_table.items():
            table_map[tid].columns = cols

        # 4. Generate compact DDL prompt snippet with LIMIT 20 guardrails
        pruned_ddl = self._build_pruned_ddl_prompt(list(table_map.values()))

        # 5. Compute full unpruned schema DDL to measure token savings
        full_ddl = self._build_full_unpruned_ddl()

        token_pruned = estimate_token_count(pruned_ddl)
        token_full = max(token_pruned, estimate_token_count(full_ddl))
        savings_percent = (
            round(((token_full - token_pruned) / token_full * 100.0), 2) if token_full > 0 else 0.0
        )

        table_names = [t.table_name for t in table_map.values()]
        file_paths = {t.table_name: t.blob_path for t in table_map.values() if t.blob_path}
        retained_columns = {
            t.table_name: [c.column_name for c in t.columns] for t in table_map.values()
        }
        column_roles = {
            c.column_name: c.role for t in table_map.values() for c in t.columns if c.role
        }

        return PrunedSchemaContext(
            table_names=table_names,
            table_ids=selected_table_ids,
            ddl_prompt_snippet=pruned_ddl,
            file_paths=file_paths,
            retained_columns=retained_columns,
            token_count_pruned=token_pruned,
            token_count_full=token_full,
            token_savings_percent=savings_percent,
            column_roles=column_roles,
        )

    def _build_pruned_ddl_prompt(self, tables: List[PrunedTable]) -> str:
        """Construct the prompt snippet containing pruned DDL with constraints."""
        lines = [
            "-- ============================================================================",
            "-- PRUNED DATABASE SCHEMA (Token-Efficient Context for Current Query)",
            "-- QUERY CONSTRAINTS:",
            "-- 1. Read-only queries only. Never run DROP, DELETE, UPDATE, INSERT, ALTER.",
            "-- 2. MUST append 'LIMIT 20' to SELECT statements unless doing scalar aggregation (COUNT, SUM, AVG).",
            "-- ============================================================================",
            "",
        ]

        for t in tables:
            lines.append(f"-- Table: {t.table_name} ({t.display_name})")
            lines.append(f"-- Description: {t.description}")
            if t.blob_path:
                lines.append(f"-- Blob File Path: {t.blob_path}")
            lines.append(f"CREATE TABLE {t.table_name} (")

            col_defs = []
            for col in t.columns:
                pk_flag = " PRIMARY KEY" if col.is_primary_key else ""
                fk_flag = " -- FOREIGN KEY" if col.is_foreign_key else ""
                samples_str = (
                    f" -- samples: {col.sample_values}"
                    if col.sample_values and not col.is_foreign_key
                    else ""
                )
                role_str = f" -- role: {col.role}" if col.role else ""
                col_defs.append(
                    f"    {col.column_name} {col.data_type}{pk_flag},"
                    f"{fk_flag}{role_str}{samples_str}"
                )

            if not col_defs:
                col_defs.append("    -- (Columns filtered out by pruning)")

            lines.append("\n".join(col_defs))
            lines.append(");\n")

        return "\n".join(lines)

    def _build_full_unpruned_ddl(self) -> str:
        """Construct the complete unpruned schema for token savings benchmark."""
        all_tables = self.db_manager.list_tables()
        if not all_tables:
            return ""

        lines = ["-- FULL UNPRUNED DATABASE SCHEMA\n"]
        for t in all_tables:
            lines.append(f"CREATE TABLE {t.table_name} (")
            cols = self.db_manager.get_columns_for_table(t.id)
            col_defs = []
            for col in cols:
                pk_flag = " PRIMARY KEY" if col.is_primary_key else ""
                samples_str = f" -- samples: {col.sample_values}" if col.sample_values else ""
                col_defs.append(f"    {col.column_name} {col.data_type}{pk_flag},{samples_str}")
            lines.append("\n".join(col_defs))
            lines.append(");\n")

        return "\n".join(lines)
