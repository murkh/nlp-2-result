"""
Metadata Extractor and Profiling Module for Structured and Unstructured Datasets.
Extracts column statistics, sample values, primary/foreign key relationships,
synthesizes semantic descriptions, and generates 1536-dim vector embeddings.
"""

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.config import Settings, get_settings
from src.database.models import ColumnMetadata, TableMetadata

# =============================================================================
# Embedding Service (Mock, OpenAI, FastEmbed)
# =============================================================================


class EmbeddingService:
    """
    Unified embedding service supporting OpenAI text-embedding-3-small,
    FastEmbed (local ONNX model), and deterministic token-projection Mock embeddings.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.provider = self.settings.embedding_provider
        self.dim = self.settings.embedding_dim
        self._fastembed_model = None
        self._openai_client = None

        if self.provider == "fastembed":
            try:
                from fastembed import TextEmbedding

                self._fastembed_model = TextEmbedding(model_name=self.settings.embedding_model)
            except Exception:
                self.provider = "mock"

        elif self.provider == "openai" and self.settings.openai_api_key:
            try:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
            except Exception:
                self.provider = "mock"

    def embed_text(self, text: str) -> List[float]:
        """Generate a 1536-dimensional vector embedding for a single text."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate 1536-dimensional vector embeddings for a list of texts."""
        if not texts:
            return []

        if self.provider == "openai" and self._openai_client:
            try:
                resp = self._openai_client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=texts,
                    dimensions=self.dim,
                )
                return [item.embedding for item in resp.data]
            except Exception:
                pass  # Fallback to mock

        if self.provider == "fastembed" and self._fastembed_model:
            try:
                embeddings_gen = self._fastembed_model.embed(texts)
                results = []
                for emb in embeddings_gen:
                    vec = emb.tolist()
                    # Pad or truncate to self.dim
                    if len(vec) < self.dim:
                        vec = vec + [0.0] * (self.dim - len(vec))
                    else:
                        vec = vec[: self.dim]
                    results.append(vec)
                return results
            except Exception:
                pass  # Fallback to mock

        # Deterministic semantic hash projection (Mock provider)
        return [self._compute_mock_embedding(t) for t in texts]

    def _compute_mock_embedding(self, text: str) -> List[float]:
        """
        Compute a deterministic 1536-dimensional unit vector using n-gram token projections.
        Ensures texts sharing semantic keywords have high cosine similarity.
        """
        clean_text = text.lower().strip()
        tokens = re.findall(r"\w+", clean_text)
        if not tokens:
            tokens = ["empty"]

        vec = [0.0] * self.dim
        for token in tokens:
            # Deterministic projection of token into vector space
            token_bytes = token.encode("utf-8")
            h1 = int(hashlib.md5(token_bytes).hexdigest(), 16)
            h2 = int(hashlib.sha256(token_bytes).hexdigest(), 16)

            # Activate 8 distinct dimensions per token
            for i in range(8):
                idx = (h1 + (i * 997)) % self.dim
                sign = 1.0 if ((h2 >> (i * 3)) & 1) == 0 else -1.0
                weight = 1.0 + (len(token) * 0.1)
                vec[idx] += sign * weight

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec


# =============================================================================
# Tabular Data Profiler & Metadata Extractor
# =============================================================================


@dataclass
class ColumnProfile:
    """Statistical and semantic profile of a single column."""

    column_name: str
    data_type: str
    is_primary_key: bool
    is_foreign_key: bool
    foreign_target_table: Optional[str]
    foreign_target_column: Optional[str]
    null_percentage: float
    distinct_values_count: int
    sample_values: List[Any]
    description: str


@dataclass
class TableProfile:
    """Statistical and semantic profile of a tabular dataset."""

    table_name: str
    display_name: str
    description: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]


class MetadataExtractor:
    """
    Inspects tabular columns, deduces SQL data types, computes null/distinct stats,
    identifies primary/foreign keys, crafts semantic descriptions, and generates embeddings.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()

    def profile_table(
        self,
        table_name: str,
        column_names: List[str],
        rows: List[List[Any]],
        display_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> TableProfile:
        """
        Generate complete statistical profile and semantic metadata for a tabular dataset.
        """
        row_count = len(rows)
        column_count = len(column_names)
        clean_display_name = (
            display_name or table_name.replace("tbl_", "").replace("_", " ").title()
        )

        col_profiles: List[ColumnProfile] = []

        # Analyze column by column
        for col_idx, col_name in enumerate(column_names):
            col_values = [row[col_idx] for row in rows if col_idx < len(row)]
            profile = self._profile_column(col_name, col_values, row_count, table_name)
            col_profiles.append(profile)

        # Refine primary key detection: if multiple or none marked, pick the best candidate
        pk_candidates = [p for p in col_profiles if p.is_primary_key]
        if len(pk_candidates) > 1:
            # Prefer column named 'id' or ending with '_id'
            for p in pk_candidates:
                if p.column_name.lower() in ("id", f"{table_name.lower()}_id"):
                    for other in pk_candidates:
                        if other != p:
                            other.is_primary_key = False
                    break

        # Synthesize table description if not provided
        if not table_description:
            key_cols = [
                p.column_name
                for p in col_profiles
                if p.is_primary_key
                or p.is_foreign_key
                or "name" in p.column_name.lower()
                or "amount" in p.column_name.lower()
                or "date" in p.column_name.lower()
            ]
            col_summary = ", ".join(key_cols[:6]) if key_cols else ", ".join(column_names[:6])
            table_description = (
                f"Table {table_name} containing {row_count} records with columns: {col_summary}."
            )

        return TableProfile(
            table_name=table_name,
            display_name=clean_display_name,
            description=table_description,
            row_count=row_count,
            column_count=column_count,
            columns=col_profiles,
        )

    def _profile_column(
        self, col_name: str, values: List[Any], total_rows: int, table_name: str
    ) -> ColumnProfile:
        """Profile statistics, data type, key status, and sample values of a single column."""
        clean_name = col_name.strip()
        lower_name = clean_name.lower()

        # Non-null values
        non_null_values = [
            v
            for v in values
            if v is not None
            and str(v).strip() != ""
            and str(v).lower() != "nan"
            and str(v).lower() != "null"
        ]
        null_count = total_rows - len(non_null_values)
        null_percentage = round((null_count / total_rows * 100.0), 2) if total_rows > 0 else 0.0

        # Distinct values count
        distinct_set = set()
        sample_values = []
        for v in non_null_values:
            val_str = str(v)
            if val_str not in distinct_set:
                distinct_set.add(val_str)
                if len(sample_values) < 5:
                    sample_values.append(v)

        distinct_count = len(distinct_set)

        # Deduce data type
        data_type = self._deduce_sql_type(non_null_values, lower_name)

        # Primary / Foreign Key heuristics
        is_pk = False
        is_fk = False
        foreign_table = None
        foreign_col = None

        base_table = table_name.lower().rstrip("s")
        if lower_name in ("id", f"{table_name.lower()}_id", f"{base_table}_id"):
            if null_count == 0 and (distinct_count == total_rows or total_rows <= 1):
                is_pk = True
        elif lower_name.endswith("_id") or lower_name.endswith("id"):
            is_fk = True
            # Extract target table name candidate (e.g. 'customer_id' -> 'customers')
            base_target = lower_name.replace("_id", "").replace("id", "")
            if base_target:
                foreign_table = f"{base_target}s" if not base_target.endswith("s") else base_target
                foreign_col = "id"

        # Unique constraint heuristic
        if (
            not is_pk
            and not is_fk
            and null_count == 0
            and total_rows > 1
            and distinct_count == total_rows
        ):
            if "id" in lower_name or "code" in lower_name or "key" in lower_name:
                is_pk = True

        # Generate semantic description
        description = self._generate_column_description(
            clean_name, data_type, is_pk, is_fk, foreign_table, sample_values
        )

        return ColumnProfile(
            column_name=clean_name,
            data_type=data_type,
            is_primary_key=is_pk,
            is_foreign_key=is_fk,
            foreign_target_table=foreign_table,
            foreign_target_column=foreign_col,
            null_percentage=null_percentage,
            distinct_values_count=distinct_count,
            sample_values=sample_values,
            description=description,
        )

    def _deduce_sql_type(self, values: List[Any], col_name: str) -> str:
        """Infer PostgreSQL-compatible column data type from sample values and column name."""
        if not values:
            return "TEXT"

        # Check boolean
        if all(
            isinstance(v, bool) or str(v).lower() in ("true", "false", "t", "f", "1", "0")
            for v in values[:50]
        ):
            if (
                col_name.startswith("is_")
                or col_name.startswith("has_")
                or all(isinstance(v, bool) for v in values[:10])
            ):
                return "BOOLEAN"

        # Check integer
        is_int = True
        max_val = 0
        for v in values[:100]:
            try:
                iv = int(str(v).strip())
                max_val = max(max_val, abs(iv))
            except (ValueError, TypeError):
                is_int = False
                break
        if is_int:
            return "BIGINT" if max_val > 2147483647 else "INTEGER"

        # Check float / double
        is_float = True
        for v in values[:100]:
            try:
                float(str(v).strip())
            except (ValueError, TypeError):
                is_float = False
                break
        if is_float:
            return "DOUBLE PRECISION"

        # Check datetime / date
        if (
            "date" in col_name
            or "time" in col_name
            or "created" in col_name
            or "updated" in col_name
        ):
            sample_str = str(values[0]).strip()
            if re.match(r"^\d{4}-\d{2}-\d{2}(T|\s)\d{2}:\d{2}", sample_str):
                return "TIMESTAMPTZ"
            if re.match(r"^\d{4}-\d{2}-\d{2}$", sample_str):
                return "DATE"

        # Check JSON
        if any(isinstance(v, (dict, list)) for v in values[:10]):
            return "JSONB"

        return "TEXT"

    def _generate_column_description(
        self,
        col_name: str,
        data_type: str,
        is_pk: bool,
        is_fk: bool,
        foreign_table: Optional[str],
        sample_values: List[Any],
    ) -> str:
        """Construct a semantic description for a column used during schema retrieval."""
        parts = []
        if is_pk:
            parts.append(f"Primary key unique identifier for this record.")
        elif is_fk and foreign_table:
            parts.append(f"Foreign key reference linking to {foreign_table}.")
        else:
            human_name = col_name.replace("_", " ").title()
            parts.append(f"{human_name} field ({data_type}).")

        if sample_values:
            samples_preview = ", ".join([repr(s) for s in sample_values[:3]])
            parts.append(f"Representative values: [{samples_preview}].")

        return " ".join(parts)

    def extract_and_embed_metadata(
        self,
        dataset_id: str,
        table_name: str,
        column_names: List[str],
        rows: List[List[Any]],
        display_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> Tuple[TableMetadata, List[ColumnMetadata]]:
        """
        Profile table and columns, compute vector embeddings, and return
        ready-to-insert TableMetadata and ColumnMetadata model instances.
        """
        profile = self.profile_table(
            table_name=table_name,
            column_names=column_names,
            rows=rows,
            display_name=display_name,
            table_description=table_description,
        )

        # Generate table embedding
        table_text_to_embed = f"Table: {profile.table_name} - {profile.display_name}. {profile.description} Columns: {', '.join(column_names)}"
        table_embedding = self.embedding_service.embed_text(table_text_to_embed)

        table_meta = TableMetadata(
            dataset_id=dataset_id,
            table_name=profile.table_name,
            display_name=profile.display_name,
            description=profile.description,
            row_count=profile.row_count,
            column_count=profile.column_count,
            embedding=table_embedding,
        )

        # Generate column embeddings
        col_texts_to_embed = [
            f"Table {profile.table_name}, Column {c.column_name} ({c.data_type}): {c.description}"
            for c in profile.columns
        ]
        col_embeddings = self.embedding_service.embed_texts(col_texts_to_embed)

        column_metas: List[ColumnMetadata] = []
        for col_prof, emb in zip(profile.columns, col_embeddings):
            col_meta = ColumnMetadata(
                table_id=table_meta.id,
                column_name=col_prof.column_name,
                data_type=col_prof.data_type,
                is_primary_key=col_prof.is_primary_key,
                is_foreign_key=col_prof.is_foreign_key,
                foreign_target_table=col_prof.foreign_target_table,
                foreign_target_column=col_prof.foreign_target_column,
                null_percentage=col_prof.null_percentage,
                distinct_values_count=col_prof.distinct_values_count,
                sample_values=col_prof.sample_values,
                description=col_prof.description,
                embedding=emb,
            )
            column_metas.append(col_meta)

        return table_meta, column_metas
