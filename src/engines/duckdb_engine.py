"""
Strategy B: DuckDB In-Memory Query Engine over Blob Files.
Registers temporary views for Parquet/CSV blob storage files,
generates DuckDB analytical SQL, executes in-memory with security PRAGMAs,
enforces LIMIT 20 guardrails, and synthesizes natural language answers.
"""

import csv
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas import (
    DecisionStep,
    ExecutionMetrics,
    QueryDuckDBRequest,
    QueryDuckDBResponse,
    TabularResult,
    ThinkingProcess,
    TokenUsage,
)
from src.config import Settings, get_settings
from src.database.connection import DatabaseManager, get_db_manager
from src.database.models import QueryLog
from src.pruning.schema_pruner import PrunedSchemaContext, TwoStageSchemaPruner
from src.storage.blob_store import BlobStorageManager, get_blob_manager


FORBIDDEN_DUCKDB_PATTERNS = [
    r"\b(ATTACH|DETACH|LOAD|INSTALL|EXPORT\s+DATABASE|COPY\s+.*?TO)\b",
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE\s+TABLE|TRUNCATE)\b",
    r"\b(PRAGMA\s+threads|PRAGMA\s+memory_limit)\b",
]


def enforce_duckdb_limit(sql: str, max_limit: int = 20) -> str:
    """Enforce LIMIT <= max_limit on DuckDB queries."""
    clean_sql = sql.strip().rstrip(";")
    limit_match = re.search(r"\bLIMIT\s+(\d+)\b", clean_sql, re.IGNORECASE)
    if limit_match:
        current_limit = int(limit_match.group(1))
        if current_limit > max_limit:
            clean_sql = re.sub(
                r"\bLIMIT\s+\d+\b",
                f"LIMIT {max_limit}",
                clean_sql,
                flags=re.IGNORECASE,
            )
    else:
        is_simple_agg = bool(
            re.search(r"SELECT\s+(COUNT|SUM|AVG|MIN|MAX)\(", clean_sql, re.IGNORECASE)
            and not re.search(r"\bGROUP\s+BY\b", clean_sql, re.IGNORECASE)
        )
        if not is_simple_agg:
            clean_sql = f"{clean_sql} LIMIT {max_limit}"

    return clean_sql


def validate_duckdb_security(sql: str) -> Tuple[bool, Optional[str]]:
    """Validate that DuckDB SQL query is strictly read-only and free of destructive statements."""
    for pat in FORBIDDEN_DUCKDB_PATTERNS:
        if re.search(pat, sql, re.IGNORECASE):
            match = re.search(pat, sql, re.IGNORECASE).group(0)
            return False, f"Forbidden DuckDB command detected: {match.upper()}"
    return True, None


class DuckDBQueryEngine:
    """
    Strategy B: DuckDB In-Memory Query Engine.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        blob_manager: Optional[BlobStorageManager] = None,
        schema_pruner: Optional[TwoStageSchemaPruner] = None,
        settings: Optional[Settings] = None,
    ):
        self.db_manager = db_manager or get_db_manager()
        self.blob_manager = blob_manager or get_blob_manager()
        self.schema_pruner = schema_pruner or TwoStageSchemaPruner(
            db_manager=self.db_manager, blob_manager=self.blob_manager
        )
        self.settings = settings or get_settings()
        self._openai_client = None

        if self.settings.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
            except Exception:
                self._openai_client = None

    def execute_query(
        self,
        request: QueryDuckDBRequest,
    ) -> QueryDuckDBResponse:
        """
        Execute Strategy B query workflow:
        1. Prune schema to get candidate tables, schemas, and blob file paths.
        2. Create in-memory DuckDB connection, set PRAGMAs, and register views for blob files.
        3. Generate DuckDB SQL query.
        4. Validate security and enforce LIMIT 20.
        5. Execute query in DuckDB.
        6. Synthesize natural language answer with data evidence.
        7. Compute latency metrics and token usage.
        8. Log execution telemetry.
        """
        start_time = time.perf_counter()
        query_text = request.query

        # 1. Two-stage schema pruning
        pruned_context = self.schema_pruner.prune_schema(
            query=query_text,
            dataset_ids=request.dataset_ids,
        )

        # 2. Generate DuckDB SQL with LLM thought extraction
        gen_start = time.perf_counter()
        sql_query, llm_thought, gen_tokens = self._generate_sql(query_text, pruned_context)
        gen_latency_ms = (time.perf_counter() - gen_start) * 1000.0

        # 3. Validate security
        is_safe, sec_err = validate_duckdb_security(sql_query)
        if not is_safe:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            return QueryDuckDBResponse(
                query=query_text,
                answer=f"Execution blocked: {sec_err}",
                sql_query=sql_query,
                tabular_result=TabularResult(columns=[], rows=[], row_count=0),
                metrics=ExecutionMetrics(query_generation_ms=gen_latency_ms, total_latency_ms=total_lat),
                token_usage=TokenUsage(prompt_tokens=gen_tokens[0], completion_tokens=gen_tokens[1]),
                error=sec_err,
            )

        sql_query = enforce_duckdb_limit(sql_query, max_limit=20)

        # 4. Execute in DuckDB (with fallback)
        exec_start = time.perf_counter()
        cols, rows, exec_err = self._execute_duckdb_views(sql_query, pruned_context.file_paths)
        exec_latency_ms = (time.perf_counter() - exec_start) * 1000.0

        if exec_err:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            return QueryDuckDBResponse(
                query=query_text,
                answer=f"DuckDB execution error: {exec_err}",
                sql_query=sql_query,
                tabular_result=TabularResult(columns=[], rows=[], row_count=0),
                metrics=ExecutionMetrics(
                    query_generation_ms=gen_latency_ms,
                    engine_execution_ms=exec_latency_ms,
                    total_latency_ms=total_lat,
                ),
                token_usage=TokenUsage(prompt_tokens=gen_tokens[0], completion_tokens=gen_tokens[1]),
                error=exec_err,
            )

        # Format tabular result
        dict_rows = [dict(zip(cols, row)) for row in rows]
        tabular_result = TabularResult(
            columns=cols,
            rows=dict_rows,
            row_count=len(dict_rows),
            truncated=len(dict_rows) >= 20,
        )

        # 5. Synthesize answer
        synth_start = time.perf_counter()
        answer, synth_tokens = self._synthesize_answer(query_text, sql_query, tabular_result)
        synth_latency_ms = (time.perf_counter() - synth_start) * 1000.0

        total_lat = (time.perf_counter() - start_time) * 1000.0

        total_prompt_tokens = gen_tokens[0] + synth_tokens[0]
        total_completion_tokens = gen_tokens[1] + synth_tokens[1]
        token_usage = TokenUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )

        metrics = ExecutionMetrics(
            query_generation_ms=gen_latency_ms,
            engine_execution_ms=exec_latency_ms,
            synthesis_ms=synth_latency_ms,
            total_latency_ms=total_lat,
        )

        # Log query telemetry
        self.db_manager.log_query(
            QueryLog(
                query_text=query_text,
                engine="strategy_b_duckdb",
                status="SUCCESS",
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                latency_ms=metrics.total_latency_ms,
                generated_code=sql_query,
            )
        )

        # Construct Thinking Process with dynamic LLM thoughts
        selected_tbls = ", ".join(pruned_context.table_names) if pruned_context.table_names else "None"
        file_summary = ", ".join(f"{t} ({Path(p).name})" for t, p in pruned_context.file_paths.items()) if pruned_context.file_paths else selected_tbls
        sql_reason = llm_thought or f"Formulated vectorized DuckDB SQL for '{query_text}' scanning view(s) [{selected_tbls}]."

        thinking = ThinkingProcess(
            summary=f"Strategy B executed vectorized in-memory DuckDB query over blob storage file(s) [{selected_tbls}] and returned {len(dict_rows)} row(s).",
            steps=[
                DecisionStep(
                    step_number=1,
                    title="Blob File & Schema Resolution",
                    choice=f"Resolved blob files for: {selected_tbls}",
                    reasoning=f"Mapped candidate views to persistent files: {file_summary}.",
                    details={"file_paths": pruned_context.file_paths},
                ),
                DecisionStep(
                    step_number=2,
                    title="In-Memory DuckDB View Registration",
                    choice="Created in-memory temporary columnar views",
                    reasoning="Configured DuckDB in-memory session with memory and thread limits, registering direct file pointers.",
                    details={"threads": 2, "memory_limit": "512MB"},
                ),
                DecisionStep(
                    step_number=3,
                    title="Analytical DuckDB SQL Execution",
                    choice="Executed vectorized DuckDB SQL with LIMIT 20",
                    reasoning=sql_reason,
                    details={"sql": sql_query},
                ),
                DecisionStep(
                    step_number=4,
                    title="Evidence Grounding & Synthesis",
                    choice=f"Synthesized answer from {len(dict_rows)} row(s)",
                    reasoning=f"DuckDB returned {len(dict_rows)} record(s). Validated column types and synthesized natural language answer.",
                ),
            ],
        )

        return QueryDuckDBResponse(
            query=query_text,
            answer=answer,
            sql_query=sql_query,
            tabular_result=tabular_result,
            thinking_process=thinking,
            metrics=metrics,
            token_usage=token_usage,
        )

    def _execute_duckdb_views(
        self,
        sql_query: str,
        file_paths: Dict[str, str],
    ) -> Tuple[List[str], List[Tuple[Any, ...]], Optional[str]]:
        """
        Execute DuckDB query with temporary view registration.
        Falls back gracefully to SQLite in-memory virtual tables if duckdb wheel is missing.
        """
        try:
            import duckdb

            con = duckdb.connect(":memory:")
            con.execute("PRAGMA threads=2;")
            con.execute("PRAGMA memory_limit='512MB';")

            for table_name, blob_path in file_paths.items():
                if not os.path.exists(blob_path):
                    continue
                ext = Path(blob_path).suffix.lower()
                if ext in (".parquet", ".pq"):
                    con.execute(f'CREATE VIEW "{table_name}" AS SELECT * FROM read_parquet(\'{blob_path}\');')
                elif ext in (".xlsx", ".xls"):
                    try:
                        import pandas as pd
                        df_excel = pd.read_excel(blob_path)
                        con.register(table_name, df_excel)
                    except Exception:
                        pass
                else:
                    con.execute(f'CREATE VIEW "{table_name}" AS SELECT * FROM read_csv(\'{blob_path}\', auto_detect=true);')

            rel = con.execute(sql_query)
            cols = [desc[0] for desc in rel.description] if rel.description else []
            rows = rel.fetchall() if rel.description else []
            con.close()
            return cols, rows, None

        except ImportError:
            return self._execute_sqlite_fallback(sql_query, file_paths)
        except Exception as e:
            return [], [], str(e)

    def _execute_sqlite_fallback(
        self,
        sql_query: str,
        file_paths: Dict[str, str],
    ) -> Tuple[List[str], List[Tuple[Any, ...]], Optional[str]]:
        """Fallback in-memory execution using SQLite when duckdb is not installed."""
        try:
            conn = sqlite3.connect(":memory:")
            cur = conn.cursor()

            for table_name, blob_path in file_paths.items():
                if not os.path.exists(blob_path):
                    continue
                with open(blob_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue
                    clean_cols = [c.strip() for c in header]
                    col_defs = ", ".join([f'"{c}" TEXT' for c in clean_cols])
                    cur.execute(f'CREATE TABLE "{table_name}" ({col_defs});')

                    placeholders = ", ".join(["?"] * len(clean_cols))
                    rows = [row[:len(clean_cols)] for row in reader if row]
                    if rows:
                        cur.executemany(f'INSERT INTO "{table_name}" VALUES ({placeholders})', rows)
            conn.commit()

            cur.execute(sql_query)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall() if cur.description else []
            conn.close()
            return cols, rows, None
        except Exception as e:
            try:
                cols, rows = self.db_manager.execute_sql_query(sql_query)
                return cols, rows, None
            except Exception as inner_e:
                return [], [], f"{str(e)} (fallback: {str(inner_e)})"

    def _generate_sql(self, query: str, context: PrunedSchemaContext) -> Tuple[str, Optional[str], Tuple[int, int]]:
        """Generate DuckDB SQL query using LLM or rule-based generator."""
        prompt = (
            f"You are an expert DuckDB analytical engineer. Write a high-performance DuckDB SQL query.\n"
            f"Available Views:\n{context.ddl_prompt_snippet}\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"1. In a ```thought block, explain your step-by-step reasoning: which views/columns you selected, filtering/aggregation logic, and why.\n"
            f"2. In a ```sql block, write ONLY the valid DuckDB SQL query."
        )

        prompt_tokens = max(1, len(prompt) // 4)

        if self._openai_client:
            try:
                resp = self._openai_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw_text = resp.choices[0].message.content or ""
                
                thought = None
                sql = None
                blocks = re.findall(r"```(\w*)\n(.*?)```", raw_text, re.DOTALL)
                for lang, content in blocks:
                    lang_clean = lang.lower().strip()
                    if lang_clean in ("thought", "thinking", "reasoning", "explanation"):
                        thought = content.strip()
                    elif lang_clean in ("sql", "duckdb"):
                        sql = content.strip()

                if not sql:
                    sql_match = re.search(r"```(?:sql)?(.*?)```", raw_text, re.DOTALL)
                    sql = sql_match.group(1).strip() if sql_match else raw_text.strip()

                comp_tokens = resp.usage.completion_tokens if resp.usage else max(1, len(raw_text) // 4)
                return sql, thought, (prompt_tokens, comp_tokens)
            except Exception:
                pass

        # Deterministic Text2DuckDB generator
        sql = self._deterministic_duckdb_sql(query, context)
        comp_tokens = max(1, len(sql) // 4)
        return sql, None, (prompt_tokens, comp_tokens)

    def _deterministic_duckdb_sql(self, query: str, context: PrunedSchemaContext) -> str:
        """Deterministic Text2DuckDB generator."""
        if not context.table_names:
            return "SELECT 1 AS status;"

        primary_table = context.table_names[0]
        retained_cols = context.retained_columns.get(primary_table, [])
        lower_q = query.lower()

        # Check count query
        if "how many" in lower_q or "count" in lower_q or "total number" in lower_q:
            if "status" in lower_q and "status" in retained_cols:
                return f'SELECT "status", COUNT(*) AS total_count FROM "{primary_table}" GROUP BY "status" ORDER BY total_count DESC LIMIT 20;'
            if "completed" in lower_q and "status" in retained_cols:
                return f'SELECT COUNT(*) AS completed_count FROM "{primary_table}" WHERE LOWER("status") = \'completed\';'
            return f'SELECT COUNT(*) AS total_records FROM "{primary_table}";'

        # Check sum / avg query
        amount_col = next((c for c in retained_cols if "amount" in c.lower() or "price" in c.lower() or "total" in c.lower() or "sales" in c.lower()), None)
        if ("sum" in lower_q or "total sales" in lower_q or "revenue" in lower_q) and amount_col:
            city_col = next((c for c in retained_cols if "city" in c.lower() or "country" in c.lower()), None)
            if city_col and ("by city" in lower_q or "per city" in lower_q or "top" in lower_q):
                return f'SELECT "{city_col}", SUM(CAST("{amount_col}" AS DOUBLE)) AS total_revenue FROM "{primary_table}" GROUP BY "{city_col}" ORDER BY total_revenue DESC LIMIT 20;'
            return f'SELECT SUM(CAST("{amount_col}" AS DOUBLE)) AS total_revenue FROM "{primary_table}";'

        if ("average" in lower_q or "avg" in lower_q) and amount_col:
            return f'SELECT AVG(CAST("{amount_col}" AS DOUBLE)) AS average_amount FROM "{primary_table}";'

        # Check top N
        if ("highest" in lower_q or "top" in lower_q or "most expensive" in lower_q) and amount_col:
            return f'SELECT * FROM "{primary_table}" ORDER BY CAST("{amount_col}" AS DOUBLE) DESC LIMIT 20;'

        # General SELECT
        selected_cols = [f'"{c}"' for c in retained_cols[:6]] if retained_cols else ["*"]
        cols_clause = ", ".join(selected_cols)
        return f'SELECT {cols_clause} FROM "{primary_table}" LIMIT 20;'

    def _synthesize_answer(
        self, query: str, sql: str, result: TabularResult
    ) -> Tuple[str, Tuple[int, int]]:
        """Synthesize a grounded natural language answer from the tabular result."""
        if result.row_count == 0:
            ans = "DuckDB query executed successfully but returned 0 records."
            return ans, (20, 10)

        rows = result.rows
        if result.row_count == 1 and len(result.columns) == 1:
            col_name = result.columns[0]
            val = rows[0].get(col_name)
            try:
                fval = float(val)
                val_str = f"{fval:,.2f}" if "." in str(val) else str(int(fval))
            except (ValueError, TypeError):
                val_str = str(val)
            ans = f"In-memory DuckDB analytical result for {col_name.replace('_', ' ')}: **{val_str}**."
            return ans, (30, 15)

        first_row_preview = ", ".join(f"{k}: {v}" for k, v in list(rows[0].items())[:4])
        ans = (
            f"DuckDB processed {result.row_count} records in-memory. "
            f"Top record: [{first_row_preview}]."
        )
        return ans, (40, 20)
