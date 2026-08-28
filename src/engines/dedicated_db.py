"""
Strategy A: PostgreSQL Dedicated Database Query Engine.
Generates PostgreSQL dialect Text2SQL using two-stage schema pruning,
executes queries in read-only transactions with LIMIT 20 guardrails,
and synthesizes evidence-backed natural language answers.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas import (
    DecisionStep,
    ExecutionMetrics,
    QueryDedicatedDBRequest,
    QueryDedicatedDBResponse,
    TabularResult,
    ThinkingProcess,
    TokenUsage,
)
from src.config import Settings, get_settings
from src.database.connection import DatabaseManager, get_db_manager
from src.database.models import QueryLog
from src.pruning.schema_pruner import PrunedSchemaContext, TwoStageSchemaPruner
from src.storage.blob_store import BlobStorageManager, get_blob_manager


FORBIDDEN_SQL_PATTERNS = [
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
    r"\b(ATTACH|DETACH|VACUUM|REINDEX|COPY|IMPORT|EXPORT)\b",
    r"\bpg_catalog\b",
    r"\binformation_schema\b",
]


def enforce_sql_limit(sql: str, max_limit: int = 20) -> str:
    """Enforce LIMIT <= max_limit on SQL SELECT queries."""
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


def validate_sql_security(sql: str) -> Tuple[bool, Optional[str]]:
    """Validate that SQL query is strictly read-only and free of destructive statements."""
    for pat in FORBIDDEN_SQL_PATTERNS:
        if re.search(pat, sql, re.IGNORECASE):
            match = re.search(pat, sql, re.IGNORECASE).group(0)
            return False, f"Forbidden SQL statement detected: {match.upper()}"
    return True, None


class DedicatedDBEngine:
    """
    Strategy A: PostgreSQL Text2SQL Engine.
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
        request: QueryDedicatedDBRequest,
    ) -> QueryDedicatedDBResponse:
        """
        Execute Strategy A query workflow:
        1. Prune schema to relevant tables/columns.
        2. Generate PostgreSQL SQL query.
        3. Validate security and enforce LIMIT 20.
        4. Execute query in read-only mode.
        5. Synthesize natural language answer with data evidence.
        6. Compute latency metrics and token usage.
        7. Log execution telemetry.
        """
        start_time = time.perf_counter()
        query_text = request.query

        # 1. Two-stage schema pruning
        pruned_context = self.schema_pruner.prune_schema(
            query=query_text,
            dataset_ids=request.dataset_ids,
        )

        # 2. Generate SQL query with LLM thought extraction
        gen_start = time.perf_counter()
        sql_query, llm_thought, gen_tokens = self._generate_sql(query_text, pruned_context)
        gen_latency_ms = (time.perf_counter() - gen_start) * 1000.0

        # 3. Validate security and guardrails
        is_safe, sec_err = validate_sql_security(sql_query)
        if not is_safe:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            return QueryDedicatedDBResponse(
                query=query_text,
                answer=f"Execution blocked: {sec_err}",
                sql_query=sql_query,
                tabular_result=TabularResult(columns=[], rows=[], row_count=0),
                metrics=ExecutionMetrics(query_generation_ms=gen_latency_ms, total_latency_ms=total_lat),
                token_usage=TokenUsage(prompt_tokens=gen_tokens[0], completion_tokens=gen_tokens[1]),
                error=sec_err,
            )

        sql_query = enforce_sql_limit(sql_query, max_limit=20)

        # 4. Execute query against database
        exec_start = time.perf_counter()
        cols, rows, exec_err = self._run_query_safe(sql_query)
        exec_latency_ms = (time.perf_counter() - exec_start) * 1000.0

        if exec_err:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            return QueryDedicatedDBResponse(
                query=query_text,
                answer=f"Database execution error: {exec_err}",
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
                engine="strategy_a_dedicated_db",
                status="SUCCESS",
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                latency_ms=metrics.total_latency_ms,
                generated_code=sql_query,
            )
        )

        # Construct Thinking Process with dynamic LLM thoughts
        selected_tbls = ", ".join(pruned_context.table_names) if pruned_context.table_names else "None"
        retained_summary = ", ".join(
            f"{t}: [{', '.join(cols[:4])}{'...' if len(cols) > 4 else ''}]"
            for t, cols in pruned_context.retained_columns.items()
        )
        pruning_reason = f"Vector search selected {len(pruned_context.table_names)} table(s). Retained schema: {retained_summary}."
        sql_reason = llm_thought or f"Formulated PostgreSQL query for '{query_text}' against table(s) [{selected_tbls}]."

        thinking = ThinkingProcess(
            summary=f"Strategy A executed PostgreSQL Text2SQL against table(s) [{selected_tbls}] and returned {len(dict_rows)} row(s).",
            steps=[
                DecisionStep(
                    step_number=1,
                    title="Two-Stage Schema Pruning",
                    choice=f"Selected table(s): {selected_tbls}",
                    reasoning=pruning_reason,
                    details={"retained_columns": pruned_context.retained_columns},
                ),
                DecisionStep(
                    step_number=2,
                    title="PostgreSQL Query Generation",
                    choice="Generated PostgreSQL SQL Query",
                    reasoning=sql_reason,
                    details={"sql": sql_query},
                ),
                DecisionStep(
                    step_number=3,
                    title="Security & Guardrail Check",
                    choice="Passed Read-Only Validation & LIMIT 20",
                    reasoning="Verified query contains no mutations (DROP, DELETE, UPDATE) and enforces 20-row safety ceiling.",
                    details={"limit_enforced": 20},
                ),
                DecisionStep(
                    step_number=4,
                    title="Execution & Grounded Synthesis",
                    choice=f"Retrieved {len(dict_rows)} row(s)",
                    reasoning=f"Database returned {len(dict_rows)} record(s). Grounded natural language response in result evidence.",
                ),
            ],
        )

        return QueryDedicatedDBResponse(
            query=query_text,
            answer=answer,
            sql_query=sql_query,
            tabular_result=tabular_result,
            thinking_process=thinking,
            metrics=metrics,
            token_usage=token_usage,
        )

    def _run_query_safe(self, sql_query: str) -> Tuple[List[str], List[Tuple[Any, ...]], Optional[str]]:
        """Execute query wrapped in read-only block."""
        try:
            cols, rows = self.db_manager.execute_sql_query(sql_query)
            return cols, rows, None
        except Exception as e:
            return [], [], str(e)

    def _generate_sql(self, query: str, context: PrunedSchemaContext) -> Tuple[str, Optional[str], Tuple[int, int]]:
        """Generate PostgreSQL SQL query using LLM or rule-based generator."""
        prompt = (
            f"You are an expert PostgreSQL data analyst. Write a valid read-only PostgreSQL query.\n"
            f"Schema:\n{context.ddl_prompt_snippet}\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"1. In a ```thought block, explain your step-by-step reasoning: which tables/columns you selected, filtering/aggregation logic, and why.\n"
            f"2. In a ```sql block, write ONLY the valid PostgreSQL SELECT query."
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
                    elif lang_clean in ("sql", "postgresql", "postgres"):
                        sql = content.strip()

                if not sql:
                    sql_match = re.search(r"```(?:sql)?(.*?)```", raw_text, re.DOTALL)
                    sql = sql_match.group(1).strip() if sql_match else raw_text.strip()

                comp_tokens = resp.usage.completion_tokens if resp.usage else max(1, len(raw_text) // 4)
                return sql, thought, (prompt_tokens, comp_tokens)
            except Exception:
                pass

        # Deterministic Text2SQL generator
        sql = self._deterministic_text2sql(query, context)
        comp_tokens = max(1, len(sql) // 4)
        return sql, None, (prompt_tokens, comp_tokens)

    def _deterministic_text2sql(self, query: str, context: PrunedSchemaContext) -> str:
        """Deterministic Text2SQL generator mapping query intent to valid PostgreSQL query."""
        if not context.table_names:
            return "SELECT 1 AS status;"

        primary_table = context.table_names[0]
        retained_cols = context.retained_columns.get(primary_table, [])
        lower_q = query.lower()

        # Check for count query
        if "how many" in lower_q or "count" in lower_q or "total number" in lower_q:
            if "status" in lower_q and "status" in retained_cols:
                return f'SELECT "status", COUNT(*) AS total_count FROM "{primary_table}" GROUP BY "status" ORDER BY total_count DESC LIMIT 20;'
            if "completed" in lower_q and "status" in retained_cols:
                return f'SELECT COUNT(*) AS completed_count FROM "{primary_table}" WHERE LOWER("status") = \'completed\';'
            return f'SELECT COUNT(*) AS total_records FROM "{primary_table}";'

        # Check for sum / average query
        amount_col = next((c for c in retained_cols if "amount" in c.lower() or "price" in c.lower() or "total" in c.lower() or "sales" in c.lower()), None)
        if ("sum" in lower_q or "total sales" in lower_q or "revenue" in lower_q) and amount_col:
            city_col = next((c for c in retained_cols if "city" in c.lower() or "country" in c.lower()), None)
            if city_col and ("by city" in lower_q or "per city" in lower_q or "top" in lower_q):
                return f'SELECT "{city_col}", SUM("{amount_col}") AS total_revenue FROM "{primary_table}" GROUP BY "{city_col}" ORDER BY total_revenue DESC LIMIT 20;'
            return f'SELECT SUM("{amount_col}") AS total_revenue FROM "{primary_table}";'

        if ("average" in lower_q or "avg" in lower_q) and amount_col:
            return f'SELECT AVG("{amount_col}") AS average_amount FROM "{primary_table}";'

        # Check for top N / highest
        if ("highest" in lower_q or "top" in lower_q or "most expensive" in lower_q) and amount_col:
            return f'SELECT * FROM "{primary_table}" ORDER BY "{amount_col}" DESC LIMIT 20;'

        # General SELECT
        selected_cols = [f'"{c}"' for c in retained_cols[:6]] if retained_cols else ["*"]
        cols_clause = ", ".join(selected_cols)
        return f'SELECT {cols_clause} FROM "{primary_table}" LIMIT 20;'

    def _synthesize_answer(
        self, query: str, sql: str, result: TabularResult
    ) -> Tuple[str, Tuple[int, int]]:
        """Synthesize a grounded natural language answer from the tabular result."""
        if result.row_count == 0:
            ans = "The query returned no matching records from the database."
            return ans, (20, 10)

        rows = result.rows
        # Single scalar row output (e.g. COUNT, SUM, AVG)
        if result.row_count == 1 and len(result.columns) == 1:
            col_name = result.columns[0]
            val = rows[0].get(col_name)
            if isinstance(val, float):
                val_str = f"{val:,.2f}"
            else:
                val_str = str(val)
            ans = f"Based on the database records, the {col_name.replace('_', ' ')} is **{val_str}**."
            return ans, (30, 15)

        # Multi-row tabular summary
        first_row_preview = ", ".join(f"{k}: {v}" for k, v in list(rows[0].items())[:4])
        ans = (
            f"Retrieved {result.row_count} records from the database. "
            f"Top result: [{first_row_preview}]."
        )
        return ans, (40, 20)
