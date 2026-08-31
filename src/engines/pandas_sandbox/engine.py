"""
Strategy C: Sandboxed Python DataFrame Execution Engine.
Orchestrates prompt generation, AST security validation,
isolated subprocess execution, and data-backed response synthesis.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas import (
    DecisionStep,
    ExecutionMetrics,
    QueryPandasSandboxRequest,
    QueryPandasSandboxResponse,
    SandboxSecurityReport,
    TabularResult,
    ThinkingProcess,
    TokenUsage,
)
from src.config import Settings, get_settings
from src.database.connection import DatabaseManager, get_db_manager
from src.database.models import QueryLog
from src.engines.pandas_sandbox.ast_validator import validate_python_code
from src.engines.pandas_sandbox.runner import execute_sandboxed_code
from src.llm import get_openai_client
from src.pruning.schema_pruner import PrunedSchemaContext, TwoStageSchemaPruner
from src.storage.blob_store import BlobStorageManager, get_blob_manager


class PandasSandboxEngine:
    """
    Strategy C: Sandboxed Python DataFrame Engine.
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
        self._openai_client = get_openai_client(self.settings)

    def execute_query(
        self,
        request: QueryPandasSandboxRequest,
    ) -> QueryPandasSandboxResponse:
        """
        Execute Strategy C query workflow:
        1. Prune schema to get relevant file paths and column metadata.
        2. Generate Python transformation code.
        3. Validate AST security before spawning subprocess.
        4. Execute code in isolated subprocess with timeout and memory limits.
        5. Extract standardized tabular result from JSON output.
        6. Synthesize natural language answer.
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

        if not pruned_context.table_names or not pruned_context.file_paths:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            return QueryPandasSandboxResponse(
                query=query_text,
                answer="No structured datasets found. Please upload a CSV, Parquet, or Excel file in the Ingestion Hub first.",
                python_code="# No datasets available",
                tabular_result=TabularResult(columns=[], rows=[], row_count=0),
                security_report=SandboxSecurityReport(ast_passed=True, violations=[], exit_code=0),
                metrics=ExecutionMetrics(query_generation_ms=0.0, total_latency_ms=total_lat),
                token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
                error="No structured datasets found",
            )

        # 2. Generate Python code with LLM thought extraction
        gen_start = time.perf_counter()
        python_code, llm_thought, gen_tokens = self._generate_python_code(
            query_text, pruned_context
        )
        gen_latency_ms = (time.perf_counter() - gen_start) * 1000.0

        # Fallback: ensure dataset loader is present if code references df without defining it
        if (
            ("df." in python_code or "df[" in python_code)
            and "pd.read_" not in python_code
            and pruned_context.table_names
        ):
            first_tbl = pruned_context.table_names[0]
            first_path = pruned_context.file_paths.get(first_tbl, "")
            if first_path:
                if first_path.endswith(".parquet"):
                    python_code = (
                        f"import pandas as pd\ndf = pd.read_parquet({first_path!r})\n" + python_code
                    )
                elif first_path.endswith(".xlsx") or first_path.endswith(".xls"):
                    python_code = (
                        f"import pandas as pd\ndf = pd.read_excel({first_path!r})\n" + python_code
                    )
                else:
                    python_code = (
                        f"import pandas as pd\ndf = pd.read_csv({first_path!r})\n" + python_code
                    )

        # 3. AST Security validation
        is_safe, err_msg = validate_python_code(python_code)
        if not is_safe:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            sec_report = SandboxSecurityReport(
                ast_passed=False,
                violations=[err_msg],
                timeout_occurred=False,
                memory_limit_exceeded=False,
                exit_code=-1,
            )
            return QueryPandasSandboxResponse(
                query=query_text,
                answer=f"Execution blocked by security sandbox: {err_msg}",
                python_code=python_code,
                tabular_result=TabularResult(columns=[], rows=[], row_count=0),
                security_report=sec_report,
                metrics=ExecutionMetrics(
                    query_generation_ms=gen_latency_ms, total_latency_ms=total_lat
                ),
                token_usage=TokenUsage(
                    prompt_tokens=gen_tokens[0], completion_tokens=gen_tokens[1]
                ),
                error=err_msg,
            )

        # 4. Subprocess execution
        exec_start = time.perf_counter()
        timeout_sec = self.settings.sandbox_timeout_sec
        max_mem = self.settings.sandbox_max_memory_mb
        success, result_data, stderr_msg, exit_code = execute_sandboxed_code(
            code=python_code,
            timeout_seconds=timeout_sec,
            max_memory_mb=max_mem,
        )
        exec_latency_ms = (time.perf_counter() - exec_start) * 1000.0

        timed_out = exit_code == 124 or "timed out" in stderr_msg.lower()
        sec_report = SandboxSecurityReport(
            ast_passed=True,
            violations=[],
            timeout_occurred=timed_out,
            memory_limit_exceeded=False,
            exit_code=exit_code,
        )

        if not success:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            return QueryPandasSandboxResponse(
                query=query_text,
                answer=f"Python sandbox execution failed: {stderr_msg}",
                python_code=python_code,
                tabular_result=TabularResult(columns=[], rows=[], row_count=0),
                security_report=sec_report,
                metrics=ExecutionMetrics(
                    query_generation_ms=gen_latency_ms,
                    engine_execution_ms=exec_latency_ms,
                    total_latency_ms=total_lat,
                ),
                token_usage=TokenUsage(
                    prompt_tokens=gen_tokens[0], completion_tokens=gen_tokens[1]
                ),
                error=stderr_msg,
            )

        # 5. Extract Tabular Result
        cols = result_data.get("columns", [])
        rows = result_data.get("rows", [])
        truncated = result_data.get("truncated", len(rows) >= 20)

        tabular_result = TabularResult(
            columns=cols,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )

        # 6. Synthesize answer
        synth_start = time.perf_counter()
        answer, synth_tokens = self._synthesize_answer(query_text, python_code, tabular_result)
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
                engine="strategy_c_pandas_sandbox",
                status="SUCCESS",
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                latency_ms=metrics.total_latency_ms,
                generated_code=python_code,
            )
        )

        # Construct Thinking Process with dynamic LLM thoughts
        selected_tbls = (
            ", ".join(pruned_context.table_names) if pruned_context.table_names else "None"
        )
        file_summary = (
            ", ".join(f"{t} ({Path(p).name})" for t, p in pruned_context.file_paths.items())
            if pruned_context.file_paths
            else selected_tbls
        )
        py_reason = (
            llm_thought
            or f"Formulated vectorized Pandas DataFrame transformation for '{query_text}' on file(s) [{file_summary}]."
        )

        thinking = ThinkingProcess(
            summary=f"Strategy C generated vectorized Python transformation code, passed AST security validation, and executed in an isolated subprocess sandbox returning {len(rows)} row(s).",
            steps=[
                DecisionStep(
                    step_number=1,
                    title="Blob File & Schema Resolution",
                    choice=f"Resolved blob files for: {selected_tbls}",
                    reasoning=f"Extracted direct storage paths: {file_summary}.",
                    details={"file_paths": pruned_context.file_paths},
                ),
                DecisionStep(
                    step_number=2,
                    title="Vectorized Python Transformation Code Generation",
                    choice="Generated vectorized Pandas script",
                    reasoning=py_reason,
                    details={"code": python_code},
                ),
                DecisionStep(
                    step_number=3,
                    title="AST Security & Sandbox Verification",
                    choice="Passed AST Whitelist & Subprocess Isolation",
                    reasoning="Static analysis verified code contains no forbidden imports (os, sys, subprocess) or dunder escapes. Subprocess allocated CPU watchdog limits.",
                    details={
                        "ast_passed": sec_report.ast_passed,
                        "violations": sec_report.violations,
                        "exit_code": sec_report.exit_code,
                    },
                ),
                DecisionStep(
                    step_number=4,
                    title="DataFrame Result Extraction & Synthesis",
                    choice=f"Extracted {len(rows)} row(s) from JSON protocol",
                    reasoning=f"Parsed standardized sandbox JSON stdout protocol into tabular records and synthesized natural language answer.",
                ),
            ],
        )

        return QueryPandasSandboxResponse(
            query=query_text,
            answer=answer,
            python_code=python_code,
            tabular_result=tabular_result,
            security_report=sec_report,
            thinking_process=thinking,
            metrics=metrics,
            token_usage=token_usage,
        )

    def _generate_python_code(
        self, query: str, context: PrunedSchemaContext
    ) -> Tuple[str, Optional[str], Tuple[int, int]]:
        """Generate sandboxed Python DataFrame code using LLM or deterministic generator."""
        file_paths_str = "\n".join(f"- {tbl}: {p}" for tbl, p in context.file_paths.items())
        prompt = (
            f"You are an expert Python data engineer. Write clean vectorized Pandas transformation code.\n"
            f"Dataset Files:\n{file_paths_str}\n"
            f"Schema:\n{context.ddl_prompt_snippet}\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"1. In a ```thought block, explain your step-by-step reasoning: how you read the file(s), filter, aggregate, and assign result.\n"
            f"2. In a ```python block, write ONLY the executable Python script.\n"
            f"3. You MUST load the dataset file using `pd.read_csv('/exact/path/to/file.csv')` or `pd.read_parquet(...)` using the exact paths given in Dataset Files.\n"
            f"4. Assign the final DataFrame, Series, or scalar output to a variable named `result`.\n"
            f"5. Ensure `result = result.head(20)` if returning a DataFrame."
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
                code = None
                blocks = re.findall(r"```(\w*)\n(.*?)```", raw_text, re.DOTALL)
                for lang, content in blocks:
                    lang_clean = lang.lower().strip()
                    if lang_clean in ("thought", "thinking", "reasoning", "explanation"):
                        thought = content.strip()
                    elif lang_clean in ("python", "py"):
                        code = content.strip()

                if not code:
                    code_match = re.search(r"```(?:python)?(.*?)```", raw_text, re.DOTALL)
                    code = code_match.group(1).strip() if code_match else raw_text.strip()

                comp_tokens = (
                    resp.usage.completion_tokens if resp.usage else max(1, len(raw_text) // 4)
                )
                return code, thought, (prompt_tokens, comp_tokens)
            except Exception:
                pass

        # Deterministic Python code generator
        code = self._deterministic_pandas_code(query, context)
        comp_tokens = max(1, len(code) // 4)
        return code, None, (prompt_tokens, comp_tokens)

    def _deterministic_pandas_code(self, query: str, context: PrunedSchemaContext) -> str:
        """Deterministic Python code generator mapping query intent to Pandas script."""
        if not context.table_names or not context.file_paths:
            return "result = {'status': 1}"

        primary_table = context.table_names[0]
        blob_path = context.file_paths.get(primary_table, "")
        retained_cols = context.retained_columns.get(primary_table, [])
        lower_q = query.lower()

        if blob_path.endswith(".parquet"):
            loader = f"""import pandas as pd
df = pd.read_parquet({blob_path!r})
"""
        else:
            loader = f"""import pandas as pd
df = pd.read_csv({blob_path!r})
"""

        # Logic 1: Count queries
        if "how many" in lower_q or "count" in lower_q or "total number" in lower_q:
            if "status" in lower_q and "status" in retained_cols:
                return loader + """res = df.groupby('status').size().reset_index(name='total_count')
result = res.sort_values(by='total_count', ascending=False).head(20)
"""
            if "completed" in lower_q and "status" in retained_cols:
                return (
                    loader + """count = len(df[df['status'].astype(str).str.lower() == 'completed'])
result = {'completed_count': count}
"""
                )
            return loader + """result = {'total_records': len(df)}
"""

        # Logic 2: Sum / average queries
        amount_col = next(
            (
                c
                for c in retained_cols
                if "amount" in c.lower()
                or "price" in c.lower()
                or "total" in c.lower()
                or "sales" in c.lower()
            ),
            None,
        )
        if ("sum" in lower_q or "total sales" in lower_q or "revenue" in lower_q) and amount_col:
            city_col = next(
                (c for c in retained_cols if "city" in c.lower() or "country" in c.lower()), None
            )
            if city_col and ("by city" in lower_q or "per city" in lower_q or "top" in lower_q):
                return (
                    loader
                    + f"""df[{amount_col!r}] = pd.to_numeric(df[{amount_col!r}], errors='coerce').fillna(0)
res = df.groupby({city_col!r})[{amount_col!r}].sum().reset_index(name='total_revenue')
result = res.sort_values(by='total_revenue', ascending=False).head(20)
"""
                )
            return (
                loader + f"""total = float(pd.to_numeric(df[{amount_col!r}], errors='coerce').sum())
result = {{'total_revenue': total}}
"""
            )

        if ("average" in lower_q or "avg" in lower_q) and amount_col:
            return (
                loader
                + f"""avg_val = float(pd.to_numeric(df[{amount_col!r}], errors='coerce').mean())
result = {{'average_amount': avg_val}}
"""
            )

        # Logic 3: Top N / highest
        if ("highest" in lower_q or "top" in lower_q or "most expensive" in lower_q) and amount_col:
            return (
                loader
                + f"""df[{amount_col!r}] = pd.to_numeric(df[{amount_col!r}], errors='coerce').fillna(0)
result = df.sort_values(by={amount_col!r}, ascending=False).head(20)
"""
            )

        # Default select
        return loader + """result = df.head(20)
"""

    def _synthesize_answer(
        self, query: str, code: str, result: TabularResult
    ) -> Tuple[str, Tuple[int, int]]:
        """Synthesize natural language answer with data evidence."""
        if result.row_count == 0:
            ans = "The Python DataFrame transformation returned 0 records."
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
            ans = f"Sandboxed Python DataFrame result for {col_name.replace('_', ' ')}: **{val_str}**."
            return ans, (30, 15)

        first_row_preview = ", ".join(f"{k}: {v}" for k, v in list(rows[0].items())[:4])
        ans = (
            f"Processed {result.row_count} records via sandboxed Python. "
            f"Top record: [{first_row_preview}]."
        )
        return ans, (40, 20)
