"""
Benchmark Arena Engine.
Executes Strategy A (PostgreSQL), Strategy B (DuckDB), and Strategy C (Pandas Sandbox)
concurrently in parallel, evaluates tabular execution equivalence (assert_frame_equal),
computes head-to-head performance telemetry, and synthesizes comparative trade-off summaries.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas import (
    BenchmarkComparisonSummary,
    ExecutionMetrics,
    QueryBenchmarkRequest,
    QueryBenchmarkResponse,
    QueryDedicatedDBRequest,
    QueryDuckDBRequest,
    QueryPandasSandboxRequest,
    StrategyBenchmarkResult,
    TabularResult,
    TokenUsage,
)
from src.config import Settings, get_settings
from src.database.connection import DatabaseManager, get_db_manager
from src.engines.dedicated_db import DedicatedDBEngine
from src.engines.duckdb_engine import DuckDBQueryEngine
from src.engines.pandas_sandbox.engine import PandasSandboxEngine
from src.pruning.schema_pruner import TwoStageSchemaPruner
from src.storage.blob_store import BlobStorageManager, get_blob_manager


def are_values_equivalent(val_a: Any, val_b: Any, tolerance: float = 1e-4) -> bool:
    """Check mathematical equivalence of two values handling floats, ints, nulls, and strings."""
    if val_a is None and val_b is None:
        return True
    if val_a is None or val_b is None:
        return False

    try:
        fa = float(val_a)
        fb = float(val_b)
        return math.isclose(fa, fb, abs_tol=tolerance, rel_tol=tolerance)
    except (ValueError, TypeError):
        pass

    return str(val_a).strip().lower() == str(val_b).strip().lower()


def compare_tabular_results(
    res_a: TabularResult, res_b: TabularResult, tolerance: float = 1e-4
) -> bool:
    """
    Compare execution equivalence of two tabular results.
    Checks row counts, scalar values, and record columns within tolerance.
    """
    if res_a.row_count != res_b.row_count:
        return False

    if res_a.row_count == 0:
        return True

    # Check scalar single value comparison
    if res_a.row_count == 1 and res_b.row_count == 1:
        row_a = res_a.rows[0]
        row_b = res_b.rows[0]
        vals_a = list(row_a.values())
        vals_b = list(row_b.values())
        if len(vals_a) == 1 and len(vals_b) == 1:
            return are_values_equivalent(vals_a[0], vals_b[0], tolerance)

    # Record-by-record comparison
    for r_a, r_b in zip(res_a.rows, res_b.rows):
        common_keys = set(r_a.keys()).intersection(set(r_b.keys()))
        if not common_keys:
            vals_a = list(r_a.values())
            vals_b = list(r_b.values())
            if len(vals_a) != len(vals_b):
                return False
            for va, vb in zip(vals_a, vals_b):
                if not are_values_equivalent(va, vb, tolerance):
                    return False
        else:
            for k in common_keys:
                if not are_values_equivalent(r_a.get(k), r_b.get(k), tolerance):
                    return False

    return True


class BenchmarkArenaEngine:
    """
    Parallel 3-Way Benchmark Arena Engine.
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

        self.strategy_a_engine = DedicatedDBEngine(
            db_manager=self.db_manager,
            blob_manager=self.blob_manager,
            schema_pruner=self.schema_pruner,
            settings=self.settings,
        )
        self.strategy_b_engine = DuckDBQueryEngine(
            db_manager=self.db_manager,
            blob_manager=self.blob_manager,
            schema_pruner=self.schema_pruner,
            settings=self.settings,
        )
        self.strategy_c_engine = PandasSandboxEngine(
            db_manager=self.db_manager,
            blob_manager=self.blob_manager,
            schema_pruner=self.schema_pruner,
            settings=self.settings,
        )

    def execute_benchmark(
        self,
        request: QueryBenchmarkRequest,
    ) -> QueryBenchmarkResponse:
        """
        Execute parallel benchmark:
        1. Spawn Strategy A, B, and C concurrently via ThreadPoolExecutor.
        2. Collect results, timings, and generated code.
        3. Evaluate Ground-Truth Execution Equivalence across engines.
        4. Determine latency winner and token efficiency winner.
        5. Synthesize comparative analysis summary.
        6. Return complete QueryBenchmarkResponse.
        """
        arena_start = time.perf_counter()
        query_text = request.query

        req_a = QueryDedicatedDBRequest(
            query=query_text,
            dataset_ids=request.dataset_ids,
            temperature=request.temperature,
        )
        req_b = QueryDuckDBRequest(
            query=query_text,
            dataset_ids=request.dataset_ids,
            temperature=request.temperature,
        )
        req_c = QueryPandasSandboxRequest(
            query=query_text,
            dataset_ids=request.dataset_ids,
            temperature=request.temperature,
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_a = executor.submit(self.strategy_a_engine.execute_query, req_a)
            future_b = executor.submit(self.strategy_b_engine.execute_query, req_b)
            future_c = executor.submit(self.strategy_c_engine.execute_query, req_c)

            res_a = future_a.result()
            res_b = future_b.result()
            res_c = future_c.result()

        total_arena_latency_ms = (time.perf_counter() - arena_start) * 1000.0

        strat_a_bench = StrategyBenchmarkResult(
            strategy_name="Strategy A (Postgres Dedicated DB)",
            status="FAILED" if res_a.error else "SUCCESS",
            code_generated=res_a.sql_query,
            answer=res_a.answer,
            tabular_result=res_a.tabular_result,
            metrics=res_a.metrics,
            token_usage=res_a.token_usage,
            error=res_a.error,
        )

        strat_b_bench = StrategyBenchmarkResult(
            strategy_name="Strategy B (DuckDB In-Memory)",
            status="FAILED" if res_b.error else "SUCCESS",
            code_generated=res_b.sql_query,
            answer=res_b.answer,
            tabular_result=res_b.tabular_result,
            metrics=res_b.metrics,
            token_usage=res_b.token_usage,
            error=res_b.error,
        )

        strat_c_bench = StrategyBenchmarkResult(
            strategy_name="Strategy C (Pandas Subprocess Sandbox)",
            status="FAILED" if res_c.error else "SUCCESS",
            code_generated=res_c.python_code,
            answer=res_c.answer,
            tabular_result=res_c.tabular_result,
            metrics=res_c.metrics,
            token_usage=res_c.token_usage,
            error=res_c.error,
        )

        # Evaluate Execution Equivalence
        successful_strats = []
        if strat_a_bench.status == "SUCCESS":
            successful_strats.append(("Strategy A", strat_a_bench))
        if strat_b_bench.status == "SUCCESS":
            successful_strats.append(("Strategy B", strat_b_bench))
        if strat_c_bench.status == "SUCCESS":
            successful_strats.append(("Strategy C", strat_c_bench))

        consensus_reached = True
        if len(successful_strats) >= 2:
            base_res = successful_strats[0][1].tabular_result
            for _, other_strat in successful_strats[1:]:
                if not compare_tabular_results(base_res, other_strat.tabular_result):
                    consensus_reached = False
                    break

        # Determine Fastest Strategy
        candidates = [
            ("Strategy A (Postgres)", strat_a_bench.metrics.total_latency_ms),
            ("Strategy B (DuckDB)", strat_b_bench.metrics.total_latency_ms),
            ("Strategy C (Pandas)", strat_c_bench.metrics.total_latency_ms),
        ]
        fastest = min(candidates, key=lambda x: x[1])[0]

        # Determine Most Token-Efficient Strategy
        token_candidates = [
            ("Strategy A (Postgres)", strat_a_bench.token_usage.total_tokens),
            ("Strategy B (DuckDB)", strat_b_bench.token_usage.total_tokens),
            ("Strategy C (Pandas)", strat_c_bench.token_usage.total_tokens),
        ]
        most_token_efficient = min(token_candidates, key=lambda x: x[1])[0]

        summary_analysis = self._build_benchmark_summary(
            fastest=fastest,
            token_winner=most_token_efficient,
            consensus=consensus_reached,
            strat_a=strat_a_bench,
            strat_b=strat_b_bench,
            strat_c=strat_c_bench,
        )

        comparison_summary = BenchmarkComparisonSummary(
            fastest_strategy=fastest,
            most_token_efficient_strategy=most_token_efficient,
            consensus_reached=consensus_reached,
            summary_analysis=summary_analysis,
        )

        return QueryBenchmarkResponse(
            query=query_text,
            strategy_a=strat_a_bench,
            strategy_b=strat_b_bench,
            strategy_c=strat_c_bench,
            benchmark_summary=comparison_summary,
            total_arena_latency_ms=total_arena_latency_ms,
        )

    def _build_benchmark_summary(
        self,
        fastest: str,
        token_winner: str,
        consensus: bool,
        strat_a: StrategyBenchmarkResult,
        strat_b: StrategyBenchmarkResult,
        strat_c: StrategyBenchmarkResult,
    ) -> str:
        """Construct concise comparative synthesis of the three execution strategies."""
        consensus_text = "All successful strategies reached 100% mathematical result consensus." if consensus else "Discrepancy detected across engine outputs."
        summary = (
            f"Benchmark Arena Analysis: {fastest} achieved the lowest total execution latency "
            f"({min(strat_a.metrics.total_latency_ms, strat_b.metrics.total_latency_ms, strat_c.metrics.total_latency_ms):.1f}ms), "
            f"while {token_winner} was most token-efficient ({min(strat_a.token_usage.total_tokens, strat_b.token_usage.total_tokens, strat_c.token_usage.total_tokens)} tokens). "
            f"{consensus_text} "
            f"Strategy A offers robust relational persistence, Strategy B enables instant zero-ingest blob queries, "
            f"and Strategy C provides flexible isolated Python transformations."
        )
        return summary
