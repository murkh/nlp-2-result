"""
Structured Execution Equivalence Evaluation Suite.
Evaluates Ground-Truth execution equivalence between generated SQL/Python DataFrame queries
and golden baseline outputs across Strategy A (PostgreSQL), Strategy B (DuckDB), and Strategy C (Pandas).

Computes:
  1. DataFrame Normalization (casing, column order, 4-decimal float rounding, row sorting)
  2. Execution Equivalence via pd.testing.assert_frame_equal with 1e-4 tolerance
  3. First-Pass Syntax Success Rate (%)
  4. Result Set Equivalence Rate (%)
  5. Latency breakdown (p50, p95, p99, mean, min, max)
  6. Token cost estimation ($)
"""

from dataclasses import dataclass, field
import math
import numbers
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.evaluation.compat import pd, np, DataFrame, Series


# Model pricing table per 1M tokens ($ USD)
MODEL_TOKEN_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"prompt": 5.00, "completion": 15.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
}


def estimate_token_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4o-mini",
) -> float:
    """
    Estimate monetary USD cost for LLM prompt and completion token consumption.
    """
    pricing = MODEL_TOKEN_PRICING.get(model.lower(), MODEL_TOKEN_PRICING["gpt-4o-mini"])
    prompt_cost = (prompt_tokens / 1_000_000.0) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000.0) * pricing["completion"]
    return round(prompt_cost + completion_cost, 6)


def normalize_dataframe(df_input: Any) -> pd.DataFrame:
    """
    Normalizes DataFrame structure, column names, column ordering, float rounding,
    string whitespace, and row sorting for deterministic equivalence comparison.

    Accepts:
      - pd.DataFrame
      - Polars DataFrame (.to_pandas())
      - List of dict records
      - Dict with 'rows' and 'columns' or Dict of column lists
      - None or empty objects
    """
    if df_input is None:
        return pd.DataFrame()

    # Handle TabularResult or Dict with rows/columns
    if hasattr(df_input, "rows") and hasattr(df_input, "columns"):
        rows = getattr(df_input, "rows") or []
        cols = getattr(df_input, "columns") or None
        df = pd.DataFrame(rows, columns=cols)
    elif isinstance(df_input, dict):
        if "rows" in df_input:
            df = pd.DataFrame(df_input.get("rows", []), columns=df_input.get("columns"))
        else:
            df = pd.DataFrame(df_input)
    elif hasattr(df_input, "to_pandas"):
        df = df_input.to_pandas()
    elif isinstance(df_input, (list, tuple)):
        df = pd.DataFrame(list(df_input))
    elif isinstance(df_input, pd.DataFrame):
        df = df_input.copy()
    else:
        try:
            df = pd.DataFrame(df_input)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        # Normalize column names even if empty
        if len(df.columns) > 0:
            df.columns = [str(c).strip().lower() for c in df.columns]
            df = df.reindex(sorted(df.columns), axis=1)
        return df.reset_index(drop=True)

    df_clean = df.copy()

    # 1. Normalize column names: strip whitespace, lowercase
    df_clean.columns = [str(c).strip().lower() for c in df_clean.columns]

    # 2. Sort columns alphabetically
    df_clean = df_clean.reindex(sorted(df_clean.columns), axis=1)

    # 3. Process each column according to data type
    for col in df_clean.columns:
        series = df_clean[col]

        # Check numeric types
        if pd.api.types.is_numeric_dtype(series):
            # If all non-null values are integer-like, cast float to float rounded to 4 decimals
            # Replace NaNs/Infs consistently
            series_num = pd.to_numeric(series, errors="coerce")
            df_clean[col] = series_num.round(4)
        elif pd.api.types.is_bool_dtype(series):
            df_clean[col] = series.astype(bool)
        elif pd.api.types.is_datetime64_any_dtype(series):
            df_clean[col] = pd.to_datetime(series, errors="coerce")
        else:
            # Check if object column actually contains numbers
            try:
                converted = pd.to_numeric(series, errors="raise")
                df_clean[col] = converted.round(4)
            except (ValueError, TypeError):
                # Standardize strings
                def clean_str(val):
                    if val is None or pd.isna(val) or str(val).strip().lower() in ("none", "null", "nan", ""):
                        return np.nan
                    return str(val).strip()

                df_clean[col] = series.apply(clean_str)

    # 4. Row sorting across all columns for order-independent result set comparison
    sort_cols = list(df_clean.columns)
    if sort_cols:
        try:
            # Fillna temporarily with string representation for stable row sorting
            sort_key_df = df_clean.astype(str)
            sorted_indices = sort_key_df.sort_values(by=sort_cols).index
            df_clean = df_clean.loc[sorted_indices].reset_index(drop=True)
        except Exception:
            try:
                df_clean = df_clean.sort_values(by=sort_cols).reset_index(drop=True)
            except Exception:
                df_clean = df_clean.reset_index(drop=True)
    else:
        df_clean = df_clean.reset_index(drop=True)

    return df_clean


def check_execution_equivalence(
    df_generated: Any,
    df_golden: Any,
    tolerance: float = 1e-4,
) -> Tuple[bool, str]:
    """
    Validates Ground-Truth execution equivalence between generated and golden DataFrames.
    Tolerates column reordering, row permutations, and float variations within 1e-4.

    Returns:
      (True, "Execution Equivalent") on match
      (False, error_reason) on mismatch
    """
    norm_gen = normalize_dataframe(df_generated)
    norm_gold = normalize_dataframe(df_golden)

    # 1. Shape comparison
    if norm_gen.shape != norm_gold.shape:
        return False, f"Shape mismatch: Generated {norm_gen.shape} vs Golden {norm_gold.shape}"

    # 2. Column schema comparison
    if list(norm_gen.columns) != list(norm_gold.columns):
        return False, f"Column mismatch: Generated {list(norm_gen.columns)} vs Golden {list(norm_gold.columns)}"

    # 3. Empty DataFrames are equal
    if norm_gen.empty and norm_gold.empty:
        return True, "Execution Equivalent"

    # 4. Deep value equivalence comparison using pandas assert_frame_equal
    try:
        pd.testing.assert_frame_equal(
            norm_gen,
            norm_gold,
            check_like=True,
            atol=tolerance,
            rtol=tolerance,
            check_dtype=False,
            check_exact=False,
        )
        return True, "Execution Equivalent"
    except AssertionError as e:
        err_msg = str(e).strip()
        return False, err_msg


def assert_frame_equivalence(
    df_generated: Any,
    df_golden: Any,
    tolerance: float = 1e-4,
) -> None:
    """
    Asserts DataFrame execution equivalence, raising AssertionError on mismatch.
    """
    is_equiv, msg = check_execution_equivalence(df_generated, df_golden, tolerance=tolerance)
    if not is_equiv:
        raise AssertionError(f"DataFrame Execution Equivalence Failed: {msg}")


def calculate_syntax_first_pass_rate(
    attempts: Sequence[Union[bool, Dict[str, Any]]],
) -> float:
    """
    Computes first-pass syntax success rate from a list of boolean success flags or result dicts.
    Returns float in [0.0, 1.0].
    """
    if not attempts:
        return 0.0

    success_count = 0
    for a in attempts:
        if isinstance(a, bool):
            if a:
                success_count += 1
        elif isinstance(a, dict):
            # Check error or status
            has_error = bool(a.get("error")) or a.get("status") == "FAILED" or a.get("success") is False
            if not has_error:
                success_count += 1
        else:
            if bool(a):
                success_count += 1

    rate = success_count / len(attempts)
    return round(rate, 4)


def calculate_equivalence_rate(
    results: Sequence[Union[bool, Tuple[bool, str], Dict[str, Any]]],
) -> float:
    """
    Computes percentage/ratio of test cases reaching execution equivalence.
    Returns float in [0.0, 1.0].
    """
    if not results:
        return 0.0

    equiv_count = 0
    for r in results:
        if isinstance(r, bool):
            if r:
                equiv_count += 1
        elif isinstance(r, (tuple, list)) and len(r) >= 1:
            if bool(r[0]):
                equiv_count += 1
        elif isinstance(r, dict):
            if r.get("equivalent") or r.get("is_equivalent") or r.get("consensus"):
                equiv_count += 1
        else:
            if bool(r):
                equiv_count += 1

    rate = equiv_count / len(results)
    return round(rate, 4)


def compute_latency_statistics(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """
    Computes comprehensive latency breakdown in milliseconds (mean, median/p50, p95, p99, min, max).
    """
    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "total_ms": 0.0,
        }

    arr = np.array(latencies_ms, dtype=float)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "total_ms": round(float(np.sum(arr)), 2),
    }


# =============================================================================
# Structured Benchmark Evaluation Engine
# =============================================================================

@dataclass
class StructuredBenchmarkRecord:
    """Detailed result for a single structured evaluation case."""
    test_id: str
    query: str
    engine: str
    syntax_success: bool
    is_equivalent: bool
    error: Optional[str] = None
    diff_reason: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class StructuredBenchmarkResult:
    """Aggregated outcome of structured benchmark evaluation."""
    total_cases: int
    syntax_first_pass_rate: float
    equivalence_rate: float
    latency_stats: Dict[str, float]
    token_summary: Dict[str, Any]
    per_engine_stats: Dict[str, Dict[str, Any]]
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "syntax_first_pass_rate": self.syntax_first_pass_rate,
            "equivalence_rate": self.equivalence_rate,
            "latency_stats": self.latency_stats,
            "token_summary": self.token_summary,
            "per_engine_stats": self.per_engine_stats,
            "details": self.details,
        }

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "syntax_first_pass_rate_pct": round(self.syntax_first_pass_rate * 100, 2),
            "equivalence_rate_pct": round(self.equivalence_rate * 100, 2),
            "mean_latency_ms": self.latency_stats.get("mean_ms", 0.0),
            "p95_latency_ms": self.latency_stats.get("p95_ms", 0.0),
            "total_tokens": self.token_summary.get("total_tokens", 0),
            "total_cost_usd": self.token_summary.get("total_cost_usd", 0.0),
        }

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.details)

    def to_dataframe(self) -> pd.DataFrame:
        return self.to_pandas()


class StructuredEquivalenceEvaluator:
    """
    Evaluates structured Text2SQL / DataFrame execution benchmark suites.
    """

    def __init__(self, tolerance: float = 1e-4, default_model: str = "gpt-4o-mini"):
        self.tolerance = tolerance
        self.default_model = default_model

    def evaluate_case(
        self,
        query: str,
        df_generated: Any,
        df_golden: Any,
        engine: str = "dedicated_db",
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: Optional[str] = None,
        test_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> StructuredBenchmarkRecord:
        """
        Evaluate a single test case comparing generated DataFrame with golden baseline.
        """
        tid = test_id or f"case_{abs(hash(query)) % 100000}"
        syntax_ok = error is None and df_generated is not None

        is_equiv = False
        diff_reason = None

        if syntax_ok:
            is_equiv, diff_reason = check_execution_equivalence(
                df_generated=df_generated,
                df_golden=df_golden,
                tolerance=self.tolerance,
            )
        else:
            diff_reason = f"Syntax/Execution Error: {error}"

        total_tok = prompt_tokens + completion_tokens
        cost = estimate_token_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model or self.default_model,
        )

        return StructuredBenchmarkRecord(
            test_id=tid,
            query=query,
            engine=engine,
            syntax_success=syntax_ok,
            is_equivalent=is_equiv,
            error=error,
            diff_reason=diff_reason,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tok,
            estimated_cost_usd=cost,
        )

    def evaluate_benchmark(
        self,
        test_cases: Sequence[Dict[str, Any]],
    ) -> StructuredBenchmarkResult:
        """
        Evaluate a benchmark suite of structured execution test cases.
        """
        records: List[StructuredBenchmarkRecord] = []

        for i, tc in enumerate(test_cases):
            query = tc.get("query") or tc.get("question") or f"Test Case #{i+1}"
            df_gen = tc.get("df_generated") if tc.get("df_generated") is not None else (tc.get("generated_df") if tc.get("generated_df") is not None else tc.get("generated_result"))
            df_gold = tc.get("df_golden") if tc.get("df_golden") is not None else (tc.get("golden_df") if tc.get("golden_df") is not None else tc.get("golden_result"))
            engine = tc.get("engine") or tc.get("strategy") or "unknown"
            latency = float(tc.get("latency_ms") or tc.get("latency") or 0.0)
            p_tok = int(tc.get("prompt_tokens") or 0)
            c_tok = int(tc.get("completion_tokens") or 0)
            err = tc.get("error")
            tid = tc.get("test_id") or f"case_{i+1}"
            model = tc.get("model") or self.default_model

            record = self.evaluate_case(
                query=query,
                df_generated=df_gen,
                df_golden=df_gold,
                engine=engine,
                latency_ms=latency,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                error=err,
                test_id=tid,
                model=model,
            )
            records.append(record)

        total_cases = len(records)
        if total_cases == 0:
            return StructuredBenchmarkResult(
                total_cases=0,
                syntax_first_pass_rate=0.0,
                equivalence_rate=0.0,
                latency_stats=compute_latency_statistics([]),
                token_summary={"total_tokens": 0, "total_cost_usd": 0.0},
                per_engine_stats={},
                details=[],
            )

        syntax_passes = sum(1 for r in records if r.syntax_success)
        equiv_passes = sum(1 for r in records if r.is_equivalent)

        syntax_rate = round(syntax_passes / total_cases, 4)
        equiv_rate = round(equiv_passes / total_cases, 4)

        all_latencies = [r.latency_ms for r in records]
        latency_stats = compute_latency_statistics(all_latencies)

        total_prompt_tok = sum(r.prompt_tokens for r in records)
        total_comp_tok = sum(r.completion_tokens for r in records)
        total_all_tok = sum(r.total_tokens for r in records)
        total_cost = round(sum(r.estimated_cost_usd for r in records), 6)

        token_summary = {
            "prompt_tokens": total_prompt_tok,
            "completion_tokens": total_comp_tok,
            "total_tokens": total_all_tok,
            "total_cost_usd": total_cost,
        }

        # Per-engine breakdown
        engines = sorted(list(set(r.engine for r in records)))
        per_engine_stats: Dict[str, Dict[str, Any]] = {}

        for eng in engines:
            eng_records = [r for r in records if r.engine == eng]
            e_total = len(eng_records)
            e_syntax = sum(1 for r in eng_records if r.syntax_success)
            e_equiv = sum(1 for r in eng_records if r.is_equivalent)
            e_lat = [r.latency_ms for r in eng_records]

            per_engine_stats[eng] = {
                "total_cases": e_total,
                "syntax_first_pass_rate": round(e_syntax / e_total, 4) if e_total else 0.0,
                "equivalence_rate": round(e_equiv / e_total, 4) if e_total else 0.0,
                "mean_latency_ms": round(float(np.mean(e_lat)), 2) if e_lat else 0.0,
                "p95_latency_ms": round(float(np.percentile(e_lat, 95)), 2) if e_lat else 0.0,
                "total_tokens": sum(r.total_tokens for r in eng_records),
                "total_cost_usd": round(sum(r.estimated_cost_usd for r in eng_records), 6),
            }

        details_list = [
            {
                "test_id": r.test_id,
                "query": r.query,
                "engine": r.engine,
                "syntax_success": r.syntax_success,
                "is_equivalent": r.is_equivalent,
                "error": r.error,
                "diff_reason": r.diff_reason,
                "latency_ms": r.latency_ms,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "estimated_cost_usd": r.estimated_cost_usd,
            }
            for r in records
        ]

        return StructuredBenchmarkResult(
            total_cases=total_cases,
            syntax_first_pass_rate=syntax_rate,
            equivalence_rate=equiv_rate,
            latency_stats=latency_stats,
            token_summary=token_summary,
            per_engine_stats=per_engine_stats,
            details=details_list,
        )
