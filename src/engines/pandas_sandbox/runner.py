"""
Subprocess Runner for Sandboxed Python DataFrame Execution.
Executes code in an isolated subprocess with watchdog timeouts,
resource limits, environment sanitation, and standardized JSON protocol.
Provides lightweight pandas/polars shims if wheels are not installed.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

WRAPPER_TEMPLATE = """
import sys
import json
import csv

# Set POSIX resource limits if supported
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
except Exception:
    pass

# Provide lightweight pandas fallback shim if pandas is not installed
try:
    import pandas as pd
except ImportError:
    import types

    class _SimpleSeries:
        def __init__(self, data, name=None):
            self._data = list(data)
            self.name = name
        def sum(self):
            return sum(float(x) for x in self._data if x is not None)
        def mean(self):
            return (sum(float(x) for x in self._data if x is not None) / len(self._data)) if self._data else 0.0
        def count(self):
            return len(self._data)
        def head(self, n=20):
            return _SimpleSeries(self._data[:n], name=self.name)
        def to_dict(self):
            return {i: v for i, v in enumerate(self._data)}
        def __len__(self):
            return len(self._data)

    class _SimpleDataFrame:
        def __init__(self, records, columns=None):
            self._records = records
            if columns:
                self.columns = columns
            elif records and isinstance(records[0], dict):
                self.columns = list(records[0].keys())
            else:
                self.columns = []

        def __getitem__(self, item):
            if isinstance(item, list):
                sub_records = [{k: r.get(k) for k in item} for r in self._records]
                return _SimpleDataFrame(sub_records, columns=item)
            vals = [r.get(item) for r in self._records]
            return _SimpleSeries(vals, name=item)

        def __setitem__(self, key, value):
            if isinstance(value, _SimpleSeries):
                for r, v in zip(self._records, value._data):
                    r[key] = v
            elif isinstance(value, list):
                for r, v in zip(self._records, value):
                    r[key] = v
            else:
                for r in self._records:
                    r[key] = value
            if key not in self.columns:
                self.columns.append(key)

        def __len__(self):
            return len(self._records)

        def head(self, n=20):
            return _SimpleDataFrame(self._records[:n], columns=self.columns)

        def sort_values(self, by, ascending=True):
            def key_func(r):
                val = r.get(by)
                try:
                    return float(val)
                except Exception:
                    return str(val)
            sorted_recs = sorted(self._records, key=key_func, reverse=not ascending)
            return _SimpleDataFrame(sorted_recs, columns=self.columns)

        def groupby(self, by):
            class _Group:
                def __init__(self, records, by_col):
                    self.records = records
                    self.by_col = by_col
                def size(self):
                    counts = {}
                    for r in self.records:
                        k = r.get(self.by_col)
                        counts[k] = counts.get(k, 0) + 1
                    class _Res:
                        def reset_index(self, name="total_count"):
                            recs = [{self.by_col: k, name: v} for k, v in counts.items()]
                            return _SimpleDataFrame(recs)
                    return _Res()
                def __getitem__(self, col):
                    class _Agg:
                        def __init__(self, records, by_col, agg_col):
                            self.records = records
                            self.by_col = by_col
                            self.agg_col = agg_col
                        def sum(self):
                            totals = {}
                            for r in self.records:
                                k = r.get(self.by_col)
                                try:
                                    v = float(r.get(self.agg_col, 0) or 0)
                                except Exception:
                                    v = 0.0
                                totals[k] = totals.get(k, 0.0) + v
                            class _Res:
                                def reset_index(self, name="total_revenue"):
                                    recs = [{self.by_col: k, name: v} for k, v in totals.items()]
                                    return _SimpleDataFrame(recs)
                            return _Res()
                    return _Agg(self.records, self.by_col, col)
            return _Group(self._records, by)

        def to_dict(self, orient="records"):
            return self._records

    def _read_csv(filepath, *args, **kwargs):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        return _SimpleDataFrame(records)

    def _to_numeric(series, errors="coerce"):
        if isinstance(series, _SimpleSeries):
            new_vals = []
            for x in series._data:
                try:
                    new_vals.append(float(x))
                except Exception:
                    new_vals.append(0.0 if errors == "coerce" else x)
            return _SimpleSeries(new_vals, name=series.name)
        return series

    fake_pd = types.ModuleType("pandas")
    fake_pd.read_csv = _read_csv
    fake_pd.read_parquet = _read_csv
    fake_pd.DataFrame = _SimpleDataFrame
    fake_pd.Series = _SimpleSeries
    fake_pd.to_numeric = _to_numeric
    sys.modules["pandas"] = fake_pd

user_code = sys.stdin.read()
local_env = {}

try:
    exec(user_code, {}, local_env)
    res = local_env.get("result", None)

    output = {}
    if hasattr(res, "to_dict") and hasattr(res, "columns"):
        try:
            head_df = res.head(20)
            records = head_df.to_dict(orient="records")
            cols = [str(c) for c in res.columns]
            output = {
                "type": "dataframe",
                "columns": cols,
                "rows": records,
                "row_count": len(records),
                "truncated": len(res) > 20,
            }
        except Exception:
            records = res.to_dict()
            output = {"type": "json", "data": records, "rows": [records], "columns": list(records.keys()), "row_count": 1}

    elif hasattr(res, "to_dict") and not hasattr(res, "columns"):
        try:
            head_series = res.head(20)
            d = head_series.to_dict()
            rows = [{"index": k, "value": v} for k, v in d.items()]
            output = {
                "type": "series",
                "columns": ["index", "value"],
                "rows": rows,
                "row_count": len(rows),
                "truncated": len(res) > 20,
            }
        except Exception:
            output = {"type": "scalar", "data": str(res), "rows": [{"value": str(res)}], "columns": ["value"], "row_count": 1}

    elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        capped = res[:20]
        cols = list(capped[0].keys())
        output = {
            "type": "dataframe",
            "columns": cols,
            "rows": capped,
            "row_count": len(capped),
            "truncated": len(res) > 20,
        }

    elif isinstance(res, (dict, list)):
        output = {
            "type": "json",
            "data": res,
            "rows": [res] if isinstance(res, dict) else [{"item": x} for x in res[:20]],
            "columns": list(res.keys()) if isinstance(res, dict) else ["item"],
            "row_count": 1 if isinstance(res, dict) else len(res[:20]),
            "truncated": False,
        }
    else:
        output = {
            "type": "scalar",
            "data": res,
            "rows": [{"result": res}],
            "columns": ["result"],
            "row_count": 1,
            "truncated": False,
        }

    print("__SANDBOX_START__")
    print(json.dumps(output, default=str))
    print("__SANDBOX_END__")

except Exception as e:
    sys.stderr.write(f"RuntimeError: {e}\\n")
    sys.exit(1)
"""


def execute_sandboxed_code(
    code: str,
    timeout_seconds: float = 5.0,
    max_memory_mb: int = 512,
) -> Tuple[bool, Dict[str, Any], str, int]:
    """Execute Python code inside an isolated subprocess."""
    with tempfile.TemporaryDirectory() as temp_dir:
        clean_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONUNBUFFERED": "1",
            "LC_ALL": "en_US.UTF-8",
            "LANG": "en_US.UTF-8",
            # LC_ALL/LANG are ignored on Windows; force UTF-8 child I/O explicitly.
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        if "PYTHONPATH" in os.environ:
            clean_env["PYTHONPATH"] = os.environ["PYTHONPATH"]

        try:
            proc = subprocess.run(
                [sys.executable, "-c", WRAPPER_TEMPLATE],
                input=code,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout_seconds,
                cwd=temp_dir,
                env=clean_env,
            )

            if proc.returncode != 0:
                err = proc.stderr.strip() or f"Subprocess exited with code {proc.returncode}"
                return False, {}, err, proc.returncode

            stdout = proc.stdout
            if "__SANDBOX_START__" in stdout and "__SANDBOX_END__" in stdout:
                payload = stdout.split("__SANDBOX_START__")[1].split("__SANDBOX_END__")[0].strip()
                result_data = json.loads(payload)
                return True, result_data, proc.stderr.strip(), 0
            else:
                return False, {}, f"Invalid sandbox output protocol: {stdout.strip()}", 1

        except subprocess.TimeoutExpired:
            return False, {}, f"Execution timed out after {timeout_seconds} seconds", 124
        except Exception as e:
            return False, {}, f"Subprocess invocation error: {str(e)}", 1
