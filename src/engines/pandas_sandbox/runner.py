"""
Subprocess Runner for Sandboxed Python DataFrame Execution.
Executes code in an isolated subprocess with watchdog timeouts,
resource limits, environment sanitation, and standardized JSON protocol.
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

# Set POSIX resource limits if supported
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
except Exception:
    pass

import pandas as pd

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
