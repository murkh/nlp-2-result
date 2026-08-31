"""
HTTP API Client for Multi-Agent Knowledge Base Q&A Platform Backend.
Handles communication between Streamlit UI and FastAPI backend endpoints.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import httpx
import pandas as pd


class BackendClient:
    """Client for interacting with the Multi-Agent Q&A FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0, max_retries: int = 3):
        url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    # -------------------------------------------------------------------------
    # Internal HTTP Helpers
    # -------------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform GET request with automatic retry."""
        query_str = ""
        if params:
            import urllib.parse

            query_str = "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        url = f"{self.base_url}{endpoint}{query_str}"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    res = client.get(url)
                    if res.status_code == 200:
                        return res.json()
                    return {"error": res.text, "status_code": res.status_code}
            except Exception as e:
                last_error = e

        return {
            "error": str(last_error) if last_error else "Request failed",
            "status": "unreachable",
        }

    def _post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform POST request with automatic retry."""
        url = f"{self.base_url}{endpoint}"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    if files is not None:
                        res = client.post(url, files=files, data=data or {})
                    else:
                        res = client.post(url, json=json_data or {})
                    if res.status_code in (200, 201):
                        return res.json()
                    return {"error": res.text, "status_code": res.status_code}
            except Exception as e:
                last_error = e

        return {
            "error": str(last_error) if last_error else "Request failed",
            "status": "unreachable",
        }

    # -------------------------------------------------------------------------
    # System & Ingestion
    # -------------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Check backend health status."""
        try:
            res = self._get("/health")
            if "status" in res and res.get("status") == "healthy":
                return res
            if "status_code" in res and res["status_code"] == 200:
                return res
            return {"status": "unhealthy", "error": res.get("error", "Non-200 status code")}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def list_datasets(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered datasets, optionally filtered by category."""
        try:
            params = {}
            if category:
                params["category"] = category
            res = self._get("/datasets", params=params)
            if isinstance(res, dict):
                return res.get("datasets", [])
            elif isinstance(res, list):
                return res
            return []
        except Exception as e:
            print(f"[BackendClient] list_datasets error: {e}")
            return []

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieve detailed metadata for a single dataset."""
        try:
            res = self._get(f"/datasets/{dataset_id}")
            return res
        except Exception as e:
            return {"error": str(e)}

    def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload and ingest a structured or unstructured dataset file."""
        try:
            files = {"file": (filename, file_bytes, "application/octet-stream")}
            data = {}
            if display_name:
                data["display_name"] = display_name
            if description:
                data["description"] = description

            return self._post("/ingest", files=files, data=data)
        except Exception as e:
            return {"error": f"Ingestion exception: {str(e)}"}

    # -------------------------------------------------------------------------
    # Conversational Agent Q&A
    # -------------------------------------------------------------------------

    def query_agent(
        self,
        query: str,
        session_id: Optional[str] = None,
        suggested_strategy: Optional[str] = None,
        dataset_ids: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Execute query through LangGraph Multi-Agent Supervisor."""
        payload: Dict[str, Any] = {
            "query": query,
            "temperature": temperature,
        }
        if session_id:
            payload["session_id"] = session_id
        if suggested_strategy:
            payload["suggested_strategy"] = suggested_strategy
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids

        try:
            res = self._post("/query/agent", json_data=payload)
            if "error" in res and "answer" not in res:
                res["answer"] = f"Error: {res.get('error')}"
            return res
        except Exception as e:
            return {
                "error": str(e),
                "answer": f"Backend connection error: {str(e)}",
            }

    # -------------------------------------------------------------------------
    # Dedicated Execution Engines
    # -------------------------------------------------------------------------

    def query_dedicated_db(
        self,
        query: str,
        dataset_ids: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Strategy A: Dedicated PostgreSQL Text2SQL query engine."""
        payload: Dict[str, Any] = {"query": query, "temperature": temperature}
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids
        try:
            return self._post("/query/dedicated-db", json_data=payload)
        except Exception as e:
            return {"error": str(e)}

    def query_duckdb(
        self,
        query: str,
        dataset_ids: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Strategy B: In-Memory DuckDB query engine."""
        payload: Dict[str, Any] = {"query": query, "temperature": temperature}
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids
        try:
            return self._post("/query/duckdb", json_data=payload)
        except Exception as e:
            return {"error": str(e)}

    def query_pandas_sandbox(
        self,
        query: str,
        dataset_ids: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Strategy C: Sandboxed Python DataFrame execution."""
        payload: Dict[str, Any] = {"query": query, "temperature": temperature}
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids
        try:
            return self._post("/query/pandas-sandbox", json_data=payload)
        except Exception as e:
            return {"error": str(e)}

    def query_unstructured_rag(
        self,
        query: str,
        top_k: int = 5,
        dataset_ids: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Unstructured Hybrid RAG with dense + sparse search and bracketed citations."""
        payload: Dict[str, Any] = {"query": query, "top_k": top_k, "temperature": temperature}
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids
        try:
            return self._post("/query/unstructured-rag", json_data=payload)
        except Exception as e:
            return {"error": str(e)}

    def query_benchmark(
        self,
        query: str,
        include_raw_data: bool = True,
        dataset_ids: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """3-Way Parallel Benchmark Arena comparing Strategy A, B, and C."""
        payload: Dict[str, Any] = {
            "query": query,
            "include_raw_data": include_raw_data,
            "temperature": temperature,
        }
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids
        try:
            return self._post("/query/benchmark", json_data=payload)
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # Utility / Conversion Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def tabular_result_to_dataframe(tabular_result: Union[Dict[str, Any], Any]) -> Any:
        """Convert TabularResult dict into a clean DataFrame."""
        if not tabular_result:
            return pd.DataFrame()

        if isinstance(tabular_result, dict):
            rows = tabular_result.get("rows", [])
            cols = tabular_result.get("columns", [])
            if rows:
                df = pd.DataFrame(rows)
                if cols and hasattr(df, "columns"):
                    valid_cols = [c for c in cols if c in df.columns]
                    extra_cols = [c for c in df.columns if c not in valid_cols]
                    df = df[valid_cols + extra_cols]
                return df
            elif cols:
                return pd.DataFrame(columns=cols)
            return pd.DataFrame()

        if hasattr(tabular_result, "rows") and tabular_result.rows:
            return pd.DataFrame(tabular_result.rows)
        return pd.DataFrame()

    @staticmethod
    def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on standard model pricing."""
        # $0.15 / 1M prompt, $0.60 / 1M completion
        cost = (prompt_tokens * 0.00000015) + (completion_tokens * 0.0000006)
        return round(cost, 6)

    @staticmethod
    def format_latency(latency_ms: float) -> str:
        """Format latency into human readable string."""
        if latency_ms >= 1000.0:
            return f"{latency_ms / 1000.0:.2f} s"
        return f"{latency_ms:.1f} ms"
