"""
Observability and Telemetry Tracing Module.
Provides Langfuse integration with child span tracing, token consumption aggregation,
and local in-memory fallback mode for offline, testing, and decoupled execution.
"""

import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from src.config import Settings, get_settings


class TelemetrySpan:
    """Represents an isolated span within a request trace."""

    def __init__(
        self,
        name: str,
        span_id: Optional[str] = None,
        parent_trace: Optional["TelemetryTrace"] = None,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        langfuse_span: Any = None,
    ):
        self.span_id = span_id or str(uuid.uuid4())
        self.name = name
        self.parent_trace = parent_trace
        self.input_data = input_data
        self.output_data = None
        self.metadata = metadata or {}
        self.langfuse_span = langfuse_span
        self.start_time = time.perf_counter()
        self._end_time: Optional[float] = None
        self._latency_ms: float = 0.0
        self.status: str = "RUNNING"
        self.error: Optional[str] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    @property
    def end_time(self) -> Optional[float]:
        return self._end_time

    @property
    def latency_ms(self) -> float:
        if self._end_time is not None:
            return self._latency_ms
        return round((time.perf_counter() - self.start_time) * 1000.0, 2)

    @latency_ms.setter
    def latency_ms(self, val: float):
        self._latency_ms = val

    def record_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Record token consumption for this span and propagate to parent trace."""
        self.prompt_tokens += max(0, prompt_tokens)
        self.completion_tokens += max(0, completion_tokens)
        self.total_tokens = self.prompt_tokens + self.completion_tokens

        if self.parent_trace:
            self.parent_trace.record_tokens(prompt_tokens, completion_tokens)

        if self.langfuse_span:
            try:
                self.langfuse_span.update(
                    usage={
                        "input": self.prompt_tokens,
                        "output": self.completion_tokens,
                        "total": self.total_tokens,
                    }
                )
            except Exception:
                pass

    def end(
        self,
        output_data: Any = None,
        status: str = "SUCCESS",
        error: Optional[str] = None,
    ):
        """Mark span execution as completed."""
        if self._end_time is not None:
            return
        self._end_time = time.perf_counter()
        self._latency_ms = max(0.01, round((self._end_time - self.start_time) * 1000.0, 2))
        self.output_data = output_data
        self.status = status
        self.error = error

        if self.langfuse_span:
            try:
                self.langfuse_span.end(
                    output=output_data,
                    status_message=error if error else None,
                    level="ERROR" if status == "ERROR" else "DEFAULT",
                )
            except Exception:
                pass

    def __enter__(self) -> "TelemetrySpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.end(status="ERROR", error=str(exc_val))
        else:
            self.end(status="SUCCESS")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize span details."""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }


class TelemetryTrace:
    """Root trace capturing multi-step agent and query execution telemetry."""

    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        langfuse_trace: Any = None,
    ):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.name = name
        self.session_id = session_id
        self.metadata = metadata or {}
        self.langfuse_trace = langfuse_trace
        self.start_time = time.perf_counter()
        self._end_time: Optional[float] = None
        self._latency_ms: float = 0.0
        self.status: str = "RUNNING"
        self.error: Optional[str] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.spans: List[TelemetrySpan] = []

    @property
    def end_time(self) -> Optional[float]:
        return self._end_time

    @property
    def latency_ms(self) -> float:
        if self._end_time is not None:
            return self._latency_ms
        return round((time.perf_counter() - self.start_time) * 1000.0, 2)

    @latency_ms.setter
    def latency_ms(self, val: float):
        self._latency_ms = val

    def start_span(
        self,
        name: str,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetrySpan:
        """Create and register a child span under this trace."""
        lf_span = None
        if self.langfuse_trace:
            try:
                lf_span = self.langfuse_trace.span(
                    name=name,
                    input=input_data,
                    metadata=metadata or {},
                )
            except Exception:
                lf_span = None

        span = TelemetrySpan(
            name=name,
            parent_trace=self,
            input_data=input_data,
            metadata=metadata,
            langfuse_span=lf_span,
        )
        self.spans.append(span)
        return span

    def record_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Aggregate token counts into the root trace."""
        self.prompt_tokens += max(0, prompt_tokens)
        self.completion_tokens += max(0, completion_tokens)
        self.total_tokens = self.prompt_tokens + self.completion_tokens

        if self.langfuse_trace:
            try:
                self.langfuse_trace.update(
                    metadata={
                        **self.metadata,
                        "prompt_tokens": self.prompt_tokens,
                        "completion_tokens": self.completion_tokens,
                        "total_tokens": self.total_tokens,
                    }
                )
            except Exception:
                pass

    def end(self, status: str = "SUCCESS", error: Optional[str] = None):
        """Mark root trace execution as finished."""
        if self._end_time is not None:
            return
        self._end_time = time.perf_counter()
        self._latency_ms = round((self._end_time - self.start_time) * 1000.0, 2)
        self.status = status
        self.error = error

        for span in self.spans:
            if span._end_time is None:
                span.end(status=status, error=error)

        if self.langfuse_trace:
            try:
                self.langfuse_trace.update(
                    output={
                        "status": self.status,
                        "latency_ms": self._latency_ms,
                        "total_tokens": self.total_tokens,
                        "error": self.error,
                    }
                )
            except Exception:
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace and child spans into a dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "session_id": self.session_id,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }

    # Enable tuple-like unpacking: `trace_obj, telemetry_record = trace`
    def __iter__(self):
        yield self.langfuse_trace
        yield self.to_dict()

    def __getitem__(self, item):
        if item == 0:
            return self.langfuse_trace
        elif item == 1:
            return self.to_dict()
        elif isinstance(item, str):
            return self.to_dict().get(item)
        raise IndexError(f"Index {item} out of range for TelemetryTrace")


class ObservabilityManager:
    """
    Centralized Observability Manager.
    Integrates with Langfuse for distributed tracing while providing full local in-memory fallback.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        cfg = settings or get_settings()
        self.public_key = public_key or cfg.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or cfg.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or cfg.langfuse_host or os.getenv("LANGFUSE_HOST", "http://localhost:3000")

        self.enabled = bool(self.public_key and self.secret_key)
        self.client = None
        self._traces: Dict[str, TelemetryTrace] = {}
        self._recent_trace_ids: List[str] = []

        if self.enabled:
            try:
                from langfuse import Langfuse

                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
            except Exception:
                # Fallback silently to local tracing
                self.enabled = False
                self.client = None

    def create_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetryTrace:
        """Create a new root trace and track in local memory."""
        trace_id = str(uuid.uuid4())
        lf_trace = None

        if self.enabled and self.client:
            try:
                lf_trace = self.client.trace(
                    id=trace_id,
                    name=name,
                    session_id=session_id,
                    metadata=metadata or {},
                )
            except Exception:
                lf_trace = None

        trace = TelemetryTrace(
            name=name,
            trace_id=trace_id,
            session_id=session_id,
            metadata=metadata,
            langfuse_trace=lf_trace,
        )

        self._traces[trace_id] = trace
        self._recent_trace_ids.append(trace_id)
        if len(self._recent_trace_ids) > 1000:
            oldest = self._recent_trace_ids.pop(0)
            self._traces.pop(oldest, None)

        return trace

    @contextmanager
    def start_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[TelemetryTrace]:
        """
        Context manager for root trace lifecycle.
        Supports both `as trace` and `as (trace_obj, telemetry_record)`.
        """
        trace = self.create_trace(name=name, session_id=session_id, metadata=metadata)
        try:
            yield trace
            trace.end(status="SUCCESS")
        except Exception as exc:
            trace.end(status="ERROR", error=str(exc))
            raise
        finally:
            self.flush()

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trace dictionary by ID from local cache."""
        trace = self._traces.get(trace_id)
        return trace.to_dict() if trace else None

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent traces in reverse chronological order."""
        ids = self._recent_trace_ids[-limit:]
        return [self._traces[tid].to_dict() for tid in reversed(ids) if tid in self._traces]

    def flush(self):
        """Flush pending traces to Langfuse backend if connected."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception:
                pass

    def clear(self):
        """Clear local in-memory trace cache."""
        self._traces.clear()
        self._recent_trace_ids.clear()


# Global Singleton Tracer
_global_tracer: Optional[ObservabilityManager] = None


def get_tracer() -> ObservabilityManager:
    """Retrieve global singleton ObservabilityManager."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = ObservabilityManager()
    return _global_tracer


tracer = get_tracer()
