"""
Unit and Integration Tests for Langfuse Observability & Local Fallback Tracing System.
Tests root trace lifecycle, child spans, token consumption aggregation,
latency tracking, local cache retrieval, and Langfuse mock integration.
"""

import unittest
import uuid
from unittest.mock import MagicMock, patch

from src.observability.telemetry import (
    ObservabilityManager,
    TelemetrySpan,
    TelemetryTrace,
    get_tracer,
)


class TestObservability(unittest.TestCase):
    """Test suite for Telemetry tracing, spans, token aggregation, and local fallback."""

    def setUp(self):
        self.tracer = ObservabilityManager(public_key=None, secret_key=None)

    def tearDown(self):
        self.tracer.clear()

    # -------------------------------------------------------------------------
    # Root Trace Lifecycle Tests
    # -------------------------------------------------------------------------

    def test_trace_creation_and_attributes(self):
        """Verify root trace initialization and basic metadata attributes."""
        trace = self.tracer.create_trace(
            name="test_root_trace",
            session_id="session_xyz",
            metadata={"environment": "test", "version": "1.0"},
        )
        self.assertIsNotNone(trace.trace_id)
        self.assertEqual(trace.name, "test_root_trace")
        self.assertEqual(trace.session_id, "session_xyz")
        self.assertEqual(trace.status, "RUNNING")
        self.assertGreaterEqual(trace.latency_ms, 0.0)
        self.assertEqual(trace.prompt_tokens, 0)
        self.assertEqual(trace.completion_tokens, 0)
        self.assertEqual(trace.total_tokens, 0)
        self.assertEqual(len(trace.spans), 0)

    def test_trace_context_manager_success(self):
        """Verify start_trace context manager automatically records status SUCCESS and latency."""
        with self.tracer.start_trace(name="ctx_trace", session_id="sess_1") as trace:
            self.assertEqual(trace.status, "RUNNING")
            trace.record_tokens(prompt_tokens=10, completion_tokens=20)

        self.assertEqual(trace.status, "SUCCESS")
        self.assertIsNotNone(trace.end_time)
        self.assertGreaterEqual(trace.latency_ms, 0.0)
        self.assertEqual(trace.total_tokens, 30)

    def test_trace_context_manager_exception_capture(self):
        """Verify start_trace context manager captures runtime errors and sets status ERROR."""
        with self.assertRaises(ValueError):
            with self.tracer.start_trace(name="failing_trace") as trace:
                raise ValueError("Database connection timed out")

        self.assertEqual(trace.status, "ERROR")
        self.assertIn("Database connection timed out", trace.error)

    # -------------------------------------------------------------------------
    # Child Span & Token Aggregation Tests
    # -------------------------------------------------------------------------

    def test_child_span_lifecycle(self):
        """Verify child span creation, token recording, and status progression."""
        trace = self.tracer.create_trace(name="parent_trace")
        span = trace.start_span("sub_action", input_data={"param": "value"})

        self.assertEqual(span.name, "sub_action")
        self.assertEqual(span.status, "RUNNING")
        self.assertEqual(len(trace.spans), 1)

        span.record_tokens(prompt_tokens=40, completion_tokens=60)
        span.end(output_data={"result": "done"})

        self.assertEqual(span.status, "SUCCESS")
        self.assertEqual(span.prompt_tokens, 40)
        self.assertEqual(span.completion_tokens, 60)
        self.assertEqual(span.total_tokens, 100)
        self.assertGreater(span.latency_ms, 0.0)

        # Verify parent trace aggregated the tokens
        self.assertEqual(trace.prompt_tokens, 40)
        self.assertEqual(trace.completion_tokens, 60)
        self.assertEqual(trace.total_tokens, 100)

    def test_multi_span_token_aggregation(self):
        """Verify multiple sequential spans aggregate their tokens into the parent trace."""
        with self.tracer.start_trace(name="agent_workflow") as trace:
            # Span 1: Router
            span1 = trace.start_span("router_node")
            span1.record_tokens(prompt_tokens=25, completion_tokens=10)
            span1.end()

            # Span 2: Engine
            span2 = trace.start_span("duckdb_engine")
            span2.record_tokens(prompt_tokens=150, completion_tokens=45)
            span2.end()

            # Span 3: Synthesizer
            span3 = trace.start_span("synthesizer_node")
            span3.record_tokens(prompt_tokens=80, completion_tokens=50)
            span3.end()

        self.assertEqual(len(trace.spans), 3)
        self.assertEqual(trace.prompt_tokens, 255)
        self.assertEqual(trace.completion_tokens, 105)
        self.assertEqual(trace.total_tokens, 360)

    def test_span_context_manager(self):
        """Verify span __enter__ and __exit__ context manager lifecycle."""
        trace = self.tracer.create_trace(name="trace_with_ctx_span")
        with trace.start_span("scoped_step") as span:
            span.record_tokens(10, 10)

        self.assertEqual(span.status, "SUCCESS")
        self.assertIsNotNone(span.end_time)

    # -------------------------------------------------------------------------
    # Serialization & Cache Tests
    # -------------------------------------------------------------------------

    def test_trace_to_dict_structure(self):
        """Verify dictionary serialization conforms to standard schema."""
        trace = self.tracer.create_trace(name="serial_trace", session_id="s1")
        span = trace.start_span("child_1")
        span.record_tokens(15, 25)
        span.end()
        trace.end()

        d = trace.to_dict()
        self.assertEqual(d["trace_id"], trace.trace_id)
        self.assertEqual(d["name"], "serial_trace")
        self.assertEqual(d["session_id"], "s1")
        self.assertEqual(d["status"], "SUCCESS")
        self.assertEqual(d["prompt_tokens"], 15)
        self.assertEqual(d["completion_tokens"], 25)
        self.assertEqual(d["total_tokens"], 40)
        self.assertEqual(len(d["spans"]), 1)
        self.assertEqual(d["spans"][0]["name"], "child_1")

    def test_trace_cache_and_retrieval(self):
        """Verify storing and retrieving recent traces from manager in-memory cache."""
        t1 = self.tracer.create_trace(name="trace_1")
        t2 = self.tracer.create_trace(name="trace_2")

        self.assertIsNotNone(self.tracer.get_trace(t1.trace_id))
        self.assertIsNotNone(self.tracer.get_trace(t2.trace_id))
        self.assertIsNone(self.tracer.get_trace("nonexistent_id"))

        recent = self.tracer.get_recent_traces(limit=5)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["trace_id"], t2.trace_id)
        self.assertEqual(recent[1]["trace_id"], t1.trace_id)

    def test_trace_tuple_unpacking(self):
        """Verify trace supports tuple unpacking for backwards compatibility."""
        trace = self.tracer.create_trace(name="unpack_trace")
        lf_obj, record = trace
        self.assertIsNone(lf_obj)
        self.assertEqual(record["name"], "unpack_trace")
        self.assertEqual(trace[1]["name"], "unpack_trace")
        self.assertEqual(trace["name"], "unpack_trace")

    # -------------------------------------------------------------------------
    # Langfuse Mock & Fallback Mode Tests
    # -------------------------------------------------------------------------

    def test_local_fallback_when_credentials_absent(self):
        """Verify seamless local execution when Langfuse credentials are not provided."""
        manager = ObservabilityManager(public_key="", secret_key="")
        self.assertFalse(manager.enabled)
        self.assertIsNone(manager.client)

        with manager.start_trace(name="local_only_trace") as trace:
            span = trace.start_span("local_span")
            span.record_tokens(5, 5)
            span.end()

        self.assertEqual(trace.status, "SUCCESS")
        self.assertEqual(trace.total_tokens, 10)

    def test_langfuse_mock_client_integration(self):
        """Verify Langfuse API calls are made when enabled with valid client mock."""
        mock_lf_client = MagicMock()
        mock_lf_trace = MagicMock()
        mock_lf_span = MagicMock()

        mock_lf_client.trace.return_value = mock_lf_trace
        mock_lf_trace.span.return_value = mock_lf_span

        manager = ObservabilityManager(
            public_key="pk_test", secret_key="sk_test", host="http://mock:3000"
        )
        manager.enabled = True
        manager.client = mock_lf_client

        with manager.start_trace(name="integrated_trace", session_id="sess_mock") as trace:
            span = trace.start_span("integrated_span")
            span.record_tokens(prompt_tokens=30, completion_tokens=70)
            span.end(output_data={"status": "ok"})

        mock_lf_client.trace.assert_called_once()
        mock_lf_trace.span.assert_called_once()
        mock_lf_span.update.assert_called()
        mock_lf_span.end.assert_called_once()
        mock_lf_client.flush.assert_called()

    def test_global_singleton_getter(self):
        """Verify get_tracer() returns a valid ObservabilityManager singleton."""
        t = get_tracer()
        self.assertIsInstance(t, ObservabilityManager)


if __name__ == "__main__":
    unittest.main()
