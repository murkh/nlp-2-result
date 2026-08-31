"""
Unit and Integration Tests for Supervisor Intent Router, LangGraph Multi-Agent State Machine,
Synthesizer Node, and Conversational Agent API Route.
"""

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

from src.agent.graph import build_multi_agent_graph, get_agent_graph, route_intent, run_agent
from src.agent.nodes.chitchat_node import chitchat_node
from src.agent.nodes.clarify_node import clarify_node
from src.agent.nodes.router_node import supervisor_router_node
from src.agent.nodes.structured_node import structured_node
from src.agent.nodes.synthesizer_node import format_markdown_table, synthesizer_node
from src.agent.nodes.unstructured_node import unstructured_node
from src.routing import SupervisorRouter
from src.agent.state import AgentState
from src.api.routes import agent
from src.api.schemas import QueryAgentRequest, QueryAgentResponse
from src.database.connection import get_db_manager
from src.ingestion.metadata_extractor import EmbeddingService, MetadataExtractor
from src.ingestion.structured import StructuredIngestionEngine
from src.ingestion.unstructured import UnstructuredIngestionEngine
from src.storage.blob_store import get_blob_manager
from tests.conftest import requires_llm


class TestAgentRouter(unittest.TestCase):
    """Test suite for supervisor intent classification, graph execution, and response synthesis."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_agent_blobs_"))
        self.db_manager = get_db_manager(in_memory=True)
        self.blob_manager = get_blob_manager(base_path=self.temp_dir)
        self.embedding_service = EmbeddingService()
        self.meta_extractor = MetadataExtractor(embedding_service=self.embedding_service)

        self.structured_engine = StructuredIngestionEngine(
            db_manager=self.db_manager,
            blob_manager=self.blob_manager,
            metadata_extractor=self.meta_extractor,
        )
        self.unstructured_engine = UnstructuredIngestionEngine(
            db_manager=self.db_manager,
            blob_manager=self.blob_manager,
            embedding_service=self.embedding_service,
        )

        sample_csv = (
            "order_id,customer_id,order_date,status,total_amount,shipping_city\n"
            "101,501,2024-01-10 10:00:00,completed,150.50,New York\n"
            "102,502,2024-01-11 11:30:00,completed,280.00,San Francisco\n"
            "103,501,2024-01-12 14:15:00,shipped,75.25,New York\n"
            "104,503,2024-01-13 09:45:00,cancelled,45.00,Chicago\n"
            "105,504,2024-01-14 16:20:00,completed,510.80,Austin\n"
        )
        self.struct_ds = self.structured_engine.ingest_file(
            file_input=sample_csv,
            filename="agent_orders.csv",
            display_name="Agent Orders",
        )

        sample_md = (
            "# Company Operations Handbook\n\n"
            "## Incident Response Protocol\n"
            "When a Severity 1 incident occurs, page the on-call engineer and open an incident Slack channel. "
            "A post-mortem document must be published within 48 hours of resolution.\n\n"
            "## Return Window Policy\n"
            "Customer return requests are accepted within 30 days of delivery for full refund.\n"
        )
        self.unstruct_ds = self.unstructured_engine.ingest_file(
            file_input=sample_md,
            filename="agent_operations_policy.md",
            display_name="Operations Policy",
        )

        self.router = SupervisorRouter(db_manager=self.db_manager)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Intent Classification Tests
    # -------------------------------------------------------------------------

    def test_router_greeting_and_chitchat(self):
        """Verify conversational greetings and capability inquiries route to GREETING_OR_CHITCHAT."""
        greetings = [
            "Hello",
            "Hi there!",
            "Good morning",
            "Who are you?",
            "What can you do?",
            "What are your capabilities?",
            "Help me",
            "Thank you!",
            "Goodbye",
        ]
        for q in greetings:
            decision = self.router.classify_intent(q)
            self.assertEqual(
                decision.intent,
                "GREETING_OR_CHITCHAT",
                f"Query '{q}' should classify as GREETING_OR_CHITCHAT, got {decision.intent}",
            )
            self.assertGreaterEqual(decision.confidence, 0.9)
            self.assertIsNone(decision.suggested_strategy)
            self.assertIsNone(decision.clarification_question)

    def test_router_ambiguous_queries(self):
        """Verify underspecified and ambiguous queries route to AMBIGUOUS_QUERY with candidate datasets."""
        ambiguous = [
            "data",
            "show data",
            "show me data",
            "tell me more",
            "orders",
            "sales",
            "overview",
            "summary",
            "status",
        ]
        for q in ambiguous:
            decision = self.router.classify_intent(q)
            self.assertEqual(
                decision.intent,
                "AMBIGUOUS_QUERY",
                f"Query '{q}' should classify as AMBIGUOUS_QUERY, got {decision.intent}",
            )
            self.assertIsNotNone(decision.clarification_question)
            self.assertIn("broad", decision.clarification_question.lower())
            self.assertGreaterEqual(len(decision.relevant_datasets), 1)

    @requires_llm
    def test_router_structured_queries(self):
        """Verify calculation, aggregation, and filtering queries route to STRUCTURED_QUERY."""
        queries = [
            "How many completed orders are there?",
            "What is the total revenue in 2024?",
            "Average order amount by shipping city",
            "Find highest price product in sales table",
            "Count total rows in orders dataset",
        ]
        for q in queries:
            decision = self.router.classify_intent(q)
            self.assertEqual(
                decision.intent,
                "STRUCTURED_QUERY",
                f"Query '{q}' should classify as STRUCTURED_QUERY, got {decision.intent}",
            )
            self.assertGreaterEqual(decision.confidence, 0.75)
            self.assertIn(decision.suggested_strategy, ["duckdb", "dedicated_db", "pandas_sandbox"])

    @requires_llm
    def test_router_strategy_hints(self):
        """Engine-hinted queries stay STRUCTURED_QUERY and pick a valid engine.

        The keyword-to-engine mapping is now a prompt instruction, not a heuristic
        fallback, so the exact engine is the classifier's call, not a hard guarantee.
        """
        hinted = [
            "Run pandas dataframe analysis to count completed orders",
            "Query dedicated postgres table for total sales",
            "How many total orders?",
        ]
        for q in hinted:
            decision = self.router.classify_intent(q)
            self.assertEqual(decision.intent, "STRUCTURED_QUERY", q)
            self.assertIn(
                decision.suggested_strategy,
                ["duckdb", "dedicated_db", "pandas_sandbox"],
                q,
            )

    @requires_llm
    def test_router_unstructured_queries(self):
        """Verify documentation and policy questions route to UNSTRUCTURED_QUERY."""
        queries = [
            "What is the incident response protocol?",
            "What is the customer return window policy?",
            "Explain the on-call post-mortem guidelines",
            "What are the security compliance rules?",
        ]
        for q in queries:
            decision = self.router.classify_intent(q)
            self.assertEqual(
                decision.intent,
                "UNSTRUCTURED_QUERY",
                f"Query '{q}' should classify as UNSTRUCTURED_QUERY, got {decision.intent}",
            )
            self.assertGreaterEqual(decision.confidence, 0.75)

    @requires_llm
    def test_router_mixed_greeting_with_question(self):
        """Verify polite greetings combined with questions prioritize the substantive intent."""
        d_struct = self.router.classify_intent("Hello! How many completed orders are there?")
        self.assertEqual(d_struct.intent, "STRUCTURED_QUERY")

        d_unstruct = self.router.classify_intent(
            "Hi there, what is the incident response protocol?"
        )
        self.assertEqual(d_unstruct.intent, "UNSTRUCTURED_QUERY")

    # -------------------------------------------------------------------------
    # Graph Flow & Node Execution Tests
    # -------------------------------------------------------------------------

    def test_route_intent_helper(self):
        """Verify route_intent correctly maps state intent to target nodes."""
        self.assertEqual(route_intent({"intent": "GREETING_OR_CHITCHAT"}), "chitchat")
        self.assertEqual(route_intent({"intent": "AMBIGUOUS_QUERY"}), "clarification")
        self.assertEqual(route_intent({"intent": "STRUCTURED_QUERY"}), "structured_agent")
        self.assertEqual(route_intent({"intent": "UNSTRUCTURED_QUERY"}), "unstructured_agent")
        self.assertEqual(route_intent({}), "chitchat")

    def test_graph_greeting_execution(self):
        """Verify graph execution on greeting short-circuits with zero tool calls."""
        res = run_agent(query="Hello, who are you?", session_id="sess_greet_1")
        self.assertEqual(res["intent"], "GREETING_OR_CHITCHAT")
        self.assertIn("assistant", res["final_answer"].lower())
        self.assertEqual(len(res["execution_result"]), 0)
        self.assertEqual(len(res["citations"]), 0)
        self.assertEqual(res["telemetry"]["route"], "GREETING_OR_CHITCHAT")
        self.assertTrue(res["telemetry"]["execution_success"])
        self.assertGreater(res["telemetry"]["total_tokens"], 0)

    def test_graph_clarification_execution(self):
        """Verify graph execution on ambiguous query returns proactive clarification."""
        res = run_agent(query="data", session_id="sess_clarify_1")
        self.assertEqual(res["intent"], "AMBIGUOUS_QUERY")
        self.assertIsNotNone(res["clarification_message"])
        self.assertIn("broad", res["final_answer"].lower())
        self.assertEqual(res["telemetry"]["route"], "AMBIGUOUS_QUERY")
        self.assertGreaterEqual(len(res["candidate_datasets"]), 1)

    @requires_llm
    def test_graph_structured_execution(self):
        """Verify graph execution on structured query runs engine and synthesizes table/answer."""
        res = run_agent(query="How many total orders are there?", session_id="sess_struct_1")
        self.assertEqual(res["intent"], "STRUCTURED_QUERY")
        self.assertIsNotNone(res["generated_code"])
        self.assertGreater(len(res["execution_result"]), 0)
        self.assertIn("5", res["final_answer"])
        self.assertEqual(res["telemetry"]["route"], "STRUCTURED_QUERY")
        self.assertTrue(res["telemetry"]["execution_success"])
        self.assertGreater(res["telemetry"]["total_tokens"], 0)

    @requires_llm
    def test_graph_unstructured_execution(self):
        """Verify graph execution on unstructured query retrieves chunks and includes citations."""
        res = run_agent(
            query="What is the incident response protocol and post-mortem SLA?",
            session_id="sess_unstruct_1",
        )
        self.assertEqual(res["intent"], "UNSTRUCTURED_QUERY")
        self.assertGreater(len(res["retrieved_chunks"]), 0)
        self.assertGreater(len(res["citations"]), 0)
        self.assertIn("incident", res["final_answer"].lower())
        self.assertIn("Doc: Operations Policy", str(res["citations"]))
        self.assertEqual(res["telemetry"]["route"], "UNSTRUCTURED_QUERY")
        self.assertTrue(res["telemetry"]["execution_success"])

    # -------------------------------------------------------------------------
    # Synthesizer Formatting Tests
    # -------------------------------------------------------------------------

    def test_synthesizer_formats_markdown_table(self):
        """Verify synthesizer node converts rows to GitHub-flavored Markdown table."""
        rows = [
            {"order_id": 101, "status": "completed", "total_amount": 150.50},
            {"order_id": 102, "status": "completed", "total_amount": 280.00},
        ]
        state: AgentState = {
            "query": "Show completed orders",
            "session_id": "test_synth_1",
            "intent": "STRUCTURED_QUERY",
            "suggested_strategy": "duckdb",
            "execution_result": rows,
            "execution_columns": ["order_id", "status", "total_amount"],
            "execution_error": None,
            "citations": [],
            "telemetry": {},
        }
        res = synthesizer_node(state)
        self.assertIn("| order_id | status | total_amount |", res["final_answer"])
        self.assertIn("| 101 | completed | 150.5 |", res["final_answer"])
        self.assertGreater(len(res["citations"]), 0)

    def test_synthesizer_truncates_large_results(self):
        """Verify synthesizer caps displayed table rows at 20."""
        rows = [{"row_id": i, "val": f"item_{i}"} for i in range(35)]
        table_md = format_markdown_table(rows, columns=["row_id", "val"], max_rows=20)
        self.assertIn("| row_id | val |", table_md)
        self.assertIn("| 0 | item_0 |", table_md)
        self.assertIn("| 19 | item_19 |", table_md)
        self.assertNotIn("| 25 | item_25 |", table_md)
        self.assertIn("truncated", table_md)

    def test_synthesizer_handles_error(self):
        """Verify synthesizer formats clean error message when engine fails."""
        state: AgentState = {
            "query": "Select bad column",
            "session_id": "test_synth_err",
            "intent": "STRUCTURED_QUERY",
            "execution_error": "SyntaxError: table not found",
            "telemetry": {},
        }
        res = synthesizer_node(state)
        self.assertIn("encountered an issue", res["final_answer"])
        self.assertIn("SyntaxError", res["final_answer"])

    # -------------------------------------------------------------------------
    # API Route Integration Tests
    # -------------------------------------------------------------------------

    @requires_llm
    def test_api_agent_endpoint(self):
        """Verify POST /query/agent endpoint processes requests and returns QueryAgentResponse."""
        req_greet = QueryAgentRequest(query="Hello there!", session_id="api_sess_1")
        resp_greet = asyncio.run(agent.query_agent_endpoint(req_greet))
        self.assertIsInstance(resp_greet, QueryAgentResponse)
        self.assertEqual(resp_greet.intent, "GREETING_OR_CHITCHAT")
        self.assertEqual(resp_greet.session_id, "api_sess_1")

        req_struct = QueryAgentRequest(
            query="How many orders are completed?", session_id="api_sess_2"
        )
        resp_struct = asyncio.run(agent.query_agent_endpoint(req_struct))
        self.assertEqual(resp_struct.intent, "STRUCTURED_QUERY")
        self.assertGreater(resp_struct.token_usage.total_tokens, 0)
        self.assertGreaterEqual(resp_struct.metrics.total_latency_ms, 0.0)

        req_unstruct = QueryAgentRequest(
            query="What is the return window policy?", session_id="api_sess_3"
        )
        resp_unstruct = asyncio.run(agent.query_agent_endpoint(req_unstruct))
        self.assertEqual(resp_unstruct.intent, "UNSTRUCTURED_QUERY")
        self.assertIn("30 days", resp_unstruct.answer)


if __name__ == "__main__":
    unittest.main()
