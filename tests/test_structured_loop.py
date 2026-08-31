"""
Structured self-correction loop tests.

The deterministic suite covers the pure logic -- error classification, the probe
allowlist, budget routing, reducer discipline and graph shape -- with no LLM and
no network. The `@requires_llm` suite exercises the loop end to end against real
fixture data.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.agent.graph import (
    build_multi_agent_graph,
    build_single_pass_graph,
    initial_state,
    run_agent,
)
from src.agent.nodes.loop import generate as generate_nodes
from src.agent.nodes.loop.explore import _parse_plan, has_probes, tool_executor_node
from src.agent.nodes.loop.probes import (
    DISTINCT_VALUE_LIMIT,
    ProbeRejected,
    build_probe_sql,
)
from src.agent.nodes.loop.reflect import (
    classify_outcome,
    escalation_node,
    route_after_reflect,
)
from src.agent.nodes.loop.schema import has_schema
from src.agent.state import AgentState
from src.config import Settings, get_settings
from src.database.connection import get_db_manager
from src.engines.duckdb_engine import FORBIDDEN_DUCKDB_PATTERNS
from src.feedback import (
    EXECUTION_ERROR,
    PROBE,
    classify_error,
    observation,
    observation_prompt_block,
    render_observations,
)
from src.ingestion.metadata_extractor import EmbeddingService, MetadataExtractor
from src.ingestion.structured import StructuredIngestionEngine
from src.storage.blob_store import get_blob_manager
from tests.conftest import SAMPLE_CSV_TEXT, requires_llm

RETAINED = {"orders": ["order_id", "status", "total_amount", "shipping_city"]}


def loop_state(**overrides) -> AgentState:
    """Minimal structured-branch state for a node under test."""
    state: AgentState = {
        "query": "how many completed orders",
        "session_id": "test-session",
        "suggested_strategy": "duckdb",
        "pruned_tables": {
            "table_names": ["orders"],
            "retained_columns": RETAINED,
            "column_roles": {"status": "display"},
            "file_paths": {"orders": "/tmp/orders.csv"},
        },
        "schema_ddl": "CREATE TABLE orders (...)",
        "generated_code": "SELECT COUNT(*) FROM orders",
        "execution_result": [{"count": 3}],
        "execution_columns": ["count"],
        "execution_error": None,
        "observations": [],
        "loop_iterations": 1,
        "probe_plan": [],
        "reflection_class": None,
        "telemetry": {},
    }
    state.update(overrides)
    return state


class TestErrorTaxonomy(unittest.TestCase):
    """Real DuckDB and PostgreSQL error strings map onto correction classes."""

    def test_missing_column(self):
        for message in (
            'Binder Error: Referenced column "statuz" not found in FROM clause!',
            'column "statuz" does not exist',
            "no such column: statuz",
        ):
            self.assertEqual(classify_error(message), "missing_column", message)

    def test_missing_table(self):
        for message in (
            "Catalog Error: Table with name ordrs does not exist!",
            'relation "ordrs" does not exist',
            "no such table: ordrs",
        ):
            self.assertEqual(classify_error(message), "missing_table", message)

    def test_syntax(self):
        for message in (
            'Parser Error: syntax error at or near "SELEC"',
            'syntax error at or near ")"',
        ):
            self.assertEqual(classify_error(message), "syntax", message)

    def test_unrecognized_and_empty_are_misc(self):
        self.assertEqual(classify_error("connection reset by peer"), "misc")
        self.assertEqual(classify_error(""), "misc")


class TestProbeAllowlist(unittest.TestCase):
    """The model names identifiers; this module writes the SQL."""

    def test_inspect_values_builds_bounded_distinct_query(self):
        sql, label = build_probe_sql("inspect_values", "orders", "status", RETAINED)
        self.assertIn('SELECT DISTINCT "status"', sql)
        self.assertIn('FROM "orders"', sql)
        self.assertIn(f"LIMIT {DISTINCT_VALUE_LIMIT}", sql)
        self.assertEqual(label, "orders.status")

    def test_sample_rows_needs_no_column(self):
        sql, label = build_probe_sql("sample_rows", "orders", None, RETAINED)
        self.assertIn('SELECT * FROM "orders"', sql)
        self.assertEqual(label, "orders")

    def test_identifiers_resolve_case_insensitively_to_schema_spelling(self):
        sql, _ = build_probe_sql("inspect_values", "ORDERS", "STATUS", RETAINED)
        self.assertIn('"orders"', sql)
        self.assertIn('"status"', sql)

    def test_unknown_table_rejected(self):
        with self.assertRaises(ProbeRejected):
            build_probe_sql("inspect_values", "secrets", "status", RETAINED)

    def test_unknown_column_rejected(self):
        with self.assertRaises(ProbeRejected):
            build_probe_sql("inspect_values", "orders", "password", RETAINED)

    def test_unknown_tool_rejected(self):
        with self.assertRaises(ProbeRejected):
            build_probe_sql("drop_table", "orders", "status", RETAINED)

    def test_injection_attempt_in_identifier_is_rejected_not_escaped(self):
        with self.assertRaises(ProbeRejected):
            build_probe_sql("sample_rows", 'orders"; DROP TABLE orders; --', None, RETAINED)


class TestToolExecutorRejection(unittest.TestCase):
    """A rejected probe must not reach an engine."""

    def test_out_of_schema_probe_executes_no_sql(self):
        executed = []

        class RecordingAdapter:
            supports_probes = True

            def execute(self, sql, schema_context):
                executed.append(sql)
                return [], [], None

        original = generate_nodes.get_adapter
        from src.agent.nodes.loop import explore

        explore.get_adapter = lambda strategy: RecordingAdapter()
        try:
            state = loop_state(
                probe_plan=[{"tool": "inspect_values", "table": "secrets", "column": "x"}]
            )
            update = tool_executor_node(state)
        finally:
            explore.get_adapter = original

        self.assertEqual(executed, [])
        self.assertEqual(update["observations"][0]["kind"], "probe_rejected")

    def test_probe_plan_is_capped_by_budget(self):
        raw = '```json\n[{"tool":"sample_rows","table":"orders"}, {"tool":"sample_rows","table":"orders"}, {"tool":"sample_rows","table":"orders"}]\n```'
        self.assertEqual(len(_parse_plan(raw, 2)), 2)

    def test_unparseable_plan_returns_none(self):
        self.assertIsNone(_parse_plan("no fenced block here", 4))
        self.assertIsNone(_parse_plan("```json\nnot json\n```", 4))
        self.assertIsNone(_parse_plan('```json\n{"tool": "sample_rows"}\n```', 4))

    def test_empty_plan_parses_to_empty_list(self):
        self.assertEqual(_parse_plan("```json\n[]\n```", 4), [])


class TestOutcomeClassification(unittest.TestCase):
    def test_error_beats_everything(self):
        self.assertEqual(classify_outcome(loop_state(execution_error="boom")), "error")

    def test_zero_rows_is_a_correction_signal(self):
        self.assertEqual(classify_outcome(loop_state(execution_result=[])), "empty_result")

    def test_single_all_null_row_is_degenerate(self):
        state = loop_state(execution_result=[{"total": None, "city": None}])
        self.assertEqual(classify_outcome(state), "degenerate")

    def test_real_rows_are_ok(self):
        self.assertEqual(classify_outcome(loop_state()), "ok")


class TestBudgetRouting(unittest.TestCase):
    """The budget is a state counter and a conditional edge, never a recursion limit."""

    def setUp(self):
        get_settings.cache_clear()

    def tearDown(self):
        get_settings.cache_clear()

    def test_ok_forwards_immediately(self):
        state = loop_state(reflection_class="ok")
        self.assertEqual(route_after_reflect(state), "continue")

    def test_error_below_budget_retries(self):
        state = loop_state(reflection_class="error", loop_iterations=1, execution_error="boom")
        self.assertEqual(route_after_reflect(state), "retry")

    def test_error_at_budget_escalates(self):
        state = loop_state(reflection_class="error", loop_iterations=2, execution_error="boom")
        self.assertEqual(route_after_reflect(state), "escalate")

    def test_empty_result_at_budget_is_forwarded_not_escalated(self):
        """Zero rows that executed cleanly is a real answer, not a failure."""
        state = loop_state(
            reflection_class="empty_result",
            loop_iterations=2,
            execution_result=[],
            execution_error=None,
        )
        self.assertEqual(route_after_reflect(state), "continue")


class TestEscalation(unittest.TestCase):
    def test_reports_failure_without_inventing_an_answer(self):
        observations = [
            observation(
                EXECUTION_ERROR, attempt=1, error="Catalog Error: no table", correction_class="missing_table"
            ),
            observation(PROBE, label="orders.status", result="completed, shipped"),
        ]
        update = escalation_node(loop_state(observations=observations, loop_iterations=2))

        self.assertIn("could not build a working query", update["final_answer"])
        self.assertIn("Catalog Error: no table", update["final_answer"])
        self.assertIn("completed, shipped", update["final_answer"])
        self.assertIs(update["telemetry"]["execution_success"], False)
        self.assertIs(update["telemetry"]["loop"]["escalated"], True)
        self.assertEqual(update["citations"], [])


class TestValidatorNode(unittest.TestCase):
    """Validation is a fast-fail; the engine remains the real boundary."""

    def test_every_forbidden_pattern_is_refused(self):
        statements = [
            "DROP TABLE orders",
            "DELETE FROM orders",
            "UPDATE orders SET status = 'x'",
            "INSERT INTO orders VALUES (1)",
            "ATTACH 'evil.db'",
            "COPY orders TO '/tmp/out.csv'",
        ]
        for sql in statements:
            update = generate_nodes.code_validator_node(loop_state(generated_code=sql))
            self.assertIsNotNone(update["execution_error"], sql)
            self.assertEqual(update["observations"][0]["kind"], EXECUTION_ERROR)

    def test_read_only_select_passes(self):
        update = generate_nodes.code_validator_node(
            loop_state(generated_code="SELECT COUNT(*) FROM orders")
        )
        self.assertIsNone(update["execution_error"])

    def test_forbidden_pattern_list_is_covered(self):
        self.assertTrue(FORBIDDEN_DUCKDB_PATTERNS)

    def test_invalid_code_skips_execution(self):
        self.assertEqual(generate_nodes.is_valid(loop_state(execution_error="bad")), "reflect")
        self.assertEqual(generate_nodes.is_valid(loop_state(execution_error=None)), "execute")


class TestObservationRendering(unittest.TestCase):
    def test_order_is_preserved(self):
        entries = [
            observation(PROBE, label="a", result="1"),
            observation(PROBE, label="b", result="2"),
        ]
        self.assertLess(
            render_observations(entries).index("a"),
            render_observations(entries).index("b"),
        )

    def test_no_observations_adds_nothing_to_a_prompt(self):
        """The first attempt's prompt must be byte-identical to the single-pass prompt."""
        self.assertEqual(observation_prompt_block([]), "")
        self.assertEqual(observation_prompt_block(None), "")

    def test_error_observation_carries_corrective_advice(self):
        entry = observation(
            EXECUTION_ERROR, attempt=1, error="boom", correction_class="missing_column"
        )
        rendered = observation_prompt_block([entry])
        self.assertIn("boom", rendered)
        self.assertIn("column names listed in the schema", rendered)


class TestStateDiscipline(unittest.TestCase):
    """Only sequential trunk nodes may write the counter."""

    def test_only_explorer_planner_writes_loop_iterations(self):
        from src.agent.nodes.loop import explore, reflect, schema

        writers = []
        for module in (explore, generate_nodes, reflect, schema):
            source = Path(module.__file__).read_text()
            if '"loop_iterations":' in source:
                writers.append(module.__name__)

        self.assertEqual(
            writers,
            ["src.agent.nodes.loop.explore", "src.agent.nodes.loop.schema"],
            "loop_iterations must only be written at loop entry and on reset",
        )

    def test_observations_uses_an_accumulating_reducer(self):
        source = Path("src/agent/state.py").read_text()
        self.assertIn("observations: Annotated[List[Dict[str, Any]], operator.add]", source)

    def test_initial_state_seeds_loop_keys(self):
        state = initial_state("q")
        self.assertEqual(state["loop_iterations"], 0)
        self.assertEqual(state["observations"], [])


class TestGraphShape(unittest.TestCase):
    def _edges(self, compiled):
        return {(e.source, e.target) for e in compiled.get_graph().edges}

    def _has_cycle(self, compiled) -> bool:
        adjacency = {}
        for source, target in self._edges(compiled):
            adjacency.setdefault(source, []).append(target)

        visiting, done = set(), set()

        def walk(node):
            visiting.add(node)
            for nxt in adjacency.get(node, []):
                if nxt in visiting:
                    return True
                if nxt not in done and walk(nxt):
                    return True
            visiting.discard(node)
            done.add(node)
            return False

        return any(n not in done and walk(n) for n in list(adjacency))

    def test_single_pass_graph_is_acyclic(self):
        """Benchmark and evaluation runs need a fixed number of steps."""
        self.assertFalse(self._has_cycle(build_single_pass_graph()))

    def test_agentic_graph_has_the_correction_cycle(self):
        self.assertTrue(self._has_cycle(build_multi_agent_graph()))

    def test_single_pass_graph_has_no_loop_only_nodes(self):
        nodes = set(build_single_pass_graph().get_graph().nodes)
        self.assertNotIn("explorer_planner", nodes)
        self.assertNotIn("reflector", nodes)
        self.assertNotIn("escalation", nodes)

    def test_retry_edge_returns_to_the_planner(self):
        edges = self._edges(build_multi_agent_graph())
        self.assertIn(("reflector", "explorer_planner"), edges)
        self.assertIn(("reflector", "escalation"), edges)
        self.assertIn(("reflector", "projection_critic"), edges)

    def test_pandas_strategy_gets_no_probes(self):
        from src.agent.nodes.loop.engine_adapter import get_adapter, probes_available

        schema_context = {"retained_columns": RETAINED}
        self.assertFalse(probes_available(get_adapter("pandas_sandbox"), schema_context))
        self.assertTrue(probes_available(get_adapter("duckdb"), schema_context))

    def test_missing_schema_routes_out_of_the_loop(self):
        self.assertEqual(has_schema({"pruned_tables": {"table_names": []}}), "no_schema")
        self.assertEqual(has_schema({"pruned_tables": {"table_names": ["orders"]}}), "generate")

    def test_empty_probe_plan_skips_the_tool_executor(self):
        self.assertEqual(has_probes(loop_state(probe_plan=[])), "generate")
        self.assertEqual(
            has_probes(loop_state(probe_plan=[{"tool": "sample_rows", "table": "orders"}])),
            "probe",
        )

    def test_unknown_strategy_is_a_hard_error(self):
        from src.agent.nodes.loop.engine_adapter import get_adapter

        with self.assertRaises(ValueError):
            get_adapter("sqlite_someday")


class TestConfigDefaults(unittest.TestCase):
    def test_budget_and_reflection_defaults(self):
        settings = Settings()
        self.assertEqual(settings.structured_loop_max_iters, 2)
        self.assertTrue(settings.schema_exploration_enabled)
        self.assertFalse(
            settings.reflection_enabled,
            "execution feedback carries the gains; introspection is opt-in",
        )


class TestCorrectionGainMetric(unittest.TestCase):
    """
    The number that decides whether the loop ships: what correction bought,
    against the tokens it cost.
    """

    def setUp(self):
        from src.evaluation.structured_equivalence import StructuredEquivalenceEvaluator

        self.evaluator = StructuredEquivalenceEvaluator()
        self.golden = [{"n": 3}]

    def _case(self, test_id, rows, iterations, error=None):
        return {
            "test_id": test_id,
            "query": "how many",
            "engine": "duckdb",
            "df_generated": rows,
            "df_golden": self.golden,
            "iterations": iterations,
            "error": error,
        }

    def test_single_pass_run_reports_no_gain(self):
        result = self.evaluator.evaluate_benchmark(
            [self._case("a", self.golden, 1), self._case("b", None, 1, error="boom")]
        )
        self.assertEqual(result.equivalence_rate, result.first_pass_equivalence_rate)
        self.assertEqual(result.correction_gain, 0.0)
        self.assertEqual(result.mean_iterations, 1.0)

    def test_case_fixed_on_retry_shows_up_as_gain(self):
        result = self.evaluator.evaluate_benchmark(
            [self._case("a", self.golden, 1), self._case("b", self.golden, 2)]
        )
        self.assertEqual(result.equivalence_rate, 1.0)
        self.assertEqual(result.first_pass_equivalence_rate, 0.5)
        self.assertEqual(result.correction_gain, 0.5)
        self.assertEqual(result.mean_iterations, 1.5)

    def test_iterations_are_reported_per_case(self):
        result = self.evaluator.evaluate_benchmark([self._case("b", self.golden, 2)])
        self.assertEqual(result.details[0]["iterations"], 2)
        self.assertIn("correction_gain_pct", result.summary_dict())


@requires_llm
class TestLoopEndToEnd(unittest.TestCase):
    """Real LLM, real DuckDB, real fixture CSV. No stubs."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="test_loop_blobs_"))
        cls.db_manager = get_db_manager(in_memory=True)
        cls.blob_manager = get_blob_manager(base_path=cls.temp_dir)
        embedding_service = EmbeddingService()
        ingestion = StructuredIngestionEngine(
            db_manager=cls.db_manager,
            blob_manager=cls.blob_manager,
            metadata_extractor=MetadataExtractor(embedding_service=embedding_service),
        )
        csv_path = cls.temp_dir / "orders.csv"
        csv_path.write_text(SAMPLE_CSV_TEXT)
        ingestion.ingest_file(csv_path, "orders.csv", "text/csv")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_answerable_query_stays_at_one_iteration(self):
        """No-regression gate: the happy path must not pay for the loop."""
        state = run_agent("how many orders are there in total?", agentic=True)
        loop = state["telemetry"].get("loop") or {}
        self.assertEqual(loop.get("iterations"), 1)
        self.assertEqual(loop.get("outcome"), "ok")
        self.assertIsNone(state.get("execution_error"))

    def test_case_mismatch_query_is_grounded_by_a_probe(self):
        state = run_agent("how many COMPLETED orders are there?", agentic=True)
        self.assertIsNone(state.get("execution_error"))
        self.assertTrue(state.get("execution_result"))

    def test_single_pass_graph_answers_without_loop_telemetry(self):
        state = run_agent("how many orders are there in total?", agentic=False)
        self.assertIsNone(state.get("execution_error"))
        self.assertNotIn("loop", state["telemetry"])


if __name__ == "__main__":
    unittest.main()
