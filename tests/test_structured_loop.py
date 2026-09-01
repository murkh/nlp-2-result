"""
Structured self-correction loop tests.

The deterministic suite covers the pure logic -- error classification, code
validation, budget routing, reducer discipline and graph shape -- with no LLM and
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
from src.agent.nodes.loop.reflect import (
    classify_outcome,
    escalation_node,
    route_after_reflect,
)
from src.agent.nodes.loop.schema import has_schema
from src.agent.state import AgentState
from src.config import Settings, get_settings
from src.database.connection import get_db_manager
from src.feedback import (
    EXECUTION_ERROR,
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
        "pruned_tables": {
            "table_names": ["orders"],
            "retained_columns": RETAINED,
            "column_roles": {"status": "display"},
            "file_paths": {"orders": "/tmp/orders.csv"},
        },
        "schema_ddl": "CREATE TABLE orders (...)",
        "generated_code": "result = {'count': len(df)}",
        "execution_result": [{"count": 3}],
        "execution_columns": ["count"],
        "execution_error": None,
        "observations": [],
        "loop_iterations": 1,
        "reflection_class": None,
        "telemetry": {},
    }
    state.update(overrides)
    return state


class TestErrorTaxonomy(unittest.TestCase):
    """Real engine error strings map onto correction classes."""

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
                EXECUTION_ERROR,
                attempt=1,
                error="KeyError: 'statuz'",
                correction_class="missing_column",
            ),
        ]
        update = escalation_node(loop_state(observations=observations, loop_iterations=2))

        self.assertIn("could not build a working query", update["final_answer"])
        self.assertIn("KeyError: 'statuz'", update["final_answer"])
        self.assertIs(update["telemetry"]["execution_success"], False)
        self.assertIs(update["telemetry"]["loop"]["escalated"], True)
        self.assertEqual(update["citations"], [])


class TestValidatorNode(unittest.TestCase):
    """Validation is a fast-fail; the sandbox remains the real boundary."""

    def test_every_forbidden_construct_is_refused(self):
        snippets = [
            "import os\nresult = {'n': 1}",
            "from subprocess import run\nresult = {'n': 1}",
            "result = {'n': eval('1+1')}",
            "open('/etc/passwd').read()",
            "result = {'n': ().__class__}",
            "result = {'n': ",
        ]
        for code in snippets:
            update = generate_nodes.code_validator_node(loop_state(generated_code=code))
            self.assertIsNotNone(update["execution_error"], code)
            self.assertIsInstance(update["execution_error"], str, code)
            self.assertEqual(update["observations"][0]["kind"], EXECUTION_ERROR)

    def test_whitelisted_dataframe_code_passes(self):
        update = generate_nodes.code_validator_node(
            loop_state(generated_code="import pandas as pd\nresult = {'count': len(df)}")
        )
        self.assertIsNone(update["execution_error"])

    def test_invalid_code_skips_execution(self):
        self.assertEqual(generate_nodes.is_valid(loop_state(execution_error="bad")), "reflect")
        self.assertEqual(generate_nodes.is_valid(loop_state(execution_error=None)), "execute")


class TestObservationRendering(unittest.TestCase):
    def test_order_is_preserved(self):
        entries = [
            observation(EXECUTION_ERROR, attempt=1, error="a", correction_class="misc"),
            observation(EXECUTION_ERROR, attempt=2, error="b", correction_class="misc"),
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

    def test_only_code_generator_writes_loop_iterations(self):
        from src.agent.nodes.loop import reflect, schema

        writers = []
        for module in (generate_nodes, reflect, schema):
            source = Path(module.__file__).read_text()
            if '"loop_iterations":' in source:
                writers.append(module.__name__)

        self.assertEqual(
            writers,
            ["src.agent.nodes.loop.generate", "src.agent.nodes.loop.schema"],
            "loop_iterations must only be written at attempt entry and on reset",
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
        """Evaluation runs need a fixed number of steps."""
        self.assertFalse(self._has_cycle(build_single_pass_graph()))

    def test_agentic_graph_has_the_correction_cycle(self):
        self.assertTrue(self._has_cycle(build_multi_agent_graph()))

    def test_single_pass_graph_has_no_loop_only_nodes(self):
        nodes = set(build_single_pass_graph().get_graph().nodes)
        self.assertNotIn("reflector", nodes)
        self.assertNotIn("escalation", nodes)

    def test_no_graph_retains_the_removed_sql_nodes(self):
        for compiled in (build_single_pass_graph(), build_multi_agent_graph()):
            nodes = set(compiled.get_graph().nodes)
            self.assertNotIn("explorer_planner", nodes)
            self.assertNotIn("tool_executor", nodes)
            self.assertNotIn("projection_critic", nodes)

    def test_retry_edge_returns_to_the_generator(self):
        edges = self._edges(build_multi_agent_graph())
        self.assertIn(("reflector", "code_generator"), edges)
        self.assertIn(("reflector", "escalation"), edges)
        self.assertIn(("reflector", "synthesizer"), edges)

    def test_missing_schema_routes_out_of_the_loop(self):
        self.assertEqual(has_schema({"pruned_tables": {"table_names": []}}), "no_schema")
        self.assertEqual(has_schema({"pruned_tables": {"table_names": ["orders"]}}), "generate")

    def test_generator_owns_the_attempt_counter(self):
        """The retry target must advance the budget or the cycle cannot terminate."""
        engine = generate_nodes.get_sandbox_engine

        class StubEngine:
            def generate_code(self, *args, **kwargs):
                return "result = {'n': 1}", "thought", (1, 1)

            def apply_dataset_loader(self, code, **kwargs):
                return code

        generate_nodes.get_sandbox_engine = lambda: StubEngine()
        try:
            first = generate_nodes.code_generator_node(loop_state(loop_iterations=0))
            second = generate_nodes.code_generator_node(loop_state(loop_iterations=1))
        finally:
            generate_nodes.get_sandbox_engine = engine

        self.assertEqual(first["loop_iterations"], 1)
        self.assertEqual(second["loop_iterations"], 2)


class TestConfigDefaults(unittest.TestCase):
    def test_budget_and_reflection_defaults(self):
        settings = Settings()
        self.assertEqual(settings.structured_loop_max_iters, 2)
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
            "engine": "pandas_sandbox",
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
    """Real LLM, real sandbox, real fixture CSV. No stubs."""

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

    def test_case_mismatch_query_still_answers(self):
        state = run_agent("how many COMPLETED orders are there?", agentic=True)
        self.assertIsNone(state.get("execution_error"))
        self.assertTrue(state.get("execution_result"))

    def test_single_pass_graph_answers_without_loop_telemetry(self):
        state = run_agent("how many orders are there in total?", agentic=False)
        self.assertIsNone(state.get("execution_error"))
        self.assertNotIn("loop", state["telemetry"])


if __name__ == "__main__":
    unittest.main()
