"""
Unit tests for the Projection Critic node and the column-role heuristics that feed it.

The gate tests matter most: they prove the node makes no LLM call and no second
database round trip on queries whose projection is already adequate.
"""

import unittest
from unittest.mock import patch

from src.agent.nodes.projection_critic import (
    is_projection_thin,
    is_widening_safe,
    projection_critic_node,
)
from src.pruning.schema_pruner import classify_column_role

PO_CONTEXT = {
    "table_names": ["purchase_orders"],
    "retained_columns": {
        "purchase_orders": [
            "purchase_order_id",
            "purchase_order_no",
            "purchase_order_value",
            "grn_total_amount",
            "created_date",
        ]
    },
    "column_roles": {
        "purchase_order_no": "display",
        "purchase_order_value": "measure",
        "grn_total_amount": "measure",
        "created_date": "date",
    },
    "file_paths": {"purchase_orders": "/tmp/po.csv"},
}


class TestColumnRoleClassification(unittest.TestCase):
    """Roles must mark the human-readable identifier and the measures, not the id."""

    def test_display_columns(self):
        self.assertEqual(classify_column_role("purchase_order_no", "VARCHAR"), "display")
        self.assertEqual(classify_column_role("customer_name", "TEXT"), "display")
        self.assertEqual(classify_column_role("item_code", "VARCHAR"), "display")
        self.assertEqual(classify_column_role("status", "VARCHAR"), "display")

    def test_measure_columns(self):
        self.assertEqual(classify_column_role("purchase_order_value", "DECIMAL"), "measure")
        self.assertEqual(classify_column_role("grn_total_amount", "DOUBLE"), "measure")
        self.assertEqual(classify_column_role("unit_price", "FLOAT"), "measure")
        self.assertEqual(classify_column_role("quantity", "INTEGER"), "measure")

    def test_date_columns(self):
        self.assertEqual(classify_column_role("created_date", "DATE"), "date")
        self.assertEqual(classify_column_role("delivered_at", "TIMESTAMP"), "date")

    def test_id_and_unknown_columns_have_no_role(self):
        # The id keeps its PK/FK handling; it must not be promoted to display.
        self.assertIsNone(classify_column_role("purchase_order_id", "VARCHAR"))
        self.assertIsNone(classify_column_role("vendor_id", "INTEGER"))
        self.assertIsNone(classify_column_role("some_flag", "BOOLEAN"))

    def test_numeric_column_named_like_a_display_is_not_display(self):
        # A numeric 'amount' is a measure, never a label.
        self.assertEqual(classify_column_role("total_amount", "DECIMAL"), "measure")


class TestProjectionGate(unittest.TestCase):
    """The deterministic gate. No LLM is constructed by any test in this class."""

    def test_bare_id_with_comparison_is_thin(self):
        sql = (
            "SELECT purchase_order_id FROM purchase_orders "
            "WHERE grn_total_amount > purchase_order_value LIMIT 20"
        )
        thin, missing = is_projection_thin(sql, PO_CONTEXT)
        self.assertTrue(thin)
        # The compared measures and the human-readable identifier are all required.
        self.assertIn("grn_total_amount", missing)
        self.assertIn("purchase_order_value", missing)
        self.assertIn("purchase_order_no", missing)

    def test_select_star_is_not_thin(self):
        sql = "SELECT * FROM purchase_orders WHERE grn_total_amount > purchase_order_value"
        thin, missing = is_projection_thin(sql, PO_CONTEXT)
        self.assertFalse(thin)
        self.assertEqual(missing, [])

    def test_pure_scalar_aggregate_is_not_thin(self):
        thin, _ = is_projection_thin("SELECT COUNT(*) FROM purchase_orders", PO_CONTEXT)
        self.assertFalse(thin)

    def test_already_wide_projection_is_not_thin(self):
        sql = (
            "SELECT purchase_order_id, purchase_order_no, purchase_order_value, "
            "grn_total_amount FROM purchase_orders "
            "WHERE grn_total_amount > purchase_order_value LIMIT 20"
        )
        thin, missing = is_projection_thin(sql, PO_CONTEXT)
        self.assertFalse(thin)
        self.assertEqual(missing, [])

    def test_unparseable_or_empty_input_is_not_thin(self):
        self.assertFalse(is_projection_thin("", PO_CONTEXT)[0])
        self.assertFalse(is_projection_thin("SELECT 1", PO_CONTEXT)[0])
        self.assertFalse(is_projection_thin("SELECT po_id FROM t", {})[0])


class TestWideningSafety(unittest.TestCase):
    """A rewrite may only add columns. Anything else is rejected."""

    ORIGINAL = (
        "SELECT purchase_order_id FROM purchase_orders "
        "WHERE grn_total_amount > purchase_order_value LIMIT 20"
    )

    def test_added_columns_accepted(self):
        widened = (
            "SELECT purchase_order_id, purchase_order_no, grn_total_amount "
            "FROM purchase_orders WHERE grn_total_amount > purchase_order_value LIMIT 20"
        )
        safe, reason = is_widening_safe(self.ORIGINAL, widened)
        self.assertTrue(safe, reason)

    def test_altered_where_clause_rejected(self):
        widened = (
            "SELECT purchase_order_id, purchase_order_no FROM purchase_orders "
            "WHERE grn_total_amount > 100 LIMIT 20"
        )
        safe, reason = is_widening_safe(self.ORIGINAL, widened)
        self.assertFalse(safe)
        self.assertIn("row-selecting clause", reason)

    def test_dropped_column_rejected(self):
        widened = (
            "SELECT purchase_order_no FROM purchase_orders "
            "WHERE grn_total_amount > purchase_order_value LIMIT 20"
        )
        safe, reason = is_widening_safe(self.ORIGINAL, widened)
        self.assertFalse(safe)
        self.assertIn("dropped", reason)

    def test_forbidden_statement_rejected(self):
        safe, _ = is_widening_safe(self.ORIGINAL, "DROP TABLE purchase_orders")
        self.assertFalse(safe)

    def test_empty_rewrite_rejected(self):
        self.assertFalse(is_widening_safe(self.ORIGINAL, "")[0])


class TestCriticNode(unittest.TestCase):
    """Node-level behavior: free pass-through, and fall-through on every failure."""

    THIN_STATE = {
        "suggested_strategy": "duckdb",
        "generated_code": (
            "SELECT purchase_order_id FROM purchase_orders "
            "WHERE grn_total_amount > purchase_order_value LIMIT 20"
        ),
        "execution_result": [{"purchase_order_id": "PO-1"}],
        "execution_columns": ["purchase_order_id"],
        "execution_error": None,
        "pruned_tables": PO_CONTEXT,
        "telemetry": {},
    }

    def test_wide_projection_makes_no_llm_call(self):
        state = dict(self.THIN_STATE)
        state["generated_code"] = "SELECT * FROM purchase_orders WHERE grn_total_amount > 1"
        with patch("src.agent.nodes.projection_critic._widen_sql") as widen:
            self.assertEqual(projection_critic_node(state), {})
            widen.assert_not_called()

    def test_upstream_error_passes_through(self):
        state = dict(self.THIN_STATE, execution_error="boom")
        with patch("src.agent.nodes.projection_critic._widen_sql") as widen:
            self.assertEqual(projection_critic_node(state), {})
            widen.assert_not_called()

    def test_pandas_strategy_passes_through(self):
        state = dict(self.THIN_STATE, suggested_strategy="pandas_sandbox")
        with patch("src.agent.nodes.projection_critic._widen_sql") as widen:
            self.assertEqual(projection_critic_node(state), {})
            widen.assert_not_called()

    def test_zero_rows_passes_through(self):
        state = dict(self.THIN_STATE, execution_result=[])
        with patch("src.agent.nodes.projection_critic._widen_sql") as widen:
            self.assertEqual(projection_critic_node(state), {})
            widen.assert_not_called()

    def test_llm_failure_keeps_original_result(self):
        with patch(
            "src.agent.nodes.projection_critic._widen_sql", side_effect=RuntimeError("no llm")
        ):
            self.assertEqual(projection_critic_node(dict(self.THIN_STATE)), {})

    def test_unsafe_rewrite_keeps_original_result(self):
        bad = "SELECT purchase_order_id, purchase_order_no FROM purchase_orders WHERE 1=1"
        with patch(
            "src.agent.nodes.projection_critic._widen_sql", return_value=(bad, (10, 5))
        ):
            self.assertEqual(projection_critic_node(dict(self.THIN_STATE)), {})

    def test_reexecution_failure_keeps_original_result(self):
        good = (
            "SELECT purchase_order_id, purchase_order_no, purchase_order_value, "
            "grn_total_amount FROM purchase_orders "
            "WHERE grn_total_amount > purchase_order_value LIMIT 20"
        )
        with patch(
            "src.agent.nodes.projection_critic._widen_sql", return_value=(good, (10, 5))
        ), patch(
            "src.agent.nodes.projection_critic._re_execute",
            return_value=([], [], "table not found"),
        ):
            self.assertEqual(projection_critic_node(dict(self.THIN_STATE)), {})

    def test_successful_widening_replaces_result_and_clears_stale_answer(self):
        good = (
            "SELECT purchase_order_id, purchase_order_no, purchase_order_value, "
            "grn_total_amount FROM purchase_orders "
            "WHERE grn_total_amount > purchase_order_value LIMIT 20"
        )
        widened_cols = [
            "purchase_order_id",
            "purchase_order_no",
            "purchase_order_value",
            "grn_total_amount",
        ]
        widened_rows = [
            {
                "purchase_order_id": "PO-1",
                "purchase_order_no": "PO-1001",
                "purchase_order_value": 100.0,
                "grn_total_amount": 120.0,
            }
        ]
        with patch(
            "src.agent.nodes.projection_critic._widen_sql", return_value=(good, (10, 5))
        ), patch(
            "src.agent.nodes.projection_critic._re_execute",
            return_value=(widened_cols, widened_rows, None),
        ):
            out = projection_critic_node(dict(self.THIN_STATE))

        self.assertEqual(out["execution_columns"], widened_cols)
        self.assertEqual(out["execution_result"], widened_rows)
        self.assertEqual(out["generated_code"], good)
        # The engine's narrow one-line preview must not survive into synthesis.
        self.assertIsNone(out["final_answer"])
        self.assertTrue(out["telemetry"]["projection_critic"]["fired"])

    def test_disabled_by_setting(self):
        from src.config import get_settings

        get_settings.cache_clear()
        with patch.dict("os.environ", {"PROJECTION_CRITIC_ENABLED": "false"}):
            get_settings.cache_clear()
            with patch("src.agent.nodes.projection_critic._widen_sql") as widen:
                self.assertEqual(projection_critic_node(dict(self.THIN_STATE)), {})
                widen.assert_not_called()
        get_settings.cache_clear()


class TestGraphTopology(unittest.TestCase):
    """The critic sits on the structured path only."""

    def test_structured_path_traverses_critic(self):
        from src.agent.graph import build_multi_agent_graph

        graph = build_multi_agent_graph()
        nodes = graph.get_graph().nodes
        self.assertIn("projection_critic", nodes)

        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        self.assertIn(("structured_agent", "projection_critic"), edges)
        self.assertIn(("projection_critic", "synthesizer"), edges)
        # The unstructured path is untouched.
        self.assertIn(("unstructured_agent", "synthesizer"), edges)
        self.assertNotIn(("unstructured_agent", "projection_critic"), edges)


if __name__ == "__main__":
    unittest.main()
