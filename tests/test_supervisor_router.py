"""
Unit and integration tests for the tiered Supervisor Intent Classifier and Router.

Runs fully offline: embeddings use the mock provider forced by tests/conftest.py,
and the Tier 2 LLM path is exercised through an injected fake client. One live
test is gated behind requires_llm.
"""

import unittest
from types import SimpleNamespace

from src.config import get_settings
from src.database.connection import get_db_manager
from src.llm import LLMUnavailableError
from src.routing import SemanticRouteIndex, SupervisorRouter, TenantCatalog
from src.routing.schemas import LLMIntentDecision
from src.routing.semantic_index import humanize, is_distinctive_name, synthesize_anchors
from tests.conftest import requires_llm


class _FakeParseEndpoint:
    """Stands in for client.beta.chat.completions."""

    def __init__(self, parsed=None, error=None):
        self._parsed = parsed
        self._error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(parsed=self._parsed, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_client(parsed=None, error=None):
    endpoint = _FakeParseEndpoint(parsed=parsed, error=error)
    client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=endpoint)),
    )
    return client, endpoint


class TestSupervisorRouter(unittest.TestCase):
    """Tier behaviour, tenant anchors, strategy inference, and resilience."""

    def setUp(self):
        self.db_manager = get_db_manager(in_memory=True)
        self.settings = get_settings()
        self.index = SemanticRouteIndex(settings=self.settings)
        self.router = SupervisorRouter(
            db_manager=self.db_manager,
            settings=self.settings,
            semantic_index=self.index,
        )
        self.index.register_or_update_tenant(
            TenantCatalog(
                org_id="acme",
                structured_tables=["inventory_q3"],
                unstructured_docs=["travel_reimbursement_handbook.md"],
            )
        )

    # -------------------------------------------------------------------------
    # Tier 1 - local fast path
    # -------------------------------------------------------------------------

    def test_pure_greetings_hit_semantic_fastpath(self):
        """Greetings resolve locally at high confidence with no execution strategy."""
        for query in ["hello", "hey there", "good morning", "thanks a lot", "goodbye"]:
            decision = self.router.classify_intent(query)
            self.assertEqual(decision.intent, "GREETING_OR_CHITCHAT", query)
            self.assertEqual(decision.route_engine, "semantic_fastpath", query)
            self.assertGreaterEqual(decision.confidence, self.settings.router_semantic_threshold)
            self.assertIsNone(decision.suggested_strategy, query)
            self.assertIsNone(decision.clarification_question, query)

    def test_tenant_table_resolves_to_structured_query(self):
        """A query naming a registered tenant table routes structured with that dataset."""
        decision = self.router.classify_intent(
            "how many rows are in inventory_q3", org_id="acme"
        )
        self.assertEqual(decision.intent, "STRUCTURED_QUERY")
        self.assertEqual(decision.route_engine, "semantic_fastpath")
        self.assertIn("inventory_q3", decision.relevant_datasets)
        self.assertEqual(decision.suggested_strategy, "duckdb")

    def test_tenant_doc_resolves_to_unstructured_query(self):
        """A query naming a registered tenant document routes to RAG."""
        decision = self.router.classify_intent(
            "what does the travel reimbursement handbook say", org_id="acme"
        )
        self.assertEqual(decision.intent, "UNSTRUCTURED_QUERY")
        self.assertIn("travel_reimbursement_handbook.md", decision.relevant_datasets)
        self.assertIsNone(decision.suggested_strategy)

    def test_tenant_anchors_are_isolated(self):
        """One tenant's dataset names never leak into another tenant's scoring."""
        self.index.register_or_update_tenant(TenantCatalog(org_id="other"))

        _, acme_score, acme_datasets = self.index.score_query(
            "acme", "how many rows are in inventory_q3"
        )
        _, other_score, other_datasets = self.index.score_query(
            "other", "how many rows are in inventory_q3"
        )

        self.assertEqual(acme_datasets, ["inventory_q3"])
        self.assertNotIn("inventory_q3", other_datasets)
        self.assertGreater(acme_score, other_score)

    def test_missing_tenant_index_falls_back_to_global(self):
        """An unregistered org_id classifies on global anchors without raising."""
        decision = self.router.classify_intent("hello there", org_id="never-registered")
        self.assertEqual(decision.intent, "GREETING_OR_CHITCHAT")

    def test_register_or_update_tenant_is_idempotent(self):
        """Re-registering an unchanged catalog does not duplicate anchors."""
        before = len(self.index.tenant_matrices["acme"])
        self.index.register_or_update_tenant(
            TenantCatalog(
                org_id="acme",
                structured_tables=["inventory_q3"],
                unstructured_docs=["travel_reimbursement_handbook.md"],
            )
        )
        self.assertEqual(len(self.index.tenant_matrices["acme"]), before)

    # -------------------------------------------------------------------------
    # Tier 3 - ambiguity guardrail
    # -------------------------------------------------------------------------

    def test_bare_keywords_trigger_clarification(self):
        """Bare domain nouns return a clarification question, never an execution route."""
        for query in ["data", "orders", "status", "inventory"]:
            decision = self.router.classify_intent(query, org_id="acme")
            self.assertEqual(decision.intent, "AMBIGUOUS_QUERY", query)
            self.assertEqual(decision.route_engine, "heuristic_guardrail", query)
            self.assertTrue(decision.clarification_question, query)
            self.assertIsNone(decision.suggested_strategy, query)

    def test_clarification_lists_tenant_datasets(self):
        """The clarification question names the tenant's own registered assets."""
        decision = self.router.classify_intent("data", org_id="acme")
        self.assertIn("inventory_q3", decision.clarification_question)
        self.assertIn("inventory_q3", decision.relevant_datasets)

    def test_guardrail_does_not_swallow_greetings(self):
        """Short greetings are exempt from the bare-keyword guardrail."""
        decision = self.router.classify_intent("hi")
        self.assertEqual(decision.intent, "GREETING_OR_CHITCHAT")

    # -------------------------------------------------------------------------
    # Tier 2 - LLM fallback
    # -------------------------------------------------------------------------

    def _grey_zone_router(self, client):
        """A router whose thresholds force every query into the grey zone."""
        settings = self.settings.model_copy(
            update={"router_semantic_threshold": 0.999, "router_ambiguity_threshold": 0.0}
        )
        return SupervisorRouter(
            db_manager=self.db_manager,
            settings=settings,
            semantic_index=self.index,
            llm_client=client,
        )

    def test_grey_zone_uses_llm_fallback(self):
        """A grey-zone score defers to structured-output parsing exactly once."""
        client, endpoint = _fake_client(
            parsed=LLMIntentDecision(
                intent="STRUCTURED_QUERY",
                confidence=0.66,
                reasoning="Aggregation over a table.",
                suggested_strategy="duckdb",
                clarification_question=None,
            )
        )
        router = self._grey_zone_router(client)

        decision = router.classify_intent("revenue trend breakdown last quarter", org_id="acme")

        self.assertEqual(decision.route_engine, "llm_fallback")
        self.assertEqual(decision.intent, "STRUCTURED_QUERY")
        self.assertAlmostEqual(decision.confidence, 0.66)
        self.assertEqual(len(endpoint.calls), 1)
        self.assertIs(endpoint.calls[0]["response_format"], LLMIntentDecision)

    def test_multi_turn_history_forces_llm_fallback(self):
        """Conversational context bypasses the fast path and reaches the LLM."""
        client, endpoint = _fake_client(
            parsed=LLMIntentDecision(
                intent="STRUCTURED_QUERY",
                confidence=0.8,
                reasoning="Follow-up on the prior aggregation.",
                suggested_strategy="duckdb",
                clarification_question=None,
            )
        )
        router = SupervisorRouter(
            db_manager=self.db_manager,
            settings=self.settings,
            semantic_index=self.index,
            llm_client=client,
        )

        decision = router.classify_intent(
            "how many rows are in inventory_q3",
            org_id="acme",
            history=[
                {"role": "user", "content": "show me the inventory"},
                {"role": "assistant", "content": "Which quarter?"},
            ],
        )

        self.assertEqual(decision.route_engine, "llm_fallback")
        self.assertEqual(len(endpoint.calls), 1)
        roles = [m["role"] for m in endpoint.calls[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_llm_failure_is_a_hard_failure(self):
        """An unreachable classifier raises; it never downgrades to a guess."""
        client, endpoint = _fake_client(error=RuntimeError("connection reset"))
        router = self._grey_zone_router(client)

        with self.assertRaises(LLMUnavailableError):
            router.classify_intent("total revenue by region", org_id="acme")
        self.assertEqual(len(endpoint.calls), 1)

    def test_llm_refusal_is_a_hard_failure(self):
        """A structured-output response with no parsed payload raises."""
        client, _ = _fake_client(parsed=None)
        router = self._grey_zone_router(client)

        with self.assertRaises(LLMUnavailableError):
            router.classify_intent("total revenue by region", org_id="acme")

    def test_llm_ambiguous_without_question_falls_back_to_guardrail(self):
        """An AMBIGUOUS_QUERY verdict always carries a clarification question."""
        client, _ = _fake_client(
            parsed=LLMIntentDecision(
                intent="AMBIGUOUS_QUERY",
                confidence=0.4,
                reasoning="Underspecified.",
                suggested_strategy=None,
                clarification_question=None,
            )
        )
        router = self._grey_zone_router(client)

        decision = router.classify_intent("something about that thing", org_id="acme")

        self.assertEqual(decision.intent, "AMBIGUOUS_QUERY")
        self.assertTrue(decision.clarification_question)

    # -------------------------------------------------------------------------
    # Strategy inference, catalog caching, anchor synthesis
    # -------------------------------------------------------------------------

    def test_strategy_inference(self):
        """Engine hints in the query override the duckdb default."""
        self.assertEqual(self.router._infer_strategy("count orders"), "duckdb")
        self.assertEqual(
            self.router._infer_strategy("use pandas to count orders"), "pandas_sandbox"
        )
        self.assertEqual(
            self.router._infer_strategy("query the dedicated postgres table"), "dedicated_db"
        )

    def test_classify_intent_does_not_query_the_catalog(self):
        """The catalog is read once per org_id, never inside classify_intent."""
        calls = []
        original = self.db_manager.list_datasets

        def counting_list_datasets(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        self.db_manager.list_datasets = counting_list_datasets
        try:
            for _ in range(5):
                self.router.classify_intent("hello", org_id="fresh-org")
        finally:
            self.db_manager.list_datasets = original

        self.assertEqual(len(calls), 1)

    def test_synthesize_anchors_needs_no_manual_tagging(self):
        """Table and document names alone yield usable query phrasings."""
        self.assertEqual(humanize("sales_orders.csv"), "sales orders")
        structured = synthesize_anchors("sales_orders", "structured")
        self.assertIn("query sales orders", structured)
        self.assertIn("how many sales orders", structured)
        unstructured = synthesize_anchors("hr_handbook.md", "unstructured")
        self.assertIn("search hr handbook for details", unstructured)

    # -------------------------------------------------------------------------
    # Auto-learning guards - bad names must not become routing anchors
    # -------------------------------------------------------------------------

    def test_generic_names_are_refused_as_anchors(self):
        """Names too vague to identify a dataset produce no anchors at all."""
        for name in ["data", "data.csv", "orders", "Sheet1", "untitled", "q3", "2024", "v2", ""]:
            self.assertFalse(is_distinctive_name(name), name)
            self.assertEqual(synthesize_anchors(name, "structured"), [], name)

        for name in ["inventory_q3", "sales_orders", "hr_handbook.md", "travel_policy_eu"]:
            self.assertTrue(is_distinctive_name(name), name)

    def test_generic_table_does_not_hijack_unrelated_queries(self):
        """A table literally named 'data' must not claim every query."""
        self.index.register_or_update_tenant(
            TenantCatalog(org_id="loose", structured_tables=["data", "inventory_q3"])
        )
        decision = self.router.classify_intent(
            "what does the security policy say about data retention", org_id="loose"
        )
        self.assertNotIn("data", decision.relevant_datasets)
        self.assertEqual(decision.intent, "UNSTRUCTURED_QUERY")

    def test_dataset_names_match_on_word_boundaries(self):
        """A name must not be claimed by a longer, unrelated word containing it."""
        self.index.register_or_update_tenant(
            TenantCatalog(org_id="bounded", structured_tables=["metric_ton_shipments"])
        )
        patterns = [p for p, _, _ in self.index._tenant_lexicon["bounded"]]

        self.assertFalse(
            any(p.search("how many metric_ton_shipments_archive_v2 rows") for p in patterns)
        )
        self.assertTrue(any(p.search("how many metric_ton_shipments were sent") for p in patterns))
        self.assertTrue(any(p.search("count the metric ton shipments") for p in patterns))

        intent, score, datasets = self.index.score_query(
            "bounded", "how many metric_ton_shipments were sent"
        )
        self.assertEqual(intent, "STRUCTURED_QUERY")
        self.assertEqual(datasets, ["metric_ton_shipments"])
        self.assertGreaterEqual(score, 0.95)

    def test_longer_dataset_name_wins_over_its_prefix(self):
        """Overlapping names resolve to the most specific match first."""
        self.index.register_or_update_tenant(
            TenantCatalog(
                org_id="overlap",
                structured_tables=["regional_shipments", "regional_shipments_europe"],
            )
        )
        patterns = [p.pattern for p, _, _ in self.index._tenant_lexicon["overlap"]]
        self.assertGreater(len(patterns[0]), len(patterns[-1]))

    def test_sync_catalog_reads_dataset_categories(self):
        """sync_catalog splits the dataset catalog by category."""
        catalog = self.router.sync_catalog(org_id="synced")
        self.assertEqual(catalog.org_id, "synced")
        self.assertIn("synced", self.index.tenant_catalogs)

    @requires_llm
    def test_live_llm_fallback_returns_valid_intent(self):
        """The real structured-output call yields a schema-valid decision."""
        settings = self.settings.model_copy(
            update={"router_semantic_threshold": 0.999, "router_ambiguity_threshold": 0.0}
        )
        router = SupervisorRouter(
            db_manager=self.db_manager, settings=settings, semantic_index=self.index
        )

        decision = router.classify_intent(
            "how many rows are in inventory_q3", org_id="acme"
        )

        self.assertEqual(decision.route_engine, "llm_fallback")
        self.assertIn(
            decision.intent,
            [
                "GREETING_OR_CHITCHAT",
                "AMBIGUOUS_QUERY",
                "STRUCTURED_QUERY",
                "UNSTRUCTURED_QUERY",
            ],
        )


if __name__ == "__main__":
    unittest.main()
