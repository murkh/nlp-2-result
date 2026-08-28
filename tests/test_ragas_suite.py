"""
Unit and Integration Tests for Ragas Unstructured Evaluation Suite (Milestone 4).
Verifies:
  - Faithfulness metric (perfect on grounded, degraded on hallucinated, zero on empty context)
  - Answer Relevancy metric (perfect on identical/direct, zero on unrelated or empty)
  - Context Precision metric (ranking order, MRR/Precision@k, nan-safe on empty ground truth)
  - Context Recall metric (ground-truth fact containment, partial retrieval, zero on absent)
  - RagasEvaluator batch runner, custom metrics validation, and result representations.
"""

import unittest

from src.evaluation.compat import np, pd
from src.evaluation.ragas_suite import (
    RagasEvaluationResult,
    RagasEvaluator,
    answer_relevancy,
    calculate_answer_relevancy,
    calculate_context_precision,
    calculate_context_recall,
    calculate_faithfulness,
    context_precision,
    context_recall,
    faithfulness,
)


class TestRagasSuite(unittest.TestCase):
    """Test suite for Ragas evaluation metrics and RagasEvaluator harness."""

    def test_faithfulness_perfect_score_when_grounded(self):
        """Verify faithfulness is 1.0 when all answer claims are supported by context."""
        context = ["Subprocess sandboxing enforces AST whitelisting and 512MB RAM."]
        question = "How is subprocess memory limited?"
        answer = "Subprocess sandboxing enforces 512MB RAM limit."
        score = calculate_faithfulness(question=question, answer=answer, contexts=context)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_faithfulness_zero_on_empty_contexts(self):
        """Verify boundary condition: zero contexts returns 0.0 faithfulness."""
        score = calculate_faithfulness(
            question="What is DuckDB?", answer="DuckDB is in-memory SQL.", contexts=[]
        )
        self.assertEqual(score, 0.0)

        score_none = calculate_faithfulness(
            question="What is DuckDB?", answer="DuckDB is in-memory SQL.", contexts=None
        )
        self.assertEqual(score_none, 0.0)

    def test_faithfulness_zero_on_empty_answer(self):
        """Verify boundary condition: empty answer returns 0.0 faithfulness."""
        context = ["Some context document."]
        score = calculate_faithfulness(question="What is it?", answer="", contexts=context)
        self.assertEqual(score, 0.0)

    def test_faithfulness_degraded_on_hallucinated_claims(self):
        """Verify faithfulness drops when answer includes hallucinated or ungrounded facts."""
        context = ["Returns are accepted within 30 days of initial purchase with receipt."]
        question = "What is the return policy?"
        hallucinated_answer = (
            "Returns are accepted within 30 days. Customers also receive a free $500 gift card."
        )
        score = calculate_faithfulness(
            question=question, answer=hallucinated_answer, contexts=context
        )
        self.assertLess(score, 1.0)
        self.assertGreaterEqual(score, 0.0)

    def test_answer_relevancy_identical_question_answer(self):
        """Verify identical question and answer yields perfect 1.0 score."""
        q = "What is Strategy A?"
        ans = "What is Strategy A?"
        score = calculate_answer_relevancy(question=q, answer=ans)
        self.assertEqual(score, 1.0)

    def test_answer_relevancy_direct_factual_answer(self):
        """Verify high answer relevancy when answer directly addresses the query."""
        q = "How does DuckDB execute queries?"
        ans = "DuckDB executes queries directly in-memory over blob Parquet files."
        score = calculate_answer_relevancy(question=q, answer=ans)
        self.assertGreater(score, 0.8)

    def test_answer_relevancy_zero_on_completely_unrelated_answer(self):
        """Verify answer relevancy is 0.0 when answer is completely off-topic."""
        q = "How does DuckDB work?"
        ans = "The recipe for chocolate cake includes sugar, flour, and cocoa powder."
        score = calculate_answer_relevancy(question=q, answer=ans)
        self.assertEqual(score, 0.0)

    def test_answer_relevancy_zero_on_empty_inputs(self):
        """Verify answer relevancy is 0.0 on empty question or answer."""
        self.assertEqual(calculate_answer_relevancy("", "Some answer"), 0.0)
        self.assertEqual(calculate_answer_relevancy("Some question", ""), 0.0)

    def test_context_precision_perfect_at_rank_one(self):
        """Verify context precision is 1.0 when the first retrieved chunk contains ground truth."""
        contexts = [
            "Relevant context chunk A containing target info.",
            "Irrelevant background noise B.",
            "Irrelevant background noise C.",
        ]
        gt = "Relevant context chunk A containing target info."
        score = calculate_context_precision(
            question="What is chunk A?", contexts=contexts, ground_truth=gt
        )
        self.assertEqual(score, 1.0)

    def test_context_precision_lower_when_relevant_chunk_ranked_lower(self):
        """Verify context precision decreases when relevant context appears later in the ranking."""
        contexts_good = ["Target chunk info.", "Distractor chunk."]
        contexts_bad = ["Distractor chunk.", "Target chunk info."]
        gt = "Target chunk info."

        score_good = calculate_context_precision(
            question="Query", contexts=contexts_good, ground_truth=gt
        )
        score_bad = calculate_context_precision(
            question="Query", contexts=contexts_bad, ground_truth=gt
        )

        self.assertGreater(score_good, score_bad)

    def test_context_precision_nan_safe_on_empty_ground_truth(self):
        """Verify context precision safely returns 0.0 (nan-safe) on empty ground truth."""
        score = calculate_context_precision(question="", contexts=["Some context"], ground_truth="")
        self.assertEqual(score, 0.0)

    def test_context_precision_zero_on_empty_contexts(self):
        """Verify context precision safely returns 0.0 on empty contexts."""
        score = calculate_context_precision(question="Q", contexts=[], ground_truth="Target")
        self.assertEqual(score, 0.0)

    def test_context_recall_full_recalled_facts(self):
        """Verify context recall is 1.0 when all ground-truth facts are in contexts."""
        contexts = [
            "Fact A is true and verified. Fact B is also established.",
            "Supplementary notes.",
        ]
        gt = "Fact A; Fact B"
        score = calculate_context_recall(ground_truth=gt, contexts=contexts)
        self.assertEqual(score, 1.0)

    def test_context_recall_partial_and_zero_recalled_facts(self):
        """Verify context recall is proportional to recalled facts."""
        contexts = ["Fact A is confirmed in production."]
        gt = "Fact A; Fact B"
        score_partial = calculate_context_recall(ground_truth=gt, contexts=contexts)
        self.assertAlmostEqual(score_partial, 0.5, places=2)

        contexts_none = ["Completely unrelated documentation."]
        score_zero = calculate_context_recall(ground_truth=gt, contexts=contexts_none)
        self.assertEqual(score_zero, 0.0)

    def test_context_recall_empty_inputs(self):
        """Verify context recall is 0.0 on empty ground truth or contexts."""
        self.assertEqual(calculate_context_recall("", ["Context"]), 0.0)
        self.assertEqual(calculate_context_recall("Ground truth", []), 0.0)

    def test_evaluator_invalid_metric_name_raises_value_error(self):
        """Verify initializing evaluator with invalid metric name raises ValueError."""
        with self.assertRaises(ValueError):
            RagasEvaluator(metrics=["unsupported_metric_xyz"])

    def test_evaluator_custom_metric_selection(self):
        """Verify evaluator can be initialized with a custom subset of metrics."""
        evaluator = RagasEvaluator(metrics=["faithfulness", "answer_relevancy"])
        self.assertEqual(evaluator.metric_names, ["faithfulness", "answer_relevancy"])

        result = evaluator.evaluate_single(
            question="What is 512MB RAM?",
            answer="512MB RAM is the sandbox memory limit.",
            contexts=["Subprocess sandbox has 512MB RAM limit."],
        )
        self.assertIn("faithfulness", result)
        self.assertIn("answer_relevancy", result)
        self.assertNotIn("context_precision", result)

    def test_evaluator_batch_test_cases_dataframe_and_dicts(self):
        """Verify batch evaluation over list of dicts and pandas DataFrame."""
        test_cases = [
            {
                "question": "What is the return window?",
                "contexts": ["Returns are accepted within 30 days of purchase."],
                "answer": "Returns are accepted within 30 days.",
                "ground_truth": "The return window is 30 days.",
            },
            {
                "question": "What is the sandbox timeout?",
                "contexts": ["Subprocess sandbox timeout watchdog terminates after 5.0 seconds."],
                "answer": "The sandbox timeout is 5.0 seconds.",
                "ground_truth": "Sandbox timeout is 5.0 seconds.",
            },
        ]

        evaluator = RagasEvaluator()
        result_from_dicts = evaluator.evaluate_test_cases(test_cases)

        self.assertIsInstance(result_from_dicts, RagasEvaluationResult)
        self.assertIn("faithfulness", result_from_dicts.summary)
        self.assertIn("answer_relevancy", result_from_dicts.summary)
        self.assertIn("context_precision", result_from_dicts.summary)
        self.assertIn("context_recall", result_from_dicts.summary)
        self.assertGreaterEqual(result_from_dicts.summary["faithfulness"], 0.9)
        self.assertGreaterEqual(result_from_dicts.summary["answer_relevancy"], 0.8)

        # Verify DataFrame conversion
        df_res = result_from_dicts.to_dataframe()
        self.assertEqual(len(df_res), 2)
        self.assertIn("faithfulness", df_res.columns)

        # Verify running from DataFrame input directly
        df_input = pd.DataFrame(test_cases)
        result_from_df = evaluator.evaluate(df_input)
        self.assertEqual(len(result_from_df.details), 2)
        self.assertAlmostEqual(
            result_from_dicts.summary["faithfulness"],
            result_from_df.summary["faithfulness"],
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
