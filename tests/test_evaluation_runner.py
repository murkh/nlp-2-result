"""
Unit and Integration Tests for Evaluation Runner & CLI (Milestone 4).
Verifies:
  - EvaluationRunner structured, unstructured, and combined runs
  - JSON artifact serialization and persistence
  - Table formatting output
  - CLI execution with various flags (--mode, --format, --dataset, --output-dir)
"""

import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.runner import (
    EvaluationRunner,
    format_evaluation_tables,
    run_eval_cli,
)
from src.evaluation.ragas_suite import RagasEvaluationResult
from src.evaluation.structured_equivalence import StructuredBenchmarkResult


class TestEvaluationRunner(unittest.TestCase):
    """Test suite for EvaluationRunner and CLI runner."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_eval_runner_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_structured_suite_default_cases(self):
        """Verify structured benchmark run with built-in default cases."""
        runner = EvaluationRunner(output_dir=self.temp_dir)
        res = runner.run_structured_suite()

        self.assertIsInstance(res, StructuredBenchmarkResult)
        self.assertGreater(res.total_cases, 0)
        self.assertEqual(res.syntax_first_pass_rate, 1.0)
        self.assertEqual(res.equivalence_rate, 1.0)
        self.assertIn("mean_ms", res.latency_stats)
        self.assertIn("total_tokens", res.token_summary)

    def test_run_unstructured_suite_default_cases(self):
        """Verify unstructured Ragas run with built-in default cases."""
        runner = EvaluationRunner(output_dir=self.temp_dir)
        res = runner.run_unstructured_suite()

        self.assertIsInstance(res, RagasEvaluationResult)
        self.assertIn("faithfulness", res.summary)
        self.assertIn("answer_relevancy", res.summary)
        self.assertIn("context_precision", res.summary)
        self.assertIn("context_recall", res.summary)
        self.assertGreater(len(res.details), 0)

    def test_run_all_and_artifact_generation(self):
        """Verify run_all produces combined results and persists JSON artifacts."""
        runner = EvaluationRunner(output_dir=self.temp_dir)
        res = runner.run_all()

        self.assertIn("structured", res)
        self.assertIn("unstructured", res)
        self.assertIn("artifacts", res)

        summary_path = Path(res["artifacts"]["summary_path"])
        details_path = Path(res["artifacts"]["details_path"])

        self.assertTrue(summary_path.exists())
        self.assertTrue(details_path.exists())

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_json = json.load(f)
            self.assertIn("structured_summary", summary_json)
            self.assertIn("unstructured_summary", summary_json)

        with open(details_path, "r", encoding="utf-8") as f:
            details_json = json.load(f)
            self.assertIn("structured_details", details_json)
            self.assertIn("unstructured_details", details_json)

    def test_format_evaluation_tables(self):
        """Verify table formatting generates readable summary sections."""
        runner = EvaluationRunner(output_dir=self.temp_dir)
        s_res = runner.run_structured_suite()
        u_res = runner.run_unstructured_suite()

        table_str = format_evaluation_tables(structured_result=s_res, unstructured_result=u_res)
        self.assertIn("STRUCTURED EXECUTION EQUIVALENCE BENCHMARK SUMMARY", table_str)
        self.assertIn("RAGAS UNSTRUCTURED EVALUATION SUMMARY", table_str)
        self.assertIn("Per-Engine Breakdown", table_str)

    def test_cli_runner_mode_all(self):
        """Verify CLI execution in 'all' mode."""
        args = ["--mode", "all", "--output-dir", str(self.temp_dir)]
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            exit_code = run_eval_cli(args)
            self.assertEqual(exit_code, 0)
            output = fake_out.getvalue()
            self.assertIn("STRUCTURED EXECUTION EQUIVALENCE", output)
            self.assertIn("RAGAS UNSTRUCTURED EVALUATION", output)

    def test_cli_runner_mode_structured_json(self):
        """Verify CLI execution in 'structured' mode with JSON format."""
        args = ["--mode", "structured", "--format", "json", "--output-dir", str(self.temp_dir)]
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            exit_code = run_eval_cli(args)
            self.assertEqual(exit_code, 0)
            output = fake_out.getvalue().strip()
            data = json.loads(output)
            self.assertIn("total_cases", data)
            self.assertIn("syntax_first_pass_rate", data)
            self.assertIn("equivalence_rate", data)

    def test_cli_runner_mode_unstructured_json(self):
        """Verify CLI execution in 'unstructured' mode with JSON format."""
        args = ["--mode", "unstructured", "--format", "json", "--output-dir", str(self.temp_dir)]
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            exit_code = run_eval_cli(args)
            self.assertEqual(exit_code, 0)
            output = fake_out.getvalue().strip()
            data = json.loads(output)
            self.assertIn("summary", data)
            self.assertIn("faithfulness", data["summary"])


if __name__ == "__main__":
    unittest.main()
