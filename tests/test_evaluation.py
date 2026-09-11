import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fyp.evaluate import build_parser, run, summarize_predictions
from fyp.execution import (ensure_docker, extract_first_code, grade_execution,
                           parse_scalar, wrap_last_expression)


class EvaluationTests(unittest.TestCase):
    def test_missing_evaluation_checksum_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "eval_questions.jsonl").write_text(json.dumps({
                "id": "one", "category": "arithmetic_only", "problem": "1 + 1", "answer": "2"}) + "\n")
            args = build_parser().parse_args(["--data-dir", root, "--model", "unused",
                                             "--output", str(folder / "prediction.jsonl")])
            for manifest in ({}, {"prepared_hashes": {}}, {"prepared_hashes": []}):
                (folder / "manifest.json").write_text(json.dumps(manifest))
                with self.subTest(manifest=manifest), self.assertRaisesRegex(ValueError, "checksum"):
                    run(args)
            self.assertFalse(args.output.exists())

    def test_first_code_only(self):
        self.assertEqual(extract_first_code("reason\n```python\n1+1\n```\n```python\n9\n```"), "1+1")
        self.assertIsNone(extract_first_code("```json\n{}\n```\n```python\n1\n```"))
        self.assertIsNone(extract_first_code("print(1)"))
        self.assertEqual(extract_first_code("```\n1\n```"), "1")

    def test_strict_scalar(self):
        for value in ("", "1\n2", "answer: 2", "nan", "NaN", "inf", "-Infinity", "1_000", "1e400", "1e-400", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_scalar(value)
        self.assertEqual(parse_scalar("  -1.25e2\n"), -125)
        self.assertEqual(parse_scalar("0"), 0)
        self.assertEqual(parse_scalar("9007199254740993"), float(9007199254740993))

    def test_tolerance_and_zero(self):
        def grade(stdout, expected):
            return grade_execution({"status": "ok", "stdout": stdout}, expected)["correct"]
        self.assertTrue(grade("100.05", 100))
        self.assertFalse(grade("100.2", 100))
        self.assertTrue(grade("0", 0))
        self.assertFalse(grade("1e-12", 0))
        self.assertFalse(grade("1\n2", 1))
        self.assertFalse(grade("", 1))
        self.assertFalse(grade("NaN", 1))

    def test_ast_no_auto_call(self):
        definition = ast.parse(wrap_last_expression("def solve():\n    return 7"))
        self.assertEqual(len(definition.body), 1)
        self.assertIsInstance(definition.body[0], ast.FunctionDef)
        transformed = ast.parse(wrap_last_expression("def solve():\n    return 7\nsolve()"))
        self.assertEqual(transformed.body[-1].value.func.id, "print")
        direct_print = ast.parse(wrap_last_expression("print(7)"))
        self.assertEqual(ast.dump(direct_print), ast.dump(ast.parse("print(7)")))

    def test_ungraded_denominator(self):
        row = {"grade": grade_execution({"status": "not_executed"}, 7)}
        self.assertIsNone(row["grade"]["correct"])
        summary = summarize_predictions([row])
        self.assertEqual(summary["graded"], 0)
        self.assertEqual(summary["ungraded"], 1)
        self.assertIsNone(summary["accuracy"])
        self.assertIsNone(grade_execution({"status": "execution_error"}, 7)["correct"])
        self.assertIsNone(grade_execution({"status": "ok", "stdout": "1"}, "not numeric")["correct"])
        self.assertFalse(grade_execution({"status": "code_error"}, 7)["correct"])

    def test_no_docker_no_fallback(self):
        with patch("fyp.execution.sys.platform", "linux"), patch("fyp.execution.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Docker is unavailable"):
                ensure_docker("python:3.11-slim")


if __name__ == "__main__":
    unittest.main()
