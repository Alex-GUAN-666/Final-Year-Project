import copy
import json
import tempfile
import unittest
from pathlib import Path

from fyp.compare import compare_runs
from fyp.evaluate import summarize_predictions
from fyp.execution import GRADER_PROTOCOL, grade_execution


class ComparisonTests(unittest.TestCase):
    def fixture(self, directory, name, correct):
        path = Path(directory) / (name + ".jsonl")
        rows = [{"id": "q" + str(index), "question_id": "problem" + str(index),
                 "problem": "A synthetic test fixture only", "answer": 2,
                 "category": "arithmetic_only" if index < 2 else "cot_and_code",
                 "raw_response": "```python\nprint(" + ("2" if value else "3") + ")\n```",
                 "first_fenced_code": "print(" + ("2" if value else "3") + ")",
                 "execution": {"status": "ok", "stdout": "2" if value else "3", "stderr": ""},
                 "grade": {"correct": value, "status": "correct" if value else "incorrect"}}
                for index, value in enumerate(correct)]
        metadata = {"selected_row_ids": [row["id"] for row in rows], "data_sha256": "a" * 64,
                    "chat_template_sha256": "b" * 64, "tokenizer_vocabulary_sha256": "c" * 64,
                    "prompt_sha256": {"arithmetic_only": "d" * 64, "cot_and_code": "e" * 64},
                    "inference_runtime": {"device_type": "cpu", "model_dtype": "torch.float32",
                                          "torch_version": "2.6.0+cpu", "transformers_version": "4.53.2", "seed": 3407},
                    "generation": {"max_new_tokens": 1024, "do_sample": False, "num_beams": 1,
                                   "eos_token_id": 1, "pad_token_id": 1}}
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        Path(str(path) + ".meta.json").write_text(json.dumps({"selected_row_ids": metadata["selected_row_ids"],
            "source_metadata": metadata, "protocol": "deferred_docker_numeric_output_scoring",
            "docker_image_id": "sha256:" + "f" * 64, "timeout_seconds": 10,
            "output_byte_limit": 65536, "grader_protocol": GRADER_PROTOCOL}))
        self.write_summary(path, rows)
        return path

    def write_summary(self, path, rows):
        summary = summarize_predictions(rows) | {"completed": True}
        summary["by_category"] = {
            category: summarize_predictions([row for row in rows if row["category"] == category])
            for category in ("arithmetic_only", "cot_and_code")
        }
        Path(str(path) + ".summary.json").write_text(json.dumps(summary))

    def test_paired_metrics_and_changed_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.fixture(directory, "base", [True, False, False])
            merged = self.fixture(directory, "merged", [False, True, True])
            result = compare_runs(base, merged)
            self.assertEqual(result["overall"]["base"]["correct"], 1)
            self.assertEqual(result["overall"]["merged"]["correct"], 2)
            self.assertAlmostEqual(result["overall"]["delta_percentage_points"], 100 / 3)
            self.assertEqual(result["by_category"]["arithmetic_only"]["delta_percentage_points"], 0)
            self.assertEqual(result["paired_changed_ids"], {"improved": ["q1", "q2"], "regressed": ["q0"]})

    def test_refuse_mismatched_protocol_and_ungraded_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.fixture(directory, "base", [True, False])
            merged = self.fixture(directory, "merged", [True, True])
            meta_path = Path(str(merged) + ".meta.json")
            original = json.loads(meta_path.read_text())
            for key in ("data_sha256", "chat_template_sha256", "tokenizer_vocabulary_sha256", "generation", "prompt_sha256"):
                metadata = copy.deepcopy(original)
                source = metadata["source_metadata"]
                if key == "generation":
                    source[key]["max_new_tokens"] = 100
                elif key == "prompt_sha256":
                    source[key]["arithmetic_only"] = "f" * 64
                else:
                    source[key] = "f" * 64
                meta_path.write_text(json.dumps(metadata))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "not comparable"):
                    compare_runs(base, merged)
            meta_path.write_text(json.dumps(original))
            rows = [json.loads(line) for line in merged.read_text().splitlines()]
            rows[0]["grade"] = {"correct": None, "status": "ungraded"}
            merged.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "not fully"):
                compare_runs(base, merged)

    def test_refuse_different_questions_or_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.fixture(directory, "base", [True, False])
            merged = self.fixture(directory, "merged", [True, True])
            original = [json.loads(line) for line in merged.read_text().splitlines()]
            for key, value in (("problem", "different problem"), ("answer", 3), ("category", "cot_and_code")):
                rows = copy.deepcopy(original)
                rows[0][key] = value
                merged.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "questions differ"):
                    compare_runs(base, merged)

    def test_refuse_different_docker_image(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.fixture(directory, "base", [True, False])
            merged = self.fixture(directory, "merged", [True, True])
            path = Path(str(merged) + ".meta.json")
            metadata = json.loads(path.read_text())
            metadata["docker_image_id"] = "sha256:" + "a" * 64
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Scoring conditions.*docker_image_id"):
                compare_runs(base, merged)

    def test_refuse_different_or_missing_numerical_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.fixture(directory, "base", [True, False])
            merged = self.fixture(directory, "merged", [True, True])
            path = Path(str(merged) + ".meta.json")
            original = json.loads(path.read_text())
            changes = {"device_type": "cuda", "model_dtype": "torch.float16",
                       "torch_version": "2.7.0+cpu", "transformers_version": "4.54.0", "seed": 1234}
            for key, value in changes.items():
                metadata = copy.deepcopy(original)
                metadata["source_metadata"]["inference_runtime"][key] = value
                path.write_text(json.dumps(metadata))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "not comparable.*inference_runtime"):
                    compare_runs(base, merged)
            metadata = copy.deepcopy(original)
            del metadata["source_metadata"]["inference_runtime"]
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "runtime provenance"):
                compare_runs(base, merged)

    def test_refuse_stale_grades_code_and_missing_execution_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.fixture(directory, "base", [True])
            merged = self.fixture(directory, "merged", [True])
            original = json.loads(merged.read_text())
            changes = [
                ("execution", {"status": "ok", "stdout": "999", "stderr": ""}, "saved grade"),
                ("execution", {"status": "timeout", "stdout": "2", "stderr": ""}, "saved grade"),
                ("execution", None, "execution evidence"),
                ("execution", {"status": "ok", "stdout": "2"}, "execution evidence"),
                ("execution", {"status": [], "stdout": "2", "stderr": ""}, "execution evidence"),
                ("first_fenced_code", "print(999)", "extracted code"),
                ("raw_response", "```python\nprint(999)\n```", "extracted code"),
            ]
            for key, value, error in changes:
                row = copy.deepcopy(original)
                row[key] = value
                merged.write_text(json.dumps(row) + "\n")
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, error):
                    compare_runs(base, merged)

    def test_refuse_stale_summary_totals_and_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.fixture(directory, "base", [True, False, True])
            merged = self.fixture(directory, "merged", [True, False, True])
            path = Path(str(merged) + ".summary.json")
            original = json.loads(path.read_text())
            for key, value in (("correct", 3), ("graded", 2), ("ungraded", 1),
                               ("accuracy", 1), ("statuses", {"correct": 3}),
                               ("by_category", {})):
                summary = copy.deepcopy(original)
                summary[key] = value
                path.write_text(json.dumps(summary))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "summary disagrees"):
                    compare_runs(base, merged)
            summary = copy.deepcopy(original)
            del summary["accuracy"]
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "summary disagrees"):
                compare_runs(base, merged)

    def test_no_code_is_valid_failure_but_cannot_claim_executed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.fixture(directory, "base", [False])
            merged = self.fixture(directory, "merged", [True])
            row = json.loads(base.read_text())
            row["raw_response"] = "No fenced Python was generated."
            row["first_fenced_code"] = None
            row["execution"] = {"status": "no_code", "stdout": "", "stderr": "No first Python fence"}
            row["grade"] = grade_execution(row["execution"], row["answer"])
            base.write_text(json.dumps(row) + "\n")
            self.write_summary(base, [row])
            self.assertEqual(compare_runs(base, merged)["overall"]["base"]["correct"], 0)
            row["execution"] = {"status": "ok", "stdout": "999", "stderr": ""}
            row["grade"] = grade_execution(row["execution"], row["answer"])
            base.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "code presence"):
                compare_runs(base, merged)


if __name__ == "__main__":
    unittest.main()
