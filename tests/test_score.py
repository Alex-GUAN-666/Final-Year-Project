import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fyp.execution import ensure_docker
from fyp.score import load_completed_predictions, run


class DeferredScoringTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory) / "predictions.jsonl"
        path.write_text(json.dumps({"id": "q1", "category": "arithmetic_only", "answer": 2,
                                    "raw_response": "```python\n1+1\n```"}) + "\n")
        Path(str(path) + ".meta.json").write_text(json.dumps({"selected_row_ids": ["q1"]}))
        Path(str(path) + ".summary.json").write_text(json.dumps({"completed": True, "total": 1}))
        return path

    def test_require_completed_consistent_sidecars_before_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            args = argparse.Namespace(predictions=path, output=Path(directory) / "scored.jsonl",
                                      docker_image="python:3.11-slim", timeout=10, output_byte_limit=1024)
            summary = Path(str(path) + ".summary.json")
            for value in ({"completed": False, "total": 1}, {"completed": True, "total": 2}):
                summary.write_text(json.dumps(value))
                with patch("fyp.score.ensure_docker") as docker, self.assertRaises(ValueError):
                    run(args)
                docker.assert_not_called()
            summary.unlink()
            with self.assertRaisesRegex(ValueError, "sidecars are required"):
                load_completed_predictions(path)

    def test_malformed_ids_and_raw_response(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            rows, _, _ = load_completed_predictions(path)
            self.assertEqual(rows[0]["id"], "q1")
            Path(str(path) + ".meta.json").write_text(json.dumps({"selected_row_ids": ["wrong"]}))
            with self.assertRaisesRegex(ValueError, "IDs/count"):
                load_completed_predictions(path)
            path.write_text(json.dumps({"id": "q1", "answer": 2, "category": "arithmetic_only"}))
            with self.assertRaisesRegex(ValueError, "Malformed prediction"):
                load_completed_predictions(path)

    def test_native_windows_requires_wsl(self):
        with patch("fyp.execution.sys.platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "Linux or WSL2"):
                ensure_docker("python:3.11-slim")

    def test_reparse_raw_response_without_executing_host_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            record = json.loads(path.read_text())
            record["first_fenced_code"] = "print(999)"
            path.write_text(json.dumps(record) + "\n")
            original = path.read_bytes()
            args = argparse.Namespace(predictions=path, output=Path(directory) / "scored.jsonl",
                                      docker_image="python:3.11-slim", timeout=10, output_byte_limit=1024)
            with patch("fyp.score.ensure_docker", return_value="sha256:abc"), \
                    patch("fyp.score.execute_docker", return_value={"status": "ok", "stdout": "2", "stderr": ""}) as executor, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                run(args)
            executor.assert_called_once_with("1+1", "sha256:abc", 10, 1024)
            self.assertTrue(json.loads(args.output.read_text())["grade"]["correct"])
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
