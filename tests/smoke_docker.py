"""Real Docker integration on synthetic programs, with no model benchmark claim.

Build docker/Dockerfile.eval first, then run this script on Linux/WSL2.
No generated or historical Python program is ever run on the host.
No ML dependencies, pretrained weights or external model services are needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fyp.data import file_hash, write_jsonl
from fyp.evaluate import summarize_predictions
from fyp.execution import ensure_docker, execute_docker, grade_execution


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def fixture_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def executor_container_ids():
    result = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--filter", "name=^/fyp-exec-"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return set(result.stdout.split())


def write_prediction_fixture(path, correct_second_answer):
    """The differing responses are explicit fixtures, not generated model output."""
    rows = []
    for index, (problem, answer, category) in enumerate([
        ("Synthetic integration question: 2 + 2", "4", "arithmetic_only"),
        ("Synthetic integration question: 3 + 3", "6", "cot_and_code"),
    ], start=1):
        response_expression = "2 + 2" if index == 1 else ("3 + 3" if correct_second_answer else "3 + 2")
        rows.append({
            "id": f"synthetic-docker:{index}", "question_id": fixture_hash(problem),
            "problem": problem, "answer": answer, "category": category,
            "raw_response": "```python\n" + response_expression + "\n```",
            # Deliberately stale: score must re-extract the raw response instead.
            "first_fenced_code": "print(999)",
            "execution": {"status": "not_executed", "stdout": "", "stderr": ""},
            "grade": {"status": "ungraded", "correct": None},
            "fixture_only": True,
        })
    metadata = {
        "protocol": "synthetic_integration_fixture_no_model_used",
        "selected_row_ids": [row["id"] for row in rows],
        "data_sha256": fixture_hash("synthetic-docker-questions-v1"),
        "chat_template_sha256": fixture_hash("synthetic-no-model-chat-template"),
        "tokenizer_vocabulary_sha256": fixture_hash("synthetic-no-model-vocabulary"),
        "prompt_sha256": {category: fixture_hash("synthetic-prompt-" + category)
                          for category in ("arithmetic_only", "cot_and_code")},
        "generation": {"max_new_tokens": 16, "do_sample": False, "num_beams": 1,
                       "eos_token_id": 1, "pad_token_id": 1},
        "inference_runtime": {"device_type": "synthetic_fixture", "model_dtype": "not_applicable",
                              "torch_version": "not_used", "transformers_version": "not_used", "seed": 0},
        "fixture_only": True, "model_used": False,
    }
    summary = summarize_predictions(rows) | {"completed": True, "fixture_only": True}
    write_jsonl(path, rows)
    for suffix, content in ((".meta.json", metadata), (".summary.json", summary)):
        Path(str(path) + suffix).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def run_cli(*arguments):
    result = subprocess.run([sys.executable, "-m", *arguments], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError("Integration CLI failed: " + " ".join(arguments)
                           + "\n" + result.stdout + "\n" + result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="fyp-eval:ci", help="Already built local evaluator image")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    image_id = ensure_docker(args.image)
    initial_containers = executor_container_ids()
    cases = []

    numeric = execute_docker("20 + 22", image_id, timeout=15)
    require(numeric["status"] == "ok", "Final-expression execution failed: " + repr(numeric))
    require(grade_execution(numeric, "42")["correct"] is True, "Numeric output was not correctly graded")
    cases.append("numeric_final_expression_and_grading")

    scientific = execute_docker(
        "import numpy as np\nimport scipy.special\nimport sympy as sp\n"
        "float(np.sum([1, 2])) + float(scipy.special.expit(0)) + float(sp.Rational(1, 2))",
        image_id, timeout=20,
    )
    require(scientific["status"] == "ok" and grade_execution(scientific, "4")["correct"] is True,
            "Evaluator scientific dependencies failed: " + repr(scientific))
    cases.append("evaluator_numpy_scipy_sympy_imports")

    failing = execute_docker("raise ValueError('synthetic-error')", image_id, timeout=15)
    require(failing["status"] == "code_error" and "synthetic-error" in failing["stderr"],
            "Runtime exception was not captured as code_error: " + repr(failing))
    require(grade_execution(failing, "42")["correct"] is False, "Code errors must be graded incorrect")
    cases.append("runtime_error_captured_and_graded")

    timed = execute_docker("import time\ntime.sleep(60)", image_id, timeout=3)
    require(timed["status"] == "timeout", "Long-running program was not stopped: " + repr(timed))
    cases.append("wall_clock_timeout")

    capped = execute_docker("print('x' * 100000)", image_id, timeout=15, output_limit=1024)
    require(capped["status"] == "output_limit", "Large output was not stopped: " + repr(capped))
    require(sum(len(capped[key].encode("utf-8")) for key in ("stdout", "stderr")) <= 1024,
            "Captured ASCII fixture output exceeds the byte cap")
    cases.append("bounded_output_capture")

    with tempfile.TemporaryDirectory(prefix="fyp-docker-integration-") as temporary:
        folder = Path(temporary)
        for label, correct_second in (("base", False), ("merged", True)):
            source = folder / (label + ".jsonl")
            destination = folder / (label + "-scored.jsonl")
            write_prediction_fixture(source, correct_second)
            original_hashes = {suffix: file_hash(Path(str(source) + suffix))
                               for suffix in ("", ".meta.json", ".summary.json")}
            run_cli("fyp.score", "--predictions", str(source), "--output", str(destination),
                    "--docker-image", image_id, "--timeout", "15")
            scored = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
            require(all(row["execution"]["status"] == "ok" for row in scored), "Deferred execution failed")
            require(all(row["first_fenced_code"] != "print(999)" for row in scored),
                    "Deferred scoring trusted stale extracted code")
            metadata = json.loads(Path(str(destination) + ".meta.json").read_text(encoding="utf-8"))
            require(metadata["docker_image_id"] == image_id, "Immutable Docker image ID was not recorded")
            for suffix, original_hash in original_hashes.items():
                require(file_hash(Path(str(source) + suffix)) == original_hash, "Scoring modified a source fixture")
        cases.append("deferred_scoring_reextracts_code_and_preserves_inputs")

        comparison_path = folder / "comparison.json"
        run_cli("fyp.compare", "--base", str(folder / "base-scored.jsonl"),
                "--merged", str(folder / "merged-scored.jsonl"), "--output", str(comparison_path))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        overall = comparison["overall"]
        require(overall["base"] == {"correct": 1, "total": 2, "accuracy": 0.5}, "Wrong synthetic base score")
        require(overall["merged"] == {"correct": 2, "total": 2, "accuracy": 1.0}, "Wrong synthetic merged score")
        require(overall["delta_percentage_points"] == 50.0, "Wrong synthetic paired score difference")
        require(comparison["paired_changed_ids"] == {"improved": ["synthetic-docker:2"], "regressed": []},
                "Wrong synthetic paired changes")
        require(comparison["by_category"]["arithmetic_only"]["delta_percentage_points"] == 0.0
                and comparison["by_category"]["cot_and_code"]["delta_percentage_points"] == 100.0,
                "Category-specific synthetic paired comparison failed")
        cases.append("paired_comparison_and_category_totals")

    require(executor_container_ids() == initial_containers, "An integration execution container was left behind")
    cases.append("execution_containers_removed")
    report = {
        "test": "real_docker_synthetic_execution_scoring_comparison", "passed": True,
        "docker_image_requested": args.image, "docker_image_id": image_id,
        "passed_checks": cases, "passed_check_count": len(cases),
        "programs_are_synthetic": True, "model_used": False,
        "original_4b_model_used": False, "historical_predictions_used": False,
        "generated_python_executed_on_host": False, "benchmark_accuracy": None,
        "scope": "Software integration only; manually specified test responses do not establish model improvement.",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
