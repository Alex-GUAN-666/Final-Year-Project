#!/usr/bin/env python3
"""Replay saved historical responses in restricted Docker, without model inference.

Build docker/Dockerfile.eval first. This checks execution-output scores for the
preserved responses; it does not regenerate responses, recover missing weights,
verify mathematical proofs, or establish an independent held-out model gain.
Historical results are immutable inputs. Any grade disagreement remains visible
and produces a failing exit status, even when aggregate correct counts agree.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fyp.execution import GRADER_PROTOCOL, ensure_docker, execute_docker, extract_first_code, grade_execution
from fyp.sources import source_path
from scripts.reconcile_evaluation_files import notebook_records, read_csv


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_verified_source(path: Path) -> tuple[dict, dict]:
    """Cross-check enriched records against archived notebooks and reference CSV.

    Parsing and hashing are the only host operations on generated Python here.
    Report fields alone are never accepted as evidence of notebook content.
    """
    raw = path.read_bytes()
    source = json.loads(raw)
    reference_path = ROOT / "data/raw/random_samples_for_inference.csv"
    with reference_path.open(encoding="utf-8-sig", newline="") as handle:
        reference = list(csv.DictReader(handle))
    provenance = {
        "enriched_report_sha256": digest(raw),
        "reference_csv_sha256": digest(reference_path.read_bytes()),
        "notebooks": {}, "uploaded_csvs": {},
    }
    require(set(source["models"]) == {"base", "merged"}, "Expected base and merged records")
    require(len(reference) >= 20, "Reference CSV has fewer than 20 questions")
    uploaded = {}
    for model in ("base", "merged"):
        data = source["models"][model]
        filename = data["notebook"]
        require(Path(filename).name == filename, "Notebook must be an archive basename")
        notebook = notebook_records(source_path(filename), model)
        provenance["notebooks"][model] = {"filename": filename, "sha256": notebook["sha256"]}
        require(data["sha256"] == notebook["sha256"], "Notebook digest differs from report: " + model)
        require(len(data["records"]) == len(notebook["records"]) == 20, "Expected 20 rows per model")
        for number, (row, saved) in enumerate(zip(data["records"], notebook["records"]), 1):
            location = f"{model} question {number}"
            for key in ("question_number", "total", "question", "llm_response",
                        "extracted_code", "recorded_is_correct"):
                require(row[key] == saved[key], f"Notebook/report {key} mismatch: {location}")
            require(type(row["recorded_is_correct"]) is bool, "Correctness must be Boolean: " + location)
            require(row["question_number"] == number and row["total"] == 20, "Invalid row order: " + location)
            require(row["question"] == reference[number - 1]["arithmetical question"],
                    "Reference question mismatch: " + location)
            require(row["gold_from_reference_csv"] == reference[number - 1]["answer to the question"],
                    "Reference answer mismatch: " + location)
            code = extract_first_code(row["llm_response"])
            require((code or "") == saved["extracted_code"], "Current/historical code extraction differs: " + location)
            require(digest(row["llm_response"]) == row["response_sha256"], "Response digest mismatch: " + location)
            require(digest(row["question"]) == row["question_sha256"], "Question digest mismatch: " + location)
            if saved["execution_result_observed"]:
                require(row["execution_result_observed"] is True
                        and row["recorded_execution_result"] == saved["recorded_execution_result"],
                        "Saved notebook execution result mismatch: " + location)
            elif row["execution_result_observed"]:
                matches = row.get("exact_matching_uploaded_files", [])
                require(bool(matches), "Observed stdout has no uploaded-CSV provenance: " + location)
                for csv_name in matches:
                    require(Path(csv_name).name == csv_name, "CSV must be an archive basename")
                    if csv_name not in uploaded:
                        uploaded[csv_name] = read_csv(source_path(csv_name))
                        provenance["uploaded_csvs"][csv_name] = uploaded[csv_name]["sha256"]
                    csv_rows = uploaded[csv_name]["rows"]
                    require(len(csv_rows) == 20, "Expected 20 uploaded CSV rows")
                    match = csv_rows[number - 1]
                    require(match["question"] == row["question"]
                            and match["llm_response"] == row["llm_response"]
                            and match["extracted_code"] == row["extracted_code"]
                            and match["execution_result"] == row["recorded_execution_result"]
                            and (match["is_correct"] == "True") == row["recorded_is_correct"],
                            "Uploaded stdout provenance mismatch: " + location)
            else:
                require(row["recorded_execution_result"] is None, "Unobserved stdout must stay null: " + location)
        correct = sum(row["recorded_is_correct"] for row in data["records"])
        require(data["correct"] == notebook["correct"] == correct
                and data["total"] == notebook["total"] == 20
                and data["accuracy"] == correct / 20, "Historical score inconsistency: " + model)
    return source, provenance


def compare_row(model: str, row: dict, execution: dict) -> dict:
    grade = grade_execution(execution, row["gold_from_reference_csv"])
    observed = row["execution_result_observed"]
    old_result = row["recorded_execution_result"]
    # Historical error strings were returned by an in-process exception handler;
    # Docker reports exceptions on stderr. Do not fabricate matching stdout.
    stdout_comparable = bool(observed and not old_result.startswith("Error:")
                             and execution["status"] in {"ok", "no_code"})
    code = extract_first_code(row["llm_response"])
    return {
        "model": model, "question_number": row["question_number"],
        "question": row["question"], "question_sha256": row["question_sha256"],
        "reference_answer": row["gold_from_reference_csv"],
        "response_sha256": row["response_sha256"],
        "first_fenced_code": code, "code_sha256": digest(code) if code is not None else None,
        "saved_correct": row["recorded_is_correct"],
        "saved_execution_result": old_result,
        "saved_execution_result_observed": observed,
        "replayed_execution": execution, "replayed_grade": grade,
        "correctness_matches_saved": (grade["correct"] == row["recorded_is_correct"]
                                      if grade["correct"] is not None else None),
        "saved_stdout_comparable": stdout_comparable,
        "saved_stdout_matches_after_strip": (old_result == execution["stdout"].strip()
                                             if stdout_comparable else None),
        "overlaps_reconstructed_training_questions": row["overlaps_reconstructed_training_questions"],
    }


def summarize(rows: list[dict], saved: dict) -> dict:
    graded = [row for row in rows if row["replayed_grade"]["correct"] is not None]
    correct = sum(row["replayed_grade"]["correct"] is True for row in graded)
    disagreements = [row["question_number"] for row in rows if row["correctness_matches_saved"] is False]
    ungraded = [row["question_number"] for row in rows if row["correctness_matches_saved"] is None]
    stdout_mismatches = [row["question_number"] for row in rows
                         if row["saved_stdout_matches_after_strip"] is False]
    complete = len(graded) == saved["total"]
    return {
        "historical_saved": {"correct": saved["correct"], "total": saved["total"], "accuracy": saved["accuracy"]},
        "replayed": {"correct": correct, "total": saved["total"], "graded": len(graded),
                     "accuracy": correct / saved["total"] if complete else None},
        "all_rows_graded": complete,
        "all_correctness_flags_match_saved": complete and not disagreements,
        "correctness_disagreement_question_numbers": disagreements,
        "ungraded_question_numbers": ungraded,
        "comparable_saved_stdout_mismatch_question_numbers": stdout_mismatches,
        "execution_status_counts": dict(Counter(row["replayed_execution"]["status"] for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/cot_historical_results_enriched.json")
    parser.add_argument("--output", "--report", dest="output", type=Path,
                        help="New JSON report; existing files are never overwritten")
    parser.add_argument("--docker-image", "--image", dest="docker_image", default="fyp-eval:ci")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--output-byte-limit", type=int, default=65536)
    parser.add_argument("--check-inputs", action="store_true",
                        help="Verify all source provenance only; no Docker or execution claims")
    args = parser.parse_args()
    if args.timeout <= 0 or args.output_byte_limit <= 0:
        parser.error("Execution limits must be positive")
    if not args.check_inputs and args.output is None:
        parser.error("--output is required for execution replay")
    if args.output is not None and args.output.exists():
        raise FileExistsError("Refusing to overwrite " + str(args.output))
    source, provenance = load_verified_source(args.source)
    if args.check_inputs:
        print(json.dumps({"source_provenance_verified": True, "response_count": 40,
                          "extractable_program_count": sum(bool(row["extracted_code"])
                              for data in source["models"].values() for row in data["records"]),
                          "programs_executed": False, "model_used": False}, indent=2))
        return 0
    image_id = ensure_docker(args.docker_image)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_type": "restricted-Docker execution replay of saved historical responses",
        "model_used": False, "original_4b_weights_used": False,
        "responses_regenerated": False, "generated_python_executed_on_host": False,
        "historical_source_results_modified": False,
        "scope": "Execution-output reproducibility only. Saved responses are inputs, not new model predictions."
                 " This does not establish proof validity or independent held-out generalization.",
        "source_provenance": provenance,
        "docker_image_requested": args.docker_image, "docker_image_id": image_id,
        "grader_protocol": GRADER_PROTOCOL,
        "timeout_seconds_per_program": args.timeout, "output_byte_limit": args.output_byte_limit,
        "training_evaluation_overlap": source["training_evaluation_overlap"],
        "stdout_policy": "Unobserved saved stdout remains null; new Docker stdout is recorded separately."
                         " Saved error strings and Docker stderr use different conventions.",
        "models": {},
    }
    for model in ("base", "merged"):
        saved = source["models"][model]
        rows = []
        for row in saved["records"]:
            code = extract_first_code(row["llm_response"])
            try:
                execution = execute_docker(code, image_id, timeout=args.timeout,
                                           output_limit=args.output_byte_limit)
            except (RuntimeError, OSError) as exc:
                execution = {"status": "execution_error", "stdout": "", "stderr": str(exc)}
            rows.append(compare_row(model, row, execution))
            print(f"{model} Q{row['question_number']:02d}: saved={row['recorded_is_correct']} "
                  f"replayed={rows[-1]['replayed_grade']['correct']} status={execution['status']}", flush=True)
        report["models"][model] = summarize(rows, saved) | {"records": rows}
    all_graded = all(data["all_rows_graded"] for data in report["models"].values())
    flags_match = all(data["all_correctness_flags_match_saved"] for data in report["models"].values())
    stdout_matches = all(not data["comparable_saved_stdout_mismatch_question_numbers"]
                         for data in report["models"].values())
    report["all_rows_graded"] = all_graded
    report["all_correctness_flags_match_saved"] = flags_match
    report["all_comparable_saved_stdout_match"] = stdout_matches
    report["exit_status_policy"] = "0: all grades and comparable stdout agree; 1: disagreement; 2: ungraded execution"
    report["exit_status"] = 2 if not all_graded else (0 if flags_match and stdout_matches else 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({model: {key: value for key, value in data.items() if key != "records"}
                      for model, data in report["models"].items()}, indent=2))
    return report["exit_status"]


if __name__ == "__main__":
    raise SystemExit(main())
