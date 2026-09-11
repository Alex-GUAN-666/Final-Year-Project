#!/usr/bin/env python3
"""Verify recovered evidence and software without model dependencies or downloads.

All audit outputs are temporary. Only an explicitly requested --report is saved.
This does not train/infer a 4B model, rerun tokenization, or generate new accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
PREPARED_FILES = {
    "train_candidates.jsonl", "eval_questions.jsonl",
    "source_dispositions.jsonl", "manifest.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(report):
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory(prefix="fyp-review-") as temporary:
        work = Path(temporary)

        def scrub(value):
            return value.replace(str(work), "<temporary>").replace(str(ROOT), "<repository>")

        def run_step(name, arguments):
            print(f"Checking {name}...", flush=True)
            start = time.monotonic()
            command = [sys.executable, *map(str, arguments)]
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=300)
            step = {"name": name, "command": ["python", *[scrub(str(x)) for x in arguments]],
                    "returncode": result.returncode,
                    "duration_seconds": round(time.monotonic() - start, 3),
                    "stdout": scrub(result.stdout), "stderr": scrub(result.stderr)}
            report["steps"].append(step)
            require(result.returncode == 0,
                    f"{name} failed (exit {result.returncode}):\n{step['stderr'] or step['stdout']}")
            return result

        artifact_path = work / "artifacts.json"
        run_step("bundled source integrity", ["scripts/audit_artifacts.py", "--output", artifact_path])
        artifacts = read_json(artifact_path)
        bundled = [item for item in artifacts["files"] if item.get("sha256_verified")]
        require(len(bundled) == 24, "Expected 24 bundled source/public-derivative hashes")
        report["source_integrity"] = {
            "verified_bundled_files": len(bundled),
            "source_manifest_sha256": digest(ROOT / "source_manifest.json"),
            "original_byte_identical_files": sum(item["original_byte_identical"] for item in bundled),
        }

        # audit_data.py has a fixed repository-relative reports/ destination.
        # Run an unchanged copy with copied inputs so the checkout stays untouched.
        isolated = work / "data-audit"
        (isolated / "scripts").mkdir(parents=True)
        (isolated / "data/raw").mkdir(parents=True)
        (isolated / "archive/scripts").mkdir(parents=True)
        shutil.copyfile(ROOT / "scripts/audit_data.py", isolated / "scripts/audit_data.py")
        for script in (ROOT / "archive/scripts").glob("*.py"):
            shutil.copyfile(script, isolated / "archive/scripts" / script.name)
        for name in ("arithmetic_logic_data_revised_v2.jsonl", "arithmetic_problems_v3.csv",
                     "random_samples_for_inference.csv"):
            shutil.copyfile(ROOT / "data/raw" / name, isolated / "data/raw" / name)
        run_step("static arithmetic and reference-data audit", [isolated / "scripts/audit_data.py"])

        recovered_dir = work / "recovered"
        run_step("recovered distillation and saved membership audit",
                 ["scripts/audit_recovered_distillation.py", "--output-dir", recovered_dir])
        recovered = read_json(recovered_dir / "recovered_data_audit.json")
        selection = recovered["final_token_filter"]
        overlap = recovered["overlap"]["historical_eval"]["final_retained_question_matches"]
        require(selection["count"] == 143 and overlap == 5,
                "Saved reconstructed membership must retain 143 distilled rows with 5 historical overlaps")
        require(selection["original_tokenizer_artifact_authenticated"] is False,
                "The absent original tokenizer must not be reported as authenticated")
        report["saved_token_membership"] = {
            "retained_distilled_rows": selection["count"],
            "historical_evaluation_overlap": overlap,
            "token_filter_report_sha256": digest(ROOT / "reports/historical_token_filter.json"),
            "tokenizer_executed": False, "original_tokenizer_authenticated": False,
            "interpretation": "Saved membership is cross-checked against recovered source rows; tokenization is not rerun.",
        }

        reconciliation_dir = work / "reconciliation"
        run_step("direct notebook and evaluation-CSV reconciliation",
                 ["scripts/reconcile_evaluation_files.py", "--output-dir", reconciliation_dir])
        reconciliation = read_json(reconciliation_dir / "evaluation_reconciliation.json")
        historical = reconciliation["historical_scores"]
        require((historical["base"]["correct"], historical["base"]["total"]) == (12, 20),
                "Saved base notebook must contain 12/20 successes")
        require((historical["merged"]["correct"], historical["merged"]["total"]) == (15, 20),
                "Saved merged notebook must contain 15/20 successes")
        require(reconciliation["csvs_identical_bytes"] is True,
                "The two supplied evaluation CSVs should be identical base-run copies")
        for file in reconciliation["files"].values():
            require(file["identified_historical_roles"] == ["base"] and file["rescored_correct"] == 12,
                    "Each recovered CSV must match/rescore the saved base run")
            require(not file["labels_inconsistent_with_saved_stdout"], "Base CSV labels disagree with saved stdout")
        historical_overlap = reconciliation["training_evaluation_overlap"]
        require(historical_overlap["overlapping_question_numbers"] == [5, 12, 13, 14, 20],
                "Historical overlapping question identities differ")
        require(len(reconciliation["merged_stdout_still_unknown_rows"]) == 15,
                "Do not invent stdout for the 15 unobserved merged successes")
        report["historical_evidence"] = {
            "saved_notebook_scores": historical,
            "duplicate_csv_payloads": True, "both_csvs_match": "base",
            "overlapping_question_numbers": historical_overlap["overlapping_question_numbers"],
            "merged_success_stdout_unknown_count": 15,
            "interpretation": "Recorded numeric-output results, not new inference or an independent held-out gain.",
        }

        recovered_notebooks_path = work / "recovered-notebooks.json"
        run_step("recovered notebook settings and recorded evaluation evidence",
                 ["scripts/verify_recovered_notebooks.py", "--output", recovered_notebooks_path])
        recovered_notebooks = read_json(recovered_notebooks_path)
        require(recovered_notebooks["status"] == "passed"
                and recovered_notebooks["model_inference_executed"] is False,
                "Recovered notebook validation must pass without model inference")
        require(recovered_notebooks == read_json(ROOT / "reports/recovered_notebook_evidence.json"),
                "Recovered notebook report differs from its source-derived committed evidence")
        report["recovered_notebook_evidence"] = {
            "verified_evaluation_runs": len(recovered_notebooks["evaluation_runs"]),
            "recorded_correct_counts": [r["correct"] for r in recovered_notebooks["evaluation_runs"]],
            "arithmetic_active_model_path": recovered_notebooks["evaluation_runs"][0]["recorded_model"]["active_model_path"],
            "original_arithmetic_paired_comparison_recovered": False,
            "model_resource": recovered_notebooks["model_resource"]["kaggle_input"],
            "model_inference_executed": False,
        }

        prepared = work / "prepared"
        run_step("fixed question-disjoint split regeneration", ["-m", "fyp.prepare", "--output-dir", prepared])
        committed = ROOT / "data/prepared/question_disjoint_v1"
        require({p.name for p in prepared.iterdir()} == PREPARED_FILES,
                "Regenerated prepared file inventory differs")
        require({p.name for p in committed.iterdir()} == PREPARED_FILES,
                "Bundled prepared file inventory differs")
        manifest = read_json(prepared / "manifest.json")
        require(set(manifest["prepared_hashes"]) == PREPARED_FILES - {"manifest.json"},
                "Prepared checksums must cover all three data files")
        hashes = {}
        for name in sorted(PREPARED_FILES):
            require((prepared / name).read_bytes() == (committed / name).read_bytes(),
                    f"Prepared file does not reproduce byte for byte: {name}")
            hashes[name] = digest(prepared / name)
            if name != "manifest.json":
                require(hashes[name] == manifest["prepared_hashes"][name], f"Prepared checksum mismatch: {name}")
        require(manifest["training_candidates"] == 436 and manifest["evaluation_questions"] == 187,
                "New split must contain 436 training candidates and 187 evaluation questions")
        require(manifest["train_eval_normalized_question_overlap"] == 0,
                "New training and evaluation questions overlap")
        report["prepared_split"] = {
            "protocol": manifest["protocol"], "byte_identical_files": sorted(PREPARED_FILES),
            "sha256": hashes, "training_candidates": 436, "evaluation_questions": 187,
            "normalized_question_overlap": 0, "token_filter_executed": False,
        }

        tests = run_step("standard-library unit tests",
                         ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"])
        match = re.search(r"Ran (\d+) tests? in", tests.stdout + "\n" + tests.stderr)
        require(match is not None and int(match.group(1)) > 0, "No unit tests were discovered")
        report["unit_tests"] = {"passed": int(match.group(1)), "returncode": tests.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional new JSON report path; never overwrite an existing file")
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error(f"Report already exists; choose a new path: {args.report}")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "stdlib_reviewer_reconstruction_check_v1", "status": "started",
        "scope": "Source integrity, static data audits, saved-result reconciliation, split regeneration and unit tests",
        "python_version": sys.version.split()[0], "steps": [],
        "model_training_executed": False, "model_inference_executed": False,
        "tokenizer_executed": False, "docker_execution_executed": False,
        "new_model_accuracy_generated": False,
        "limitations": [
            "Original fine-tuned weights and original tokenizer artifacts are absent.",
            "Historical reasoning results have 5/20 reconstructed training-question overlap.",
            "The thesis arithmetic paired comparison remains absent; a separate qwen-3/transformers/4b/1 run with 17/20 is recovered.",
            "This command does not rerun a model or guarantee historical scores from new training.",
            "Static code inspection and numeric saved-output checks do not verify all mathematical proofs.",
        ],
    }
    print("FYP reviewer check: saved evidence and software only; no model downloads or new model accuracy.", flush=True)
    start = time.monotonic()
    try:
        verify(report)
        report["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
    report["duration_seconds"] = round(time.monotonic() - start, 3)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        print(f"Report: {args.report}", flush=True)
    if report["status"] != "passed":
        print(f"FAILED: {report['error']}", file=sys.stderr)
        return 1
    print(f"PASS: 24 source hashes; saved reasoning scores 12/20 and 15/20; 5 overlapping questions.")
    print(f"PASS: 436 training candidates / 187 evaluation questions reproduce byte for byte; "
          f"{report['unit_tests']['passed']} unit tests passed.")
    print("No 4B training, inference, fresh tokenization, Docker execution, or new model accuracy was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
