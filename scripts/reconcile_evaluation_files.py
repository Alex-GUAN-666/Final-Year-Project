#!/usr/bin/env python3
"""Reconcile uploaded evaluation CSVs against original saved notebook outputs.

Uses only the standard library. Does not load a model or execute any model-written
code. Numeric rescoring reads recorded execution_result strings only. A matching
filename is never accepted as evidence of model identity.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import sys

FIELDS = ["question", "ground_truth", "llm_response", "extracted_code", "execution_result", "is_correct"]
FENCE = re.compile(r"```+(?:python|py)?\s*\n(.*?)\n```+", re.DOTALL)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fyp.sources import source_path


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def numeric_equal(a: str, b: str) -> bool:
    try:
        return Decimal(a) == Decimal(b)
    except InvalidOperation:
        return False


def rescore_saved_result(row: dict) -> tuple[bool, str]:
    """Follow the archived scalar-output metric, without running extracted_code."""
    if not row["extracted_code"]:
        return False, "no_matching_code_block"
    result = row["execution_result"]
    if not result:
        return False, "empty_execution_stdout"
    if result.startswith("Error:"):
        return False, "recorded_execution_error"
    try:
        output = Decimal(result)
        gold = Decimal(str(row["ground_truth"]))
        passed = math.isclose(output, gold, rel_tol=1e-3)
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return False, "non_numeric_stdout"
    return passed, "correct_numeric_stdout" if passed else "wrong_numeric_stdout"


def static_code(code: str) -> dict:
    if not code:
        return {"has_code": False, "syntax_error": None, "last_ast_node": None}
    try:
        module = ast.parse(code)
        return {"has_code": True, "syntax_error": None,
                "last_ast_node": type(module.body[-1]).__name__ if module.body else None}
    except SyntaxError as error:
        return {"has_code": True, "syntax_error": str(error), "last_ast_node": None}


def read_csv(path: Path) -> dict:
    raw = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected evaluation schema in {path.name}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No evaluation rows in {path}")
    for i, row in enumerate(rows, 1):
        if set(row) != set(FIELDS) or any(v is None for v in row.values()):
            raise ValueError(f"Malformed CSV row {i} in {path.name}")
        if row["is_correct"] not in {"True", "False"}:
            raise ValueError(f"Unexpected correctness label row {i} in {path.name}")
    return {"filename": path.name, "sha256": sha(raw), "bytes": len(raw),
            "rows": rows, "row_count": len(rows), "field_count": len(FIELDS),
            "all_field_characters_read": sum(len(v) for r in rows for v in r.values())}


def notebook_records(path: Path, role: str) -> dict:
    raw = path.read_bytes()
    nb = json.loads(raw)
    candidates = []
    declarations = []
    for cell_number, cell in enumerate(nb["cells"], 1):
        source = "".join(cell.get("source", []))
        declarations.extend(re.findall(r"^LOAD_BASE_MODEL\s*=\s*(True|False)[ \t]*(?:#[^\n]*)?$", source, re.M))
        stdout = "".join("".join(o.get("text", [])) for o in cell.get("outputs", [])
                         if o.get("name") == "stdout")
        if "--- Processing Question 1/" in stdout and "--- Evaluation Finished ---" in stdout:
            candidates.append((cell_number, cell, stdout))
    if len(candidates) != 1:
        raise ValueError(f"Expected one complete historical run in {path.name}, found {len(candidates)}")
    expected = "True" if role == "base" else "False"
    if declarations != [expected]:
        raise ValueError(f"Model configuration ambiguity in {path.name}: {declarations}")
    cell_number, cell, stdout = candidates[0]
    incorrect_tail = stdout.split("--- Incorrect Samples ---", 1)[1]
    incorrect = {}
    for m in re.finditer(r"\[\d+\] Question: (.*?)\n    Ground Truth: (.*?)\n    LLM Result:   (.*?)\n-{20}", incorrect_tail, re.S):
        question, gold, result = m.groups()
        incorrect[question] = {"gold": gold, "result": result}
    chunks = re.split(r"--- Processing Question (\d+)/(\d+) ---\n", stdout)
    records = []
    for offset in range(1, len(chunks), 3):
        number, total, body = int(chunks[offset]), int(chunks[offset + 1]), chunks[offset + 2]
        header = re.search(r"^Question: (.*?)\n\(System (\d+): (.*?)\)\nIt cost ([0-9.]+) seconds for reference\n", body, re.S)
        if header is None:
            raise ValueError(f"Cannot parse question header {number} in {path.name}")
        question, route, route_text, seconds = header.groups()
        flag = re.search(r"\nis_correct:  (True|False)\n", body[header.end():])
        if flag is None:
            raise ValueError(f"Cannot parse recorded flag {number} in {path.name}")
        response = body[header.end():header.end() + flag.start()]
        correct = flag.group(1) == "True"
        blocks = FENCE.findall(response)
        known = incorrect.get(question)
        if not correct and known is None:
            raise ValueError(f"Missing incorrect-result summary row {number} in {path.name}")
        records.append({"question_number": number, "total": total, "question": question,
                        "llm_response": response, "extracted_code": blocks[0].strip() if blocks else "",
                        "recorded_is_correct": correct, "route": route,
                        "generation_seconds_recorded": float(seconds),
                        "recorded_execution_result": known["result"] if known else None,
                        "execution_result_observed": known is not None,
                        "gold_from_incorrect_log": known["gold"] if known else None})
    reported = int(re.search(r"Correct Predictions: (\d+)", stdout).group(1))
    reported_total = int(re.search(r"Total Questions: (\d+)", stdout).group(1))
    if len(records) != reported_total or sum(r["recorded_is_correct"] for r in records) != reported:
        raise ValueError(f"Summary and per-question flags disagree in {path.name}")
    if [r["question_number"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError(f"Unexpected question order in {path.name}")
    return {"notebook": path.name, "sha256": sha(raw), "model_role_in_source": role,
            "load_base_model_in_source": expected == "True", "evaluation_cell_one_based": cell_number,
            "saved_execution_metadata": cell.get("metadata", {}).get("execution", {}),
            "correct": reported, "total": reported_total, "accuracy": reported / reported_total,
            "records": records}


def compare_csv_to_notebook(rows: list[dict], historical: dict, reference: list[dict]) -> dict:
    if len(rows) != len(historical["records"]):
        return {"row_count_matches": False, "complete_observable_match": False}
    fields = Counter()
    mismatches = []
    for number, (row, saved) in enumerate(zip(rows, historical["records"]), 1):
        checks = {
            "question_exact": row["question"] == saved["question"],
            "response_exact": row["llm_response"] == saved["llm_response"],
            "code_exact": row["extracted_code"] == saved["extracted_code"],
            "correctness_flag_exact": (row["is_correct"] == "True") == saved["recorded_is_correct"],
            "reference_question_exact": row["question"] == reference[number - 1]["arithmetical question"],
            "reference_gold_numeric_equal": numeric_equal(row["ground_truth"], reference[number - 1]["answer to the question"]),
        }
        if saved["execution_result_observed"]:
            checks["observed_execution_result_exact"] = row["execution_result"] == saved["recorded_execution_result"]
            checks["observed_ground_truth_numeric_equal"] = numeric_equal(row["ground_truth"], saved["gold_from_incorrect_log"])
        for key, value in checks.items():
            fields[key + "_compared"] += 1
            fields[key + "_matched"] += int(value)
        if not all(checks.values()):
            mismatches.append({"question_number": number, "fields": [k for k, v in checks.items() if not v]})
    return {"row_count_matches": True, "counts": dict(fields),
            "complete_observable_match": not mismatches, "mismatches": mismatches}


ROW_NOTES = {
    1: "Numeric 4 matches gold. The prose sufficiency check incorrectly simplifies 2y + 2sqrt((x+y)y) >= 4y to a right side of 3y; output scoring does not verify the proof.",
    2: "SymPy solve output is indexed by a Symbol as though a dictionary; recorded execution error. Prose equations do not correctly use the two step counts.",
    3: "Prose gives 64 but no matching Python block. Fails the recorded execution metric despite the textual answer.",
    4: "Vieta and cube-sum calculation return recorded stdout 12.0, equivalent to gold 12.",
    5: "Prose states 66, then repeats tokens. No matching Python block.",
    6: "Code prints a list of symbolic perimeter expressions, including nonphysical branches. Full stdout is not a scalar decimal and is rejected.",
    7: "SymPy solve output is incorrectly indexed with a Symbol; recorded execution error. Prose additionally claims an unsupported speed of 3.",
    8: "Recorded stdout 1 matches gold; executable code tests p=5 only. The general argument is in prose and is not scored for validity.",
    9: "Prose gives 36 but code uses the first two roots without requiring positive lengths; recorded stdout is 0.",
    10: "Enumeration of digit strings returns recorded stdout 233. A prose statement about maximum length is imprecise but code counts lengths 1 through 12.",
    11: "Code uses 2222222220 rather than the smallest valid number 2220; recorded quotient 148148148 differs from gold 148.",
    12: "Five widths sum to 20; recorded stdout 4.0 is equivalent to gold 4.",
    13: "Choosing two unordered co-presidents gives 105. Irrelevant trailing tokens do not affect the first extracted code block.",
    14: "Numeric result 28 matches gold; code returns the literal after the preceding derivation.",
    15: "Gold 59 is correct for the standard remainder puzzle, but contradicts the supplied even-number-per-line clause. Recorded pass is retained and dataset defect is flagged separately.",
    16: "Code tests a=15,b=77 and returns 2. Prose incorrectly restricts a/3 and b/7 and does not prove the universal result; both a,b being odd would establish it.",
    17: "Recurrence is evaluated at 21 and recorded stdout is 28.",
    18: "Recorded stdout 15.0 matches gold. Code checks endpoints for the example p=7.5, while prose gives the general piecewise-linear endpoint argument.",
    19: "Six-term AM-GM and equality point (1,2,3) give numeric 6, consistent with recorded stdout.",
    20: "Prose falsely claims infinitely many positive integer solutions and emits no matching code block. Gold is C(99,2)=4851.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "base-csv": source_path("evaluation_results_base_cot_v5(1).csv"),
        "merged-csv": source_path("evaluation_results_merged_cot_v5(1).csv"),
        "base-notebook": source_path("notebookd84987e9c1_base_cot.ipynb"),
        "merged-notebook": source_path("notebookd84987e9c1_merged_cot.ipynb"),
        "reference-data": ROOT / "data/raw/random_samples_for_inference.csv",
        "historical-json": ROOT / "reports/cot_historical_results.json",
        "token-filter": ROOT / "reports/historical_token_filter.json",
        "distilled-data": ROOT / "data/raw/results_v3.jsonl",
        "output-dir": ROOT / "reports",
    }
    for option, default in defaults.items():
        parser.add_argument("--" + option, type=Path, default=default,
                            help=f"Default: {default.relative_to(ROOT)}")
    args = parser.parse_args()
    csv.field_size_limit(16 * 1024 * 1024)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    uploads = {"named_base": read_csv(args.base_csv), "named_merged": read_csv(args.merged_csv)}
    notebooks = {"base": notebook_records(args.base_notebook, "base"),
                 "merged": notebook_records(args.merged_notebook, "merged")}
    with args.reference_data.open(encoding="utf-8-sig", newline="") as f:
        reference = list(csv.DictReader(f))
    old = json.loads(args.historical_json.read_text(encoding="utf-8"))
    token_filter = json.loads(args.token_filter.read_text(encoding="utf-8"))
    retained_lines = [row["line"] for row in token_filter["lengths"] if row["retained"]]
    if len(retained_lines) != token_filter["retained_distilled"]:
        raise ValueError("Token-filter manifest disagrees with retained source-line count")
    distilled = [json.loads(line) for line in args.distilled_data.read_text(encoding="utf-8").splitlines() if line.strip()]
    retained_questions = {}
    for line in retained_lines:
        retained_questions.setdefault(distilled[line - 1]["problem"], []).append(line)
    overlap_mapping = [{"question_number": row["question_number"],
                        "retained_distilled_source_lines": retained_questions.get(row["question"], [])}
                       for row in notebooks["base"]["records"]]
    overlapping_rows = [row["question_number"] for row in overlap_mapping if row["retained_distilled_source_lines"]]
    overlap_note = (
        f"The official-tokenizer reconstruction retains {len(retained_lines)} distilled training rows; "
        f"{len(overlapping_rows)}/{len(overlap_mapping)} historical evaluation questions are exact matches "
        f"to those retained rows (evaluation rows {', '.join(map(str, overlapping_rows))}). "
        "These historical scores do not establish an independent held-out generalization gain. "
        "The original training-tokenizer revision and exact merged checkpoint remain unverified."
    )
    overlap = {"status": "exact_question_overlap_with_reconstructed_training_subset",
               "official_tokenizer_revision": token_filter["official_revision"],
               "token_filter_manifest": args.token_filter.name,
               "token_filter_manifest_sha256": sha(args.token_filter.read_bytes()),
               "distilled_file_sha256": sha(args.distilled_data.read_bytes()),
               "retained_distilled_count": len(retained_lines),
               "evaluation_count": len(overlap_mapping), "overlap_count": len(overlapping_rows),
               "overlapping_question_numbers": overlapping_rows, "mappings": overlap_mapping,
               "interpretation": overlap_note}
    prior_checks = {}
    for role, data in notebooks.items():
        prior = old["models"][role]
        checks = [prior["notebook_sha256"] == data["sha256"], len(prior["records"]) == len(data["records"])]
        fields_to_check = ("question", "llm_response", "extracted_code", "recorded_is_correct", "recorded_execution_result")
        checks.extend(all(a[key] == b[key] for key in fields_to_check)
                      for a, b in zip(prior["records"], data["records"]))
        prior_checks[role] = all(checks)
        if not all(checks):
            raise ValueError(f"Prior recovered JSON differs from direct notebook extraction: {role}")
    file_reports = {}
    detailed_rows = []
    for name, upload in uploads.items():
        matches = {role: compare_csv_to_notebook(upload["rows"], data, reference)
                   for role, data in notebooks.items()}
        identified_roles = [role for role, match in matches.items() if match["complete_observable_match"]]
        row_audits = []
        for number, row in enumerate(upload["rows"], 1):
            rescored, category = rescore_saved_result(row)
            blocks = FENCE.findall(row["llm_response"])
            first = blocks[0].strip() if blocks else ""
            record = {"question_number": number, "question": row["question"],
                      "ground_truth": row["ground_truth"], "execution_result": row["execution_result"],
                      "recorded_is_correct": row["is_correct"] == "True",
                      "rescored_from_saved_stdout": rescored,
                      "label_consistent_with_saved_stdout": rescored == (row["is_correct"] == "True"),
                      "failure_or_success_category": category,
                      "first_block_matches_csv_extracted_code": first == row["extracted_code"],
                      "matching_fenced_block_count": len(blocks), "static_code": static_code(row["extracted_code"]),
                      "full_field_sha256": {key: sha(value) for key, value in row.items()},
                      "field_characters": {key: len(value) for key, value in row.items()},
                      "manual_review": ROW_NOTES.get(number, "") if identified_roles == ["base"] else ""}
            row_audits.append(record)
        file_reports[name] = {k: v for k, v in upload.items() if k != "rows"}
        file_reports[name].update({"identified_historical_roles": identified_roles,
                                   "comparisons": matches,
                                   "recorded_correct": sum(r["recorded_is_correct"] for r in row_audits),
                                   "rescored_correct": sum(r["rescored_from_saved_stdout"] for r in row_audits),
                                   "labels_inconsistent_with_saved_stdout": [r["question_number"] for r in row_audits if not r["label_consistent_with_saved_stdout"]],
                                   "categories": dict(Counter(r["failure_or_success_category"] for r in row_audits)),
                                   "row_audits": row_audits})
        detailed_rows.append((name, row_audits))
    same_bytes = args.base_csv.read_bytes() == args.merged_csv.read_bytes()
    enriched = {"evidence_type": "historical evidence reconstruction; no model inference and no generated-code execution",
                "execution_result_policy": "Only exactly matched CSV rows augment previously missing saved stdout. Unobserved merged successful stdout stays null.",
                "training_evaluation_overlap": overlap,
                "models": {}}
    for role, notebook in notebooks.items():
        matched_names = [name for name, report in file_reports.items() if report["identified_historical_roles"] == [role]]
        records = []
        for number, saved in enumerate(notebook["records"], 1):
            record = dict(saved)
            record["question_sha256"] = sha(saved["question"])
            record["response_sha256"] = sha(saved["llm_response"])
            record["gold_from_reference_csv"] = reference[number - 1]["answer to the question"]
            record["retained_distilled_source_lines"] = retained_questions.get(saved["question"], [])
            record["overlaps_reconstructed_training_questions"] = bool(record["retained_distilled_source_lines"])
            record["result_provenance"] = "saved_notebook_incorrect_summary" if saved["execution_result_observed"] else "unobserved_in_supplied_evidence"
            record["exact_matching_uploaded_files"] = [uploads[name]["filename"] for name in matched_names]
            if matched_names:
                row = uploads[matched_names[0]]["rows"][number - 1]
                record["newly_recovered_stdout"] = not saved["execution_result_observed"]
                record["recorded_execution_result"] = row["execution_result"]
                record["execution_result_observed"] = True
                record["result_provenance"] = "exactly_matched_uploaded_csv" if record["newly_recovered_stdout"] else "saved_notebook_and_exactly_matched_uploaded_csv"
                record["rescored_from_saved_stdout"], record["stdout_category"] = rescore_saved_result(row)
            else:
                record["newly_recovered_stdout"] = False
                record["rescored_from_saved_stdout"] = None
                record["stdout_category"] = None
            records.append(record)
        enriched["models"][role] = {**{k: v for k, v in notebook.items() if k != "records"}, "records": records}
    report = {
        "audit_scope": "All six fields of every row of both uploaded CSVs; exact raw text matches against direct original notebook extraction and the prior recovered JSON; static syntax inspection only.",
        "generated_code_executed": False, "model_inference_performed": False,
        "training_evaluation_overlap": overlap,
        "csvs_identical_bytes": same_bytes,
        "unique_csv_payloads": len({value["sha256"] for value in uploads.values()}),
        "prior_json_matches_direct_notebook_extraction": prior_checks,
        "reference_csv": {"filename": args.reference_data.name, "sha256": sha(args.reference_data.read_bytes()), "rows": len(reference)},
        "files": file_reports,
        "base_newly_observed_success_stdout_rows": [r["question_number"] for r in enriched["models"]["base"]["records"] if r["newly_recovered_stdout"]],
        "merged_stdout_still_unknown_rows": [r["question_number"] for r in enriched["models"]["merged"]["records"] if not r["execution_result_observed"]],
        "historical_scores": {role: {"correct": data["correct"], "total": data["total"], "accuracy": data["accuracy"]} for role, data in notebooks.items()},
        "public_reporting": {
            "metric": "Historical generated-Python numeric-output accuracy on the same first20 reasoning questions; training/evaluation overlap is present",
            "base": "12/20 (60%), supported by original notebook and two identical CSV copies of that base run",
            "merged": "15/20 (75%), supported by the original merged notebook's per-question flags and incorrect-result summary",
            "difference": "15 percentage points; net three additional recorded successes. This is not an independent held-out gain: five of the20 questions overlap the reconstructed training subset.",
            "training_overlap_limit": overlap_note,
            "caveat": "The uploaded filename containing merged is a duplicate of the base results. It provides no independent merged or arithmetic benchmark result. Exact checkpoint identity is not verified by these files, and no model was rerun.",
            "proof_claim": "Do not call these proof-validation accuracy results. Scalar-output scoring does not validate full reasoning or proofs.",
            "data_quality": "Question15 has contradictory wording. Keep historical labels intact, flag separately, and use versioned corrected data for any future clean benchmark."
        }
    }
    (args.output_dir / "evaluation_reconciliation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "cot_historical_results_enriched.json").write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paired = []
    for base, merged in zip(enriched["models"]["base"]["records"], enriched["models"]["merged"]["records"]):
        if base["question"] != merged["question"]:
            raise ValueError("Historical base/merged questions are not aligned")
        paired.append({"question_number": base["question_number"], "question": base["question"],
                       "reference_gold": base["gold_from_reference_csv"],
                       "base_is_correct_recorded": base["recorded_is_correct"],
                       "merged_is_correct_recorded": merged["recorded_is_correct"],
                       "base_execution_result": base["recorded_execution_result"],
                       "base_result_observed": base["execution_result_observed"],
                       "merged_execution_result": merged["recorded_execution_result"],
                       "merged_result_observed": merged["execution_result_observed"],
                       "base_result_provenance": base["result_provenance"],
                       "merged_result_provenance": merged["result_provenance"],
                       "overlaps_reconstructed_training_questions": base["overlaps_reconstructed_training_questions"],
                       "retained_distilled_source_lines": ",".join(map(str, base["retained_distilled_source_lines"]))})
    with (args.output_dir / "cot_historical_paired_results_enriched.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    text = ["# Evaluation CSV reconciliation", "",
            "**Current evidence: the historical evaluation is not an independent held-out benchmark.** " + overlap_note, "",
            "Both newly supplied CSV files contain the same BASE historical run. The filename containing `merged` does not identify merged-model predictions.", "",
            "This audit reads every CSV field, independently extracts the original notebook outputs, checks exact full-response/code matches, and recomputes scalar-answer correctness from recorded stdout. It does not run a model or model-generated code.", "",
            "## File identity and scores", "",
            "| Uploaded file | Bytes | Rows | Recorded/rescored | Historical run matched |", "|---|---:|---:|---:|---|"]
    for file in file_reports.values():
        text.append(f"| `{file['filename']}` | {file['bytes']} | {file['row_count']} | {file['recorded_correct']}/{file['row_count']} = {file['recorded_correct']/file['row_count']:.0%} | {', '.join(file['identified_historical_roles'])} |")
    text.extend(["", f"Byte equality: **{same_bytes}**. SHA-256: `{uploads['named_base']['sha256']}`.", "",
                 "For each uploaded CSV: all20 full question strings, all20 full response strings, all20 extracted code strings, and all20 correctness flags exactly match the base notebook. All8 notebook-observed failure stdout strings also match. All20 reference answers are numerically equivalent (CSV serializes values such as `4.0`, reference data uses `4`). No response or code field matches the merged notebook, even though the questions are identical.", "",
                 "Each CSV has12 correct scalar outputs,3 missing-code failures,2 recorded execution errors,1 nonnumeric symbolic-list output, and2 wrong numeric outputs. There are0 incorrect correctness labels under the archived execution metric, and every extracted code block matches the archived first-fence regex.", "",
                 "## New evidence recovered", "",
                 "The new CSV payload supplies previously unprinted stdout for all12 successful base examples. Base stdout is now directly observed for all20 rows. Both duplicate files are preserved in the manifest; they are one payload, not two independent experiments.", "",
                 "Merged remains15/20 from its original notebook, subject to the5/20 training-overlap limitation above. Successful merged stdout remains unobserved for15 rows and stays null; it is never copied from the base CSV or the reference answers. The five merged failure outputs remain directly recovered from its notebook. These uploaded CSVs add no arithmetic-specific result evidence.", "",
                 "## Full per-row review", "",
                 "Rows refer to both identical uploaded files. Full response/code field hashes and literal execution_result strings are retained in the JSON report; full model response and code are retained in the enriched historical JSON.", "",
                 "| Row | Gold | Saved stdout | Flag/rescore | Finding |", "|---:|---:|---|---|---|"])
    for row in file_reports["named_base"]["row_audits"]:
        result = row["execution_result"]
        if result.startswith("Error:"):
            shown = "Symbol-indexing execution error"
        elif row["failure_or_success_category"] == "non_numeric_stdout":
            shown = "Symbolic list (full value in JSON)"
        else:
            shown = result if result else "empty"
        text.append(f"| {row['question_number']} | {row['ground_truth']} | {shown} | {'pass' if row['recorded_is_correct'] else 'fail'} / {'pass' if row['rescored_from_saved_stdout'] else 'fail'} | {row['manual_review']} |")
    text.extend(["", "## Public repository wording", "",
                 "> Historical saved outputs show numeric-answer execution accuracy of12/20 (60%) for the base model and15/20 (75%) for the merged model on the same20-question reasoning subset. Five of these20 questions occur in the143 distilled examples retained by the official-tokenizer reconstruction, so the15-percentage-point difference is not an independent held-out generalization gain. Base results are additionally corroborated by a recovered CSV export. The two uploaded CSV filenames represent identical copies of the base run; the merged score is supported by its original notebook output. These are historical small-sample observations, not a new model rerun or verification of full mathematical proofs.", "",
                 "Keep row15's wording defect separate from execution-label consistency:59 matches the supplied gold but cannot satisfy the extra even-number-per-line clause. For example, an even quotient with remainder1 on division by2 implies N=4k+1, which contradicts N mod4=3. Do not silently alter the historical denominator or labels.", "",
                 "## Generated audit artifacts", "",
                 "- `evaluation_reconciliation.json`: input hashes, all-row static checks, file/run matching, rescoring and reporting guidance.",
                 "- `cot_historical_results_enriched.json`:40 complete historical responses and provenance of recovered/unknown stdout.",
                 "- `cot_historical_paired_results_enriched.csv`:20 aligned base/merged rows with explicit observed flags.",
                 "- `reconcile_evaluation_files.py`: standard-library recovery script; no checkpoint or CUDA required. Run `python scripts/reconcile_evaluation_files.py` from the repository; defaults resolve relative to the script, so another working directory also works.", ""])
    (args.output_dir / "evaluation_reconciliation.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"identical_csv_payloads": same_bytes,
                      "matched_roles": {k: v["identified_historical_roles"] for k, v in file_reports.items()},
                      "scores": report["historical_scores"],
                      "training_evaluation_overlap": {k: overlap[k] for k in ("overlap_count", "evaluation_count", "overlapping_question_numbers", "interpretation")},
                      "new_base_stdout_rows": report["base_newly_observed_success_stdout_rows"],
                      "label_inconsistency_counts": {k: len(v["labels_inconsistent_with_saved_stdout"]) for k, v in file_reports.items()},
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
