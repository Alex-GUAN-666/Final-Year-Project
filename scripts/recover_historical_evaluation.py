"""Extract historical CoT evaluation evidence without executing generated code.

Output values are only recovered where the stored notebook actually printed them.
Correct samples' execution output was not printed, so it remains null.
This is the preserved notebook-only extraction. For updated CSV provenance and
training/evaluation overlap, run reconcile_evaluation_files.py afterward.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fyp.sources import source_path

SOURCE = ROOT / "data" / "raw"
NOTEBOOKS = ROOT / "archive" / "notebooks"
DEST = ROOT / "reports"
DEST.mkdir(parents=True, exist_ok=True)
STEMS = {
    "base": "notebookd84987e9c1_base_cot",
    "merged": "notebookd84987e9c1_merged_cot",
}
FENCE = re.compile(r"```+(?:python|py)?\s*\n(.*?)\n```+", re.DOTALL)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():
    with (SOURCE / "random_samples_for_inference.csv").open(newline="", encoding="utf-8") as f:
        dataset = list(csv.DictReader(f))
    train = [json.loads(line) for line in (SOURCE / "arithmetic_logic_data_revised_v2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    training_questions = {r["problem"] for r in train}
    models = {}
    all_records = []
    for model, stem in STEMS.items():
        notebook_path = source_path(stem + ".ipynb")
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        # stderr is deliberately separate: Jupyter asynchronous stream ordering can
        # place a question's execution error after the next question's header.
        stdout = "".join("".join(o.get("text", [])) for o in nb["cells"][13]["outputs"] if o.get("name") == "stdout")
        stderr = "".join("".join(o.get("text", [])) for o in nb["cells"][13]["outputs"] if o.get("name") == "stderr")
        tail = stdout.split("--- Incorrect Samples ---", 1)[1]
        incorrect = {}
        for match in re.finditer(r"\[\d+\] Question: (.*?)\n    Ground Truth: (.*?)\n    LLM Result:   (.*?)\n-{20}", tail, re.DOTALL):
            question, gold, result = match.groups()
            incorrect[question] = (gold, result)
        chunks = re.split(r"--- Processing Question (\d+)/(\d+) ---\n", stdout)
        records = []
        for pos in range(1, len(chunks), 3):
            number, total, body = int(chunks[pos]), int(chunks[pos + 1]), chunks[pos + 2]
            header = re.search(r"^Question: (.*?)\n\(System (\d+): (.*?)\)\nIt cost ([0-9.]+) seconds for reference\n", body, re.DOTALL)
            assert header, (model, number, "header not found")
            question, route, route_text, seconds = header.groups()
            flag = re.search(r"\nis_correct:  (True|False)\n", body[header.end():])
            assert flag, (model, number, "flag not found")
            response = body[header.end():header.end() + flag.start()]
            correct = flag.group(1) == "True"
            blocks = FENCE.findall(response)
            code = blocks[0].strip() if blocks else ""
            syntax_error = None
            module = None
            if code:
                try:
                    module = ast.parse(code)
                except SyntaxError as error:
                    syntax_error = str(error)
            exact = question == dataset[number - 1]["arithmetical question"]
            assert exact, (model, number, "dataset mismatch")
            result = incorrect.get(question, (None, None))[1]
            if not correct:
                assert question in incorrect
            record = {
                "model": model,
                "notebook": stem + ".ipynb",
                "cell": 14,
                "question_number": number,
                "evaluated_sample_count": total,
                "question": question,
                "question_sha256": digest(question),
                "matches_attached_csv_row_exactly": exact,
                "gold_from_attached_csv": dataset[number - 1]["answer to the question"],
                "gold_from_saved_incorrect_log": incorrect.get(question, (None, None))[0],
                "route": route,
                "generation_seconds_recorded": float(seconds),
                "llm_response": response,
                "response_sha256": digest(response),
                "extracted_code": code,
                "regex_matching_block_count": len(blocks),
                "code_syntax_error_static": syntax_error,
                "code_final_ast_node": type(module.body[-1]).__name__ if module and module.body else None,
                "recorded_execution_result": result,
                "execution_result_observed": not correct,
                "recorded_is_correct": correct,
                "exact_question_overlap_with_attached_synthetic_jsonl": question in training_questions,
            }
            records.append(record)
            all_records.append(record)
        assert len(records) == 20
        correct_total = sum(r["recorded_is_correct"] for r in records)
        reported = int(re.search(r"Correct Predictions: (\d+)", stdout).group(1))
        assert correct_total == reported
        models[model] = {
            "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
            "sample_count": len(records),
            "correct": correct_total,
            "accuracy": correct_total / len(records),
            "reported_accuracy_percent": float(re.search(r"Accuracy: ([0-9.]+)%", stdout).group(1)),
            "prompt_reset_cell_execution_count": nb["cells"][9].get("execution_count"),
            "saved_evaluation_execution_metadata": nb["cells"][13].get("metadata", {}).get("execution"),
            "stderr": stderr,
            "records": records,
        }
    paired = []
    for a, b in zip(models["base"]["records"], models["merged"]["records"]):
        assert a["question"] == b["question"]
        paired.append({
            "question_number": a["question_number"],
            "question": a["question"],
            "gold_from_attached_csv": a["gold_from_attached_csv"],
            "base_correct_recorded": a["recorded_is_correct"],
            "merged_correct_recorded": b["recorded_is_correct"],
            "base_execution_result_recorded": a["recorded_execution_result"],
            "merged_execution_result_recorded": b["recorded_execution_result"],
            "base_result_observed": a["execution_result_observed"],
            "merged_result_observed": b["execution_result_observed"],
        })
    doc = {
        "superseded_interpretation_notice": "This notebook-only extraction is preserved for traceability. Consult cot_historical_results_enriched.json and evaluation_reconciliation.md for current evidence: official-tokenizer reconstruction finds exact training overlap in5/20 evaluated questions (rows5,12,13,14,20), so60% to75% is not an independent held-out generalization result. The two recovered CSVs are identical base-run copies and supply12 formerly unobserved successful base stdout strings; successful merged stdout remains unknown. The original tokenizer revision and merged checkpoint are unauthenticated.",
        "partial_overlap_scope_notice": "The exact_question_overlap_with_attached_synthetic_jsonl fields below check only arithmetic_logic_data_revised_v2.jsonl. They do not check the recovered results_v3.jsonl corpus or establish absence of training/evaluation overlap.",
        "evidence_type": "historical notebook output extraction; no generated code/model execution",
        "csv_rows_total": len(dataset),
        "evaluated_rows": "first 20 rows, exact same order in both notebooks",
        "correct_result_note": "Correct samples' execution stdout was not printed; null is intentional and is not copied from gold.",
        "models": models,
        "paired_records": paired,
    }
    (DEST / "cot_historical_results.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (DEST / "cot_historical_paired_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paired[0].keys())
        writer.writeheader()
        writer.writerows(paired)
    summaries = {model: {k: v for k, v in data.items() if k not in {"records", "stderr"}} for model, data in models.items()}
    print(json.dumps(summaries, indent=2))
    print("Interpretation:5/20 evaluated questions overlap reconstructed training; historical60%/75% is not an independent held-out gain. Run reconcile_evaluation_files.py for current enriched evidence.")
    print("all 40 logged questions exactly match the first20 attached CSV rows")
    print("base wrong -> merged correct", [r["question_number"] for r in paired if not r["base_correct_recorded"] and r["merged_correct_recorded"]])
    print("base correct -> merged wrong", [r["question_number"] for r in paired if r["base_correct_recorded"] and not r["merged_correct_recorded"]])
    print("overlap with attached synthetic", sum(r["exact_question_overlap_with_attached_synthetic_jsonl"] for r in models["base"]["records"]))


if __name__ == "__main__":
    main()
