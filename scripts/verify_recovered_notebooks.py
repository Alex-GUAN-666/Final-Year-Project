#!/usr/bin/env python3
"""Cross-check newly recovered notebook records without executing notebook code.

Saved model names, logs and Papermill metadata establish what the files record;
without the checkpoint bytes they cannot authenticate the model weights.
"""
from __future__ import annotations

import argparse
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

MERGED_PATH = "/kaggle/input/qwen3-4b-sfft-merged/transformers/default/1/Qwen3-4B-SFFT-merged"
BASE_PATH = "/kaggle/input/qwen-3/transformers/4b-base/1"
ARITHMETIC_PATH = "/kaggle/input/qwen-3/transformers/4b/1"
RUNS = (
    ("arithmetic_qwen_4b_run_20251122", 17, True, ARITHMETIC_PATH, False),
    ("base_reasoning_run_20251120", 12, True, BASE_PATH, False),
    ("merged_reasoning_run_20251122", 15, False, MERGED_PATH, False),
    ("base_reasoning_empty_prompt_20251120", 0, True, BASE_PATH, True),
    ("merged_reasoning_empty_prompt_20251120", 2, False, MERGED_PATH, True),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    return value if isinstance(value, str) else "".join(value)


def stdout(cell):
    return "".join(text(o.get("text", [])) for o in cell.get("outputs", [])
                   if o.get("name") == "stdout")


def source(cell):
    return text(cell.get("source", []))


def read_notebook(name):
    path = source_path(name)
    return json.loads(path.read_text(encoding="utf-8")), {
        "notebook": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def literal_assignment(statements, name):
    matches = [node.value for node in statements if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    require(len(matches) == 1, f"Expected one literal assignment for {name}")
    return ast.literal_eval(matches[0])


def recorded_sequence(nb):
    pm = nb["metadata"]["papermill"]
    require(pm.get("exception") is None and pm["start_time"] < pm["end_time"],
            "Notebook must record successful Papermill completion")
    executed = []
    previous_end = pm["start_time"]
    for number, cell in enumerate(nb["cells"], 1):
        require(not any(o.get("output_type") == "error" for o in cell.get("outputs", [])),
                f"Notebook contains a saved error output in cell {number}")
        if cell["cell_type"] != "code" or not source(cell).strip():
            continue
        meta = cell["metadata"]["papermill"]
        require(meta["status"] == "completed" and meta["exception"] is False,
                f"Cell {number} must record successful completion")
        require(previous_end <= meta["start_time"] <= meta["end_time"] <= pm["end_time"],
                f"Cell {number} is not temporally sequential")
        previous_end = meta["end_time"]
        executed.append(cell["execution_count"])
    require(executed == list(range(1, len(executed) + 1)),
            "Saved execution counters must be sequential from 1")
    return {"start_time": pm["start_time"], "end_time": pm["end_time"],
            "sequential_executed_cells": len(executed), "saved_exception": None}


def model_selection(nb, expected_flag, expected_path):
    configs = [(i, c) for i, c in enumerate(nb["cells"], 1)
               if re.search(r"^LOAD_BASE_MODEL\s*=", source(c), re.M)]
    require(len(configs) == 1, "Expected exactly one model selection cell")
    index, cell = configs[0]
    statements = ast.parse(source(cell)).body
    flag = literal_assignment(statements, "LOAD_BASE_MODEL")
    require(flag is expected_flag, "Unexpected LOAD_BASE_MODEL flag")
    branches = [node for node in statements if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == "LOAD_BASE_MODEL"]
    require(len(branches) == 1, "Expected the documented model selection branch")
    path = literal_assignment(branches[0].body if flag else branches[0].orelse, "base_model_path")
    require(path == expected_path, "Active model path differs from the documented run")
    load_cell = verify_loader(nb, path, "Base" if flag else "Merged")
    require(index < load_cell, "Model selection must precede loading")
    return {"load_base_model_flag": flag, "active_model_path": path,
            "configuration_cell": index, "loading_cell": load_cell}


def verify_loader(nb, path, label):
    message = f"{label} model loaded successfully with Unsloth."
    matches = [(i, c) for i, c in enumerate(nb["cells"], 1) if message in stdout(c)]
    require(len(matches) == 1, f"Expected one saved {label} loading message")
    index, cell = matches[0]
    nodes = ast.walk(ast.parse(source(cell)))
    calls = [node for node in nodes if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "FastLanguageModel"]
    require(len(calls) == 1 and any(k.arg == "model_name" and isinstance(k.value, ast.Name)
                                  and k.value.id == "base_model_path" for k in calls[0].keywords),
            f"Saved loader must use base_model_path ({path})")
    return index


def evaluation_records(nb):
    matches = [(i, c) for i, c in enumerate(nb["cells"], 1)
               if re.search(r"^Correct Predictions: \d+$", stdout(c), re.M)]
    require(len(matches) == 1, "Expected exactly one saved evaluation summary")
    cell_number, cell = matches[0]
    output = stdout(cell)
    chunks = re.split(r"--- Processing Question (\d+)/(\d+) ---\n", output)
    records = []
    for pos in range(1, len(chunks), 3):
        number, total, body = int(chunks[pos]), int(chunks[pos + 1]), chunks[pos + 2]
        header = re.search(r"^Question: (.*?)\n\(System (\d+): (.*?)\)\nIt cost ([0-9.]+) seconds for reference\n", body, re.S)
        require(header is not None, f"Missing question header {number}")
        flag = re.search(r"(?:^|\n)is_correct:\s+(True|False)\n", body[header.end():])
        require(flag is not None, f"Missing recorded correctness flag {number}")
        response = body[header.end():header.end() + flag.start()]
        records.append({"number": number, "total": total, "question": header.group(1),
                        "route": header.group(2), "recorded_correct": flag.group(1) == "True",
                        "response": response})
    def summary(pattern):
        found = re.findall(pattern, output, re.M)
        require(len(found) == 1, "Missing or ambiguous evaluation summary")
        return float(found[0])
    require([r["number"] for r in records] == list(range(1, 21))
            and all(r["total"] == 20 for r in records) and summary(r"^Total Questions: (\d+)$") == 20,
            "Evaluation must contain exactly the 20 ordered question records")
    correct = sum(r["recorded_correct"] for r in records)
    require(summary(r"^Correct Predictions: (\d+)$") == correct
            and summary(r"^Accuracy: ([0-9.]+)%$") == correct * 5,
            "Recorded flags disagree with the printed score")
    return cell_number, cell, correct, records


def verify():
    report = {"protocol": "recovered_notebook_records_v1", "status": "passed",
              "model_training_executed": False, "model_inference_executed": False,
              "notebook_code_executed": False, "generated_programs_executed": False,
              "new_model_accuracy_generated": False, "evaluation_runs": []}
    all_records = {}
    for name, expected_correct, flag, path, empty in RUNS:
        nb, entry = read_notebook(f"qwen3_4b_{name}.ipynb")
        entry["recorded_execution"] = recorded_sequence(nb)
        entry["recorded_model"] = model_selection(nb, flag, path)
        cell_number, cell, correct, records = evaluation_records(nb)
        require(correct == expected_correct, f"Unexpected saved score for {name}")
        require(entry["recorded_model"]["loading_cell"] < cell_number, "Loading must precede evaluation")
        resets = [i for i, c in enumerate(nb["cells"], 1)
                  if re.search(r'^code_system_prompt\s*=\s*""\s*$', source(c), re.M)
                  and re.search(r'^cot_code_system_prompt\s*=\s*""\s*$', source(c), re.M)]
        require(bool(resets) is empty and all(i < cell_number for i in resets),
                f"Unexpected empty-prompt setting for {name}")
        dataset = "arithmetic_problems_v3.csv" if path == ARITHMETIC_PATH else "random_samples_for_inference.csv"
        statements = ast.parse(source(cell)).body
        require(literal_assignment(statements, "num_for_test") == 20
                and Path(literal_assignment(statements, "input_file")).name == dataset,
                "Saved evaluation configuration differs from the documented dataset")
        with (ROOT / "data/raw" / dataset).open(encoding="utf-8-sig", newline="") as handle:
            questions = list(csv.DictReader(handle))[:20]
        require([r["question"] for r in records] == [q["arithmetical question"] for q in questions],
                f"Question records must exactly match the first 20 rows of {dataset}")
        require({r["route"] for r in records} == ({"01"} if path == ARITHMETIC_PATH else {"02"}),
                "Recorded task route differs from the documented task")
        entry.update(evaluation_cell=cell_number, correct=correct, total=20,
                     recorded_accuracy=correct / 20, empty_prompt_reset_cells=resets,
                     dataset=dataset, dataset_rows="first 20, exact question text and order")
        entry["records"] = [{"number": r["number"], "question": r["question"],
                              "recorded_correct": r["recorded_correct"],
                              "dataset_gold": questions[i]["answer to the question"],
                              "response_printed": bool(r["response"].strip()),
                              "response_sha256": hashlib.sha256(r["response"].encode()).hexdigest()
                              if r["response"].strip() else None}
                             for i, r in enumerate(records)]
        report["evaluation_runs"].append(entry)
        all_records[name] = records

    for role, run in (("base", "base_reasoning_run_20251120"), ("merged", "merged_reasoning_run_20251122")):
        old, _ = read_notebook(f"qwen3_4b_{role}_reasoning_evaluation.ipynb")
        _, _, _, previous = evaluation_records(old)
        current = all_records[run]
        require([(r["question"], r["recorded_correct"]) for r in current]
                == [(r["question"], r["recorded_correct"]) for r in previous],
                f"Recovered {role} questions and flags must match the prior archive")
        if role == "base":
            require([r["response"] for r in current] == [r["response"] for r in previous],
                    "Recovered base responses must match the prior archive")
        else:
            require(all(not r["response"].strip() for r in current)
                    and all(r["response"].strip() for r in previous),
                    "Recovered merged run omits responses; preserve the prior full-response source")
    report["prior_archive_comparison"] = {
        "base_questions_flags_and_responses_match": True,
        "merged_questions_and_flags_match": True,
        "new_merged_responses_not_printed": True,
        "old_merged_response_source_still_required": True,
    }
    nb, resource = read_notebook("qwen3_4b_merged_model_resource_20251119.ipynb")
    resource["recorded_execution"] = recorded_sequence(nb)
    resources = nb["metadata"]["kaggle"]["dataSources"]
    require(len(resources) == 1, "Resource notebook must retain exactly one Kaggle input")
    expected = {"modelId": 508779, "modelInstanceId": 493349,
                "sourceId": 653102, "sourceType": "modelInstanceVersion"}
    require(all(resources[0].get(k) == v for k, v in expected.items()), "Merged resource IDs differ")
    configs = [(i, c) for i, c in enumerate(nb["cells"], 1)
               if re.search(r"^base_model_path\s*=", source(c), re.M)]
    require(len(configs) == 1 and literal_assignment(ast.parse(source(configs[0][1])).body,
                                                   "base_model_path") == MERGED_PATH,
            "Resource notebook must select the documented merged path")
    require(configs[0][0] < verify_loader(nb, MERGED_PATH, "Merged"), "Resource path must precede loading")
    resource.update(active_model_path=MERGED_PATH, kaggle_input=resources[0],
                    weights_present=False, checkpoint_identity_authenticated=False)
    report["model_resource"] = resource
    report["limitations"] = [
        "These are saved notebook records, not new model inference or independent checkpoint authentication.",
        "Arithmetic 17/20 selected qwen-3/transformers/4b/1, not the project's merged checkpoint or 4b-base/1.",
        "The thesis arithmetic 14/20 versus 17/20 paired experiment remains unrecovered.",
        "Reasoning 12/20 and 15/20 retain 5/20 reconstructed training-question overlap.",
        "Early 0/20 and 2/20 runs used empty system prompts and are separate settings.",
        "The recovered 15/20 merged run does not print model responses; prior notebook responses remain necessary.",
        "Numeric execution flags do not establish mathematical proof validity.",
        "Kaggle input IDs and loading logs are recovery clues; original checkpoint weights are absent.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON report; existing files are not overwritten")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    try:
        report = verify()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        print("PASS: 5 saved evaluation runs (17/20, 12/20, 15/20, 0/20, 2/20), model selections, "
              "ordered execution, dataset questions and merged resource IDs.")
        print("No notebook code, model inference or generated programs were executed.")
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
