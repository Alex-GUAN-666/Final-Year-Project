#!/usr/bin/env python3
"""Audit all bundled historical files. Standard library only; executes no model code.

This is evidence recovery, not model training or a new benchmark run.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def output_text(output: dict) -> str:
    value = output.get("text", output.get("data", {}).get("text/plain", ""))
    return "".join(value) if isinstance(value, list) else str(value)


def duplicates(values: list[str]) -> dict:
    counts = Counter(values)
    return {
        "unique": len(counts),
        "duplicate_excess": sum(n - 1 for n in counts.values()),
        "repeated_values": {v: n for v, n in counts.items() if n > 1},
    }


def inspect_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    summaries = []
    prompt_resets = []
    outputs = []
    references = set()
    for number, cell in enumerate(cells, 1):
        source = "".join(cell.get("source", []))
        references.update(re.findall(r"[\w.\-]+\.(?:jsonl|csv|safetensors)", source))
        if re.search(r'(?:code_system_prompt|cot_code_system_prompt)\s*=\s*[\"\']{2}', source):
            prompt_resets.append({"cell": number, "execution_count": cell.get("execution_count"), "source": source})
        texts = [output_text(o) for o in cell.get("outputs", [])]
        joined = "\n".join(texts)
        outputs.append({
            "cell": number,
            "type": cell["cell_type"],
            "execution_count": cell.get("execution_count"),
            "source_chars": len(source),
            "text_output_chars": len(joined),
            "text_output_sha256": sha256(joined.encode()),
            "error_types": [o.get("ename") for o in cell.get("outputs", []) if o.get("output_type") == "error"],
        })
        match = re.search(r"Total Questions:\s*(\d+)\s*Correct Predictions:\s*(\d+)\s*Accuracy:\s*([\d.]+)%", joined)
        if match:
            total, correct = int(match[1]), int(match[2])
            summaries.append({
                "cell": number,
                "total": total,
                "correct": correct,
                "printed_accuracy_percent": float(match[3]),
                "recomputed_accuracy_percent": 100 * correct / total if total else None,
                "evidence_type": "historical saved output; not a new inference run",
            })
    return {
        "cells": len(cells),
        "code_cells": sum(c["cell_type"] == "code" for c in cells),
        "markdown_cells": sum(c["cell_type"] == "markdown" for c in cells),
        "cell_inventory": outputs,
        "saved_evaluation_summaries": summaries,
        "prompt_reset_cells": prompt_resets,
        "referenced_data_or_weight_names": sorted(references),
    }


def inspect_data(path: Path) -> tuple[dict, list[str]]:
    if path.suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"{path.name}: blank JSONL row {line_number}")
            rows.append(json.loads(line))
        keys = sorted(set().union(*(set(r) for r in rows))) if rows else []
        questions = [r.get("problem", "") for r in rows]
        codes = [r.get("generated_python_code", "") for r in rows]
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            keys = reader.fieldnames or []
        if any(None in row for row in rows):
            raise ValueError(f"{path.name}: CSV contains extra unlabelled columns")
        questions = [r.get("arithmetical question", r.get("question", "")) for r in rows]
        codes = [r.get("generated_python_code", r.get("code for solving the question", r.get("extracted_code", ""))) for r in rows]
    syntax_errors = []
    missing_code_rows = []
    for row_number, code in enumerate(codes, 1):
        if not isinstance(code, str) or code.strip() in ("", "None", "N/A"):
            missing_code_rows.append(row_number)
            continue
        try:
            ast.parse(code)
        except (SyntaxError, ValueError) as exc:
            syntax_errors.append({"row": row_number, "error": str(exc)})
    return {
        "rows": len(rows),
        "columns": keys,
        "questions": duplicates(questions),
        "missing_question_rows": [i for i, q in enumerate(questions, 1) if not q.strip()],
        "placeholder_solution_rows": [i for i, r in enumerate(rows, 1) if str(r.get("generated_solution", "")).strip() == "..."],
        "reference_code_syntax_errors": syntax_errors,
        "missing_or_placeholder_code_rows": missing_code_rows,
        "is_correct_flag_counts": dict(Counter(str(r.get("is_correct", "absent")) for r in rows)),
        "contains_model_prediction_columns": any(k in keys for k in ("llm_response", "execution_result")),
        "reference_code_executed": False,
    }, questions


def run(root: Path) -> dict:
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    result = {
        "purpose": "Audit supplied files and recover saved evidence; not retraining or rerunning inference.",
        "files": [], "notebooks": {}, "datasets": {}, "exact_question_overlaps": [],
        "model_training_executed": False, "model_inference_executed": False,
        "generated_code_executed": False,
    }
    questions = {}
    for entry in manifest:
        relative = entry.get("repository_path")
        if relative is None:
            result["files"].append({"name": entry["file"], "status": "paper retained outside repository", "sha256": entry["sha256"]})
            continue
        path = root / relative
        data = path.read_bytes()
        expected_bytes = entry.get("published_bytes", entry["bytes"])
        expected_hash = entry.get("published_sha256", entry["sha256"])
        if len(data) != expected_bytes or sha256(data) != expected_hash:
            raise ValueError(f"Published artifact changed: {relative}")
        result["files"].append({"name": entry["file"], "path": relative, "bytes": len(data),
                               "sha256_verified": True,
                               "original_byte_identical": "published_sha256" not in entry,
                               "original_sha256": entry["sha256"], "published_sha256": expected_hash})
        if path.suffix == ".ipynb":
            result["notebooks"][path.name] = inspect_notebook(path)
        elif path.suffix in (".csv", ".jsonl"):
            result["datasets"][path.name], questions[path.name] = inspect_data(path)
        elif path.suffix == ".py":
            ast.parse(data.decode("utf-8-sig"), filename=path.name)
    names = sorted(questions)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            overlap = sorted(set(questions[first]) & set(questions[second]))
            result["exact_question_overlaps"].append({"first": first, "second": second, "shared_unique_questions": len(overlap), "questions": overlap})
    result["missing_for_historical_reproduction"] = [
        "Original LoRA adapter plus tokenizer/config, or merged model plus tokenizer/config",
        "Original arithmetic baseline/merged predictions on the same 20 questions",
        "Original tokenizer snapshot and base weight fingerprint; public-tokenizer reconstruction is available",
    ]
    result["recovered_in_second_upload"] = "results_v3.jsonl:1125 rows; duplicated CSVs both match base output, not a separate merged run."
    result["historical_overlap"] = "Public-tokenizer reconstruction retains143 distilled rows;5of20 historical evaluated questions overlap. See reports/recovered_data_audit.json."
    result["warning"] = "Syntax and saved-output checks do not establish model correctness, proof correctness, absence of leakage, or GPU reproducibility."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    try:
        result = run(args.root.resolve())
    except (ValueError, OSError, KeyError, SyntaxError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        bundled = sum(bool(item.get("sha256_verified")) for item in result["files"])
        rows = sum(item["rows"] for item in result["datasets"].values())
        print(f"Verified {bundled} bundled artifact hashes (including declared public derivatives); audited {len(result['notebooks'])} notebooks and {rows} data records. Report: {args.output}")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
