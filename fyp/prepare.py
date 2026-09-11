"""Prepare a question-disjoint new experiment or historical candidate recovery.

This stage uses the standard library. Exact tokenizer filtering happens in train.
No source files are changed; every source row receives a disposition.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .data import ROOT, file_hash, question_id, read_csv, read_jsonl, write_jsonl

SYNTHETIC = "arithmetic_logic_data_revised_v2.jsonl"
DISTILLED = "results_v3.jsonl"
ARITHMETIC = "arithmetic_problems_v3.csv"
REASONING = "random_samples_for_inference.csv"
PROTOCOLS = ("question_disjoint_v1", "historical_candidates")
QUARANTINE = {
    15: "Additional even-per-line condition contradicts reference answer 59.",
    43: "Oddness incorrectly used as injectivity; reference answer is not uniquely determined.",
}


def source_record(name: str, row: int, problem: str, category: str) -> dict:
    qid = question_id(problem)
    return {"id": f"{name}:{row}", "question_id": qid, "problem": problem,
            "category": category, "source_file": name, "source_row": row}


def build(raw_dir: Path, prompt_dir: Path, protocol: str) -> tuple[list, list, dict, list]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol}")
    synthetic = read_jsonl(raw_dir / SYNTHETIC)
    distilled = read_jsonl(raw_dir / DISTILLED)
    arith_eval = read_csv(raw_dir / ARITHMETIC)
    math_eval = read_csv(raw_dir / REASONING)
    prompts = {
        "arithmetic_only": (prompt_dir / "arithmetic_code_only.txt").read_text(encoding="utf-8"),
        "cot_and_code": (prompt_dir / "reasoning_cot_code.txt").read_text(encoding="utf-8"),
    }
    # Exclusion uses ALL source evaluation questions, including duplicates and
    # quarantined questions, before selecting train examples or token filtering.
    holdout = {question_id(r["arithmetical question"]) for r in arith_eval + math_eval}
    rejected = []
    train = []
    seen_train = set()
    dispositions = []
    for name, records, category in ((SYNTHETIC, synthetic, "arithmetic_only"),
                                     (DISTILLED, distilled, "cot_and_code")):
        for number, source in enumerate(records, 1):
            problem = source.get("problem")
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"Missing problem: {name}:{number}")
            record = source_record(name, number, problem, category)
            eligible = name == SYNTHETIC or bool(source.get("is_correct") and source.get("generated_solution") and source.get("generated_python_code"))
            reason = None
            if not eligible:
                reason = "not_historically_eligible"
            elif protocol == "question_disjoint_v1" and record["question_id"] in holdout:
                reason = "evaluation_question_overlap"
            elif protocol == "question_disjoint_v1" and record["question_id"] in seen_train:
                reason = "duplicate_training_question"
            if reason:
                dispositions.append({**{k: record[k] for k in ("id", "source_file", "source_row", "question_id")}, "disposition": reason})
                continue
            code = source.get("generated_python_code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError(f"Eligible row lacks code: {name}:{number}")
            target = f"```python\n{code}\n```"
            if category == "cot_and_code":
                target = source["generated_solution"] + "\n\n" + target
            record["messages"] = [
                {"role": "system", "content": prompts[category]},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": target},
            ]
            train.append(record)
            seen_train.add(record["question_id"])
            dispositions.append({**{k: record[k] for k in ("id", "source_file", "source_row", "question_id")}, "disposition": "training_candidate"})
    evaluation = []
    seen_eval = set()
    for name, records, category in ((ARITHMETIC, arith_eval, "arithmetic_only"),
                                     (REASONING, math_eval, "cot_and_code")):
        for number, source in enumerate(records, 1):
            problem = source["arithmetical question"]
            record = source_record(name, number, problem, category)
            if protocol == "historical_candidates" and (name == ARITHMETIC or number > 20):
                # Arithmetic historical subset identities were not supplied.
                reason = "not_confirmed_historical_evaluation_subset"
            elif protocol == "question_disjoint_v1" and name == REASONING and number in QUARANTINE:
                reason = "quarantined_reference_problem"
            elif record["question_id"] in seen_eval:
                reason = "duplicate_evaluation_question"
            else:
                reason = None
            if reason:
                rejected.append({**record, "reason": reason, "detail": QUARANTINE.get(number) if name == REASONING else None})
                continue
            record["answer"] = source["answer to the question"]
            evaluation.append(record)
            seen_eval.add(record["question_id"])
    overlaps = sorted({r["question_id"] for r in train} & {r["question_id"] for r in evaluation})
    if protocol == "question_disjoint_v1" and overlaps:
        raise AssertionError("Training/evaluation question overlap survived exclusion")
    manifest = {
        "protocol": protocol,
        "experiment_status": "new experiment, no new model results" if protocol == "question_disjoint_v1" else "historical candidate reconstruction; contains known evaluation overlap",
        "token_filter_applied": False,
        "token_filter_stage": "train: exact tokenizer/template length <= max_seq_length; no truncation",
        "source_hashes": {name: file_hash(raw_dir / name) for name in (SYNTHETIC, DISTILLED, ARITHMETIC, REASONING)},
        "prompt_hashes": {p.name: file_hash(p) for p in sorted(prompt_dir.glob("*.txt"))},
        "template_hash": file_hash(prompt_dir / "historical_chat_template.jinja"),
        "question_normalization": "Unicode NFKC, replace $Calculate# with Calculate, whitespace collapse, casefold",
        "excluded_evaluation_question_count": len(holdout) if protocol == "question_disjoint_v1" else 0,
        "training_candidates": len(train),
        "training_by_category": dict(Counter(r["category"] for r in train)),
        "training_dispositions": dict(Counter(r["disposition"] for r in dispositions)),
        "evaluation_questions": len(evaluation),
        "evaluation_by_category": dict(Counter(r["category"] for r in evaluation)),
        "evaluation_exclusions": rejected,
        "train_eval_normalized_question_overlap": len(overlaps),
        "limitations": ["String normalization does not detect every semantic duplicate or pretraining exposure.",
                        "References are mathematical numeric answers, not formally verified proofs.",
                        "New protocol differs from historical training; original weights are absent."],
    }
    return train, evaluation, manifest, dispositions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--prompt-dir", type=Path, default=ROOT / "prompts")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="question_disjoint_v1")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = args.output_dir or ROOT / "data" / "prepared" / args.protocol
    try:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Output directory is not empty: {destination}; choose a new path")
        train, evaluation, manifest, dispositions = build(args.raw_dir, args.prompt_dir, args.protocol)
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(destination / "train_candidates.jsonl", train)
        write_jsonl(destination / "eval_questions.jsonl", evaluation)
        write_jsonl(destination / "source_dispositions.jsonl", dispositions)
        manifest["prepared_hashes"] = {name: file_hash(destination / name) for name in ("train_candidates.jsonl", "eval_questions.jsonl", "source_dispositions.jsonl")}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(f"Prepared {len(train)} training candidates and {len(evaluation)} evaluation questions.")
    print(f"Normalized train/evaluation overlap: {manifest['train_eval_normalized_question_overlap']}; tokenizer filtering still required.")
    print(destination)


if __name__ == "__main__":
    main()
