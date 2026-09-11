"""Compare complete, fully graded paired runs only when their inference protocols match."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import sha256, summarize_predictions
from .execution import GRADER_PROTOCOL, extract_first_code, grade_execution
from .score import load_completed_predictions


def inference_metadata(metadata):
    for _ in range(8):
        if "source_metadata" not in metadata:
            return metadata
        metadata = metadata["source_metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Invalid source inference metadata")
    raise ValueError("Source metadata nesting is too deep")


def validate_protocol(metadata):
    for key in ("data_sha256", "chat_template_sha256", "tokenizer_vocabulary_sha256"):
        if not isinstance(metadata.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[key]):
            raise ValueError("Missing or invalid inference provenance: " + key)
    prompts = metadata.get("prompt_sha256")
    if (not isinstance(prompts, dict) or set(prompts) != {"arithmetic_only", "cot_and_code"}
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in prompts.values())):
        raise ValueError("Missing or invalid inference prompt hashes")
    generation = metadata.get("generation")
    required = {"max_new_tokens", "do_sample", "num_beams", "eos_token_id", "pad_token_id"}
    if not isinstance(generation, dict) or not required.issubset(generation):
        raise ValueError("Missing core generation settings in inference metadata")
    # Compare every saved generation setting except library metadata; this includes
    # stopping tokens, penalties, token budgets and deterministic/sampling options.
    generation = {key: value for key, value in generation.items()
                  if key not in ("transformers_version", "_from_model_config")}
    runtime = metadata.get("inference_runtime")
    runtime_strings = ("device_type", "model_dtype", "torch_version", "transformers_version")
    if (not isinstance(runtime, dict)
            or any(not isinstance(runtime.get(key), str) or not runtime[key] for key in runtime_strings)
            or type(runtime.get("seed")) is not int):
        raise ValueError("Missing or invalid numerical inference runtime provenance")
    return {key: metadata[key] for key in ("data_sha256", "prompt_sha256", "chat_template_sha256",
                                           "tokenizer_vocabulary_sha256")} | {
                                               "generation": generation, "inference_runtime": runtime}


def scoring_protocol(metadata):
    config = metadata.get("config", {})
    if metadata.get("protocol") == "deferred_docker_numeric_output_scoring":
        timeout, output_limit = metadata.get("timeout_seconds"), metadata.get("output_byte_limit")
    elif isinstance(config, dict) and config.get("execution") == "docker":
        timeout, output_limit = config.get("execution_timeout"), config.get("output_byte_limit")
    else:
        raise ValueError("Missing Docker scoring execution provenance")
    image = metadata.get("docker_image_id")
    if (not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image)
            or not isinstance(timeout, (int, float)) or not timeout > 0
            or type(output_limit) is not int or output_limit <= 0
            or metadata.get("grader_protocol") != GRADER_PROTOCOL):
        raise ValueError("Missing or invalid Docker scoring conditions/grader protocol")
    return {"docker_image_id": image, "timeout_seconds": timeout,
            "output_byte_limit": output_limit, "grader_protocol": metadata["grader_protocol"]}


def validate_scored_predictions(label, rows, summary):
    """Reject stale grades, extracted code and summaries without rerunning programs."""
    for row in rows:
        grade = row.get("grade", {})
        if (not isinstance(grade, dict) or type(grade.get("correct")) is not bool
                or grade.get("status") != ("correct" if grade["correct"] else "incorrect")):
            raise ValueError(label + " run is not fully and consistently graded: " + row["id"])
        execution = row.get("execution")
        statuses = {"ok", "no_code", "code_error", "timeout", "output_limit",
                    "not_executed", "execution_error"}
        if (not isinstance(execution, dict) or not isinstance(execution.get("status"), str)
                or execution["status"] not in statuses
                or not isinstance(execution.get("stdout"), str)
                or not isinstance(execution.get("stderr"), str)):
            raise ValueError(label + " row lacks valid execution evidence: " + row["id"])
        extracted = extract_first_code(row["raw_response"])
        if "first_fenced_code" not in row or row["first_fenced_code"] != extracted:
            raise ValueError(label + " extracted code differs from raw response: " + row["id"])
        if (execution["status"] == "no_code") != (extracted is None):
            raise ValueError(label + " code presence disagrees with execution status: " + row["id"])
        if grade != grade_execution(execution, row["answer"]):
            raise ValueError(label + " saved grade disagrees with execution evidence: " + row["id"])
    expected = summarize_predictions(rows)
    expected["by_category"] = {
        category: summarize_predictions([row for row in rows if row["category"] == category])
        for category in ("arithmetic_only", "cot_and_code")
    }
    for key, value in expected.items():
        if key not in summary or summary[key] != value:
            raise ValueError(label + " summary disagrees with prediction rows: " + key)
        if isinstance(value, int) and type(summary[key]) is not int:
            raise ValueError(label + " summary has invalid count type: " + key)


def compare_runs(base_path, merged_path):
    base, base_meta, base_summary = load_completed_predictions(base_path)
    merged, merged_meta, merged_summary = load_completed_predictions(merged_path)
    scoring_base, scoring_merged = scoring_protocol(base_meta), scoring_protocol(merged_meta)
    for key in scoring_base:
        if scoring_base[key] != scoring_merged[key]:
            raise ValueError("Scoring conditions are not comparable; mismatch: " + key)
    source_base, source_merged = inference_metadata(base_meta), inference_metadata(merged_meta)
    protocol_base, protocol_merged = validate_protocol(source_base), validate_protocol(source_merged)
    for key in protocol_base:
        if protocol_base[key] != protocol_merged[key]:
            raise ValueError("Inference protocols are not comparable; mismatch: " + key)
    if len(base) != len(merged):
        raise ValueError("Paired runs contain different numbers of questions")
    for label, rows, metadata in (("base", base, source_base), ("merged", merged, source_merged)):
        if metadata.get("selected_row_ids") != [row["id"] for row in rows]:
            raise ValueError(label + " rows do not match original inference selected IDs")
        for row in rows:
            if not isinstance(row.get("problem"), str):
                raise ValueError("Question text is missing: " + row["id"])
    for left, right in zip(base, merged):
        for key in ("id", "problem", "answer", "category", "question_id"):
            if json.dumps(left.get(key), sort_keys=True) != json.dumps(right.get(key), sort_keys=True):
                raise ValueError("Paired questions differ in " + key + ": " + left["id"])
    validate_scored_predictions("base", base, base_summary)
    validate_scored_predictions("merged", merged, merged_summary)

    def metrics(pairs):
        total = len(pairs)
        left = sum(a["grade"]["correct"] for a, _ in pairs)
        right = sum(b["grade"]["correct"] for _, b in pairs)
        return {"base": {"correct": left, "total": total, "accuracy": left / total if total else None},
                "merged": {"correct": right, "total": total, "accuracy": right / total if total else None},
                "delta_percentage_points": 100 * (right - left) / total if total else None}

    pairs = list(zip(base, merged))
    changed = {"improved": [], "regressed": []}
    for left, right in pairs:
        if left["grade"]["correct"] != right["grade"]["correct"]:
            changed["improved" if right["grade"]["correct"] else "regressed"].append(left["id"])
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "metric": "Numeric execution-output accuracy; not proof validity",
            "base_predictions_sha256": sha256(base_path), "merged_predictions_sha256": sha256(merged_path),
            "matched_scoring_protocol": scoring_base,
            "matched_inference_protocol": protocol_base, "overall": metrics(pairs),
            "by_category": {category: metrics([(a, b) for a, b in pairs if a["category"] == category])
                            for category in ("arithmetic_only", "cot_and_code")}, "paired_changed_ids": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Completed, fully graded baseline JSONL")
    parser.add_argument("--merged", type=Path, required=True, help="Completed, fully graded adapted JSONL")
    parser.add_argument("--output", type=Path, required=True, help="New comparison report JSON; never overwritten")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite " + str(args.output))
        report = compare_runs(args.base, args.merged)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Comparison refused: {exc}\n")


if __name__ == "__main__":
    main()
