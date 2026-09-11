"""Score completed prediction files using Docker on Linux/WSL2; no ML dependencies."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import sha256, summarize_predictions
from .execution import GRADER_PROTOCOL, ensure_docker, execute_docker, extract_first_code, grade_execution


def load_completed_predictions(path):
    path = Path(path)
    meta_path, summary_path = Path(str(path) + ".meta.json"), Path(str(path) + ".summary.json")
    if not meta_path.is_file() or not summary_path.is_file():
        raise ValueError("Both prediction .meta.json and completed .summary.json sidecars are required; partial runs cannot be scored as complete.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(metadata, dict) or not isinstance(summary, dict) or summary.get("completed") is not True:
        raise ValueError("Prediction summary does not confirm a completed run")
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not isinstance(row.get("raw_response"), str) or "answer" not in row
                or row.get("category") not in ("arithmetic_only", "cot_and_code")):
            raise ValueError("Malformed prediction row: require id, raw_response, answer and valid category")
    ids = [row["id"] for row in rows]
    if (not rows or len(set(ids)) != len(ids) or type(summary.get("total")) is not int
            or summary.get("total") != len(rows)
            or metadata.get("selected_row_ids") != ids):
        raise ValueError("Prediction row IDs/count do not match completed-run sidecars")
    return rows, metadata, summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New scored JSONL; originals are preserved")
    parser.add_argument("--docker-image", default="python:3.11-slim", help="Local image, explicitly pulled/built beforehand")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--output-byte-limit", type=int, default=65536)
    return parser


def run(args):
    if args.timeout <= 0 or args.output_byte_limit <= 0:
        raise ValueError("Timeout and output limit must be positive")
    rows, source_metadata, source_summary = load_completed_predictions(args.predictions)
    meta_path, summary_path = Path(str(args.output) + ".meta.json"), Path(str(args.output) + ".summary.json")
    for path in (args.output, meta_path, summary_path):
        if path.exists():
            raise FileExistsError("Refusing to overwrite " + str(path))
    image_id = ensure_docker(args.docker_image)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": "deferred_docker_numeric_output_scoring", "selected_row_ids": [row["id"] for row in rows],
                "source_predictions_sha256": sha256(args.predictions),
                "source_metadata_sha256": sha256(str(args.predictions) + ".meta.json"),
                "source_summary_sha256": sha256(str(args.predictions) + ".summary.json"),
                "source_metadata": source_metadata, "source_summary": source_summary,
                "docker_image_requested": args.docker_image, "docker_image_id": image_id,
                "grader_protocol": GRADER_PROTOCOL,
                "timeout_seconds": args.timeout, "output_byte_limit": args.output_byte_limit,
                "metric": "Numeric execution-output accuracy; not proof validity"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    scored = []
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            code = extract_first_code(row["raw_response"])
            execution = execute_docker(code, image_id, args.timeout, args.output_byte_limit)
            record = {**row, "first_fenced_code": code, "execution": execution,
                      "grade": grade_execution(execution, row["answer"])}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            scored.append(record)
            print(f"{len(scored)}/{len(rows)} {row['id']}: {record['grade']['status']}", file=sys.stderr)
    summary = summarize_predictions(scored)
    summary["by_category"] = {category: summarize_predictions([row for row in scored if row["category"] == category])
                              for category in ("arithmetic_only", "cot_and_code")}
    summary["completed"] = True
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Scoring stopped: {exc}\n")


if __name__ == "__main__":
    main()
