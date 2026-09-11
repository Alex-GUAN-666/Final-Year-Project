"""Controlled inference with optional Docker-only numeric-output evaluation."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import verify_prepared_manifest
from .execution import GRADER_PROTOCOL, ensure_docker, execute_docker, extract_first_code, grade_execution


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_predictions(rows):
    total = len(rows)
    graded = [row for row in rows if row.get("grade", {}).get("correct") is not None]
    correct = sum(row["grade"]["correct"] is True for row in graded)
    return {"total": total, "graded": len(graded), "ungraded": total - len(graded),
            "correct": correct, "accuracy": correct / len(graded) if graded else None,
            "statuses": dict(Counter(row.get("grade", {}).get("status", "ungraded") for row in rows)),
            "metric": "Numeric execution-output accuracy; not proof validity"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Local model directory or Hugging Face model ID")
    parser.add_argument("--model-revision", default="main", help="Prefer an immutable HF commit")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL; existing files are never overwritten")
    parser.add_argument("--prompt-dir", type=Path, default=Path(__file__).resolve().parents[1] / "prompts")
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--eos-token", default="<|im_end|>", help="Explicit shared EOS override for both models")
    parser.add_argument("--execution", choices=("none", "docker"), default="none")
    parser.add_argument("--docker-image", default="python:3.11-slim", help="Local image; no implicit pull")
    parser.add_argument("--execution-timeout", type=float, default=10)
    parser.add_argument("--output-byte-limit", type=int, default=65536)
    parser.add_argument("--limit", type=int, help="Optional prefix for a smoke run; recorded in metadata")
    return parser


def run(args):
    if not 0 < args.max_new_tokens < args.max_total_tokens:
        raise ValueError("Require 0 < max-new-tokens < max-total-tokens")
    if args.execution_timeout <= 0 or args.output_byte_limit <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("Execution limits and optional row limit must be positive")
    data_path, manifest_path = args.data_dir / "eval_questions.jsonl", args.data_dir / "manifest.json"
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = verify_prepared_manifest(args.data_dir, "eval_questions.jsonl")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Evaluation row IDs must be unique")
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Evaluation set is empty")
    prompts = {"arithmetic_only": "arithmetic_code_only.txt", "cot_and_code": "reasoning_cot_code.txt"}
    for row in rows:
        if row["category"] not in prompts:
            raise ValueError("Unknown evaluation category: " + str(row["category"]))
    prompt_paths = {key: args.prompt_dir / filename for key, filename in prompts.items()}
    prompt_texts = {key: path.read_text(encoding="utf-8") for key, path in prompt_paths.items()}
    template_path = args.prompt_dir / "historical_chat_template.jinja"
    template = template_path.read_text(encoding="utf-8")
    meta_path, summary_path = Path(str(args.output) + ".meta.json"), Path(str(args.output) + ".summary.json")
    for path in (args.output, meta_path, summary_path):
        if path.exists():
            raise FileExistsError("Refusing to overwrite " + str(path))
    image_id = ensure_docker(args.docker_image) if args.execution == "docker" else None
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Inference dependencies are missing. Install the repository's model requirements in a dedicated environment; --help and unit tests need only Python.") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a GPU environment, or explicitly --device cpu (4B inference is slow and requires substantial RAM).")
    torch.manual_seed(3407)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision,
                                              local_files_only=args.local_files_only, trust_remote_code=False)
    if args.eos_token not in tokenizer.get_vocab():
        raise ValueError("Requested EOS token is absent from tokenizer vocabulary: " + args.eos_token)
    tokenizer.eos_token = args.eos_token
    tokenizer.pad_token = args.eos_token
    tokenizer.chat_template = template
    # Both models receive identical system/user content. Reference answers never enter messages.
    prepared = []
    for row in rows:
        messages = [{"role": "system", "content": prompt_texts[row["category"]]},
                    {"role": "user", "content": row["problem"]}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
        count = inputs["input_ids"].shape[-1]
        if count + args.max_new_tokens > args.max_total_tokens:
            raise ValueError(f"Row {row['id']} uses {count} prompt tokens; with completion budget it exceeds {args.max_total_tokens}. No silent truncation is permitted.")
        prepared.append((row, inputs, count))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, local_files_only=args.local_files_only,
        trust_remote_code=False, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32)
    model.to(args.device).eval()
    context = getattr(model.config, "max_position_embeddings", None)
    if context and args.max_total_tokens > context:
        raise ValueError("Requested context budget exceeds model max_position_embeddings")
    generation_config = transformers.GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "initialized; completion is recorded in summary.json",
                "protocol": "new_controlled_historical_prompts_same_explicit_template_both_models",
                "historical_exact_replay_claim": False,
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "generation": generation_config.to_dict(), "seed": 3407,
                "model_resolved_commit": getattr(model.config, "_commit_hash", None),
                "tokenizer_resolved_commit": tokenizer.init_kwargs.get("_commit_hash"),
                "tokenizer_vocabulary_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "inference_runtime": {"device_type": args.device,
                                      "model_dtype": str(next(model.parameters()).dtype),
                                      "torch_version": torch.__version__,
                                      "transformers_version": transformers.__version__, "seed": 3407},
                "data_sha256": sha256(data_path), "manifest_sha256": sha256(manifest_path),
                "data_manifest": manifest, "selected_row_ids": [row["id"] for row in rows],
                "prompt_sha256": {key: sha256(path) for key, path in prompt_paths.items()},
                "chat_template_sha256": sha256(template_path), "docker_image_id": image_id,
                "grader_protocol": GRADER_PROTOCOL,
                "warning": "No model-code host execution. Numeric output accuracy does not evaluate proof validity."}
    if Path(args.model).is_dir():
        metadata["local_model_config_sha256"] = {path.name: sha256(path) for path in Path(args.model).glob("*.json")}
        metadata["local_model_weights"] = [{"name": p.name, "bytes": p.stat().st_size} for p in Path(args.model).glob("*.safetensors")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    records = []
    with args.output.open("x", encoding="utf-8") as handle:
        for row, inputs, count in prepared:
            inputs = {key: tensor.to(args.device) for key, tensor in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=generation_config)
            completion = generated[0, count:]
            raw = tokenizer.decode(completion, skip_special_tokens=True)
            code = extract_first_code(raw)
            execution = ({"status": "not_executed", "stdout": None, "stderr": None} if args.execution == "none"
                         else execute_docker(code, image_id, args.execution_timeout, args.output_byte_limit))
            record = {**row, "model": args.model, "raw_response": raw, "first_fenced_code": code,
                      "prompt_tokens": count, "completion_tokens": len(completion),
                      "reached_completion_budget": len(completion) == args.max_new_tokens,
                      "execution": execution, "grade": grade_execution(execution, row["answer"])}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            print(f"{len(records)}/{len(rows)} {row['id']}: {record['grade']['status']}", file=sys.stderr)
    summary = summarize_predictions(records)
    summary["by_category"] = {category: summarize_predictions([row for row in records if row["category"] == category]) for category in prompts}
    summary["completed"] = True
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f"Evaluation stopped: {exc}\n")


if __name__ == "__main__":
    main()
