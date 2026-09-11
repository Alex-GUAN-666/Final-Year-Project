"""Recover exact token-filter membership without downloading model weights."""
import argparse
import hashlib
import json
from pathlib import Path

from .data import ROOT, file_hash
from .train import read_candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--revision", default="906bfd4b4dc7f14ee4320094d8b41684abff8539")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--eos-token", default="<|im_end|>")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output exists; choose a new report path")
        if args.max_seq_length < 2:
            raise ValueError("max-seq-length must be at least2")
        rows, manifest = read_candidates(args.data_dir)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision,
                    local_files_only=args.local_files_only, trust_remote_code=False)
        if args.eos_token not in tokenizer.get_vocab():
            raise ValueError("Requested EOS is not in tokenizer vocabulary")
        tokenizer.eos_token = args.eos_token
        template = ROOT / "prompts/historical_chat_template.jinja"
        tokenizer.chat_template = template.read_text(encoding="utf-8")
        evaluated = []
        for row in rows:
            n = len(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False))
            evaluated.append({"id": row["id"], "question_id": row["question_id"],
                "source_row": row["source_row"], "source_file": row["source_file"],
                "category": row["category"], "tokens": n, "retained": 2 <= n <= args.max_seq_length})
        retained = [r for r in evaluated if r["retained"]]
        result = {"protocol": manifest["protocol"], "tokenizer": args.tokenizer, "revision": args.revision,
            "tokenizer_vocab_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
            "template_sha256": file_hash(template), "eos_token": args.eos_token,
            "data_sha256": file_hash(args.data_dir / "train_candidates.jsonl"),
            "max_seq_length": args.max_seq_length, "candidates": len(rows), "retained": len(retained),
            "retained_by_category": {c: sum(r["category"] == c for r in retained) for c in ("arithmetic_only", "cot_and_code")},
            "rows": evaluated, "model_weights_loaded": False, "original_kaggle_tokenizer_authenticated": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    except ImportError as exc:
        p.exit(1, f"Install requirements-model.txt for tokenization: {exc}\n")
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f"Token filtering failed: {exc}\n")


if __name__ == "__main__":
    main()
