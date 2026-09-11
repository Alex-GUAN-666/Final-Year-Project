"""Merge a reconstructed LoRA adapter into its recorded, nonquantized base model."""
import argparse
import json
from pathlib import Path

from .train import config_fingerprint, sha, versions, write_json


def same_model(left, right):
    return left == right or (Path(left).is_dir() and Path(right).is_dir() and Path(left).resolve() == Path(right).resolve())


def resolve_base(adapter, model=None, revision=None):
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    provenance = json.loads((adapter / "reproduction_manifest.json").read_text(encoding="utf-8"))
    if provenance.get("status") != "completed":
        raise ValueError("Adapter requires the completed training reproduction manifest.")
    base = provenance["base_model"]
    expected_revision = provenance.get("resolved_revision") or provenance["requested_revision"]
    if not same_model(config.get("base_model_name_or_path", ""), base) or (model and not same_model(model, base)):
        raise ValueError("Base model does not match adapter provenance; refusing a silent model substitution.")
    if config.get("revision") != expected_revision or (revision and revision != expected_revision):
        raise ValueError("Base revision does not match adapter provenance.")
    return base, expected_revision, provenance


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model", help="Optional verification of the recorded base model; cannot substitute another model.")
    p.add_argument("--model-revision", help="Optional verification; resolved training commit takes precedence over a branch name.")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = p.parse_args(argv)
    base, revision, provenance = resolve_base(args.adapter, args.model, args.model_revision)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty.")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies. Install the documented training requirements first.") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, torch_dtype=getattr(torch, args.dtype), local_files_only=args.local_files_only, trust_remote_code=False)
    if getattr(model, "is_quantized", False):
        raise ValueError("Merging requires reloading the nonquantized base model.")
    if config_fingerprint(model.config) != provenance["base_config_sha256"]:
        raise ValueError("Reloaded base config differs from the training base; refusing merge.")
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False, local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.config.eos_token_id = tokenizer.eos_token_id
    merged.config.pad_token_id = tokenizer.pad_token_id
    merged.generation_config.eos_token_id = tokenizer.eos_token_id
    merged.generation_config.pad_token_id = tokenizer.pad_token_id
    merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output_dir)
    write_json(args.output_dir / "merge_manifest.json", {"base_model": base, "revision": revision, "dtype": args.dtype,
        "adapter_manifest_sha256": sha(args.adapter / "reproduction_manifest.json"),
        "adapter_config_sha256": sha(args.adapter / "adapter_config.json"), "safe_merge": True,
        "libraries": versions(), "source_training": provenance, "original_historical_weights": False})
    print(f"Saved merged model: {args.output_dir}")


if __name__ == "__main__":
    main()
