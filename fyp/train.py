"""Portable Transformers/PEFT reconstruction; this is not the historical Unsloth run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from .data import verify_prepared_manifest

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def config_fingerprint(config):
    value = config.to_dict()
    for key in ("_name_or_path", "transformers_version", "torch_dtype", "_commit_hash", "quantization_config", "_pre_quantization_dtype"):
        value.pop(key, None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def versions():
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "bitsandbytes"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def read_candidates(data_dir):
    folder = Path(data_dir)
    manifest = verify_prepared_manifest(folder, "train_candidates.jsonl")
    rows = [json.loads(line) for line in (folder / "train_candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(manifest, dict) or not rows:
        raise ValueError("Prepared manifest must be an object and training candidates must be nonempty.")
    ids = set()
    for row in rows:
        if not row.get("id") or row["id"] in ids:
            raise ValueError("Training candidate IDs must be present and unique.")
        ids.add(row["id"])
        messages = row.get("messages", [])
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Candidate {row['id']} must contain system/user/assistant messages.")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            raise ValueError(f"Candidate {row['id']} has empty or non-text content.")
    return rows, manifest


def pad_examples(examples, pad_token_id):
    """Mask inserted padding by position; a genuine EOS may share the PAD token ID."""
    if not examples or any(len(x["input_ids"]) < 2 for x in examples):
        raise ValueError("Causal training requires at least two tokens per example.")
    width = max(len(x["input_ids"]) for x in examples)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for example in examples:
        ids = list(example["input_ids"])
        padding = width - len(ids)
        result["input_ids"].append(ids + [pad_token_id] * padding)
        result["attention_mask"].append([1] * len(ids) + [0] * padding)
        result["labels"].append(ids + [-100] * padding)
    return result


class FullTextCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch
        return {key: torch.tensor(value, dtype=torch.long) for key, value in pad_examples(examples, self.pad_token_id).items()}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--model-revision", default="main", help="Prefer an immutable HF commit; resolved commit is recorded.")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--optim", choices=["adamw_8bit", "adamw_torch"], default="adamw_8bit")
    p.add_argument("--load-in-4bit", action="store_true", help="Changed QLoRA protocol; historical run used false.")
    p.add_argument("--eos-token", default="<|im_end|>", help="Recorded November EOS; override only for a different protocol or smoke fixture.")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=-1, help="Positive value overrides epochs, useful for a smoke run.")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=3407)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.max_seq_length, args.batch_size, args.gradient_accumulation_steps, args.epochs, args.learning_rate) <= 0 or args.warmup_steps < 0 or args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("Invalid nonpositive training parameter.")
    if args.device == "cpu" and (args.dtype != "float32" or args.optim != "adamw_torch" or args.load_in_4bit):
        raise ValueError("CPU smoke runs require --dtype float32 --optim adamw_torch without 4-bit loading.")
    rows, data_manifest = read_candidates(args.data_dir)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty; existing runs are never overwritten.")
    if args.model.startswith(("/", "./", "../", "~")) and not Path(args.model).expanduser().is_dir():
        raise FileNotFoundError(f"Local base model directory does not exist: {args.model}")
    os.environ["WANDB_DISABLED"] = "true"
    try:
        import torch
        from numpy.random import default_rng
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies. Install the documented training requirements first.") from exc
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This reconstruction supports one process; distributed training changes the protocol.")
    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES, or explicitly use CPU smoke settings.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support bfloat16.")
    if args.optim == "adamw_8bit" or args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise SystemExit("bitsandbytes is required for the chosen optimizer/quantization.") from exc
    set_seed(args.seed)
    model_ref = str(Path(args.model).expanduser().resolve()) if Path(args.model).expanduser().is_dir() else args.model
    load = dict(revision=args.model_revision, local_files_only=args.local_files_only, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, **load)
    original_eos = tokenizer.eos_token
    if args.eos_token not in tokenizer.get_vocab():
        raise ValueError("Requested EOS is absent from the tokenizer; do not silently add new model tokens.")
    tokenizer.eos_token = args.eos_token
    tokenizer.pad_token = args.eos_token
    tokenizer.padding_side = "right"
    template_path = Path(__file__).resolve().parents[1] / "prompts" / "historical_chat_template.jinja"
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    examples, token_report = [], []
    for row in rows:
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
        kept = 2 <= len(ids) <= args.max_seq_length
        token_report.append({"id": row["id"], "tokens": len(ids), "retained": kept})
        if kept:
            examples.append({"input_ids": ids, "id": row["id"]})
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "tokenization_report.json", token_report)
    if not examples:
        raise ValueError("All training candidates were discarded by the token-length filter; see tokenization_report.json.")
    examples = [examples[int(i)] for i in default_rng(42).permutation(len(examples))]
    order = [x.pop("id") for x in examples]
    dtype = getattr(torch, args.dtype)
    extra = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype), "device_map": {"": 0}} if args.load_in_4bit else {}
    model = AutoModelForCausalLM.from_pretrained(model_ref, torch_dtype=dtype, attn_implementation="sdpa", **load, **extra)
    base_hash = config_fingerprint(model.config)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0, bias="none", target_modules=TARGETS, task_type="CAUSAL_LM", revision=resolved_revision or args.model_revision))
    model.peft_config["default"].base_model_name_or_path = model_ref
    train_args = TrainingArguments(output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, weight_decay=0.01, max_grad_norm=1.0,
        lr_scheduler_type="linear", optim=args.optim, seed=args.seed, data_seed=args.seed,
        fp16=args.dtype == "float16", bf16=args.dtype == "bfloat16", use_cpu=args.device == "cpu",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="no", report_to=[], push_to_hub=False,
        dataloader_num_workers=0, dataloader_pin_memory=args.device == "cuda", remove_unused_columns=False, label_names=["labels"])
    provenance = {"protocol": "transformers_peft_reconstruction_not_historical_weights", "status": "started",
        "base_model": model_ref, "requested_revision": args.model_revision, "resolved_revision": resolved_revision,
        "base_config_sha256": base_hash, "template_sha256": sha(template_path),
        "data_sha256": sha(Path(args.data_dir) / "train_candidates.jsonl"),
        "data_manifest_sha256": sha(Path(args.data_dir) / "manifest.json"), "data_manifest": data_manifest,
        "tokenizer_original_eos": original_eos, "training_eos": tokenizer.eos_token,
        "settings": vars(args), "libraries": versions(), "lora": {"r": 32, "alpha": 64, "target_modules": TARGETS},
        "loss": "full_text_causal_ce_padding_positions_only_masked", "packing": False,
        "dataset_shuffle": "numpy.default_rng(42).permutation", "training_order_ids": order,
        "candidate_count": len(rows), "retained_count": len(examples), "discarded_count": len(rows) - len(examples),
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(out / "run_manifest.json", provenance)
    trainer = Trainer(model=model, args=train_args, train_dataset=examples, data_collator=FullTextCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    result = trainer.train()
    adapter = out / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    trainer.save_state()
    provenance.update(status="completed", metrics=result.metrics, global_step=trainer.state.global_step)
    write_json(out / "run_manifest.json", provenance)
    write_json(adapter / "reproduction_manifest.json", provenance)
    print(f"Saved adapter: {adapter}; retained {len(examples)} / {len(rows)} candidates.")


if __name__ == "__main__":
    main()
