"""Real CPU training/merge/inference integration check on a RANDOM tiny Qwen3.

No pretrained 4B model is used and no mathematical benchmark score is claimed.
Run after installing CPU ML requirements. Generated model code is never executed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    import torch
    from peft import PeftModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, AutoModelForCausalLM
    from fyp.data import file_hash, write_jsonl
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    vocab = {s: i for i, s in enumerate(["[UNK]", "<|im_end|>", "<|endoftext|>", "<start_working_out>",
                                         "1", "2", "3", "4", "+", "=", "solve", "return", "x", "y", "math", "python"])}
    with tempfile.TemporaryDirectory(prefix="fyp-tiny-") as temp:
        folder = Path(temp)
        base = folder / "base"
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                             eos_token="<|endoftext|>", pad_token="<|endoftext|>")
        config = Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=1024, eos_token_id=2, pad_token_id=2)
        model = Qwen3ForCausalLM(config)
        parameter_count = sum(p.numel() for p in model.parameters())
        model.save_pretrained(base)
        tokenizer.save_pretrained(base)
        data = folder / "prepared"
        data.mkdir()
        train_rows = []
        for n in range(4):
            train_rows.append({"id": f"smoke:{n}", "question_id": f"smoke:{n}", "problem": "1 + 1",
                               "category": "arithmetic_only", "source_file": "synthetic_smoke", "source_row": n + 1,
                               "messages": [{"role": "system", "content": "math"},
                                            {"role": "user", "content": "1 + 1"},
                                            {"role": "assistant", "content": "2"}]})
        write_jsonl(data / "train_candidates.jsonl", train_rows)
        write_jsonl(data / "eval_questions.jsonl", [{"id": "smoke-eval:1", "question_id": "smoke-eval:1", "problem": "2 + 2",
                       "category": "arithmetic_only", "answer": "4", "source_file": "synthetic_smoke", "source_row": 1}])
        (data / "manifest.json").write_text(json.dumps({"protocol": "random_tiny_integration_smoke",
            "prepared_hashes": {name: file_hash(data / name) for name in ("train_candidates.jsonl", "eval_questions.jsonl")}}))
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        commands = [
            [sys.executable, "-m", "fyp.train", "--data-dir", str(data), "--output-dir", str(folder / "trained"),
             "--model", str(base), "--model-revision", "smoke-local", "--device", "cpu", "--dtype", "float32",
             "--optim", "adamw_torch", "--max-steps", "2", "--warmup-steps", "0", "--learning-rate", "0.001",
             "--gradient-accumulation-steps", "1", "--max-seq-length", "128", "--local-files-only"],
            [sys.executable, "-m", "fyp.merge", "--adapter", str(folder / "trained/adapter"),
             "--output-dir", str(folder / "merged"), "--device", "cpu", "--dtype", "float32", "--local-files-only"],
            [sys.executable, "-m", "fyp.evaluate", "--data-dir", str(data), "--model", str(folder / "merged"),
             "--model-revision", "smoke-local", "--output", str(folder / "predictions.jsonl"), "--device", "cpu",
             "--max-total-tokens", "512", "--max-new-tokens", "8", "--limit", "1", "--local-files-only"],
        ]
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=120)
        clean_base = AutoModelForCausalLM.from_pretrained(base, local_files_only=True).eval()
        ids = torch.tensor([[4, 8, 4]])
        with torch.inference_mode():
            base_logits = clean_base(ids).logits
        adapted = PeftModel.from_pretrained(clean_base, folder / "trained/adapter", local_files_only=True).eval()
        merged = AutoModelForCausalLM.from_pretrained(folder / "merged", local_files_only=True).eval()
        with torch.inference_mode():
            adapted_logits = adapted(ids).logits
            merged_logits = merged(ids).logits
        delta = float((adapted_logits - base_logits).abs().max())
        merge_error = float((adapted_logits - merged_logits).abs().max())
        assert delta > 1e-6, "Training did not measurably change logits"
        torch.testing.assert_close(adapted_logits, merged_logits, rtol=1e-4, atol=1e-6)
        summary = json.loads((folder / "predictions.jsonl.summary.json").read_text())
        assert summary["completed"] and summary["total"] == 1
        assert summary["ungraded"] == 1 and summary["accuracy"] is None
        run = json.loads((folder / "trained/run_manifest.json").read_text())
        assert run["global_step"] == 2 and run["retained_count"] == 4
        report = {"test": "real_cpu_tiny_random_qwen3_lora_merge_inference", "passed": True,
                  "random_base_parameters": parameter_count, "training_steps": 2, "training_candidates": 4,
                  "retained_samples": 4, "maximum_logit_change_after_training": delta,
                  "maximum_adapter_vs_merged_logit_error": merge_error,
                  "prediction_rows": 1, "predictions_graded": 0, "benchmark_accuracy": None,
                  "libraries": run["libraries"], "original_4b_model_used": False,
                  "generated_python_executed": False, "docker_tested": False}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
