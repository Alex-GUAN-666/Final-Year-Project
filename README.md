# Final Year Project

[![Validate reconstruction](https://github.com/Alex-GUAN-666/Final-Year-Project/actions/workflows/validate.yml/badge.svg)](https://github.com/Alex-GUAN-666/Final-Year-Project/actions/workflows/validate.yml)

**Qwen3-4B mathematical reasoning through multi-teacher distillation and supervised fine-tuning**

A reconstruction of **Yuzhen Guan and Weizhou Lu’s** undergraduate project, *Enhancing Compact Language Models: Multi-Teacher Distillation and Supervised Fine-Tuning for Research and Application* (November 2025).

**Validation snapshot, 2026-09-11:** recovered training data, fixed question-disjoint data preparation, and runnable LoRA training → merging → inference → isolated scoring → paired comparison. All **27 unit tests passed**, and real CPU training/merge/inference passed with a randomly initialized tiny Qwen3. **Full 4B GPU training, Docker execution, and a new accuracy comparison have not been run.** Original fine-tuned weights remain unavailable. The badge tracks GitHub checks; the full 4B experiment is outside their scope. The first hosted CI run is pending.

Method: [METHODOLOGY.md](docs/METHODOLOGY.md).

中文说明：[完整核查与复现说明](docs/REVIEW_ZH.md)。Environment and command details: [RUNNING.md](docs/RUNNING.md).

## Historical evidence

| Task | Base | Historical merged | What the evidence supports |
|---|---:|---:|---|
| Arithmetic | 14/20 (70%) | 17/20 (85%) | Reported in the November thesis; paired prediction files and exact question IDs were not recovered |
| Mathematical reasoning | 12/20 (60%) | 15/20 (75%) | Saved paired notebook outputs; **5 of these 20 questions overlap the reconstructed training set** |

The increases are 15 **percentage points**, not a new result. The reasoning score measures numeric output from generated Python; it does not establish proof validity or independent held-out generalization. The two newly recovered CSVs named `base` and `merged` are byte-identical and both match the base notebook. Merged results come from the original merged notebook, not from the mislabeled CSV.

The recovered 1,125-row distilled pool reproduces 251 eligible candidates. Using the saved prompts/template and the fixed official tokenizer revision below reproduces 143 retained reasoning examples plus 309 arithmetic examples: **452 total**, matching saved training counts. This authenticates the count reconstruction, not the absent original Kaggle tokenizer or weights. See [data evidence](reports/RECOVERED_DATA_AUDIT.md) and [evaluation reconciliation](reports/evaluation_reconciliation.md).

## New experiment

The default `question_disjoint_v1` protocol excludes all source evaluation questions from training, deduplicates normalized questions, and quarantines two known inconsistent references. It has:

| Stage | Arithmetic | Reasoning/code | Total |
|---|---:|---:|---:|
| Training candidates | 309 | 127 | 436 |
| Retained at 2,048 tokens with fixed tokenizer/template | 309 | 63 | 372 |
| Evaluation questions | 100 | 87 | 187 |

Normalized question overlap is zero. This is a new split over recovered material, not a newly collected independent benchmark. String checks do not certify absence of semantic duplicates or pretraining exposure. The 87 references have not all received independent mathematical proof verification. New training is required before evaluating the fine-tuned model on this split.

## Quick start: no GPU or model downloads

Clone the repository, then run commands from its root. Python 3.12 was used for verification. The `.gitattributes` file preserves original bytes on Windows as well as Linux.

```bash
git clone https://github.com/Alex-GUAN-666/Final-Year-Project.git
cd Final-Year-Project
```

One-command reviewer check (standard library only):

```bash
python scripts/verify_reconstruction.py --report outputs/reviewer_check.json
```

This verifies source hashes, saved historical scores, question overlap, split regeneration and unit tests. It downloads no models and generates no new accuracy. Individual checks:

```bash
python scripts/audit_artifacts.py --output reports/artifact_audit_v2.json
python scripts/audit_data.py
python scripts/audit_recovered_distillation.py
python scripts/reconcile_evaluation_files.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

Audits read all bundled source files without executing generated Python. `data/prepared/question_disjoint_v1/` already contains the fixed split. To regenerate it without overwriting the supplied copy:

```bash
python -m fyp.prepare --output-dir outputs/prepared-check
```

The [validation workflow](.github/workflows/validate.yml) has three jobs: source/data audits and unit tests with byte-for-byte split regeneration; real tiny-Qwen3 CPU training, merging and inference; and Docker execution, deferred scoring and paired comparison using explicitly synthetic programs. No job downloads pretrained model weights or uses custom secrets. Synthetic scores test software behavior and are not model accuracy results. Job logs and integration reports are available on the [Actions page](https://github.com/Alex-GUAN-666/Final-Year-Project/actions).

## Real model workflow

Install the documented CPU or GPU environment in [RUNNING.md](docs/RUNNING.md). The selected model is the pretrained [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base), pinned to `906bfd4b4dc7f14ee4320094d8b41684abff8539`. Preserve that revision when comparing base and merged models.

```bash
# No model weights needed for this token-filter check.
python -m fyp.tokenize --data-dir data/prepared/question_disjoint_v1 \
  --output outputs/token_filter.json

# Expose one GPU. This full 4B run has not been executed in this review.
CUDA_VISIBLE_DEVICES=0 python -m fyp.train \
  --data-dir data/prepared/question_disjoint_v1 \
  --model Qwen/Qwen3-4B-Base \
  --model-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --output-dir outputs/sft

python -m fyp.merge --adapter outputs/sft/adapter \
  --output-dir outputs/merged --device cpu --dtype float16

python -m fyp.evaluate --data-dir data/prepared/question_disjoint_v1 \
  --model Qwen/Qwen3-4B-Base \
  --model-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --output outputs/base_predictions.jsonl
python -m fyp.evaluate --data-dir data/prepared/question_disjoint_v1 \
  --model outputs/merged --local-files-only \
  --output outputs/merged_predictions.jsonl
```

Inference defaults to no code execution, so accuracy is `null` until scoring. On Linux or WSL2 with Docker, build the proposed evaluator image, score both completed prediction files, then compare them:

```bash
docker build -f docker/Dockerfile.eval -t fyp-python-math .
python -m fyp.score --predictions outputs/base_predictions.jsonl \
  --output outputs/base_scored.jsonl --docker-image fyp-python-math
python -m fyp.score --predictions outputs/merged_predictions.jsonl \
  --output outputs/merged_scored.jsonl --docker-image fyp-python-math
python -m fyp.compare --base outputs/base_scored.jsonl \
  --merged outputs/merged_scored.jsonl --output outputs/comparison.json
```

Copy prediction `.jsonl`, `.jsonl.meta.json`, and `.jsonl.summary.json` together when moving from a GPU notebook to the scoring machine. Scoring requires complete files. The comparison refuses mismatched evaluation protocols or ungraded rows. Docker scoring is implemented and unit-tested through mocks; real Docker execution is not yet validated.

For a real, small CPU integration check with no pretrained weights:

```bash
python tests/smoke_tiny_model.py --report outputs/tiny_model_smoke.json
```

The saved [smoke report](reports/tiny_model_smoke.json) records two actual optimizer steps, changed model outputs, and agreement between adapter and merged-model logits. It is not a mathematical benchmark.

## Replay saved historical programs

On the same Linux/WSL2 Docker host, run:

```bash
python scripts/replay_historical_programs.py --check-inputs
python scripts/replay_historical_programs.py --docker-image fyp-python-math \
  --output outputs/historical-replay.json
```

All 40 saved responses are verified against the archived notebooks and references; 36 contain extractable programs and four have no code. The report preserves the saved 12/20 and 15/20 and records replayed results and disagreements separately. Replaying saved responses does not regenerate them with the original model. A mismatch is reported as a nonzero exit rather than rewriting the historical evidence.

## Reconstruction choices

Historical code uses float16, **no 4-bit loading**, LoRA rank 32/alpha 64 on seven projection layers, batch 1 with gradient accumulation 8, learning rate `3e-5`, and two epochs. The paper differs on several settings. The new implementation uses [Transformers Trainer 4.53.2](https://huggingface.co/docs/transformers/v4.53.2/en/main_classes/trainer) and PEFT rather than the historical Unsloth runtime. It explicitly uses a 2,048-token full-text causal loss and retains real EOS labels when padding uses the same token. Historical trainer truncation was not fully specified. These choices are recorded as reconstruction differences.

The exact historical prompts and template are in `prompts/`. Both new model evaluations receive the same template, EOS override, prompt, greedy decoding, and token budget. Reference answers never enter the inference prompt. Altering prompts, quantization, labels, sampling, or splitting creates a different experiment; record it separately. [Detailed boundaries](docs/REPRODUCTION_STATUS.md).

## Repository contents and credits

- `archive/`: nine historical notebook files, three historical scripts, and two result CSVs. Public notebooks omit personal local paths and rich UI caches; plain-text research evidence is retained. Two scripts use generic paths. One notebook and the two CSVs include exact duplicates.
- `data/raw/`: both original 100-row input CSVs, the 309-row arithmetic JSONL, and the recovered 1,125-row distilled JSONL.
- `fyp/`, `tests/`, `docker/`: new implementation, meaningful tests, and proposed isolated evaluator image.
- `reports/`, `docs/`: historical evidence, audits, new-protocol manifests and verification status. `v1/` preserves superseded first-upload findings.
- `source_manifest.json`: original SHA-256 hashes for all 20 supplied files plus separate publication hashes for 11 sanitized derivatives. Seven bundled files preserve their original bytes; two papers were reviewed separately. See [publication notes](docs/PUBLICATION_NOTES.md).

The papers contain student identifiers, signatures and document metadata; their full unredacted files are not included in this repository. Archived notebooks are evidence and may depend on stale interactive state or call unrestricted `exec`; use the new entrypoints for reproduction. New scoring executes generated programs only in restricted Docker containers, with no host fallback.

Original project credit belongs to **Yuzhen Guan and Weizhou Lu**. Preserve existing Unsloth and other upstream notices. Model and third-party material retain their own terms; no blanket license is assigned here. Upstream attribution for some recovered mathematical questions is unresolved. New code and documentation were prepared with AI assistance in September 2026.
