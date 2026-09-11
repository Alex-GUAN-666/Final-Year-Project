# Mathematical Reasoning with Qwen3-4B

Multi-teacher distillation and LoRA fine-tuning for arithmetic and mathematical problem solving.

[![Tests](https://github.com/Alex-GUAN-666/Final-Year-Project/actions/workflows/validate.yml/badge.svg)](https://github.com/Alex-GUAN-666/Final-Year-Project/actions/workflows/validate.yml)

Undergraduate final year project by **Yuzhen Guan and Weizhou Lu** · November 2025 report.

This project investigates whether a compact language model can improve its numerical reasoning by learning from mathematical explanations and executable Python solutions. We fine-tuned Qwen3-4B-Base with LoRA, then compared the base and fine-tuned models by executing their generated programs and checking the numerical answers.

The repository contains the original experiment records and a maintained Python implementation of data preparation, training, model merging, inference and evaluation. The implementation was reconstructed from the saved materials in September 2026.

[Methodology](docs/METHODOLOGY.md) · [Running the project](docs/RUNNING.md) · [Validation results](docs/CLOUD_VALIDATION.md) · [中文说明](docs/REVIEW_ZH.md)

## Approach

1. **Prepare training examples.** Pair mathematical questions and existing solutions with Python programs produced by a code teacher. Add a separate synthetic arithmetic dataset for shorter, code-only tasks.
2. **Select and format the data.** Use the recorded answer-check flags to select eligible distilled examples, then apply the saved chat template and a 2,048-token length limit. Reconstructing the historical selection yields 452 training examples: 309 arithmetic and 143 reasoning/code examples.
3. **Fine-tune the student.** Train LoRA adapters on Qwen3-4B-Base. The saved main SFT configuration uses rank 32, alpha 64, a learning rate of `3e-5` and two epochs. The current implementation uses Transformers and PEFT; the original notebooks used Unsloth.
4. **Evaluate generated programs.** Give both models the same questions and task prompts, then inspect the first fenced code block. Accept it if labeled Python, `py`, or left unlabeled, and compare the program's numerical output with the reference answer. The maintained evaluator runs programs in restricted Docker containers.

The recovered generator uses an existing mathematical solution as input to the code teacher. Training distills these explanation/code sequences through supervised fine-tuning; it does not use teacher logits. [METHODOLOGY.md](docs/METHODOLOGY.md) documents the teacher inputs, selection rules, loss function and differences between the original and maintained implementations.

## Results

The original project reported the following results:

| Task | Base model | Fine-tuned model | Available evidence |
|---|---:|---:|---|
| Arithmetic | 14/20 (70%) | 17/20 (85%) | Thesis results; the original paired predictions and exact question subset were not recovered |
| Mathematical reasoning | 12/20 (60%) | 15/20 (75%) | Saved notebook responses; program replay reproduces every recorded correctness flag |

The reasoning evaluation contains **five questions that overlap the reconstructed training set**, so the 15-percentage-point increase should not be treated as an independent held-out result. The metric measures numerical program output, not the validity of a mathematical proof.

The original fine-tuned weights are unavailable. The cloud checks replay saved responses and test the software with a small random model; they do not regenerate the historical responses with Qwen3-4B. Full 4B training and inference have not been rerun for this repository.

## Data for a new experiment

The default `question_disjoint_v1` split excludes all normalized source evaluation questions from training, removes repeated questions and quarantines two inconsistent references.

| Stage | Arithmetic | Reasoning/code | Total |
|---|---:|---:|---:|
| Training candidates | 309 | 127 | 436 |
| Training examples after token filtering | 309 | 63 | 372 |
| Evaluation questions | 100 | 87 | 187 |

There is no normalized question overlap between training and evaluation. This is a new split of the recovered data, with no completed 4B benchmark result yet. Semantic duplication, pretraining exposure and remaining reference-quality issues are discussed in the [data audit](docs/DATA_AUDIT.md).

## Run the project

Start with the repository checks, which require Python 3.12 and no model downloads:

```bash
git clone https://github.com/Alex-GUAN-666/Final-Year-Project.git
cd Final-Year-Project
python scripts/verify_reconstruction.py --report outputs/reviewer_check.json
```

This command verifies source hashes, reconstructs the prepared split, checks the saved evaluation records and runs the unit tests.

For model training and evaluation, follow [RUNNING.md](docs/RUNNING.md). It includes the pinned model revision, dependency installation, LoRA training, merging, inference and Docker scoring commands. Each experiment records its data hashes and configuration alongside the outputs.

GitHub Actions also runs actual training, merging and inference with a small randomly initialized Qwen3, eight Docker integration checks, and a replay of all 40 saved reasoning responses. The [validation report](docs/CLOUD_VALIDATION.md) links the tested revision, execution results and recorded limitations. The badge above shows the latest run.

## Repository guide

| Location | Contents |
|---|---|
| [`fyp/`](fyp/) | Data preparation, tokenization, training, merging, inference, scoring and comparison |
| [`data/`](data/) | Source datasets and the fixed training/evaluation split |
| [`archive/`](archive/) | Original notebooks, scripts and result exports, with a filename and provenance index |
| [`prompts/`](prompts/) | Task instructions and the saved chat template |
| [`tests/`](tests/) | Unit tests and model/Docker integration checks |
| [`docs/`](docs/) | Methodology, installation, experiment history and validation |
| [`reports/`](reports/) | Data audits, historical results and verification records |

Archived notebooks preserve research evidence and may rely on their original interactive environments. Use the documented `fyp` commands to run the maintained implementation. Original filenames and file hashes remain recorded in [`source_manifest.json`](source_manifest.json).

## Credits

Original project: **Yuzhen Guan and Weizhou Lu**, *Enhancing Compact Language Models: Multi-Teacher Distillation and Supervised Fine-Tuning for Research and Application*.

Model and third-party materials retain their own terms. Existing Unsloth notices are preserved. Source handling, repository maintenance and attribution details are documented in [PUBLICATION_NOTES.md](docs/PUBLICATION_NOTES.md).
