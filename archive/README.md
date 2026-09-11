# Historical experiment files

These files preserve the project's original experiment code and saved outputs.
Descriptive filenames make the archive easier to navigate. The rename changes no
notebook cells, outputs, CSV records, or file contents. Original upload names and
SHA-256 hashes remain in [source_manifest.json](../source_manifest.json), alongside
the current and previous repository paths.

For the executable reproduction workflow, use the commands in the
[project README](../README.md) and [running guide](../docs/RUNNING.md). Historical
notebooks retain their original environment assumptions and execution behavior.

## Training and exploratory notebooks

| File | Role | Original upload name |
|---|---|---|
| [qwen3_4b_sft_training_initial.ipynb](notebooks/qwen3_4b_sft_training_initial.ipynb) | Initial SFT training notebook | `Qwen3_(4B)_SFFT_code_0811.ipynb` |
| [qwen3_4b_sft_training_initial_duplicate.ipynb](notebooks/qwen3_4b_sft_training_initial_duplicate.ipynb) | Byte-identical duplicate supplied in a later upload | `Qwen3_(4B)_SFFT_code_0811(2).ipynb` |
| [qwen3_4b_sft_training_and_merge.ipynb](notebooks/qwen3_4b_sft_training_and_merge.ipynb) | SFT training, adapter merge, and saved experiment outputs | `notebook4d41ab3228 _train and reference.ipynb` |
| [qwen3_4b_sft_training_and_merge_executed.ipynb](notebooks/qwen3_4b_sft_training_and_merge_executed.ipynb) | Sequentially executed November 18, 2025 export of the training and merge pipeline | `qwen4b-train-code-100920859-279441290..ipynb` |
| [qwen3_4b_grpo_experiment.ipynb](notebooks/qwen3_4b_grpo_experiment.ipynb) | Separate exploratory GRPO experiment | `Qwen3_4B_GRPO_V1.ipynb` |

## Inference and evaluation notebooks

| File | Role | Original upload name |
|---|---|---|
| [qwen3_4b_code_execution_inference.ipynb](notebooks/qwen3_4b_code_execution_inference.ipynb) | Historical inference and Python execution workflow | `Qwen3_(4B)_SFTT_inference_exec_0816.ipynb` |
| [qwen3_4b_merged_inference_demo.ipynb](notebooks/qwen3_4b_merged_inference_demo.ipynb) | Merged-model inference demonstration | `notebook62f7bf13ad_reference .ipynb` |
| [qwen3_4b_base_reasoning_evaluation.ipynb](notebooks/qwen3_4b_base_reasoning_evaluation.ipynb) | Saved base-model responses and 12/20 recorded successes | `notebookd84987e9c1_base_cot.ipynb` |
| [qwen3_4b_merged_reasoning_evaluation.ipynb](notebooks/qwen3_4b_merged_reasoning_evaluation.ipynb) | Saved merged-model responses and 15/20 recorded successes | `notebookd84987e9c1_merged_cot.ipynb` |

The reasoning scores are historical numeric execution-output results. Five of the
20 questions overlap the reconstructed training set. See the
[reproduction status](../docs/REPRODUCTION_STATUS.md) for the experiment boundaries.

## Evaluation exports

| File | Verified contents | Original upload name |
|---|---|---|
| [qwen3_4b_base_reasoning_results.csv](evaluation/qwen3_4b_base_reasoning_results.csv) | Base-model evaluation export, 20 records | `evaluation_results_base_cot_v5(1).csv` |
| [qwen3_4b_base_reasoning_results_duplicate.csv](evaluation/qwen3_4b_base_reasoning_results_duplicate.csv) | Byte-identical copy of the base export | `evaluation_results_merged_cot_v5(1).csv` |

Both CSV files contain base-model outputs, including the original file labelled
`merged`. The merged-model evidence comes from its notebook. The
[CSV reconciliation](../reports/evaluation_reconciliation.md) records this finding.

The original data-generation scripts remain under `scripts/`. Raw datasets retain
their existing names under [`data/raw/`](../data/raw/) so prepared sample identities
and checksums remain stable. Older audit and cloud reports retain the filenames
used when they were created; the manifest resolves those names to the current files.
