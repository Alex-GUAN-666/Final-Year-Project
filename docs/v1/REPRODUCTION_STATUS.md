> Historical first-upload review. Superseded by ../REVIEW_ZH.md and ../REPRODUCTION_STATUS.md after recovery of results_v3.jsonl.

# Reproduction status and experiment boundaries

Review date: 2026-09-11. Page numbers below are physical PDF page numbers, counting the cover.

## What has been established

- All 15 supplied files were read by the review team: a 42-page PDF, a 466-body-block DOCX rendered to 22 pages, 203 notebook cells across 7 notebooks and their saved text outputs, all 3 Python scripts, and all 509 records across 3 data files. Figures and tables in the papers were inspected. Repeated progress-bar output was normalized for readability while exact originals were preserved.
- The November PDF describes the SFT project as its main experiment. The June DOCX and GRPO notebook represent a separate exploratory branch.
- The main SFT notebook records training and successful merging. These are historical saved outputs, not execution performed in this review.
- The saved paired CoT notebooks contain 12/20 and 15/20 evaluation summaries. Their metric is numeric agreement of executed code with a reference answer using `math.isclose(..., rel_tol=1e-3, abs_tol=0)`; it does not validate the reasoning text as a proof. The paper instead describes a `1e-6` tolerance, another protocol discrepancy.
- The CPU audit scripts were actually executed on the supplied files in this review. No CUDA GPU, original adapter, merged weights, or full distilled dataset was available. No new 4B training or inference was performed.

## Historical training settings to recover

| Item | Main notebook evidence | Paper / reproduction implication |
|---|---|---|
| Base model | Local Kaggle Qwen3 4B Base path | Exact snapshot/config/tokenizer hash missing |
| Distilled pool | 1,125 source records, 251 passing correctness/content filter, 143 within 2,048 tokens | `results_v3.jsonl` missing; teacher provenance cannot yet be checked |
| Arithmetic pool | All 309 supplied JSONL records retained | Code field used for training; text-answer issues have different impact |
| Training total | 452 | Not the evaluation denominator |
| Precision | `load_in_4bit=False`, float16 | Paper's repeated 4-bit/QLoRA description conflicts; PDF p21 screenshot also says False |
| LoRA | rank 32, alpha 64 | q/k/v/o and gate/up/down projection modules |
| Batch | 1 × 8 accumulation × 1 GPU = 8 | Paper table says 64 |
| Learning rate | `3e-5` | Paper table says `2e-4` |
| Epochs / steps | 2 / 114 | Saved main-run training output |
| Optimizer | `adamw_8bit` | Optimizer-state precision is distinct from quantization of base weights |
| Seeds | LoRA/trainer 3407; dataset shuffle 42 | Reuse and log all random sources |
| Merge | `merge_and_unload()` and save to `Qwen3-4B-SFFT-merged` | Output directory itself not supplied |

The code evidence takes priority when reconstructing an executed run. Paper-only settings should be recorded as differences rather than substituted silently.

## Current blockers

| Required item | Why it matters | Acceptable recovery |
|---|---|---|
| `results_v3.jsonl` | Contains the 143 retained reasoning examples and teacher provenance | Original file, including all 1,125 rows if possible |
| Merged model or matching adapter | Required to load the historical fine-tuned state | Model folder/link with weights, config, tokenizer; adapter_config plus adapter weights if using LoRA |
| Arithmetic paired predictions | 70%→85% is currently supported only by the paper | Original result CSVs/notebook outputs and exact evaluated question IDs/order |
| Model and software provenance | Prevent accidentally mixing versions/runs | Base snapshot, tokenizer/template, successful environment export or preserved notebook image |

An adapter can enable re-evaluation without redoing SFT. Full historical retraining still requires the missing distilled dataset. A new dataset, cleaned references, a new prompt or a different quantization scheme creates a new experiment and must be labeled accordingly.

## Corrections needed before a clean GPU workflow

1. Resolve all Kaggle/local paths through explicit configuration; fail clearly when required data/weights are missing.
2. Consolidate environment installation. Remove broken shell quoting and conflicting repeated installs; freeze versions only after a successful GPU smoke run.
3. Freeze the historical prompt/template selected from the saved run. Remove or isolate unexecuted prompt-reset cells.
4. Rebuild train/eval sample identities, check train/test overlap including the missing distilled data, and deduplicate by problem family where appropriate.
5. Record raw output, parsed code, execution status, numeric correctness, and reasoning/proof assessment separately. A constant return with a matching value is not proof verification.
6. Isolate model-generated code with no network/secrets and explicit CPU/time/memory/output limits. A timeout or AST parse alone is not a security boundary.
7. Load baseline and merged model under identical prompting, decoding, precision and evaluation settings. Preserve row-level records and full denominators, not just aggregate percentages.
8. Run a small end-to-end GPU smoke test, then the frozen historical evaluation; record any new protocol separately.

## Publication boundary

The connected GitHub account is `Alex-GUAN-666`; no FYP repository was visible in the accessible repository list. This draft has not been pushed. It contains no blanket claim that the archived notebooks run from a clean kernel. Full papers containing student IDs/signatures remain separate. Retain both project authors and original Unsloth attribution when preparing a public release.
