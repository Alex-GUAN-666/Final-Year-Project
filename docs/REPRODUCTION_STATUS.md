# Reproduction status and experiment boundaries

Updated 2026-09-11 with additional historical notebooks. This replaces the first-upload status preserved in `v1/`. The following cloud links describe the earlier verification snapshot, not validation of the later additions: [Verified run](https://github.com/Alex-GUAN-666/Final-Year-Project/actions/runs/34580694788) · [Tested commit](https://github.com/Alex-GUAN-666/Final-Year-Project/commit/cfac02190ff74c00c8facdb81b863b36c5f1981c).

## Established evidence

The earlier 20-file recovery was inventoried and read: the 42-page final PDF, the earlier DOCX rendered to 22 pages, all code/markdown cells and saved text outputs in nine notebook files, three Python scripts, and six data/result files containing 1,674 records when duplicate files are counted. Original source hashes are retained; that package contained 11 declared sanitized derivatives and 7 byte-identical source files. Two exact duplicate pairs are documented. See PUBLICATION_NOTES.md. Repeated progress output was normalized in reading copies. Reading and parsing every record does not mean every generated solution has been mathematically certified.

The later `FYP.zip` contains 25 notebooks: eight match previously received files, 16 are additional FYP versions, and one belongs to an unrelated Jigsaw competition. All 16 additional FYP versions were reviewed; six key records were selected for this repository, while the other versions remain in the original uploaded archive. The repository now contains 15 notebooks and 24 bundled source/public-derivative files. `source_manifest.json` lists 26 sources, including two unbundled papers; the public derivatives comprise 15 notebooks and two scripts. These records strengthen provenance without adding independent test questions or recovering model weights. See [RECOVERED_NOTEBOOKS.md](RECOVERED_NOTEBOOKS.md) for the version index, duplicate-code groups and exclusions.

The November report is the main SFT project. June GRPO material is a separate exploratory branch. The previously recovered November 18 Papermill notebook has a successful sequential run, 452 retained training examples, 114 optimizer steps and a successful merge. It provides better historical execution evidence, but contains no recoverable adapter or merged weight files.

The recovered 1,125-row `results_v3.jsonl` has 251 eligible solution/code records; all eligible records are classified `COMPUTATIONAL`. The training outcome is therefore best described as arithmetic and code-assisted mathematical reasoning, rather than a verified proof model. The two CSVs named base/merged are byte-identical and match the base notebook.

With the exact saved prompts/template and fixed official public tokenizer, the filter reproduces 143 distilled examples plus 309 arithmetic examples. The original Kaggle tokenizer snapshot is absent. The reconstructed final set overlaps five of the original twenty reasoning evaluation questions, rows **5, 12, 13, 14, 20**. This is a reconstruction-based leakage finding, not authentication of the original weight/tokenizer snapshot. All twenty questions occur in the pre-token-filter eligible pool.

## Historical settings versus new implementation

| Setting | Historical saved notebook | Current reconstruction |
|---|---|---|
| Model | Kaggle-local Qwen3 4B Base, original weight revision unknown | Official Qwen3-4B-Base, explicit fixed revision |
| Data | 309 arithmetic + 143 CoT/code after token filtering | Default 309 + 63 after excluding all evaluation questions and deduplicating |
| Precision | float16, `load_in_4bit=False` | Same intended GPU default; optional 4-bit flag is a different protocol |
| LoRA | Rank 32, alpha 64, q/k/v/o/gate/up/down projections | Same seven targets/rank/alpha |
| Effective batch | 1 × accumulation 8, one process | Same default; refuse implicit distributed training |
| Optimizer/LR | adamw_8bit, `3e-5`, warmup 10 | Same intended GPU defaults; tiny CPU smoke explicitly uses adamw_torch |
| Epochs/steps | 2 epochs, 114 saved steps on 452 examples | 2 epochs on a smaller new split; do not force 114 steps |
| Trainer | Historical Unsloth/TRL stack, not fully locked | Transformers 4.53.2 / PEFT 0.16.0, CPU-verified |
| Length handling | Pre-filter at 2,048; SFTConfig does not explicitly set trainer max_length | Explicit full sequence <=2,048, no later truncation |
| Loss | Historical trainer behavior incompletely recoverable | Full-text causal loss; only padding positions masked |
| Evaluation | Main reasoning records use the first 20 questions with system prompts; separate early runs execute a prompt reset | Fixed 100 arithmetic + 87 reasoning questions, same prompts and decoder settings across models |
| Code scoring | Unrestricted historical exec; numeric tolerance rel=1e-3, abs=0 | Restricted Docker execution, same numeric tolerance, defined failure handling |

Paper statements about 4-bit/QLoRA, batch 64, LR `2e-4`, and `1e-6` evaluation tolerance do not all match saved executable settings. The reconstruction records these differences; it does not change evidence to match the paper.

## Results boundaries

- Arithmetic 14/20 → 17/20 remains a **paper-reported, unverified comparison**. Four recovered runs contain identical 17/20 predictions on the first 20 arithmetic CSV rows, but select `qwen-3/transformers/4b/1`, not the project merged model or `4b-base/1`. These logs do not authenticate the original paired arithmetic comparison or its subset.
- Reasoning 12/20 → 15/20 is supported by saved paired notebook logs and a cloud replay matching all 40 individual correctness flags. Recovered sequential runs support the same scores; the new merged export omits full answers, so the earlier merged notebook is retained. Early 0/20 and 2/20 runs cleared the system prompts and represent different settings. The result CSVs add base stdout, not new merged predictions. Missing historical merged stdout remains null; newly replayed stdout is recorded separately and never substituted into the original evidence.
- The thesis reports +15 percentage-point differences on small samples; the arithmetic difference is not confirmed by the recovered logs. In addition, the reasoning comparison has reconstructed train/test overlap, and its metric is numeric code-output accuracy, not proof validity.
- The default new split has zero normalized question overlap, 372 token-retained training examples and 187 evaluation questions. Its references still have residual quality/provenance limitations. No new 4B accuracy result exists yet.

The new notebook metadata identifies a merged resource named `qwen3-4b-sfft-merged/transformers/default/1` (`modelId=508779`). This is a recovery lead, not a verified download or proof of identity with the older `lora-2-3` resource; neither checkpoint nor LoRA weights are present.

## Recorded verification snapshot

- Hash verification for all 18 bundled source/public-derivative files, full notebook/data parsing, static data audits and paired historical-result recovery.
- Exact tokenizer filtering with a downloaded official tokenizer/config snapshot; both historical and new candidate counts were checked. No 4B weights were downloaded for this step.
- Preparation and 32 passing unit tests covering overlap removal, label masking, scoring failure states, container cleanup, provenance and comparison rules; verified on the cloud runner.
- Two real optimizer steps on a random tiny Qwen3, LoRA saving, merging, reloading and inference. Adapter and merged logits agreed within numerical tolerance; see `reports/tiny_model_smoke.json`.
- Cloud repetition of real tiny-Qwen3 CPU training/merge/inference, plus eight real Docker integration checks for execution, failure handling, deferred scoring and paired comparison.
- Cloud replay of 40 saved historical responses, including 36 extractable programs and four no-code responses. All rows were graded; base 12/20 and merged 15/20 matched every saved correctness flag, with no comparable stdout mismatches and replay exit status 0. Historical inputs were unchanged.

All three GitHub workflow jobs passed for the linked historical snapshot. The additional notebooks were reviewed statically; this snapshot does not cover them or replay their arithmetic responses. [Cloud validation](CLOUD_VALIDATION.md) links the tested source, run and [retained verification record](../reports/cloud/2026-09-11/verification.json), and explains the two earlier failed runs and their fixes. The official-tokenizer count checks above were performed locally; the cloud workflow verifies prepared-data reconstruction without downloading that tokenizer.

Not executed: full Qwen3-4B training/inference, original merged-model re-evaluation, GPU optimizer validation and optional GPU quantization. Current status is recorded in `reports/validation_status.json`. The absence of old weights does not prevent a new run from the recovered data; it prevents exact re-evaluation of the historical checkpoint. Replaying saved programs does not recover those weights or generate a new model benchmark.
