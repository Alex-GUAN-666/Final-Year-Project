> **2026-09-11 supplemental review:** The distilled pool is now recovered. Public-tokenizer reconstruction reproduces 452 training examples and shows 5/20 historical reasoning evaluation questions overlap training. Older references below to missing data or unknown overlap are superseded by [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) and [the recovered-data audit](../reports/RECOVERED_DATA_AUDIT.md).

# SFT training, merge, and interactive execution audit

Reviewed 2026-09-11. This is a static audit of supplied historical notebooks, not a training rerun. Cell references are 1-based positions in the supplied file. Every code/Markdown source cell and every textual output in the three notebooks below was read, including package logs, HTML training tables, generated model text, and displayed execution results. Exact extracted notebook text and output dictionaries are retained in sibling `.full.txt` files; readable text/HTML in `.review.txt`. Original notebooks were not executed.

## Evidence hierarchy

The strongest supplied training/merge evidence is `notebook4d41ab3228 _train and reference.ipynb`, whose training and merge cells show a completed November 16, 2025 run on a Tesla T4. It starts from a **Qwen3-4B-Base checkpoint**, trains LoRA with **SFT**, and merges the adapter. The file contains no load of a prior GRPO adapter before this SFT step. The standalone GRPO notebook is therefore a separate experiment unless missing checkpoint provenance can establish a chain.

`Qwen3_(4B)_SFFT_code_0811.ipynb` documents an earlier successful August 16 SFT run on an RTX 4090 laptop GPU and saves an adapter named `sftt_save_lora_v3.8`. Its prose goal says it begins from a GRPO-finetuned model, but the implemented path instead points to a local `models--unsloth--Qwen3-4B-Base/snapshots` directory. The content of that missing local directory cannot be verified from a filename.

`Qwen3_(4B)_SFTT_inference_exec_0816.ipynb` is an interactive inference/execution demonstration, **not** valid evidence that the saved LoRA adapter was evaluated: its active code assigns `model = base_model`; loading the adapter is commented out. Its latest displayed result is for one matrix invertibility question, with no accuracy dataset or metric calculation.

## Recorded successful SFT configuration

| Item | August notebook | November training/merge notebook |
|---|---|---|
| Model load | Cell 8: local Base snapshot folder | Cell 8: `/kaggle/input/qwen-3/transformers/4b-base/1` |
| Context limit | 2048 | 2048 |
| Quantized load | `load_in_4bit=False` | `load_in_4bit=False` |
| Dtype | Auto; run reports bf16-capable RTX 4090 | `torch_dtype=torch.float16`; T4 lacks bf16 |
| Fast inference | `fast_inference=True` (vLLM) | `fast_inference=True` line commented out |
| LoRA rank / alpha | 32 / 64 | 32 / 64 |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Same |
| Checkpointing / LoRA seed | `unsloth` / 3407 | Same |
| Training rows | 452 | 452 |
| Per-device batch / accumulation | 1 / 8 | 1 / 8 |
| Epochs / optimizer steps | 2 / 114 | 2 / 114 |
| LR / scheduler / warmup | 3e-5 / linear / 10 steps | Same |
| Optimizer / weight decay / max grad norm | `adamw_8bit` / .01 / 1.0 | Same |
| Logging / trainer seed | Every 5 steps / 3407 | Same |
| Trainable parameters shown | 66,060,288 out of 4,088,528,384 (1.62%) | Same |
| Runtime shown | 6:11 | 18:39 |
| First / last displayed training loss | .9456 at step 5 / .3270 at 110 | .9850 at step 5 / .3494 at 110 |
| Saved result | Adapter and tokenizer | Merged model and tokenizer |

Neither notebook defines a held-out validation dataset, early stopping, or a train/test split. Neither explicitly configures response-only/assistant-only loss masking; training is handed a single rendered `text` column. Do not describe the above last displayed losses as held-out performance or final evaluation accuracy.

### Data preparation and missing dependency

August cells 12–14 and November cells 14–17 implement the same pipeline:

1. Load **1125** records from `results_v3.jsonl` (missing from the supplied files).
2. Keep rows only when `is_correct`, `generated_solution`, and `generated_python_code` are truthy: **251** CoT+Code rows.
3. Load **309** records from supplied `arithmetic_logic_data_revised_v2.jsonl`; take `problem` and `generated_python_code` with **no correctness filter** for this pool.
4. Render a system/user/assistant conversation. Arithmetic assistant target is one fenced code block. CoT+Code target is `generated_solution`, two newlines, and fenced `generated_python_code`.
5. Filter on full rendered token count `<=2048`: all **309** arithmetic rows remain, but CoT+Code becomes **143**. A total of **108/251** accepted distilled rows are lost at this stage.
6. Concatenate both pools, form a dataset, shuffle with seed **42**, render chat text with `add_generation_prompt=False`, and retain only `text`.

This is a **452-example supervised training dataset**, not 1125 verified training examples. Recovering `results_v3.jsonl` is necessary to reproduce the original 143 distilled targets or check overlap with evaluation questions. The complete verified teacher target includes reasoning text and generated code; reconstructing only problem/answer pairs would create a different training experiment.

### Exact prompt and template provenance

Exact static literal assignments were extracted without executing the notebook into `exact_prompt_literals.json`, keyed by notebook, 1-based cell, and variable name.

Two task prompts distinguish code-only arithmetic from reasoning plus code. They request a `solve()` function returning a numerical answer, followed by a call. The example uses `print(solve())`, although prose calls the return value the final expression. Several original typos (`caculation`, `caculate`) are preserved in exact extracts. The August source prompt example was later changed to `$Calculate#`, whereas the saved rendered training sample still contains the older `caculate` wording. Therefore its source text and saved example are from different editing/execution states.

November cell 11 sets a custom Jinja template. A system message is emitted as `system content + eos_token`; user content is emitted directly; assistant content is emitted directly and followed by EOS. **There are no explicit user/assistant role delimiters or separator between the user problem and assistant answer.** At inference it appends `<start_working_out>`. If no system message exists it injects a GRPO-style instruction using `<start_working_out>`, `<end_working_out>`, and `<SOLUTION>` tags. This is not Qwen's standard conversation template and must be preserved as a named historical setting, not silently substituted during reproduction.

August cell 15 shows `<|endoftext|>` as EOS; November cell 18 shows `<|im_end|>`. The local August tokenizer is absent and no cell explicitly defines its template. Consequently the precise August tokenizer/template configuration cannot be recovered solely by loading a fresh modern Base tokenizer.

August cell 16 and November cell 19 incorrectly call `apply_chat_template` with an already-rendered string rather than a list of messages. Their displayed result is only the fallback system instruction, not the real training example. These are misleading inspection cells; the actual training dataset was built correctly from a `messages` list earlier.

## Notebook-by-notebook complete source coverage

### `Qwen3_(4B)_SFFT_code_0811.ipynb` — 27 cells

| Cells | Content and observations |
|---|---|
| 1–3 | Installation headings and disabled installs. All `pip` commands are commented/embedded after `pass`; this does not provision dependencies in a fresh runtime. `os`, `sys`, `re`, and `requests` imported in cell 3. |
| 4–7 | Claimed GRPO→SFFT goal and configuration headings/imports. Implemented Base path does not substantiate claimed GRPO lineage. |
| 8 | Base load/LoRA config; complete package/GPU startup logs reviewed. Load succeeds and 36 QKV/O/MLP layers patched. WSL `pin_memory=False`; missing FlashInfer fallback; repeated TensorFlow CUDA plugin registration warnings, not an uncaught training failure. |
| 9–10 | System prompts described above. |
| 11–14 | Complete two-pool data prep, token filtering, shuffle, render; recorded counts above. |
| 15 | Full displayed distilled ratio example with reasoning and executable `11/26` code read; output prompt differs from current source. |
| 16 | Invalid string-as-conversation inspection described above. |
| 17–20 | SFT configuration and completed 114-step run. All 22 logged training loss points read. |
| 21–22 | `model.save_lora('/mnt/d/.../outputs/sftt_save_lora_v3.8')`; `tokenizer.save_pretrained` both print success. No merged checkpoint saved here. |
| 23 | Commented historical adapter save paths. |
| 24 | CUDA cache reset via numba and GC count 198. Destructive to the active CUDA context if run while later operations need the loaded model. |
| 25 | Commented `kill -9` recovery command. |
| 26 | Historical `nvidia-smi` from **August 13**, earlier than the August 16 training output; shows RTX 4090 16GB and no process. This is another stale display, not a post-training audit. |
| 27 | Empty cell. |

Recorded environment: Python metadata **3.10.12**; Unsloth **2025.6.6**, Transformers **4.52.4**, vLLM **0.8.5.post1**, Torch **2.6.0+cu124**, CUDA toolkit **12.4**, Triton **3.2.0**, Xformers **0.0.29.post3**. Exact `datasets`, `trl`, `peft`, `bitsandbytes` versions are not recorded by a lockfile.

### `Qwen3_(4B)_SFTT_inference_exec_0816.ipynb` — 19 cells

| Cells | Content and observations |
|---|---|
| 1–4 | Claimed adapter inference goal, imports/device setup, Base and v3.8 adapter path strings. |
| 5 | Actual Base load with `load_in_4bit=False`, `fast_inference=False`; PEFT creation block is a triple-quoted inactive string. `PeftModel.from_pretrained` is commented. Base load succeeds. |
| 6 | Active `model=base_model`, execution count 11. |
| 7–8 | Task prompts and heuristic routing. `calculate the` routes code-only first and replaces case-sensitive `Calculate` with `$Calculate#`; various proof/word-problem keywords and broad parentheses/LaTeX patterns route CoT+Code. This is a rule-based router, not a separately learned classifier. |
| 9 | All 30 hardcoded demonstration questions read: arithmetic/bitwise through algebra, probability, symbolic formulas, Lebesgue integrability, and matrix invertibility. There are no answer labels. |
| 10–11 | Latest selected question index 29: `X^T X` invertibility; 2048 generation tokens, sampling enabled, temperature .2, top_p .9, no explicit generation seed. Saved response took 28.447 s. Model explains full column rank and outputs a specific rank-deficient matrix test; no proof-quality score. |
| 12 | Markdown cell containing inactive extraction code. |
| 13 | Regex extracts fenced code. Saved output is stale XOR code (`0x1235 ^ 0xfb67`) from question 0, not the latest question 29. Execution count 13 vs later 92–94 confirms mixed interactive state. |
| 14 | `run_code` uses AST transformation, unrestricted in-process `exec`, captures stdout; then **evaluates the final print expression again**, outside the redirected stdout. This explains the extra printed `False` in output. Empty output is accepted as success; no timeout, process isolation, or execution permission boundary. The docstring saying “safe scope” is inaccurate. |
| 15 | Re-extracts latest response code; executes it and annotates the response with `False`, `Error Occurred: False`. This establishes only successful execution of a demonstration matrix, not correctness of arbitrary mathematical proof. |
| 16–18 | CUDA reset/GC, commented kill, stale earlier `nvidia-smi`; metadata and outputs disagree with a clean top-to-bottom session. |
| 19 | Empty cell. |

Model output also gives an incomplete justification for the rank identity: rank of a product being *at most* each factor does not establish `rank(X^T X)=rank(X)` on its own; the statement holds over real matrices via the null-space identity. Preserve this as model output, not endorsed proof quality.

### `notebook4d41ab3228 _train and reference.ipynb` — 36 cells

| Cells | Content and observations |
|---|---|
| 1–3 | Unsloth tutorial banner/news and explicit **LGPL-3.0 notebook license** reference. Upstream attribution must remain in any redistributed derived notebook. |
| 4–7 | Install/config heading; UV installs. Cell 5 contains an unclosed shell quote in the `datasets` install line; `%%capture` hides its output. Cell 7 repeats the corrected command. First install upgrades unpinned Unsloth, so it cannot promise the historical run environment. |
| 8 | Base model/LoRA initialization; active config above. Recorded run sees two T4s but actual trainer later reports one GPU used. |
| 9–11 | Task prompts and explicit custom Jinja template. |
| 12–17 | Imports/data prep/template display/filtering/render; counts above. File-not-found branch refers to `sys.exit` before `sys` is imported in cell 26, so a clean run missing data may raise `NameError` instead of its intended error exit. |
| 18 | Full arithmetic rendered training example read: `$Calculate# the product of 58934 and 16146967720371580991`, target multiplication code. Uses `<|im_end|>` EOS. |
| 19 | Invalid string-as-conversation display described above. |
| 20–23 | Trainer and completed 114-step run. Every table entry read. |
| 24–25 | Merge with `model.merge_and_unload()` into `merged_model`; `merged_model.save_pretrained(..., safe_serialization=True)` and tokenizer save print success. Location `/kaggle/working/Qwen3-4B-SFFT-merged`. Actual weight/tokenizer files are absent from supplied material. |
| 26–29 | Imports/device, same heuristic router, same 30-question bank. |
| 30 | Source selects `question_set[12]` (square-root equation); saved output instead shows rowing-rate question, which is index **17** in the saved list. Clear source/output drift. |
| 31 | Generation max_new_tokens **1024**, temperature .2, top_p .9, sampling enabled. Saved output is rowing-rate reasoning ending in `c=2`, then truncated mid-code at token limit. 56.353 s shown. Inference still calls `model.generate` rather than explicitly rebinding `model=merged_model`; merged-model output provenance is not explicitly demonstrated by reloading saved files. |
| 32 | Stale extracted response: **five identical** `sqrt(73)/2` code blocks from the earlier square-root question. |
| 33 | Same unrestricted/double-executing helper described above. |
| 34 | Stale five-block execution, each produces `4.272001872658765`; displayed response contains repeated Chinese self-ratings `100%` and ends in another incomplete code fence. Self-ratings are generated text and **not an evaluation metric**. |
| 35–36 | Empty cells. |

Recorded successful environment: package output Python **3.11.13** despite notebook language metadata **3.10.11**; Unsloth **2025.11.3**, Transformers **4.53.2**, vLLM **0.8.5.post1**, Torch **2.6.0+cu124**, CUDA toolkit **12.4**, Triton **3.2.0**, Xformers **0.0.29.post3**, datasets **3.6.0**, dill **0.3.8**, fsspec **2025.3.0**, requested TRL **0.22.2**. Kaggle metadata says accelerator `none` and `isGpuEnabled:false` despite saved GPU output; this notebook was edited/exported after the recorded run.

## Reproduction boundaries and practical next steps

1. Preserve these originals as historical research artifacts with their saved outputs; do not run them unmodified as a supposedly reproducible demo.
2. Recover `results_v3.jsonl`, the saved merged model/tokenizer or LoRA adapter, and any exported per-example evaluation CSVs. Weight fingerprints, base revision, teacher model provenance, and validation split are otherwise missing.
3. New executable reproduction code should parameterize data/checkpoint paths, explicitly separate Base and tuned loading, explicitly set the historical template/prompt variant, and require complete dependencies. Fixing these issues must be documented as a repaired implementation, not a bit-identical historical replay.
4. Do not execute model-generated Python in the host notebook as a “safe” helper. A future runner needs process/container isolation, time/resource limits, explicit failure accounting, and no duplicate evaluation of its final expression.
5. A proof/CoT task can be evaluated on final numeric or symbolic answers, but code execution alone does not establish formal proof correctness. Report the supplied scoring definition exactly.

Exact original execution counters, source checksums, notebook checksums, and per-cell metadata are in `coverage_manifest.json`. This audit does not claim historical logs establish authenticity of an original runtime, nor that a new GPU run has occurred.


Packaging note: historical full-text extraction files are intermediate review material and are not bundled. Exact originals are in archive/notebooks/. JSON/CSV evidence mentioned above is in reports/. The portable recovery command is python scripts/recover_historical_evaluation.py.
