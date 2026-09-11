# Supplementary notebook audit, 2026-09-11

This is a static review of the two newly uploaded notebook files. No original notebook, model generation, or generated Python was executed. All 36 cells of the new Kaggle export were checked against the prior 36-cell notebook, and all source plus saved text/HTML output was read. Widget progress values and Papermill execution metadata were inspected separately. The other upload is byte-for-byte identical to the previously completely reviewed August notebook; that existing full review is reused rather than claiming a second distinct experiment.

## Main finding

`qwen4b-train-code-100920859-279441290..ipynb` supplies a **clean, sequential November 18, 2025 Kaggle run** of the already supplied SFT/merge pipeline. Its source differs from `notebook4d41ab3228 _train and reference.ipynb` only by deleting a leading blank line in cell 25. Thus it does not introduce a new training method or different prompts. It improves historical execution evidence: all 25 nonempty code cells have sequential execution counts 1–25, Papermill reports all 36 cells completed with no exception, and the selected question, generated answer, extracted code, and execution result now agree.

The saved run trained **452 examples = 309 arithmetic + 143 CoT+Code**, completed **114 steps over 2 epochs**, then printed successful adapter merge and model/tokenizer save messages. This is historical evidence contained in the upload, not a new GPU run performed during this audit. Neither uploaded notebook contains the actual merged weights, adapter tensors, optimizer state, or tokenizer files.

## File identity and coverage

| Uploaded file | SHA-256 | Size | Relationship to previous file |
|---|---|---:|---|
| `qwen4b-train-code-100920859-279441290..ipynb` | `3790e0e38706dc4bb3ea04c3e9178505f6437a0f6b384ba97f11183ab5463f10` | 128,272 bytes | 36 cells; all source equal except one removed blank line in cell 25; fresh saved run outputs/metadata |
| `Qwen3_(4B)_SFFT_code_0811(2).ipynb` | `42cef8097628bb2d09f1fc6c51a6aa7bb1d20c462e2f56fbda44236807b14aa9` | 359,790 bytes | Exact duplicate of `Qwen3_(4B)_SFFT_code_0811.ipynb`, including all 27 cells, outputs, and metadata |

The earlier Kaggle training notebook hash is `009946a653bbb35bc787c5759cd5331740c65e9a647497d0f03e6eb745cee20e`. The August duplicate's complete interpretation remains in `docs/sft_training_inference_audit.md`. In particular, it saves an August LoRA adapter on a local WSL path and does not supply those adapter files; its prose does not establish that a GRPO checkpoint actually fed the shown SFT run.

Machine-readable per-cell hashes, source lengths, MIME types, differences, execution counters, and all Papermill status/timing values are in `coverage_manifest_v2.json`. The full source/text/output dump is `new_training_notebook.full.txt`; ANSI-cleaned readable material is `new_training_notebook.review.txt`. These are audit extracts, not runnable training scripts.

## Historical training configuration

The exact configured values are also saved in `historical_training_config_v2.json`.

| Setting | Recorded configuration |
|---|---|
| Base checkpoint path | `/kaggle/input/qwen-3/transformers/4b-base/1` |
| Model loader | `unsloth.FastLanguageModel.from_pretrained` |
| Loader context limit | 2,048 tokens |
| Load mode | `load_in_4bit=False`, `torch_dtype=torch.float16` |
| Fast inference | Commented out, therefore not explicitly enabled |
| LoRA | Rank 32, alpha 64 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient checkpointing / adapter seed | `unsloth` / 3407 |
| Training rows / epochs / steps | 452 / 2 / 114 |
| Per-device batch / accumulation | 1 / 8; total observed batch 8 |
| Training GPUs | 1 used, although 2 Tesla T4 devices detected |
| Learning rate / scheduler | 0.00003 / linear |
| Warmup / gradient norm | 10 steps / 1.0 |
| Optimizer / weight decay | `adamw_8bit` / 0.01 |
| Log interval / trainer seed | 5 steps / 3407 |
| Dataset shuffle seed | 42 |
| Reporting | `none` |
| Trainable parameters | 66,060,288 / 4,088,528,384 including adapters (1.62%) |
| Merge/save | `merge_and_unload()`, `save_pretrained(..., safe_serialization=True)`, tokenizer save |
| Saved path | `/kaggle/working/Qwen3-4B-SFFT-merged` |

The notebook sets a 2,048-token **loader limit and preprocessing filter**, but it does **not** explicitly pass `max_length` or `max_seq_length` into `SFTConfig`. The trainer's effective truncation length therefore depends on the installed patched TRL/Unsloth implementation. Do not infer that every accepted 2,048-token example was trained in full solely from the preprocessing setting. Future code should record the effective trainer configuration; explicitly enforcing a desired trainer length is a documented implementation repair unless the historical default is verified.

No held-out validation set, train/evaluation split, early stopping, packing setting, or explicit assistant-only loss mask is configured. The rendered `text` is passed to SFTTrainer. `fp16`/`bf16` are not explicitly set in SFTConfig; float16 is explicitly set only when loading the model.

### Exact data preparation

1. Load all 1,125 distilled records from `results_v3.jsonl` and 309 arithmetic records from `arithmetic_logic_data_revised_v2.jsonl`.
2. Arithmetic: retain every row's `problem` and `generated_python_code`, without checking its correctness flag.
3. Distilled: retain only rows with truthy `is_correct`, `generated_solution`, and `generated_python_code`; saved pool size 251.
4. Render each into system/user/assistant messages. Arithmetic assistant text is one fenced Python block. CoT+Code assistant text is the original `generated_solution`, two newlines, then fenced `generated_python_code`.
5. Compute `len(tokenizer.apply_chat_template(messages, tokenize=True))`; keep lengths `<=2048`. Saved retained counts are 309 arithmetic and 143 CoT+Code.
6. Concatenate arithmetic first, then CoT+Code; convert through pandas and Hugging Face Dataset; shuffle seed 42; render with `add_generation_prompt=False`; keep only `text` for the trainer.

These counts come from the saved execution logs. Verifying them against the recovered JSONL with the correct tokenizer is a separate reproducibility check. Any modern correction of wrong labels, filtering, or deduplication must create a separately named dataset variant, not overwrite the historical selection.

### Prompt and chat-template fidelity

Exact string literals are in `exact_prompt_literals_v2.json`; the Jinja template is also in `historical_chat_template.jinja`. Both prompts and the template are byte-identical at source level to the prior Kaggle notebook. The spelling mistakes `caculation` and `caculate`, example formatting, and newlines are preserved.

The arithmetic system prompt requires a fenced Python-only response defining and calling `solve()`. The reasoning system prompt requests a textual explanation followed by fenced Python. Both demonstrate `print(solve())`, while prose says the numerical result should be the final expression.

The custom template emits:

- System content followed by `eos_token`.
- User content directly followed by assistant content, with **no user/assistant role delimiters and no inserted separator**.
- EOS after assistant content.
- `<start_working_out>` only when `add_generation_prompt=True`.

If there is no system message, a fallback GRPO-style instruction is injected. That fallback does not establish that GRPO training occurred. The displayed rendered training example shows `<|im_end|>` as EOS. Fresh Qwen standard templates or a different EOS setting are not equivalent reproductions.

## November 18 run and complete cell coverage

Papermill metadata reports start `2025-11-18T08:03:03.356212`, end `2025-11-18T08:26:38.956392`, duration 1,415.60018 seconds (about 23 minutes 36 seconds). All 36 cells have status `completed`, exception `false`; the notebook-level exception is null. Metadata timestamps have no explicit timezone suffix.

| Cells | Full review findings |
|---|---|
| 1–3 | Upstream Unsloth tutorial header/news and explicit LGPL-3.0 notebook-license link. Preserve upstream attribution with any redistributed derived notebook. These cells are unchanged. |
| 4–7 | Installation sections unchanged. Cell 5 retains an unterminated shell quote in the datasets install command, with output hidden by `%%capture`; cell 7 repeats a correct command. Shell command failure may not fail the Python cell. Installation upgrades unpinned Unsloth and several packages, so it is not a reproducible environment lock. New output contains successful dependency resolutions and downgrades, with all terminal text read. |
| 8 | Base load and LoRA initialization succeed. Startup detects two Tesla T4s; 36 QKV, O, and MLP layers patched. Duplicate CUDA plugin registration warnings are present, without an uncaught exception. |
| 9–11 | Exact task prompts and custom template unchanged. |
| 12–17 | Full two-pool data loading, correctness/truthiness selection for CoT+Code, token filtering, concatenation and shuffle; saved counts 1125→251→143 distilled and 309→309 arithmetic. Widget states show map/tokenization completion. Missing-file handling still references `sys` before its explicit import in cell 26. |
| 18 | Rendered arithmetic example matches current prompt/template: `$Calculate# the product of 58934 and 16146967720371580991.` and multiplication code. |
| 19 | Existing incorrect inspection passes a rendered string to `apply_chat_template` instead of a message list. Output is only the fallback instruction. This is not the actual trainer input. |
| 20–23 | SFT configured as above. All 22 loss entries read and extracted. Training completes 114/114 steps, epoch 2/2. UI elapsed 17:30; whole training-cell duration 1074.11744 seconds. First/last logged loss .9850 at step 5 / .3494 at step 110. All logged loss values exactly match the earlier November 16 notebook, while displayed duration differs; no held-out metric is reported. |
| 24–25 | Heading still says adapters; implementation merges and saves full model plus tokenizer. The only source change is deleting one leading blank line. Cell executes at count 16, duration 24.887137 seconds, with all four success messages. This proves the upload records saving; saved tensors are not attached or validated. |
| 26–29 | Imports, device selection, heuristic prompt routing, and all 30 question strings unchanged. `CUDA_VISIBLE_DEVICES` is set after CUDA has already initialized, so new code should set device visibility before importing/initializing CUDA. |
| 30 | Selects `question_set[12]`, square-root equation. New saved output now correctly shows that same question and CoT+Code prompt. |
| 31 | Generation uses `model.generate`, max_new_tokens 1024, temperature .2, top_p .9, sampling, no independent generation seed. No saved-checkpoint reload. A warning states no attention mask is supplied while pad equals EOS. 55.06365 seconds. The answer correctly derives sqrt(73)/2, writes one complete code block, then starts repeating the reasoning and ends mid-expression, consistent with length truncation. It is an interactive demonstration, not an accuracy evaluation. |
| 32 | Extracts exactly one complete code block, matching cell 31; stale five-block output from the earlier export is gone. |
| 33 | Original `run_code` function uses unrestricted in-process exec and evaluates a final print expression twice. Its claim of a safe scope is inaccurate. No timeout or resource/isolation boundary is provided. |
| 34 | Same one-block code execution now aligns with selected question and response. Saved result `4.272001872658765`, no error; a duplicate printed number is caused by the helper's second evaluation. Annotated answer and repeated/truncated trailing text fully read. |
| 35–36 | Empty code cells, no outputs. |

No output images are present. MIME types are text/plain, text/html, and Jupyter widget view JSON. Widget state contains only progress labels and counts relevant here: loaded shards 3/3, map 452/452, tokenization 452/452. There is no encoded checkpoint payload in a widget.

### Differences from the older Kaggle export

Sources: only the leading newline at cell 25 changes. Outputs differ in cells **6, 7, 8, 17, 21, 23, 30, 31, 32, 34**, reflecting package/runtime logs, progress widgets, elapsed training time, and the consistent square-root demonstration. Other differences are cell IDs, metadata and execution counters. Full output/source diff is in `training_notebook.diff.txt`.

The new Kaggle metadata now agrees with the GPU run: `isGpuEnabled=true`, `accelerator=nvidiaTeslaT4`, Python 3.11.13, Docker image version ID 31193. It lists train dataset ID 8679486/version 13652781 and pinned Base model ID 322000, instance ID 301515, source version ID 363135. Those identifiers aid provenance but are not authenticated checkpoint files or directly usable URLs to the missing fine-tuned model. No Hugging Face push, public artifact download URL, or checkpoint content hash is recorded.

## Recorded environment and remaining repair work

Recorded packages/platform: Python 3.11.13; Unsloth 2025.11.3; Transformers 4.53.2; vLLM 0.8.5.post1; Torch 2.6.0+cu124; CUDA toolkit 12.4; GPU compute capability 7.5; Triton 3.2.0; Xformers 0.0.29.post3; datasets 3.6.0; dill 0.3.8; fsspec 2025.3.0; requested TRL 0.22.2. The banner's `CUDA: 7.5` is the GPU capability value, not CUDA toolkit version 7.5. Bfloat16 support and FlashAttention2 are both false. Exact versions of PEFT, bitsandbytes, Unsloth Zoo, NumPy, pandas, huggingface_hub, Pillow, and torchvision are not all recorded in a lockfile.

Implementation recommendations:

1. Prefer this November 18 export as archived clean-run evidence; preserve the previous export to explain history, not as another independent model improvement.
2. Parameterize paths and pin the official Base checkpoint/tokenizer revision. Preserve the historical prompts/template in an explicit configuration. Log model and tokenizer fingerprints.
3. Rebuild the recovered dataset using exact original truthiness selection and token filter, retaining row IDs and per-row lengths. Save a selection manifest so 143 distilled targets can be traced back to the 1,125 records.
4. Explicitly record effective SFT truncation and loss masking. If modern library compatibility requires changing constructor arguments or defaults, identify that as a repaired reproduction rather than a bit-identical replay.
5. Save both the adapter and tokenizer before merge, then save the merged checkpoint; reload the checkpoint in a fresh process for inference validation and compare Base versus merged with identical frozen evaluation questions and settings.
6. Replace in-process model-code execution with an opt-in isolated runner, timeout and resource limits, and evaluate the final expression once. This is a concrete flaw in the supplied helper, not proof that an observed numeric output is wrong.
7. Pass attention masks, explicitly seed each generation run, and record generated token counts/finish reasons. Consider a stop condition or larger budget only as a separate inference variant; the historical demo uses 1,024 tokens and repeats/truncates.
8. Keep demonstration success, historical evaluation metrics and a new reproduction run clearly labeled. Neither a successful print nor a final numerical answer establishes formal proof verification.

The recovered data supplied elsewhere in this upload can remove the missing-training-target blocker. These two notebooks alone do not remove the missing-model-weight blocker. Retraining a new artifact is possible in principle once dependencies, exact tokenization and GPU access are resolved, but should never be described as recovering the old merged weights.
