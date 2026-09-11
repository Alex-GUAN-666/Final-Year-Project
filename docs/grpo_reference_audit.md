# GRPO prototype and merged-model reference inference audit

Audit date: 2026-09-11. This is a static audit of the supplied originals, not a new model-training or inference run. Cell numbers below are one-based notebook positions, not execution counts.

## Coverage and evidence boundaries

- `Qwen3_4B_GRPO_V1.ipynb`: all 69 cells inspected, including every source line, every stored stdout/stderr/traceback/text representation, all 100 GRPO metric rows, all 23 displayed formatting-SFT loss rows, and all 20 printed GRPO sample responses. HTML tables were also extracted into CSV. No original code was executed.
- `notebook62f7bf13ad_reference .ipynb`: all 24 cells inspected, including every source line and all textual outputs, the full 30-question list, full prompts, and the repeated generated response in cells 16 and 20. No original code was executed.
- All nine GRPO `application/vnd.google.colaboratory.intrinsic+json` payloads were also inspected. Eight only declare a string type; cell 13's dataframe payload contains additional unique-count statistics, three full problem examples, and three full teacher-solution samples, all read in full. These are preserved in `grpo_intrinsic_payloads.txt`. Column samples are independently selected and must not be assumed to be corresponding rows.
- Notebook widget state, other renderer payloads, decorative SVG/CSS, and notebook UI JavaScript were not manually interpreted. They remain in the originals and full JSON/text dumps. There are no image/png cell outputs in these two notebooks. The audit does not claim inspection of hidden training logs or recovery of all rows from the truncated dataframe display.
- Supporting semantic transcriptions: `Qwen3_4B_GRPO_V1.semantic.txt` and `notebook62f7bf13ad_reference.semantic.txt`. These preserve source and human-readable output, with HTML table text extracted and renderer-only payloads identified.
- Historical logs establish what the uploaded notebook contains; they do not authenticate the creator, execution environment, model artifact lineage, or the paper's experiment. All GRPO code cells from cell 28 onward have `execution_count=None` despite cached outputs, including the reported completed GRPO training. Do not present them as a run performed during this reconstruction.

## Overall interpretation

The GRPO file is a standalone Qwen3-4B-Base formatting-SFT + GRPO tutorial/prototype using public OpenMathReasoning/DAPO data. It does not use the user's arithmetic CSV, multiteacher generation JSONL, or code-execution reward. Its custom chat template is relevant to the later project, but it is not by itself evidence that the final merged SFT model underwent GRPO.

The reference notebook loads an already-merged SFT model and manually runs one selected question through a heuristic code-only/CoT+code router and Python-code extraction. It is a useful demonstration of prompt engineering and execution-assisted inference, but it contains no batch evaluation, ground-truth grading, base-versus-merged comparison, or result CSV.

The GRPO file repeatedly identifies Unsloth, its public API, and public datasets and uses tutorial-style narration. This is clear evidence of upstream framework/tutorial influence; an exact upstream notebook URL, commit, and license are not recorded in the supplied notebook and were not independently matched in this audit. Attribute Unsloth and dataset sources, and avoid claiming that this complete training recipe or reward family was created from scratch.

## GRPO notebook: complete cell map

| Cells | Source and saved result | Implication |
|---|---|---|
| 1–4 | Installation headings; unpinned `pip3-autoremove`, Torch/vision/audio/xformers from CUDA 12.4 index, `unsloth vllm`; stated goal Qwen3-4B-Base → reasoning via GRPO. | Current reexecution is environment-sensitive. |
| 5 | Load `unsloth/Qwen3-4B-Base`, 2048-token context, no 4-bit quantization, vLLM enabled, GPU utilization .7. Attach rank-32 LoRA to seven projection types, alpha 64, Unsloth checkpointing, seed 3407. | Actual saved log shows T4 14.741 GB; fp16 fallback; 36 layers patched. |
| 6–11 | Define working-out/solution delimiters, four-line system prompt, and Jinja chat template. Demonstrate 1+1 then 2+2 conversation. | User/assistant role delimiters are absent; assistant text and system end with EOS; generation is prefixed with `<start_working_out>`. Must retain exact template when loading trained artifacts. |
| 12–13 | Load `unsloth/OpenMathReasoning-mini`, split `cot`; retain `expected_answer`, `problem`, `generated_solution`; numeric-answer filtering. | Saved 19,252 source rows, 7,507 numeric-answer rows; rich dataframe metadata reports 3,895 unique problem strings, 868 unique answer strings, and 7,507 unique teacher solutions. Public data, not supplied arithmetic CSV. |
| 14–17 | Strip `<think>`/`</think>` strings from generated solution and wrap complete solution in custom reasoning delimiters, then append expected answer in `<SOLUTION>`. Full radical example saved. | This retains both teacher working and teacher final explanation inside the reasoning region; not extraction of formal proofs. |
| 18–21 | Filter full formatted samples to at most 1,024 tokens, convert to text and HF Dataset. | Saved 59 samples, not the full 7,507. This is length filtering, not truncation of traces. |
| 22–24 | Formatting SFT using SFTTrainer, 2 epochs/118 steps on 59 examples; complete TrainOutput saved. | Logs support a historical completed formatting-SFT stage. |
| 25–26 | Generate answer to first retained training question about Jenifer's pennies; 1,024 new-token cap, temperature 0 but no `do_sample`; warning says temperature may be ignored. | Answer 17 with desired tags. This is a training example, not held-out validation. |
| 27–28 | Delete dataset and collect garbage; cached result 0. | `execution_count=None` begins here. |
| 29–37 | Load `open-r1/DAPO-Math-17k-Processed`, config `en`, split `train` (14,116 rows); identity `extract_hash_answer`; map system+user messages and answer. First triangle task and answer 34 shown. | Dataset is English despite later multilingual prose. |
| 38–42 | Compile end-reasoning/solution regex with optional EOS; two formatting checks return 2 with retained whitespace. | Format regex does not require a start-reasoning tag, because it is prefilled in prompts. |
| 43–44 | Exact-format reward: +3 for a match, otherwise 0. | Response suffix/structure reward. |
| 45–46 | Approximate-format reward: each of reasoning-end, solution-start, solution-end gives +.5 if present exactly once, otherwise −1. | Range −3 to +1.5. |
| 47–48 | Answer reward: missing extraction −2; exact +5; trimmed exact +3.5; numerical ratio within [.9,1.1] +2, within [.8,1.2] +1.5; otherwise −2.5; conversion/division errors −4.5. | Approximate numeric closeness is rewarded; zero ground truth triggers division error unless string match succeeds. |
| 49–50 | Extract first numeric-looking string following `<SOLUTION>`, allowing minus, decimal point, comma; examples .34, 123,456, −.234, 17. | Not a mathematical expression/equivalence checker; fractions/exponents/embedded numbers can be misread. |
| 51–52 | Additional numerical reward: exact float equality +3.5, mismatch −1.5, no match −2.5, conversion exception 0. Every fifth invocation prints first generated response. | This duplicates some final-answer reward; printed samples are not a test-set score. |
| 53–54 | Compute prompt lengths, choose 90th percentile, keep <= threshold. | Threshold 201 tokens; training log later reports 12,709 retained rows. |
| 55–56 | GRPOConfig and vLLM sampling; 202 prompt tokens + 1,846 completion tokens; 100 steps. | Unsloth changes batch size from 1 to 4 to match 4 generations. |
| 57 | Narrative describes a multilingual tutor, “GRPO-optimized SFTTrainer,” and `max_steps=2000`. | Stale/inaccurate narration: source uses GRPOTrainer, English data, and 100 steps. |
| 58 | GRPOTrainer with all four rewards and no eval dataset; 100-step completed table and TrainOutput plus 20 printed responses. | Historical training output only; no benchmark result. |
| 59–60 | Base/no-LoRA inference on raw `What is the sqrt of 101?`, temperature 1, top_k 50, 1,024-token cap. | Saved answer drifts into scraped Q&A style after 10.05. |
| 61–62 | `model.save_lora("grpo_saved_lora")`, no output. | Referenced adapter files are not among supplied files. |
| 63–64 | Open adapter safetensors; attempt to assert each tensor not all zero. | **Assertion is wrong:** zero fraction is compared against `tensor.numel()`, so multi-element all-zero tensors pass. |
| 65–66 | Template-formatted system+user prompt, loaded LoRA, temperature 1, top_k 50, 2,048-token cap. | Final tagged answer 10.049875, but reasoning includes arithmetic errors. This comparison also changes prompting and token budget. |
| 67 | “much better”/“four hour or so” narrative. | One example is insufficient evidence of performance improvement; runtime below is about 2.95 hours for GRPO. |
| 68–69 | All merged16/merged4/adapter saving and hub upload examples guarded by `if False`. | No saved merged GRPO model is demonstrated by these cells. Empty token strings are placeholders, not exposed credentials. |

### Historical environment and all explicitly configured hyperparameters

Saved GRPO environment: Unsloth 2025.6.6; Transformers 4.52.4; vLLM 0.8.5.post1; Torch 2.6.0+cu124; Triton 3.2.0; Xformers 0.0.29.post3; T4, compute capability 7.5, one GPU; no bf16 or FlashAttention2. vLLM fell back to V0 and Xformers. Log records actual GPU utilization 69.34%, 7.63 GiB weights, 1.68 GiB KV cache, 764 CUDA blocks, 0 CPU blocks; model context 2048. Version of TRL was not explicitly pinned here.

| Stage | Settings |
|---|---|
| Load | max_seq_length=2048; load_in_4bit=False; fast_inference=True; max_lora_rank=32; gpu_memory_utilization=.7 |
| LoRA | r=32; alpha=64; targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj; gradient checkpointing `unsloth`; random_state=3407 |
| Format SFT | text field `text`; batch 1; accumulation 1; warmup 5 steps; 2 epochs; lr 2e−4; logging every 5; optimizer adamw_8bit; weight_decay=.01; linear schedule; seed 3407; report_to=none |
| GRPO generation | min_p=.1; top_p=1; top_k=−1; seed 3407; stop tokenizer.eos_token; include stop string; temperature 1; four generations |
| GRPO train | lr 5e−6; weight_decay=.01; warmup_ratio=.1; linear schedule; adamw_8bit; logging every step; requested batch 1 → effective 4; accumulation 1; max_prompt_length=202; max_completion_length=1846; max_steps=100; save_steps=100; report_to=none; output_dir=`outputs` |
| Base demo | raw prompt; LoRA None; temperature 1; top_k 50; max_tokens 1024 |
| LoRA demo | full custom prompt; loaded adapter; temperature 1; top_k 50; max_tokens 2048 |

No explicit SFT validation set, GRPO validation set, held-out test set, formal-proof checker, code executor, or confidence interval appears in this GRPO notebook. Additional defaults such as KL coefficient depend on the historical TRL/Unsloth environment and are not fully captured by the notebook source.

The rich dataframe metadata establishes that multiple teacher trajectories can refer to the same problem: 7,507 rows contain only 3,895 distinct problem strings before length filtering. The three additional full teacher samples concern primes p,q,r (answer 25), an LCM/GCD ratio (15), and the odd-number product modulo 1000 (375). A reconstruction should split by normalized problem identity, rather than by trajectory row, to prevent the same task appearing in training and testing. This does not by itself establish leakage in a later notebook's separate experiment.

### Historical training metrics and limitations

Format SFT: 66,060,288 trainable parameters out of 4,088,528,384 parameters including adapters (1.62%); 118 steps; 163.3104 seconds; mean train loss 0.3485639267048593. Displayed loss decreases from 0.6449 at step 5 to 0.1804 at step 115. Extracted exactly into `grpo_format_sft_historical_metrics.csv` (23 rows).

GRPO: available training set 12,709; 100 steps; same trainable-parameter count; runtime 10,607.693 seconds; mean train loss 0.0067273743636906145. Extracted exactly into `grpo_training_historical_metrics.csv` (100 rows). Across displayed rows, mean reward −1.61875, range −7.5 to +13, first 25 mean −1.31 and last 25 mean −3.165. Forty steps have mean completion length exactly equal to the 1,846-token limit. Thus these logs do not show a simple rising-reward trend; many outputs end before answer extraction.

`grpo_logged_samples.json` preserves all 20 periodically printed question/ground-truth/response/extraction records. Nine have an extracted number; eight of those match the stored target. This is a selected training diagnostic, **not 40% test accuracy**, not 20 independently held-out tests, and not evidence for the final SFT model's two-dataset improvement.

The saved outputs also demonstrate why numerical answer match is not proof correctness: the tangent-circle example reaches target 2 after using the wrong angle-bisector ratio and treating a sin(theta) numerator as constant 1. The strongly regular graph example reaches 21 but makes unsupported theorem/existence statements. Other logged responses exhibit substantive numerical errors and reasoning loops. The base/LoRA sqrt demo changes both prompt and budget; it cannot isolate training's causal effect.

## Reference inference notebook: complete cell map

| Cells | Source and saved result | Implication |
|---|---|---|
| 1 | Describes loading base + SFT LoRA for specific-input inference. | Actual path is already merged model. |
| 2 | uv installer; standby environment variable; probes NVIDIA T4; upgrades dependencies. | Malformed unterminated shell quote in datasets command: `"datasets>=3.4.1,<4.0.`. `%%capture` hides the problem. `is_t4` is unused. |
| 3–4 | Explicit fixes vLLM .8.5.post1, xformers .0.29.post3, triton 3.2.0; HF hub >=.34; datasets >=3.4.1,<4; Transformers 4.53.2; Torch 2.6; TRL .22.2 without dependencies. | Saved datasets downgrades 4.3→3.6; dill .4→.3.8; fsspec 2025.9→2025.3. Unsloth remains unpinned. |
| 5–6 | Imports torch/Transformers before Unsloth, sets CUDA_VISIBLE_DEVICES late, selects CUDA0/CPU. | Saved CUDA0. Model loader later warns import order. CPU branch is not proof Unsloth works on CPU. |
| 7–9 | Set `/kaggle/input/lora-2-3/Qwen3-4B-SFFT-merged`; load fp16/unquantized, context 2048, max_lora_rank32, utilization .9; `model=base_model`. | Commented adapter path `/kaggle/input/lora-2-3/sftt_save_lora_v1_3` is unused; no separate PeftModel wrapping. |
| 10 | Print tokenizer/embedding sizes and pad token. | 151,669 tokenizer entries; 151,936 embeddings; equality False; pad `<|endoftext|>`. Padding embedding rows can be intentional; inequality alone is not corruption. Check valid max token IDs rather than resize blindly. |
| 11 | Two full prompts: code-only and CoT+code. | Both demand numerical `solve()` answer, even for questions whose correct answer is a theorem/condition. |
| 12 | Inspect stored custom tokenizer template. | Same working-out prefix template style as GRPO prototype; a code-only prompt still gets `<start_working_out>` prefilled. |
| 13 | Heuristic router. “calculate the” takes precedence; capitalized Calculate becomes `$Calculate#`; then broad proof keywords/LaTeX/parenthesis regex; fallback code-only. | Case-insensitive detection but case-sensitive rewrite creates inconsistent prompt variants. Broad parenthesis match routes many simple arithmetic inputs into reasoning. Return annotation `str` is wrong; returns two strings. |
| 14 | Thirty manually authored/selected questions, ten computational-style then twenty mathematical/reasoning-style items. | No labels/ground truths or dataset scoring. Typos and input ambiguity (AND/OR vs Python `and`, cyclic shift spelling) must be kept in legacy view and documented if normalized. |
| 15 | Select question index29 about X^T X invertibility; execution_count135. | Repeated manual exploration implied by counts, but prior responses are not preserved here. |
| 16 | Apply stored chat template; max_new_tokens2048; do_sample=True; temperature .2; top_p .9; use_cache True; no seed. | Recorded time 110.08667612075806 sec, one response. Final answer full column rank is correct for real X, but derivation is incomplete, code has no call, example comment is wrong, long repeated note tail. |
| 17 | Old code extraction stored as Markdown. | Not executable in Run All. |
| 18 | Active fenced-code regex accepts python, py, or empty language and runs on generated assistant response. | Extracts one `is_XTX_invertible` definition, not requested `solve()`. |
| 19 | AST wraps last expression in print; in-process `exec`; stdout capture; afterwards `eval(last_expr,scope)` again if string contains print. | Unrestricted generated code execution with no timeout/isolation; double execution for print expressions, side effects, stdout leakage; misleading “safe scope” docstring. |
| 20 | Extract, execute all blocks, add execution output into response by exact string replacement. | Stored result empty, had_error False. No output is not a correct answer. Exact replacement may miss py/unlabeled/whitespace fence forms. |
| 21 | GPU reset cleanup enclosed in a triple-quoted string. | Inert. |
| 22–24 | Commented `kill -9`, commented nvidia-smi, empty cell. | No active runtime function. |

### Observed reference output defects

The generated response says rank(X^T X)=rank(X) merely because a product's rank is at most the minimum rank; that inequality alone does not prove equality. Its example `[[1,2],[3,4],[5,6]]` actually has independent columns, contrary to the comment saying dependent. It returns a function definition without invoking it; hence the recorded empty code result is expected. The displayed delimiter is `<end_working-out>` with a hyphen, differing from `<end_working_out>`. The response ends with repeated self-referential notes and a cut-off `Note: The`; no stop-reason metadata is retained, so token-limit termination is plausible but not confirmed from the stored metadata.

The tokenizer input is not explicitly truncated and generation allows 2,048 additional tokens on top of a prompt while model context is 2,048; long questions need an explicit prompt+completion budget. Only input_ids are passed, not attention_mask; this is risky for future padded batches. `response_text.split(...)[-1]` never raises IndexError, so the fallback exception branch is ineffective. This file has no saved model output artifact beyond notebook cells.

### Historical inference environment

Log records Python 3.11.13, Unsloth 2025.11.3, Transformers 4.53.2, vLLM 0.8.5.post1, Torch 2.6.0+cu124, Triton3.2.0, Xformers0.0.29.post3, T4 14.741GB, one GPU, fp16/no bf16. Duplicate cuFFT/cuDNN/cuBLAS registration messages appear but loading succeeds. These warnings are distinct from a failed model load. Current machine possession of the merged weights is not established; the Kaggle model directory is not in the supplied file inventory.

## Reconstruction recommendations

1. Keep both originals as historical/prototype notebooks with provenance notes; do not silently turn their cached outputs into new benchmark claims.
2. Main reproducible path should use the actual SFT/data-generation files. Keep GRPO optional until its role in the final artifact can be demonstrated.
3. Preserve exact prompt strings, template, and routing choices in a versioned legacy configuration. Offer intentional fixes separately and label changed behavior.
4. Evaluate base and merged models on the same fixed input IDs, same prompt/template mode, same generation settings, same token budget and grading rules. Any code-execution condition is a separate evaluation dimension.
5. Archive historical metrics under an explicitly historical namespace. Do not mix training reward, execution success, numerical final-answer match, and proof validity.
6. Replace in-process arbitrary execution with a separately controlled execution backend; merely passing `scope={}` does not sandbox Python. A fresh working directory/timeout alone also does not stop filesystem or network access.
7. Require explicit “answer absent”/“code absent”/“execution failed”/“not graded” categories. Empty successful execution must never be counted as correct.
8. Fix the all-zero adapter assertion if retaining it: test whether `(tensor != 0).any()` for each relevant tensor, and establish changes against initial state. Merely seeing non-zero A weights is not evidence of training because LoRA initialization may make A nonzero.
9. Dependencies, model/dataset revisions, seeds, checkpoint hashes, and model artifact locations must be recorded for actual reruns. The supplied notebook logs are not complete lockfiles.

## Exact-prompt/source appendices

Exact source for the two reference prompts/router and GRPO system prompt/template is written to `grpo_reference_prompt_sources.json` beside this audit. The full notebook originals remain the authoritative byte-preserving source.


Packaging note: historical full-text extraction files are intermediate review material and are not bundled. Exact originals are in archive/notebooks/. JSON/CSV evidence mentioned above is in reports/. The portable recovery command is python scripts/recover_historical_evaluation.py.
