# Paired CoT evaluation notebook audit

> **Superseded interpretation — updated evidence, 2026-09-11.** The historical scores below are not an independent held-out generalization result. Reconstructing the 143 retained distilled training examples with the official Qwen3 tokenizer identifies exact training/evaluation question overlap in **5 of 20** evaluated rows: **5, 12, 13, 14, 20**. The original training tokenizer revision and exact merged checkpoint are still unauthenticated. Both subsequently uploaded evaluation CSVs are byte-identical copies of the **base** run, including the file named `merged`. Their complete responses/code match the base notebook and supply the 12 previously missing successful base stdout values. Merged successful stdout remains unobserved. The earlier statements below that `results_v3.jsonl` is missing, overlap is unknown, or all successful stdout is absent describe the initial audit and are superseded. Consult [current CSV reconciliation](../reports/evaluation_reconciliation.md), [current distilled-data audit](../reports/recovered_data_audit.json), and [enriched historical records](../reports/cot_historical_results_enriched.json). Original observations and scores remain below for traceability.

Audited on 2026-09-11. Cell references are **one-based**, counting the introductory Markdown cell.

Files:

- `notebookd84987e9c1_base_cot.ipynb`
- `notebookd84987e9c1_merged_cot.ipynb`

All 14 cells in each notebook, all source text, every textual output block, execution counts, and execution timestamps were inspected. Long generated responses were inspected using complete extraction with repeated lines/identical repeated substrings normalized for reading. Original text remains in the notebooks, `.full.txt` and `.output.txt` dumps; no original source was modified. No model was loaded and no model-generated code was executed during this audit.

## Main finding

The saved outputs support a **historical, paired 20-question code-output evaluation**: base **12/20 = 60%**, merged **15/20 = 75%**, an increase of **15 percentage points**, or three additional correct answers net. Both notebooks evaluate exactly the same first 20 questions from the attached 100-row `random_samples_for_inference.csv`, in the same order. This is verified by exact string equality for every logged question.

These are numeric-answer reasoning tasks evaluated through generated Python execution, **not a proof-validity benchmark**. Correct code output can pass despite invalid prose reasoning, and correct prose can fail because code is missing or not called. These scores must not be described as independently verified mathematical proof accuracy.

The saved execution history skipped cell 10, which empties both system prompts. Running all cells in order would execute that cell, remove the prompts and reduce every question's generation budget from the intended CoT branch to 1,024 tokens. The untouched notebooks therefore do not replay their historical configuration with Run All.

## Recovered evidence artifacts

- `cot_historical_results.json`: 40 full raw model responses, first extracted code blocks, recorded outcome flags, timings, route labels, question hashes, source hashes and execution metadata.
- `cot_historical_paired_results.csv`: 20 aligned base/merged outcomes.
- `reconstruct_cot_logs.py`: re-creates both artifacts from the original notebook JSON and attached CSV, without executing generated code.

Correct examples' execution output is **not printed** by the historical evaluator. Their `recorded_execution_result` is deliberately `null`; it has not been fabricated by copying the reference answer. The incorrect-example summary does print execution output, including blank strings and execution errors, and those values were recovered literally.

The CSV paths mentioned in the notebook output (`/kaggle/working/evaluation_results_base_v5.csv` and `/kaggle/working/evaluation_results_merged_v5.csv`) are historical output locations, not attached result files. The reconstructed artifacts are explicitly marked as log extraction, not a new inference run.

## Cell-by-cell source inventory

| Cell | Both notebooks, except noted differences |
|---|---|
| 1 | Markdown purpose: load base/merged SFT model and test arithmetic/CoT questions. |
| 2 | Captured installation cell. Unpinned latest Unsloth; vLLM 0.8.5 initially; Triton 3.2.0; Transformers 4.53.2; Torch 2.6.0; TRL 0.22.2. One malformed shell line has an unclosed quote: `!uv pip install "huggingface_hub>=0.34.0" "datasets>=3.4.1,<4.0.`. |
| 3 | Installs vLLM 0.8.5.post1 and xformers 0.0.29.post3, Triton 3.2.0 and bitsandbytes; saved outputs show the vLLM/xformers version replacements. |
| 4 | Corrected dependency line specifies datasets >=3.4.1,<4.0.0; saved outputs show datasets 4.3.0→3.6.0, dill 0.4.0→0.3.8, fsspec 2025.9.0→2025.3.0. |
| 5 | Imports torch/Transformers before Unsloth, standard libraries, pandas and Decimal; sets CUDA_VISIBLE_DEVICES=0 after importing torch. |
| 6 | Decimal precision 50; base notebook LOAD_BASE_MODEL=True, merged=False; hardcoded model paths; unused TEST_arithmatic=True. |
| 7 | Selects cuda:0 if available, otherwise CPU; both saved runs use cuda:0. |
| 8 | Unsloth FastLanguageModel loading with max_seq_length=2048, float16, load_in_4bit=False, max_lora_rank=32 and gpu_memory_utilization=.9. `fast_inference=True` is commented out. No adapter is attached here; merged checkpoint is loaded directly. |
| 9 | Defines the code-only and CoT+Code system prompts; both require a Python script and numeric answer. |
| 10 | Sets both prompts to empty strings; **unexecuted in both saved runs**. |
| 11 | Keyword/regex question routing; all 20 logged questions use System 02, the theoretical/CoT+Code branch. |
| 12 | `run_code`: AST parsing, wraps final expression in print unless already print, then unrestricted in-process exec; captures stdout. |
| 13 | CSV head selection, chat template, greedy generation, first-code-block extraction, execution, numeric scoring, CSV writing and summary. |
| 14 | num_for_test=20, input random_samples_for_inference.csv. Arithmetic CSV path is commented out. Contains complete 20-question historical outputs and final score. |

## Historical run identity and prompt-state mismatch

| Detail | Base | Merged |
|---|---|---|
| Model path | `/kaggle/input/qwen-3/transformers/4b-base/1` | `/kaggle/input/lora-2-3/Qwen3-4B-SFFT-merged` |
| Loading shards | 3 | 2 |
| Historical evaluation UTC | 2025-11-19 02:25:47–03:03:23 | 2025-11-19 03:17:59–03:55:36 |
| Cell 9 execution_count | 8 | 8 |
| Cell 10 execution_count | null | null |
| Cell 11 execution_count | 9 | 9 |
| Cell 14 execution_count | 14 | 12 |
| Mean recorded generation time | 112.745 sec/question | 112.831 sec/question |
| Sum recorded generation time | 2,254.899 sec | 2,256.611 sec |

Cell 10 has no execution timestamp in either notebook. Cells 9 and 11 have consecutive counts 8 and 9, respectively. This is strong saved-state evidence that the reset was skipped. The historically intended path thus used cell 9's nonempty CoT+Code prompt and the 2,048-new-token branch for all 20 questions. Exact historical token counts were not stored, so generation length is inferred from source and execution order, not independently counted.

On a clean top-to-bottom run, cell 10 would make `code_system_prompt == cot_code_system_prompt == ''`. Consequently, cell 13's `if system_prompt == code_system_prompt:` is true for every routed question, and `max_new_tokens` is 1,024. This changes both prompt content and generation length. The route log would still say CoT+Code, masking the change.

Saved environment information: Python 3.11.13; Unsloth 2025.11.3; Transformers 4.53.2; Torch 2.6.0+cu124; CUDA Toolkit 12.4; Triton 3.2.0; vLLM 0.8.5.post1; xformers 0.0.29.post3; one Tesla T4, 14.741 GB reported memory; bfloat16 unavailable. A warning explicitly says Unsloth should be imported before Transformers. cuFFT/cuDNN/cuBLAS duplicate-registration warnings appear but loading completes successfully. These logs are historical environment evidence, not a lockfile or a guarantee current package resolution reproduces it.

## Prompt and inference contract

The code-only prompt requests a self-contained script starting directly with a Python fence, no external prose, `solve()` returning a numeric answer, and a call to `solve()`. The CoT+Code prompt requests a clear step-by-step textual explanation followed by that script. Both demonstrate `print(solve())`. Exact text is retained in cell 9; spelling errors such as `caculate` and `caculation` should not be silently changed when attempting a historical replay.

Routing first checks for `calculate the`, then broad theory/proof keywords or LaTeX/parenthesis patterns; otherwise code-only. This heuristic is not a validated classifier of proof versus arithmetic. Every first-20 row triggers the theory branch, including simple average-width and combinatorics problems.

Cell 13 feeds only the selected system prompt and CSV question into `apply_chat_template`; the CSV's `generated_solution`, `generated_python_code`, and reference answer are **not** included in the inference messages. The template uses add_generation_prompt=True, padding=True, return_tensors='pt'. Generation is greedy (do_sample=False), with use_cache=True and tokenizer.eos_token_id, with no sampling temperature/top-p in effect. The attention mask is not supplied separately. Model-specific chat-template content, tokenizer revision/hash, checkpoint revision/hash, prompt token counts and generated token counts were not archived.

The declared max_seq_length=2048 and a possible 2048-new-token budget are not accompanied by an explicit prompt-plus-completion budget check. A cleaned implementation should validate the effective context limit. Numerous responses continue into repeated malformed conversation delimiters, random tokens, repeated answer text or irrelevant Java/HTML-like fragments. This shows poor termination/output discipline; it does not establish an exact EOS or truncation cause because token IDs and finish reasons are absent.

## Scoring semantics and defects

1. `df.head(num)` chooses the **first** num rows, with no runtime random sampling. The filename `random_samples_for_inference.csv` does not establish how those rows were originally sampled.
2. Only the first block matched by `r"```+(?:python|py)?\s*\n(.*?)\n```+"` is used. An unlabeled block can match; later corrected code is ignored. Same-line code without the required newlines is not matched.
3. `run_code` parses the block, wraps a final expression in print, and executes it. A function definition with no call yields no stdout and fails; it does not automatically invoke solve().
4. Every nonempty stdout value must convert in full to Decimal; explanatory text, multiple printed lines, symbolic expressions or lists are rejected. No text-answer fallback, boxed-answer parser, symbolic equivalence check, or proof grader exists.
5. The comparison is `math.isclose(Decimal(result), Decimal(str(gold)), rel_tol=1e-3)`, with default abs_tol=0. math.isclose uses float arithmetic, so setting Decimal precision to 50 does not make the comparison exact. A 0.1% relative error is accepted; zero has no absolute tolerance. For this 20-row subset, reference answers are small finite numbers; problems with huge integers elsewhere require separate care.
6. Nonnumeric output in the Decimal conversion except block sets is_correct=False but does **not** set insert_text. On a first such example it can raise UnboundLocalError; following prior examples it can reuse stale insert_text. The historical base run's row 6 prints a symbolic list, but previous numeric rows provide a stale value and the run continues. `revised_content` is not the response saved in the results CSV, so this bug is mostly dormant in the recorded run, but it is a real failure path.
7. Empty input (num=0 or no CSV rows) fails at `evaluation_results[0].keys()` before the final zero-question guard. Missing input calls sys.exit; column/type validation is absent.
8. Generated code runs inside the same Python process with unrestricted imports/builtins, no timeout, no memory/output cap, and no filesystem/network isolation. It can hang or affect the host. Refactored execution should be isolated and bounded; an AST allowlist by itself is not a security boundary.
9. Timing is printed but not saved in the original per-result CSV fields. No seed, environment manifest, dataset/model hash, prompt version or termination reason is recorded by the evaluator.

These format/code failures are **not false negatives under its documented execution metric**; they become misleading if the number is presented as all mathematical reasoning correctness. Likewise, numeric passes with bad proofs are **not proof validation**.

## Per-question outcome review

Question numbers refer to the original CSV row numbers, one-based. B=base; M=merged. A dash means the historical correct sample's exact executed stdout was not printed; it is not reconstructed from gold.

| Row | Short description / gold | B | M | Explanation of relevant behavior |
|---|---|---|---|---|
| 1 | Maximum inequality constant / 4 | Pass | Pass | Both return literal 4. Both prose derivations make an algebraic error during sufficiency checking (replace required 4y with 3y after cancelling x); scalar scorer cannot detect it. |
| 2 | Escalator / 46 | Fail | Fail | Base SymPy result incorrectly indexed by Symbol; recorded execution error. M prose finds 46, but code uses wrong speed formula and prints 23.444444444444443. |
| 3 | Students with two talents / 64 | Fail | Pass | Base prose gives 64, no code. M provides code computing 64. |
| 4 | Sum of cubic intersection coordinates / 12 | Pass | Pass | Vieta/cube-sum reasoning and successful code. |
| 5 | Nonnegative x+y+z=10 / 66 | Fail | Pass | Base says 66 in prose, then repetitive tokens, no code. M provides math.comb code. |
| 6 | Inscribed rectangle perimeter / 28 | Fail | Pass | Base prints a list of symbolic perimeters and fails numeric parsing; positivity filtering is missing. M uses (a+b)^2=196. |
| 7 | Stream speed / 2 | Fail | Pass | Base SymPy solve result incorrectly indexed by Symbol. M derives and returns 2. |
| 8 | Legendre symbol / 1 | Pass | Pass | Base tests example p=5 in code, not universal verification; M returns literal 1 after general reasoning. |
| 9 | Triangle squared-side difference / 36 | Fail | Pass | Base prose says 36, but selects first two roots including negative roots whose squares coincide; executed result 0. M computes difference of positive squared solutions. |
| 10 | Friendly 1/2-digit integers / 233 | Pass | Fail | M prose correctly totals 233, but code sums central binomial coefficients and prints 1889. |
| 11 | Smallest 0/2-digit multiple of 15, divided by 15 / 148 | Fail | Fail | B selects 2222222220 and prints 148148148. M correctly identifies 2220 and defines a correct solve(), but never calls it, so stdout is empty. |
| 12 | Average book width / 4 | Pass | Pass | Correct sum and average code. |
| 13 | Choose two co-presidents / 105 | Pass | Pass | Correct combination code; trailing repeated garbage is ignored. |
| 14 | Triple green-marble probability / 28 | Pass | Fail | M prose correctly derives 28 but emits no Python block, only repeated `toContain`. |
| 15 | Warrior remainders / 59 | Pass | Pass | Both ignore the input's additional even-number-per-line condition. See data-quality note below. |
| 16 | Least prime factor of a+b / 2 | Pass | Pass | B tests a=15,b=77. M reasoning falsely claims a+b divisible by 3 and 7/21, yet returns literal 2 and passes. Correct reason is both a,b odd, so their sum is even. |
| 17 | Recurrence f(21) / 28 | Pass | Pass | Correct recurrence implementations. |
| 18 | Minimum absolute-value sum / 15 | Pass | Fail | M correct prose and correct solve() body, but no function call, so empty stdout. |
| 19 | Weighted AM-GM minimum / 6 | Pass | Pass | Correct six-term AM-GM argument and returned 6. |
| 20 | Positive x+y+z=100 / 4851 | Fail | Pass | B claims infinitely many solutions and gives no code. M correctly uses stars-and-bars C(99,2). |

Base incorrect rows: 2,3,5,6,7,9,11,20. Merged incorrect rows: 2,10,11,14,18. Six rows improve (3,5,6,7,9,20), three regress (10,14,18), two fail in both (2,11), and nine pass in both. One sample changes the reported score by five percentage points; no population-level or proof-generalization claim is supported by this tiny subset alone.

The base run has 17/20 responses with a regex-matching code block, versus 19/20 merged. Base has three missing-code failures, two execution errors, one nonscalar-output failure and two wrong numeric results. Merged has one missing-code failure, two uncalled-function/empty-output failures and two wrong numeric results. This decomposition is useful for separating formatting, implementation and mathematical outcomes without rewriting the historical metric.

### Data-quality and leakage notes

First-20 row 15 says each line contains an **even** number of warriors. Gold 59 gives per-line quotients 29,19,14,11,9 for 2–6 lines, so it does not satisfy that literal additional condition. Indeed, the two-line condition with an even quotient implies N=4a+1, conflicting with remainder 3 under division by 4. The intended standard remainder puzzle would yield 59 after removing that clause, but the original should be retained and flagged rather than silently repaired in historical reporting.

The 20 logged questions have zero exact-string overlap with the attached `arithmetic_logic_data_revised_v2.jsonl`. This **does not establish a clean held-out evaluation**: the main `results_v3.jsonl` distilled training file referenced by training notebooks is missing from the supplied files. The evaluation CSV contains generated solutions/code as well as questions/gold; that is not direct answer leakage at inference because those columns are not used in messages, but its provenance and any overlap with the missing training corpus must still be established. No train/evaluation split manifest or dataset-family deduplication record is present in these two notebooks.

## What to preserve and what to change for a reproduction repository

- Preserve source notebooks and historical scores as clearly labelled evidence; strip potentially huge output only in separate clean working copies, retaining hashes and archived originals.
- Use one configurable base/merged evaluation script with explicit model/tokenizer revisions, prompt version, sample IDs, dataset hashes, decoding settings and JSONL results.
- Make the historical prompt route explicit. Remove the dormant prompt-reset cell from a clean notebook or turn it into a named, intentionally selected ablation. Do not equate a clean Run All result to the saved 60%/75% without preserving state.
- Keep a `historical_execution_accuracy` scoring mode for comparability. If adding text-answer fallback, automatically invoking solve(), improved symbolic parsing, or proof grading, give those separate metric names and recompute **both** base and merged under the same new rules.
- Execute generated code in an isolated, bounded runtime; ensure empty/nonnumeric output cannot crash the evaluator. Save failure categories, not just a Boolean.
- Track code-output accuracy, code-format/execute success and separately reviewed mathematical reasoning. Repair dataset wording in a versioned derivative with rationale, while leaving historical results unchanged.
- Obtain exact merged weights, tokenizer files and the missing training corpus/split before claiming end-to-end training replay or leakage-free held-out gains.



Packaging note: historical full-text extraction files are intermediate review material and are not bundled. Exact originals are in archive/notebooks/. JSON/CSV evidence mentioned above is in reports/. The portable recovery command is python scripts/recover_historical_evaluation.py.
