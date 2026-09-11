# Evaluation CSV reconciliation

**Current evidence: the historical evaluation is not an independent held-out benchmark.** The official-tokenizer reconstruction retains 143 distilled training rows; 5/20 historical evaluation questions are exact matches to those retained rows (evaluation rows 5, 12, 13, 14, 20). These historical scores do not establish an independent held-out generalization gain. The original training-tokenizer revision and exact merged checkpoint remain unverified.

Both newly supplied CSV files contain the same BASE historical run. The filename containing `merged` does not identify merged-model predictions.

This audit reads every CSV field, independently extracts the original notebook outputs, checks exact full-response/code matches, and recomputes scalar-answer correctness from recorded stdout. It does not run a model or model-generated code.

## File identity and scores

| Uploaded file | Bytes | Rows | Recorded/rescored | Historical run matched |
|---|---:|---:|---:|---|
| `evaluation_results_base_cot_v5(1).csv` | 136051 | 20 | 12/20 = 60% | base |
| `evaluation_results_merged_cot_v5(1).csv` | 136051 | 20 | 12/20 = 60% | base |

Byte equality: **True**. SHA-256: `a05b045035a3a2e84c2844d92ba549a48db8096061ef76b73ac9e43a5197825e`.

For each uploaded CSV: all20 full question strings, all20 full response strings, all20 extracted code strings, and all20 correctness flags exactly match the base notebook. All8 notebook-observed failure stdout strings also match. All20 reference answers are numerically equivalent (CSV serializes values such as `4.0`, reference data uses `4`). No response or code field matches the merged notebook, even though the questions are identical.

Each CSV has12 correct scalar outputs,3 missing-code failures,2 recorded execution errors,1 nonnumeric symbolic-list output, and2 wrong numeric outputs. There are0 incorrect correctness labels under the archived execution metric, and every extracted code block matches the archived first-fence regex.

## New evidence recovered

The new CSV payload supplies previously unprinted stdout for all12 successful base examples. Base stdout is now directly observed for all20 rows. Both duplicate files are preserved in the manifest; they are one payload, not two independent experiments.

Merged remains15/20 from its original notebook, subject to the5/20 training-overlap limitation above. Successful merged stdout remains unobserved for15 rows and stays null; it is never copied from the base CSV or the reference answers. The five merged failure outputs remain directly recovered from its notebook. These uploaded CSVs add no arithmetic-specific result evidence.

## Full per-row review

Rows refer to both identical uploaded files. Full response/code field hashes and literal execution_result strings are retained in the JSON report; full model response and code are retained in the enriched historical JSON.

| Row | Gold | Saved stdout | Flag/rescore | Finding |
|---:|---:|---|---|---|
| 1 | 4.0 | 4 | pass / pass | Numeric 4 matches gold. The prose sufficiency check incorrectly simplifies 2y + 2sqrt((x+y)y) >= 4y to a right side of 3y; output scoring does not verify the proof. |
| 2 | 46.0 | Symbol-indexing execution error | fail / fail | SymPy solve output is indexed by a Symbol as though a dictionary; recorded execution error. Prose equations do not correctly use the two step counts. |
| 3 | 64.0 | empty | fail / fail | Prose gives 64 but no matching Python block. Fails the recorded execution metric despite the textual answer. |
| 4 | 12.0 | 12.0 | pass / pass | Vieta and cube-sum calculation return recorded stdout 12.0, equivalent to gold 12. |
| 5 | 66.0 | empty | fail / fail | Prose states 66, then repeats tokens. No matching Python block. |
| 6 | 28.0 | Symbolic list (full value in JSON) | fail / fail | Code prints a list of symbolic perimeter expressions, including nonphysical branches. Full stdout is not a scalar decimal and is rejected. |
| 7 | 2.0 | Symbol-indexing execution error | fail / fail | SymPy solve output is incorrectly indexed with a Symbol; recorded execution error. Prose additionally claims an unsupported speed of 3. |
| 8 | 1.0 | 1 | pass / pass | Recorded stdout 1 matches gold; executable code tests p=5 only. The general argument is in prose and is not scored for validity. |
| 9 | 36.0 | 0 | fail / fail | Prose gives 36 but code uses the first two roots without requiring positive lengths; recorded stdout is 0. |
| 10 | 233.0 | 233 | pass / pass | Enumeration of digit strings returns recorded stdout 233. A prose statement about maximum length is imprecise but code counts lengths 1 through 12. |
| 11 | 148.0 | 148148148 | fail / fail | Code uses 2222222220 rather than the smallest valid number 2220; recorded quotient 148148148 differs from gold 148. |
| 12 | 4.0 | 4.0 | pass / pass | Five widths sum to 20; recorded stdout 4.0 is equivalent to gold 4. |
| 13 | 105.0 | 105 | pass / pass | Choosing two unordered co-presidents gives 105. Irrelevant trailing tokens do not affect the first extracted code block. |
| 14 | 28.0 | 28 | pass / pass | Numeric result 28 matches gold; code returns the literal after the preceding derivation. |
| 15 | 59.0 | 59 | pass / pass | Gold 59 is correct for the standard remainder puzzle, but contradicts the supplied even-number-per-line clause. Recorded pass is retained and dataset defect is flagged separately. |
| 16 | 2.0 | 2 | pass / pass | Code tests a=15,b=77 and returns 2. Prose incorrectly restricts a/3 and b/7 and does not prove the universal result; both a,b being odd would establish it. |
| 17 | 28.0 | 28 | pass / pass | Recurrence is evaluated at 21 and recorded stdout is 28. |
| 18 | 15.0 | 15.0 | pass / pass | Recorded stdout 15.0 matches gold. Code checks endpoints for the example p=7.5, while prose gives the general piecewise-linear endpoint argument. |
| 19 | 6.0 | 6 | pass / pass | Six-term AM-GM and equality point (1,2,3) give numeric 6, consistent with recorded stdout. |
| 20 | 4851.0 | empty | fail / fail | Prose falsely claims infinitely many positive integer solutions and emits no matching code block. Gold is C(99,2)=4851. |

## Public repository wording

> Historical saved outputs show numeric-answer execution accuracy of12/20 (60%) for the base model and15/20 (75%) for the merged model on the same20-question reasoning subset. Five of these20 questions occur in the143 distilled examples retained by the official-tokenizer reconstruction, so the15-percentage-point difference is not an independent held-out generalization gain. Base results are additionally corroborated by a recovered CSV export. The two uploaded CSV filenames represent identical copies of the base run; the merged score is supported by its original notebook output. These are historical small-sample observations, not a new model rerun or verification of full mathematical proofs.

Keep row15's wording defect separate from execution-label consistency:59 matches the supplied gold but cannot satisfy the extra even-number-per-line clause. For example, an even quotient with remainder1 on division by2 implies N=4k+1, which contradicts N mod4=3. Do not silently alter the historical denominator or labels.

## Generated audit artifacts

- `evaluation_reconciliation.json`: input hashes, all-row static checks, file/run matching, rescoring and reporting guidance.
- `cot_historical_results_enriched.json`:40 complete historical responses and provenance of recovered/unknown stdout.
- `cot_historical_paired_results_enriched.csv`:20 aligned base/merged rows with explicit observed flags.
- `reconcile_evaluation_files.py`: standard-library recovery script; no checkpoint or CUDA required. Run `python scripts/reconcile_evaluation_files.py` from the repository; defaults resolve relative to the script, so another working directory also works.
