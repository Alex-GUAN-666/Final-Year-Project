> **2026-09-11 supplemental review:** The distilled pool is now recovered. Public-tokenizer reconstruction reproduces 452 training examples and shows 5/20 historical reasoning evaluation questions overlap training. Older references below to missing data or unknown overlap are superseded by [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) and [the recovered-data audit](../reports/RECOVERED_DATA_AUDIT.md).

# Data and generation-script audit

Audit date: 2026-09-11. This report concerns the three supplied datasets and three standalone Python scripts. It does not establish a fresh model-training or inference result.

## Coverage and method

Every byte of the six files was read, decoded and hashed. Every field of all 509 dataset records was parsed, and every supplied reference program was parsed with Python's AST. All 622 lines of the three standalone scripts were read. Full-record and normalized-question duplication, cross-file overlap, missing fields, syntax, reference labels, code structure and reasoning-tag structure were checked. File hashes and per-row source locations are in `data_audit.json` and `row_audit.jsonl`.

All 309 arithmetic-training reference programs and all 100 arithmetic-evaluation reference programs are supported by a closed arithmetic AST interpreter. This independently interprets only numeric constants, local numeric assignments, whitelisted arithmetic/bitwise operators and selected `math` functions. It does not use `exec`, `eval`, import the supplied programs, run top-level `print(solve())` calls, or execute arbitrary model-generated Python. It also resolves 75 of the 100 general-math reference programs. The remaining 25 general-math programs were inspected statically; their runtime behavior was not executed by this audit. General-math reasoning is not formally verified, and parsing all field contents must not be confused with certifying every argument as mathematically sound.

The reproducible audit used the provided Linux x86-64 CPU environment, Python 3.12 and the Python standard library. Exact interpreter version is recorded in `data_audit.json`. No Ollama request, teacher generation, model download, GPU training, adapter merge or model inference was run. The supplied datasets and scripts were not modified. A supplementary SymPy runtime check was attempted but SymPy was unavailable; no SymPy-specific numeric test is claimed.

## Inventory and actual roles

| File | Bytes | Records / lines | Role established by contents |
|---|---:|---:|---|
| `arithmetic_logic_data_revised_v2.jsonl` | 108,308 | 309 records, 309 physical lines | Computational training supplement: question, type, correctness flag, short solution text and reference Python |
| `arithmetic_problems_v3.csv` | 16,139 | 100 records, 429 physical lines | Arithmetic/bitwise evaluation inputs with reference code and answers |
| `random_samples_for_inference.csv` | 550,349 | 100 records, 10,845 physical lines | General mathematical-reasoning evaluation inputs with reference reasoning, reference code and numeric answers |
| `add_more_data.py` | 3,897 | 116 lines | Appends 33 hard-coded `math` examples to an existing JSONL file |
| `generate_arithmetic_data.py` | 4,275 | 110 lines | Randomly generates 200 arithmetic/bitwise examples |
| `batch_generate_and_verify_v3_1.py` | 15,696 | 396 lines | Reads an existing reasoning dataset, classifies tasks, requests Python from Ollama, runs it and compares results |

All 509 records have the expected fields populated and syntactically valid Python reference code. These facts concern file structure, not mathematical validity.

Neither CSV contains a base-model prediction column, merged-model prediction column, paired correctness outcomes or checkpoint identifier. The word `generated` in a reference-field name does not establish that it is a measured student-model output. Neither CSV alone demonstrates improvement after fine-tuning.

## Computational training supplement

All 309 records are labeled `COMPUTATIONAL` and `is_correct=true`. By question/code content, the file contains 104 arithmetic examples, 82 bitwise-logic examples, 32 ordinary bit shifts, 48 fixed-width cyclic rotations, and 43 `math` constant/function examples. There are no proof targets in this file.

There are 309 unique normalized question strings and no duplicate full records. However, records 204 and 283 ask for the same natural logarithm of 100 with different wording and identical code ASTs. Exact string deduplication alone does not remove this semantic duplication.

Of the 276 records with numeric solution text, 270 agree with the code and six disagree. Integers were compared exactly; floating-point results used absolute tolerance 1e-9. The erroneous text labels are:

| JSONL row and physical line | Stored textual answer | Independently recomputed code result |
|---:|---:|---:|
| 205 | 9028 | 9044 |
| 207 | 12173 | 12157 |
| 209 | 2882400010 | 3168731130 |
| 210 | 2003195395 | 2014458966 |
| 211 | 3735928559 | 6230900220451885620 |
| 212 | 8509377377798222589 | 8526495043095935640 |

Rows 205 and 207 use `10101` and `11100` without a base prefix. Their code interprets these as decimal integers. If binary strings were intended, that must be stated explicitly; neither their stored textual labels nor the decimal-code results should silently be relabeled as binary results.

As an additional independent check, all 48 cyclic-rotation answers were derived from their question text by rotating fixed-width binary strings, without using the supplied code formulas or the AST interpreter. This confirmed the same four rotation-label errors at rows 209–212; the other 44 matched. See `rotation_independent_checks.json`.

Rows 277–309, all 33 appended `math` samples, have `generated_solution` equal to the literal string `...`. They have computable reference programs, but not completed natural-language explanations. These placeholders must not be described as full chain-of-thought supervision. Whether they affected training depends on which target fields the selected training notebook actually used.

The special literal marker `$Calculate#` appears in 108 questions. The supplied basic generator does not produce that marker. Its origin and intended role are not established by these scripts. Raw data should be preserved; any normalization belongs in an explicit derived dataset.

## Arithmetic evaluation inputs

There are 100 distinct questions and no duplicate full records. All 100 stored answers agree with their reference code. However, 14 questions display an operand rounded to four decimal places, while the corresponding reference code and answer use a more precise hidden operand.

Affected data rows are 7, 8, 9, 38, 41, 50, 53, 62, 65, 69, 70, 82, 86 and 88. Their physical CSV line locations, displayed-input result, stored reference result and differences are recorded in `data_audit.json`.

For example, row 7 asks for `57.7030 * e`; its reference code uses `57.70300299162965 * math.e`. The displayed question yields approximately 156.8530163475723, while the stored answer is 156.8530244796648.

**Tolerance distinction:** the 14 absolute differences range from approximately 3.7347e-6 to 9.3266e-5. Relative differences range from approximately 5.1845e-8 to 6.7563e-6. All 14 would pass the paired CoT notebooks' `math.isclose(..., rel_tol=1e-3)` comparison with default `abs_tol=0`. Consequently, these discrepancies do not by themselves imply any change to the recorded paired CoT accuracy. They would fail a 1e-9 absolute comparison, as used by the standalone generation script. This audit flags the inconsistent inputs under its stricter check and does not replace historical metrics with a different tolerance.

Future derived benchmarks should generate both the displayed question and reference answer from the same exact operand representation. Keep the original data and original metric definition available to reproduce historical scoring.

## General mathematical-reasoning evaluation inputs

The file has 100 records but only 89 distinct normalized questions. Nine duplicate groups account for the 11 repeat occurrences:

| Duplicate group | Data rows |
|---|---|
| Summer-camp talents | 3, 29 |
| Cubic/line intersection sum | 4, 66 |
| Average book width | 12, 56, 82 |
| Rectangular backyard | 21, 50 |
| Polynomial remainder | 31, 55 |
| Impossible sine equation | 34, 41 |
| Tennis-ball distributions | 35, 44, 67 |
| Integral limit | 42, 46 |
| Factorial inversion | 89, 98 |

There are no identical full records: some repeated questions have different reference code or reasoning. A benchmark score on all 100 records gives repeated questions extra weight; a future score on the 89 unique questions would be a different benchmark and must be labeled accordingly. These counts do not establish training contamination, because the full original training dataset is absent.

Every answer is numeric: 98 are integers by value, and the remaining values are 0.5 and 388.80. Problems range from arithmetic and word problems to algebra, combinatorics, calculus, number theory and complex analysis. They may require mathematical reasoning, but the file does not contain a clean set of proof-output tasks. Calling its metric “formal proof accuracy” would be unsupported.

Seventeen reference programs are simply `def solve(): return <constant>` after ignoring comments: rows 1, 8, 16, 18, 34, 37, 41, 42, 43, 49, 51, 57, 58, 60, 74, 78 and 81. Such a program can encode an already-derived answer but does not independently verify the argument used to obtain it. Row 77 defines and calls the same `solve()` twice; its syntax is valid, but a print-based executor would receive repeated output.

All 100 reasoning strings have one `<think>` and one `</think>` tag. Their combined length is 492,440 characters (minimum 1,907, median 3,783, maximum 18,482). Fifty-five exceed the standalone batch script's 3,120-character cutoff. Therefore that script with its supplied settings cannot be assumed to recreate this CSV directly without a version/configuration or lineage explanation.

### Confirmed mathematical issue: row 43

Row 43, physical CSV lines 4,876–5,012, states that

`f(x) = log(sqrt(x^2 + 1) + x) + sin(x)`, `f(a+2)=5`, and `f(b-7)=-5`, then gives `a+b=5`.

The reference argument uses oddness to infer `b-7=-(a+2)`. Oddness alone does not permit that inference without injectivity. Under the natural-log convention, the function is `asinh(x)+sin(x)` and its derivative `1/sqrt(x^2+1)+cos(x)` can be negative.

A numerical counterexample found by independent bisection of this explicitly written formula is:

| Quantity | Value |
|---|---:|
| `a` | 30.392675889797374 |
| `b` | -26.645480543028654 |
| `f(a+2)` | 4.999999999999999 |
| `f(b-7)` | -5.0000000000000036 |
| `a+b` | 3.74719534676872 |

Thus the supplied question does not determine the stated answer under that convention. The detailed counterexample is in `row43_counterexample.json`. The printed `solve(): return 5` agrees with the stored label but does not validate this mathematical claim. A future curated benchmark should quarantine this row pending clarification, rather than silently changing the historical benchmark.

### Additional wording/reasoning concerns

- Row 1 has the correct final answer 4, but its final prose verification incorrectly changes `sqrt(y(x+y)) >= y` into `sqrt(y(x+y)) >= sqrt(y)` after division. This demonstrates why correct final answers do not certify reasoning text.
- Row 15 gives the standard remainder-system answer 59, while the statement additionally says the number in each line is even. That extra condition is not respected by the usual remainder solution and needs clarification.
- Row 28 uses the convention-sensitive expression `15 / 3 [7 + 2(8 - 9)] - 5`. Its answer 20 adopts left-to-right multiplication/division. Parentheses should make the intended interpretation explicit in a derived benchmark.
- Row 64 requests `a+b+c` from a radical representation without explicitly requiring an integer, reduced, standard representation. The intended answer 12 assumes `2*sqrt(5)/5`; equivalent rescalings otherwise make the representation nonunique.

These are data-quality findings, not allegations about intent. They should be disclosed as limitations and handled with versioned curation.

## Standalone script audit

### `generate_arithmetic_data.py` (110 lines)

- Lines 5–6 hard-code a machine-specific output location and 200 examples. Lines 28–34 draw random integers and shift amounts. There is no random seed, so the exact supplied data cannot be regenerated deterministically.
- Lines 8–25 define ten arithmetic/bitwise operation templates and two three-operand templates. Lines 41–78 select operations/operands. Three-operand examples are eligible only during the first quarter of records and are then selected with probability one half; this is not a stratified 25% allocation.
- Lines 80–99 construct a zero-argument `solve()` and compute the matching solution with `eval(expression)`. Here the expression is internally constructed from numeric literals and a fixed operator set, unlike arbitrary teacher-code execution. A future implementation can use direct arithmetic instead.
- The generation loop and file write run at module import time (lines 36–110); no `if __name__ == '__main__'` guard protects imports. A reusable package should use a CLI/function and an explicit output path.
- This generator covers neither the cyclic-rotation supplement nor the main mathematical-reasoning corpus. It cannot alone recreate the provided 309-row file, whose extra 76 pre-augmentation records and marker edits are not generated here.

### `add_more_data.py` (116 lines)

- Lines 12–29 and 66–70 define 3 constant expressions, 27 function examples and 3 combined expressions: 33 examples total.
- Lines 43, 61 and 82 deliberately write `generated_solution='...'`; the correctness flag is hard-coded true. There is no independent semantic validation, split management or duplicate check.
- Lines 88–105 read an existing JSONL file, concatenate examples and write an output. If the input is absent, lines 95–97 quietly begin with an empty dataset. This can create a superficially successful but incomplete training file.
- Lines 114–115 contain local absolute paths. The supplied function prompts also contain `math.` prefixes that have been removed from the final 33 questions in the provided JSONL; the final file therefore reflects an additional edit step.

### `batch_generate_and_verify_v3_1.py` (396 lines)

The implemented flow is **existing reasoning data → rule/LLM classification → Qwen coder translation to Python for computational examples → local code execution → numeric comparison → append JSONL → checkpoint → attempted Arrow export**. It does not itself generate the original mathematical questions or all teacher chains of thought.

- **Inputs/configuration, lines 17–24:** an external Arrow reasoning dataset is required. The local Ollama model tag is `qwen-coder-32b-fp16:latest`; its underlying model digest and template are not recorded. The mutable `latest` tag is insufficient for exact reproducibility. The output filename suggests v4 while the script filename suggests v3.1.
- **Classification prompt, lines 27–43:** distinguishes computational outputs from general proofs using both the problem and the existing reasoning. This is useful prompt design evidence, but it is not a formal task taxonomy or ground-truth classifier.
- **Code prompt, lines 45–67:** asks for self-contained zero-argument `solve()`, a numerical value or list of numbers, and a `[PYTHON CODE]` fenced block. This format contract is part of the reproducible method and should be retained/versioned.
- **Heuristics, lines 77–105:** any problem containing `inf`/`infty` is assigned `PROOF`, including potentially numerical limits; some short symbolic answers with no digits are assigned `NON_COMPUTABLE`; a fraction or `degree` in the answer forces `COMPUTATIONAL`. These are practical filters with known misclassification risk.
- **Classifier fallback, lines 107–115:** failed or unexpected LLM responses default to `PROOF`; errors are not distinguished from genuine proof classification. Substring matching for `COMPUTATIONAL` can also accept a verbose negation.
- **Ollama calls, lines 117–133:** two attempts, temperature zero and a 70-second client timeout are specified. Full model version, seed, server version and prompt/response provenance are not saved.
- **Code extraction, lines 135–145:** removes fences using text replacements and returns the literal string `None` on failure. Robust parsing should distinguish missing, invalid and empty code explicitly.
- **Execution, lines 147–159:** `execute_safely` calls unrestricted `exec(code_string, scope)` in the current process, then calls `solve()`. There is no isolation, execution timeout, memory bound or filesystem/network restriction. Exceptions alone do not make this safe. Top-level code may also invoke `solve()` before the function is called again. This path was not run during the audit.
- **Comparison, lines 189–236:** exact string equality is tried first; otherwise lightweight LaTeX conversion and SymPy parsing reduce both sides to 30-digit numerical approximations and test absolute error below 1e-9. This is not a proof checker. List outputs are requested by the prompt but are not properly compared by the scalar `.evalf()` path. Nested LaTeX and mathematically equivalent expressions may be mishandled. Very large exact integers should not be compared by finite-precision float approximations. The `locals` dictionary is not a security sandbox for SymPy parsing.
- **Cutoff, lines 24 and 326–328:** `1300*2.4` is 3,120 Python string characters, not a measured token budget. Oversized rows are omitted, so retained examples are length-selected.
- **Errors/checkpoints, lines 240–253 and 364–375:** checkpointing stores only an integer index. It records no dataset fingerprint or configuration. Error records are constructed but never written before `continue`; length skips also omit an output row. Appending results and saving a checkpoint are not transactional, so an interruption between them can duplicate records on resume. Stable source IDs and recorded skip/error outcomes are needed.
- **Export, lines 385–392:** calls `final_dataset.to_feather(...)`. That method is not in the current documented Hugging Face `Dataset` API. Use a supported export such as `to_parquet` or `save_to_disk` with the matching loader, and verify the pinned environment. The current official API documents these alternatives: [Hugging Face Dataset reference](https://huggingface.co/docs/datasets/package_reference/main_classes).

## Overlap and missing lineage

There is zero exact normalized-question overlap and zero full-code-AST overlap between any pair of the three supplied datasets. This does **not** establish an uncontaminated test split, because the original main training corpus, sampling indices and source dataset IDs are missing. The 309-row arithmetic supplement must not be represented as the entire mathematical-reasoning training corpus merely because it is the only supplied JSONL.

Assets required to reproduce the standalone-generation lineage include:

1. The original Arrow reasoning input with `problem`, `generated_solution` and `expected_answer`, its source dataset revision/license, and the record IDs used.
2. The actual `results_v3.jsonl` or equivalent computational/proof training export, along with successful, failed and skipped record logs.
3. The script/version that created the additional 76 examples before the 33-row augmentation, and any `$Calculate#` or `math.` prompt edits.
4. The generator and seed for `arithmetic_problems_v3.csv`, and the selection procedure/seed for `random_samples_for_inference.csv`.
5. Frozen teacher/model digests, prompt versions and generation parameters.

Adapter weights, merged weights, dependency versions, exact train/validation/test membership and paired prediction exports are also needed for a full training-to-evaluation reproduction; those are outside the evidence contained in these six files.

## Practical repository treatment

- Preserve the supplied raw files and their SHA-256 hashes. Archive legacy scripts without encouraging readers to run unrestricted teacher-code execution.
- Store source-data audits and a per-record curation manifest beside a separate derived dataset; record every changed answer, prompt or dropped duplicate.
- Keep source/example identifiers and split the main corpus before any answer-informed generation. Never include reference reasoning, code or answers in student inference inputs.
- Use both code-format/execution measures and numerical answer accuracy with a declared tolerance. Do not present numeric exact-match as proof correctness.
- Maintain a historical benchmark view and a separately named curated benchmark view. Do not rewrite historical scores after fixing inputs, deduplicating questions or removing row 43.
- Implement the normal reproduction CLI so that data checks run without a model/GPU and training/inference fail clearly when required data or weights are absent. An isolated execution environment is required before evaluating freshly generated code.

The available files support a substantive, auditable repository draft and deterministic data checks. They do not yet support a claim that the original complete training run and its measured improvement have been freshly reproduced.
