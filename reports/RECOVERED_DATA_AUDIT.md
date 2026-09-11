# Recovered data audit

Date: 2026-09-11. This report summarizes the reproducible audit in `scripts/audit_recovered_distillation.py`; detailed per-record evidence is in `reports/recovered_data_audit.json` and `reports/recovered_data_rows.jsonl`.

## Main finding

The recovered distillation file reconstructs the saved 251-record eligible pool. Official-public-tokenizer reconstruction retains the historical **143 CoT+Code + 309 arithmetic** counts and finds **5 of the 20 historical evaluation questions overlapping the reconstructed fine-tuning data**. These are historical, small-sample results with training-overlap limitations.

**Original Kaggle tokenizer files are unavailable.** Reproducing selection counts with a frozen official tokenizer is documented reconstruction evidence, not authentication of byte-identical original tokenizer artifacts or original training membership.

## Coverage

Every byte of the 12,118,956-byte distillation JSONL and all seven fields of its 1,125 records were ingested. All relevant input and prediction CSVs and the arithmetic-training JSONL were parsed. Every non-placeholder code string was inspected through Python's AST. No supplied/generated program was executed, and this script does not train or run a model. Full structural ingestion does not certify the mathematical correctness of more than ten million characters of reasoning.

| Saved label | True | False | Null | Total |
|---|---:|---:|---:|---:|
| COMPUTATIONAL | 251 | 58 | 288 | 597 |
| PROOF | 0 | 0 | 268 | 268 |
| NON_COMPUTABLE | 0 | 0 | 260 | 260 |
| Total | 251 | 58 | 816 | 1,125 |

The historical predicate requires truthy `is_correct`, `generated_solution` and `generated_python_code`. It selects 251 computational records, with 216 unique normalized questions. Proof/non-computable records are excluded. The final task is mathematical reasoning and numeric Python-output prediction; no formal proof checker is present.

The corpus has 990 unique normalized questions, 103 repeated-question groups and four exact duplicate-record pairs. Code accounting: 528 `N/A` placeholders, 175 literal `None` placeholders, 95 syntax errors, and 327 parseable programs. All 251 eligible programs parse. A null or false flag is not always mathematical failure: the audit includes selected cases where units/LaTeX/reference rounding caused rejection of an equivalent numeric output.

## Evaluation ancestry and reconstructed overlap

All 100 general-math evaluation rows exactly match recovered source records across question, answer, reference program and reference solution. All 20 historically tested questions have eligible source counterparts. This candidate-pool overlap must be distinguished from the final length-filtered selection.

| Scope | Evaluation rows with retained counterpart | Distinct matched questions | Retained source records |
|---|---:|---:|---:|
| Full general-math evaluation CSV | 58/100 | 49 | 70 |
| Historical first 20; each new prediction CSV | 5/20 | 5 | 8 |

| Historical evaluation row | Retained source lines |
|---:|---|
| 5 | 897 |
| 12 | 764, 876, 999 |
| 13 | 988, 994 |
| 14 | 581 |
| 20 | 209 |

The reconstructed 143 records contain 112 unique questions. No exact/normalized question or exact/AST program overlap was found between the recovered distillation corpus and either the arithmetic-training JSONL or arithmetic-evaluation CSV. That does not exclude semantic or pretraining overlap.

The earlier flawed evaluation row 43 corresponds to source line 616. Its reasoning incorrectly infers a unique relation from an odd non-injective function. The reconstructed sequence has **3,739 tokens**, so this source record is excluded from the final 143; evaluation row 43 was also outside the historical first 20.

## Frozen tokenizer evidence

- Official revision: `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Model last modified: `2025-07-26T03:45:37.000Z`.
- Tokenizer JSON SHA-256: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
- Historical prompts/template; EOS `<|im_end|>`, ID 151645; maximum 2,048 tokens.
- Retained IDs: `reports/retained_source_lines.json`.
- Per-candidate token lengths and reconstruction method: `reports/historical_token_filter.json`.

The audit checks agreement between these two evidence files and records their hashes. Its report explicitly sets `selection_kind="reconstructed"` and `original_tokenizer_artifact_authenticated=false`.

## Provenance and revised experiment

Existing reasoning is passed to a code teacher and converted into Python, then checked against a reference numeric answer. The engineering work should be described as solution-conditioned code distillation, filtering, prompt design, LoRA SFT and executable-answer evaluation. The source Arrow basename and a separate notebook suggest OpenMathReasoning-mini, but exact upstream export/revision and per-record attribution are absent. Code licensing must not be assumed to license third-party data.

Historical reconstruction preserves originals, flags and duplicates. A prospective question-disjoint experiment should exclude the entire 100-row/89-unique-question evaluation pool before training: this removes 114 candidates and leaves **137 candidates / 127 unique questions before token filtering**. Changes to scoring, reference labels or data selection create a new versioned experiment and cannot retroactively repair an existing checkpoint.

## Reproduce

From the repository root, with Python's standard library only:

```bash
python scripts/audit_recovered_distillation.py
```

The script defaults to the repository containing it, so its absolute path also works from another directory. It reads repository-relative raw/evaluation files, adopts the saved reconstructed membership by default, and writes UTF-8 reports under `reports/`. Optional `--root`, `--output-dir` and `--retained-source-lines` arguments override these locations. `RECOVERED_DATA_PROVENANCE.md` provides source line/cell references.
