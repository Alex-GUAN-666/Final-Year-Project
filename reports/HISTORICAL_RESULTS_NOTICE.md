# Historical result interpretation

**The historical 60% base and 75% merged scores are not an independent held-out generalization result.** The official-tokenizer reconstruction retains 143 distilled training examples, with exact overlap in five of the 20 evaluated questions: rows 5, 12, 13, 14, and 20. The original training-tokenizer revision and exact merged checkpoint are not authenticated by the supplied files.

The files `cot_historical_results.json` and `cot_historical_paired_results.csv` are preserved first-audit extractions. Their absence of successful execution stdout and their incomplete training-overlap information have been superseded by `cot_historical_results_enriched.json`, `cot_historical_paired_results_enriched.csv`, and `evaluation_reconciliation.md`.

Both uploaded evaluation CSVs, including the one named `merged`, contain identical copies of the base run. All 20 complete responses and extracted code blocks match the original base notebook. The recovered CSV supplies actual stdout for the 12 successful base examples. The merged score remains supported by its original notebook; successful merged stdout is still unknown and is never filled from the gold answers or the base CSV.

The metric checks the numeric result of generated Python, not the validity of a complete mathematical proof. Question 15 also has contradictory wording. No model weights were loaded or model-generated code executed by this evidence reconciliation.

Run `python scripts/reconcile_evaluation_files.py` to rebuild the current evaluation reports with portable repository defaults.
