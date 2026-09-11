# Additional experiment records

The September 11, 2026 upload supplied further Kaggle exports from the original
project. Six notebooks are included to clarify model identity, execution history
and prompt settings. These are historical records, not new model inference.
The [source index](../reports/notebook_recovery_sources.json) records original
names, hashes, dates and publication transformations. All cell sources,
execution counts and plain-text outputs are preserved; widgets and rich display
payloads are removed. Other exports with repeated code or predictions remain in
the supplied archive. The unrelated Jigsaw notebook is outside this project.

## Arithmetic model identity

The original `qwen3-4b-merged-arithmetic-102115527-280966320..ipynb` records
**17/20 (85%)**, but cell 6 selects:

```python
LOAD_BASE_MODEL = True
if LOAD_BASE_MODEL:
    base_model_path = "/kaggle/input/qwen-3/transformers/4b/1"
else:
    base_model_path = "/kaggle/input/qwen3-4b-sfft-merged/transformers/default/1/Qwen3-4B-SFFT-merged"
```

Cell 8 prints `Base model loaded successfully with Unsloth.` and assigns
`model = base_model`. Cell 14 writes `evaluation_results_base_v5.csv` and prints
17 correct predictions. Papermill records a completed sequential run on
November 22, 2025. The questions exactly match the first 20 rows of the bundled
`arithmetic_problems_v3.csv`.

This is a saved result for the **Qwen `4b/1` path**, not the project merged
checkpoint. It also differs from the `4b-base/1` path used for SFT and the
reasoning baseline. The notebook name and the variable name `base_model` do not
establish fine-tuned or pretrained-base model identity.

Three December exports have identical code, questions, responses and flags,
with different execution times and environment logs. They do not add independent
questions. The thesis-reported **14/20 to 17/20** comparison remains unconfirmed:
its paired baseline/merged predictions have not been recovered. This separate
17/20 record does not authenticate the thesis's original comparison or subset.

## Reasoning execution history

| Original export suffix | Selected path | System prompts | Recorded score |
|---|---|---|---|
| `280327485` | `qwen-3/transformers/4b-base/1` | Cleared in an executed cell | 0/20 |
| `280343001` | `qwen-3/transformers/4b-base/1` | Preserved | 12/20 |
| `280392283` | Project merged resource | Cleared in an executed cell | 2/20 |
| `280930433` | Project merged resource | Preserved | 15/20 |

All four are completed Papermill runs and evaluate the first 20 rows of
`random_samples_for_inference.csv` in the same order. The November 20 base
export (`280343001`) contains the same responses and flags as the earlier base
notebook. The November 22 merged export (`280930433`) has the same questions
and flags as the earlier merged notebook, but its response-printing statement
is commented out. The earlier merged notebook remains necessary for complete
responses. These exports improve provenance, not the number of independent
samples. Identical paths do not prove that missing checkpoints contain the same
weights.

The lower-scoring runs retain the executed statements that emptied both system
prompts. The base versions also differ in result-handling code, so they are not
a controlled study of the isolated effect of prompting or fine-tuning.

The original reasoning comparison still has **5/20 questions overlapping the
reconstructed training set**. Its 60% to 75% change is a historical observation,
not an independent held-out generalization result. The score checks numerical
program output, not mathematical proof validity.

## Model recovery lead

The November 19 reference export and later merged reasoning exports load:

```text
/kaggle/input/qwen3-4b-sfft-merged/transformers/default/1/Qwen3-4B-SFFT-merged
```

The reference export's only attached model resource has `modelId=508779`,
`modelInstanceId=493349` and `sourceId=653102` (`modelInstanceVersion`). This is
a more specific recovery lead than the earlier `lora-2-3` input path. It does
not establish current availability or identify the tensors produced by a
particular training run. Original weights, adapters and tokenizer artifacts
remain unavailable.

## Verification

`python scripts/verify_recovered_notebooks.py` checks saved model selection,
loading messages, sequential execution metadata, question identities,
per-question flags, printed totals and the earlier CoT records. It is included
in `python scripts/verify_reconstruction.py` and GitHub Actions.

The committed [evidence summary](../reports/recovered_notebook_evidence.json)
contains static verification results. This reads notebook data without executing
notebook code, generating responses or reproducing 4B training. The earlier
[cloud report](CLOUD_VALIDATION.md) remains a dated snapshot; Actions records
checks of subsequent commits. The [archive index](../archive/README.md#additional-recovered-runs)
maps the six descriptive filenames to the originals.
