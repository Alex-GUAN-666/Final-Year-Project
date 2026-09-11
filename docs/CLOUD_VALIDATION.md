# Cloud validation

This page records the dated cloud-validation snapshot below. Later recovered
notebooks have their own [source and result checks](RECOVERED_NOTEBOOKS.md), included
in subsequent Actions runs. Those historical records are not new 4B inference.

**All three GitHub Actions jobs passed on 2026-09-11** in [run 34580694788](https://github.com/Alex-GUAN-666/Final-Year-Project/actions/runs/34580694788), testing [commit `cfac02190ff74c00c8facdb81b863b36c5f1981c`](https://github.com/Alex-GUAN-666/Final-Year-Project/commit/cfac02190ff74c00c8facdb81b863b36c5f1981c). The [retained verification record](../reports/cloud/2026-09-11/verification.json) records the observed results. This report describes that specific run; the repository badge follows the latest workflow state.

## What passed

| Check | Observed result | Scope |
|---|---|---|
| Source integrity and data preparation | 18 source/publication hashes passed; prepared files reproduced byte for byte | 436 training candidates, 187 evaluation questions, zero normalized train/evaluation question overlap |
| Unit tests | 32 passed | Data preparation, training contracts, evaluation, scoring, paired comparison and container cleanup |
| Tiny Qwen3 CPU integration | Passed; two real optimizer steps | Adapter saving, merging, reloading and inference using a randomly initialized miniature Qwen3 |
| Docker integration | Eight checks passed | Real restricted execution, failure handling, deferred scoring and paired comparison using synthetic fixtures |
| Historical program replay | Passed; exit status 0 | All 40 saved responses graded, including 36 extractable programs and four no-code responses |

No pretrained 4B weights were downloaded or used by these jobs. The tiny-model and synthetic-fixture tests measure software behavior, not mathematical benchmark accuracy.

## Historical results reproduced by execution

| Model | Saved correct | Replayed correct | Individual grade disagreements | Comparable saved-stdout mismatches |
|---|---:|---:|---:|---:|
| Base | 12/20 | 12/20 | 0 | 0 |
| Historical merged | 15/20 | 15/20 | 0 | 0 |

All 40 individual correctness flags matched. The replay first verified each saved response against the archived notebooks and reference questions, then executed its extracted program in Docker. No-code responses remained failures under the documented scorer. Saved historical evidence was unchanged.

Historical successful merged stdout was not preserved in the source notebook. It remains unknown in the historical records; new replay stdout is stored separately. Therefore, zero comparable-stdout mismatches does not claim that all original stdout was available.

This reproduces **execution-output scores for saved responses**. It does not regenerate those responses using the original model or recover its fine-tuned weights. Five of the 20 historical reasoning questions overlap the reconstructed training set. The result is not an independent held-out generalization finding or a formal proof-validity score.

## Issues found and fixed

The first two cloud attempts exposed a Docker cleanup race and a `Path` argument issue in the integration smoke test. The implementation and test invocation were corrected, and five cleanup regression tests were added. The subsequent run linked above passed all three jobs. The fixes did not change historical data, saved model responses or historical scores. The [first-run record](../reports/cloud/2026-09-11/first_run.json) and [second-run record](../reports/cloud/2026-09-11/second_run.json) preserve the earlier outcomes.

## Reproduce the verified revision

Use Python 3.12. Clone and check out the tested revision before running commands from the repository root:

```bash
git clone https://github.com/Alex-GUAN-666/Final-Year-Project.git
cd Final-Year-Project
git checkout cfac02190ff74c00c8facdb81b863b36c5f1981c
python scripts/verify_reconstruction.py --report outputs/reviewer_check.json
python scripts/replay_historical_programs.py --check-inputs
```

These checks require only Python's standard library. They verify sources and saved records without executing generated programs or downloading a model. The tested revision predates this results document; its code and workflow are the ones verified by the linked run.

For actual tiny-model training, use a clean virtual environment and the CPU dependencies used by the workflow:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-model.txt
python -m pip check
python tests/smoke_tiny_model.py --report outputs/tiny_model_smoke.json
```

On a Linux or WSL2 Docker host, reproduce the execution checks and saved-program replay:

```bash
docker build --tag fyp-eval:ci --file docker/Dockerfile.eval .
python tests/smoke_docker.py --image fyp-eval:ci --report outputs/docker_smoke.json
python scripts/replay_historical_programs.py --image fyp-eval:ci --report outputs/historical_replay.json
```

Report paths must be new. The historical replay returns a nonzero exit status if grades or comparable stdout disagree, or if an infrastructure error leaves rows ungraded. Preserve the report when diagnosing a difference. Environment details and the full model commands are in [RUNNING.md](RUNNING.md).

## Remaining model experiment

Full Qwen3-4B GPU training and inference have not been rerun; the original fine-tuned weights remain unavailable. GPU optimizer behavior, memory requirements and the optional 4-bit branch remain unverified. No new 4B accuracy result exists for the 187-question split.

Arithmetic 14/20 → 17/20 remains thesis-reported only: its paired predictions and exact 20-question subset were not recovered. The historical reasoning replay above does not supply missing arithmetic evidence.

Earlier local checks with the fixed official tokenizer reproduced 452 historical training examples and 372 new-split examples. The cloud workflow independently regenerated the prepared candidates and evaluation files, but did not download and rerun that tokenizer. Source, local tokenizer, cloud software and historical replay evidence are recorded separately.
