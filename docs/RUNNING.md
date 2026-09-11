# Running the reconstructed experiment

Run commands from the repository root. The primary verified environment is Linux, Python 3.12.14, torch 2.6.0+cpu, Transformers 4.53.2, PEFT 0.16.0 and Accelerate 1.8.1. `requirements-cpu.lock.txt` records the actual installed CPU environment, including transitive packages. This is not an original 2025 Kaggle lock.

## 1. Environment

For CPU audits, Python alone is sufficient. For tokenizer checks and the real tiny-model smoke test:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-model.txt
python tests/smoke_tiny_model.py --report outputs/tiny_model_smoke.json
```

To reproduce the recorded CPU package set exactly, use `python -m pip install -r requirements-cpu.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu` in a clean Python 3.12 environment. Platform and hardware can still affect numerical behavior.

For a **candidate, not yet GPU-tested** Linux CUDA environment, replace the CPU torch installation with the matching CUDA wheel and add bitsandbytes:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

Use one visible GPU. Compatibility depends on its driver and available memory. The historical nonquantized 4B load alone uses roughly 8 GB for float16 parameters; activations, LoRA parameters, gradients and optimizer state add memory. A successful tiny CPU test does not show that a specific Kaggle GPU fits the full run. Do not silently combine two GPUs or reduce the context to claim the same protocol.

The optional `--load-in-4bit` uses a different QLoRA protocol; record it as a new run. It may reduce training memory, but merging reloads a nonquantized base and still needs adequate RAM. This optional branch has not been validated on GPU. [PEFT quantization reference](https://huggingface.co/docs/peft/v0.16.0/en/developer_guides/quantization).

## 2. Fixed inputs and prompts

The bundled `data/prepared/question_disjoint_v1/` is ready to use. It contains 436 candidate training rows, 187 evaluation rows, all source-row dispositions, and content hashes. Training and evaluation reject edited prepared files whose hashes no longer match. To change the data, make a new preparation directory and preserve the old manifest.

```bash
python -m fyp.prepare --output-dir outputs/prepared-check
python -m fyp.tokenize --data-dir outputs/prepared-check --output outputs/new_token_filter.json
```

The tokenizer command downloads only tokenizer/config assets, not 4B weights. It fixes the official revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`, the exact saved custom template, and EOS `<|im_end|>` (ID 151645 for this tokenizer). The expected retained count is **372 = 309 arithmetic + 63 reasoning/code**. The saved report is `reports/new_protocol_token_filter.json`; its local tokenizer path records where the check was run and is not a dependency of your run.

Keep both prompt texts unchanged for a baseline comparison. The arithmetic target is a fenced Python program. The reasoning target concatenates the recorded solution text and its fenced program. The custom generation prefix is `<start_working_out>`; it is preserved even though historical training targets do not consistently follow that tag format. Reference answers are available to the grader only.

For historical candidate reconstruction only:

```bash
python -m fyp.prepare --protocol historical_candidates --output-dir outputs/historical-candidates
python -m fyp.tokenize --data-dir outputs/historical-candidates --output outputs/historical-token-filter.json
```

Expected counts are **560 candidates → 452 retained**. This intentionally retains historical duplicates and known test overlap. The associated evaluation set is only the confirmed first 20 reasoning CSV rows; the original arithmetic 20-row subset is unknown. Do not present this option as a clean evaluation protocol.

## 3. Train, merge and infer

Use the commands below with the explicit immutable base revision. Each output directory/file must be new or empty; existing runs are not overwritten. Training defaults are two epochs, batch 1, accumulation 8, LR `3e-5`, warmup 10, full-text causal loss and no packing. Training writes token membership, actual shuffled IDs, configuration and dependency versions, metrics, adapter and tokenizer files. It does not support resuming an interrupted run in this version. `--max-steps 2` provides a short 4B GPU trial but is not the full experiment.

Merging requires an adapter made by this implementation, plus its completed `reproduction_manifest.json`. It checks the recorded base identifier, revision and semantic config fingerprint before reloading and merging. That fingerprint is not a hash of original Kaggle weights. The original adapter is absent; the command does not pretend to recover it.

Inference uses greedy decoding, a shared explicit template/EOS, at most 1,024 generated tokens and a 2,048-token total budget. It rejects overflow instead of silently truncating. All 187 bundled prompts were checked with the fixed official tokenizer; the longest is 252 tokens. Use the same limits for both models. `--limit 2` is useful for a trial; it produces a clearly recorded subset, not a full benchmark.

`--execution none` is the default. It writes predictions incrementally, then a completion summary. A stopped run has no completed summary and is refused by the deferred scorer. Save the JSONL and both `.meta.json` and `.summary.json` sidecars together. It is expected that unexecuted predictions have `accuracy: null`.

In a Kaggle notebook, unpack/upload this repository, change the working directory to its root, and execute the commands in this guide using `!` or a `%%bash` cell after preparing the Python environment. Keep Internet enabled only when downloading dependencies/model assets is required. Preserve/download `outputs/sft/adapter/`, `outputs/merged/`, manifests and predictions before ending the session. No access to the old Kaggle account is needed for this new experiment. Current GPU availability and allocation are not assumed by this package.

```bash
# No model weights needed for this token-filter check.
python -m fyp.tokenize --data-dir data/prepared/question_disjoint_v1 \
  --output outputs/token_filter.json

# Full Qwen3-4B training requires a suitable GPU; it has not been rerun for this repository.
CUDA_VISIBLE_DEVICES=0 python -m fyp.train \
  --data-dir data/prepared/question_disjoint_v1 \
  --model Qwen/Qwen3-4B-Base \
  --model-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --output-dir outputs/sft

python -m fyp.merge --adapter outputs/sft/adapter \
  --output-dir outputs/merged --device cpu --dtype float16

python -m fyp.evaluate --data-dir data/prepared/question_disjoint_v1 \
  --model Qwen/Qwen3-4B-Base \
  --model-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --output outputs/base_predictions.jsonl
python -m fyp.evaluate --data-dir data/prepared/question_disjoint_v1 \
  --model outputs/merged --local-files-only \
  --output outputs/merged_predictions.jsonl
```

## 4. Isolated scoring and paired comparison

Use Linux or WSL2 with Docker. Native Windows execution is rejected because the bounded pipe reader uses Linux-style selectors. Real Docker execution passed eight integration checks on a GitHub-hosted Linux runner, followed by a successful replay of all 40 saved historical responses. See [cloud validation](CLOUD_VALIDATION.md) for the tested commit and run. Docker was unavailable in the earlier local review; the cloud run supplies the actual execution evidence.

`docker/Dockerfile.eval` installs fixed NumPy, SciPy and SymPy versions. Use exactly the same built image for both runs. The scorer resolves the tag to an image ID and records it. It will not pull an image implicitly. Each generated program gets no network, a read-only filesystem, an unprivileged user, dropped capabilities, CPU/memory/process limits, a temporary writable directory, a timeout and an output-byte limit. It never falls back to executing generated code on the host.

The scorer inspects the first fenced block and accepts it only if labeled Python, `py`, or left unlabeled; it does not skip an earlier block to find a later Python block. A final expression is printed; a function definition is not silently called. Runtime errors, absent code, timeouts and malformed output are scored as failures; Docker/infrastructure failures remain ungraded. Finite numeric output is compared using relative tolerance `1e-3`, absolute tolerance `0`, matching the saved historical evaluator’s tolerance. Zero requires exact numeric agreement. This evaluates output correctness, not the validity of reasoning or proof.

`fyp.compare` requires completed outputs, every row graded, identical question IDs/order/content/references and matching inference/scoring conditions, including recorded model dtype, device type, torch/Transformers versions and seed. It reports counts, accuracy, percentage-point change and improved/regressed question IDs for each category. It does not optimize the split or select a checkpoint using test scores. Keep failed or negative results as well as positive ones.

```bash
docker build -f docker/Dockerfile.eval -t fyp-python-math .
python -m fyp.score --predictions outputs/base_predictions.jsonl \
  --output outputs/base_scored.jsonl --docker-image fyp-python-math
python -m fyp.score --predictions outputs/merged_predictions.jsonl \
  --output outputs/merged_scored.jsonl --docker-image fyp-python-math
python -m fyp.compare --base outputs/base_scored.jsonl \
  --merged outputs/merged_scored.jsonl --output outputs/comparison.json
```

## 5. GitHub layout and release boundary

The archive preserves supplied sources; `fyp/` is the clean executable implementation. Keep bulky model weights and generated `outputs/` outside Git; the supplied `.gitignore` excludes them. When a full run is available, release its adapters/merged weights through an appropriate model artifact host and attach the immutable model revision, run manifest and complete paired predictions. Do not copy a historical score into the new experiment’s result file.

The [public repository](https://github.com/Alex-GUAN-666/Final-Year-Project) passed all three workflow jobs in [run 34580694788](https://github.com/Alex-GUAN-666/Final-Year-Project/actions/runs/34580694788), testing [commit `cfac021`](https://github.com/Alex-GUAN-666/Final-Year-Project/commit/cfac02190ff74c00c8facdb81b863b36c5f1981c). The [cloud report](CLOUD_VALIDATION.md) records 32 unit tests, real tiny-model CPU integration, eight Docker checks and historical program replay. The badge reflects the latest workflow state; the linked run records this specific verification.

No full 4B GPU experiment has been completed. The remaining experimental step is to run the pinned 4B commands on an available GPU, then score the new predictions and retain the resulting comparison regardless of its direction. The recorded cloud success does not verify 4B GPU memory requirements or the optional quantized branch.
