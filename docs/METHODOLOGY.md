# Methodology and how to verify the results

This document connects the recovered research materials to the executable implementation. The project studies whether supervised fine-tuning of **Qwen3-4B-Base** on mathematical explanations and executable Python can improve numerical problem solving. The final project is credited to **Yuzhen Guan and Weizhou Lu**. The September 2026 implementation is a documented reconstruction of their November 2025 work.

中文导读：本页说明数据从哪里来、教师实际做了什么、训练如何计算损失，以及评测怎样判分。历史分数可核对到保存的记录，但原始微调权重没有找回，不能保证重新训练必然得到相同分数。新的去重划分用于重新实验，其结果必须通过真实运行产生。

## 1. The research method and the recovered implementation

The thesis describes a multi-teacher framework. The supplied implementation supports a more specific account: an existing mathematical solution is translated into Python by a code teacher; a numerical checker selects usable examples; the student learns the resulting text/code sequences through SFT. A separate synthetic arithmetic corpus supplies shorter code-only examples.

| Component | What the supplied artifacts establish |
|---|---|
| Mathematical solution source | The generator reads `problem`, `generated_solution`, and `expected_answer` from a local Arrow dataset. It copies the existing explanation into the output. The Arrow basename and a separate notebook suggest OpenMathReasoning-mini, but its precise upstream export, revision, and per-record teacher identities are unavailable. |
| Code teacher | `archive/scripts/batch_generate_and_verify_v3_1.py` calls Ollama with alias `qwen-coder-32b-fp16:latest` and temperature `0.0`. The local SFT notebook describes Qwen2.5-Coder-32B. The alias is not an immutable weight revision. |
| Teacher input | Both classification and code generation receive the question **and the existing complete solution**. This is solution-conditioned code distillation. The teacher is not solving an unseen question without access to a solution. |
| Selection | The generator classifies problems, attempts code generation/execution for computational examples, and records an answer-check flag. Subsequent training selects records with truthy `is_correct`, `generated_solution`, and `generated_python_code`. |
| Student | The frozen pretrained Qwen3-4B-Base receives LoRA adapters and learns the selected serialized sequences using next-token cross-entropy. |
| Inference | The student receives only a system instruction and question. Arithmetic uses code-only instructions; general mathematical reasoning uses explanation-plus-code instructions. |
| Scoring | Extract one generated Python block, execute it, and compare its finite numeric output to the supplied reference. Explanation/proof validity is not scored. |

This is **response/sequence distillation through SFT**. The recovered SFT code does not collect teacher logits or implement a KL-divergence distillation term, temperature-weighted soft labels, or simultaneous teacher ensembling. Although the thesis mentions multiple teacher families, the materials do not establish a complete, versioned roster of successful teacher runs or their sample contributions.

| Phase | Method and evidence boundary |
|---|---|
| June 2025 exploratory work | The earlier DOCX/notebooks explore format SFT followed by GRPO and a separate embedding-based semantic router. An integrated mixture-of-experts system is not established. |
| November 2025 main project | The later thesis and SFT notebooks describe data distillation, LoRA SFT, merging, and executable-answer evaluation. The thesis explicitly excludes GRPO from this final training pipeline. |
| September 2026 reconstruction | The `fyp/` implementation repairs packaging, records data/protocol provenance, introduces a question-disjoint split, and supports controlled new experiments. It does not recover missing historical weights. |

The earlier GRPO and semantic-router explorations are not stages of the final SFT pipeline or explanations for its reported accuracy changes. The new implementation selects prompts from each dataset's recorded category; it does not claim a learned routing model. Detailed source references are in [recovered-data provenance](../reports/RECOVERED_DATA_PROVENANCE.md) and the [thesis audit](THESIS_AUDIT.md).

## 2. Data and selection

The bundled snapshots are the reproducible starting point. Recreating all raw material from the historical generator is not possible from the supplied files alone: the upstream Arrow snapshot and pinned teacher weights are absent, and the arithmetic generator scripts do not account for every revision leading to the 309-row file.

| Input in `data/raw/` | Rows | Role |
|---|---:|---|
| `results_v3.jsonl` | 1,125 | Recovered questions, existing explanations, generated code, references, and saved check flags |
| `arithmetic_logic_data_revised_v2.jsonl` | 309 | Synthetic arithmetic training snapshot |
| `arithmetic_problems_v3.csv` | 100 | Arithmetic evaluation question pool |
| `random_samples_for_inference.csv` | 100 | General mathematical evaluation question pool; 89 normalized unique questions |

The distilled pool has **251 true, 58 false, and 816 null** correctness flags. A null flag includes skipped or unexecuted examples; it does not mean an independently verified incorrect answer. The historical selection predicate yields **251 candidates**, all with parseable Python, representing 216 normalized unique questions. The arithmetic selection takes all 309 rows without checking their correctness flags. These are preserved historical selection rules, not a claim that every reference or explanation has been mathematically certified.

The generator's answer checker and model evaluator are different components: the generator accepts exact strings or a SymPy-based numeric comparison with absolute error `< 1e-9`; the saved model evaluator uses relative tolerance `1e-3`. A saved training flag is therefore not a model benchmark result. The archived generator executes code in its own process and is retained for evidence, not used by the new execution workflow.

### Historical candidate reconstruction

The historical notebooks form system/user/assistant messages, filter the serialized example to at most 2,048 tokens, combine arithmetic with eligible distilled examples, and shuffle with seed 42. They contain no explicit evaluation-question exclusion or question deduplication.

Using the supplied prompts/template and official tokenizer revision `906bfd4b4dc7f14ee4320094d8b41684abff8539` gives:

| Stage | Arithmetic | Reasoning/code | Total |
|---|---:|---:|---:|
| Candidates | 309 | 251 | 560 |
| Retained after token filtering | 309 | 143 | 452 |

The 143 reasoning/code rows contain 112 normalized unique questions. These counts agree with the saved training notebook. That agreement supports the reconstruction; it does not authenticate the missing original Kaggle tokenizer files, original weights, or every historical training detail.

The historical reasoning evaluation uses the first 20 rows of `random_samples_for_inference.csv`. Under the documented tokenizer reconstruction, **rows 5, 12, 13, 14, and 20 overlap retained training questions**. All 100 reasoning evaluation rows match recovered distillation records in question, answer, reference solution, and reference code. Thus the historical reasoning set is derived from the same source pool and cannot support an independent held-out generalization claim.

The recovered arithmetic runs use the first 20 rows of `arithmetic_problems_v3.csv`, but load the Qwen `4b/1` resource rather than the project merged model or the training base at `4b-base/1`. Their question identities therefore do not resolve the thesis's missing paired arithmetic comparison.

### New `question_disjoint_v1` experiment

`fyp.prepare` creates a fixed evaluation set and new training selection:

1. Normalize question strings with Unicode NFKC, the documented `$Calculate#` replacement, whitespace collapse, and case folding; hash the normalized strings for question identity.
2. Exclude **all 189 unique questions** in the two source evaluation pools from training, including evaluation duplicates and questions later quarantined.
3. Apply the historical eligibility predicate, then keep the first occurrence of each remaining normalized training question. This excludes 114 eligible distilled rows for evaluation overlap and 10 further duplicate rows: **251 − 114 − 10 = 127** reasoning/code candidates.
4. Deduplicate evaluation questions and quarantine general-math rows 15 and 43 because their wording/reference relationship is inconsistent or underdetermined. This leaves **100 arithmetic + 87 reasoning = 187** evaluation questions.
5. Apply the fixed tokenizer/template length filter to training candidates. No overlength training example is silently truncated.

| Stage | Arithmetic | Reasoning/code | Total |
|---|---:|---:|---:|
| Training candidates | 309 | 127 | 436 |
| Retained at 2,048 tokens | 309 | 63 | 372 |
| Evaluation questions | 100 | 87 | 187 |

Normalized training/evaluation overlap is zero. This is a new split of recovered material, not a newly collected benchmark. String-based deduplication does not exclude semantic duplicates or pretraining exposure, and all 87 retained reasoning references have not received independent mathematical proof verification. Historical results are not reassigned to this new denominator. The [prepared manifest](../data/prepared/question_disjoint_v1/manifest.json) records source hashes, all exclusions, counts, and prepared-file hashes.

## 3. Prompts, loss, and LoRA

The exact system instructions are [`arithmetic_code_only.txt`](../prompts/arithmetic_code_only.txt) and [`reasoning_cot_code.txt`](../prompts/reasoning_cot_code.txt). For training, the assistant target is a fenced Python program for arithmetic, or the saved explanation followed by a fenced Python program for reasoning. Targets are not rewritten during preparation.

The supplied [custom chat template](../prompts/historical_chat_template.jinja) concatenates system text plus EOS, user text, and assistant text plus EOS. For generation it appends `<start_working_out>`. That prefix is preserved even though the training targets do not consistently follow the corresponding tag convention. EOS and padding are explicitly `<|im_end|>`; its ID is 151645 for the pinned official tokenizer. Replacing the template, fixing its style, or changing the prefix defines a different experiment.

The new implementation minimizes full-text causal cross-entropy: each nonpadding token after the first is predicted from preceding tokens in its serialized example. System, user, and assistant tokens participate in the loss; this is not assistant-only training. Padding labels are set to `-100` **by position**, so genuine EOS tokens remain supervised even though EOS and padding share the same ID. Examples are dynamically padded within a batch, not packed together.

For an adapted projection, the effective weight is `W + (alpha / r) B A`; the pretrained weight is frozen and the low-rank adapter parameters are trainable. `fyp.merge` incorporates the learned adapter into a reloaded base model and saves a standalone checkpoint.

| Setting | New default / evidence boundary |
|---|---|
| Base | `Qwen/Qwen3-4B-Base`, explicitly pin revision `906bfd4b4dc7f14ee4320094d8b41684abff8539` in commands |
| Precision / quantization | float16; no 4-bit model loading. Optional `--load-in-4bit` is a separate QLoRA variant. |
| LoRA | Rank 32, alpha 64, dropout 0, no bias adaptation |
| Target projections | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs / batch | 2 epochs; one visible GPU; microbatch 1; gradient accumulation 8; nominal effective batch 8 |
| Optimization | AdamW 8-bit; learning rate `3e-5`; weight decay `0.01`; gradient norm limit `1.0` |
| Schedule / seeds | Linear schedule; 10 warmup steps; training seed 3407; initial dataset permutation uses `numpy.default_rng(42)` |
| Sequence policy | Complete serialized examples of 2–2,048 tokens; full-text loss; no packing |
| Runtime | Transformers 4.53.2 with PEFT; gradient checkpointing enabled; adapter and run manifest saved after completion |

The historical notebook's explicit rank, learning rate, batch, accumulation, epochs, optimizer, and nonquantized float16 load support these choices. Its saved run reports 452 examples, 114 optimizer steps, and 66,060,288 trainable parameters. Those are **historical observations**, not promised values for the 372-example new split.

The paper's learning rate `2e-4`, batch size 64, and some QLoRA descriptions conflict with the actual notebook. For reconstructing code behavior, the notebook is the stronger evidence. However, its patched Unsloth/TRL trainer did not explicitly specify every effective truncation/loss default. The new Transformers/PEFT implementation therefore records these settings rather than claiming a bit-identical historical replay. See [historical configuration](../reports/historical_training_config_v2.json).

## 4. Evaluation and the meaning of a correct answer

Base and newly merged models must receive identical prepared questions, prompts, template, EOS, token budgets, and runtime conditions. Reference answers, source solutions, and source reference programs are excluded from inference messages.

The new defaults are greedy decoding (`do_sample=False`, one beam), seed 3407, at most 1,024 generated tokens, and a 2,048-token total budget. A prompt that cannot fit with its completion allowance causes an error; it is not silently truncated. Inference saves complete raw responses and token counts before any optional execution.

Scoring examines the **first fenced block** and accepts it only when its language is Python, `py`, or unspecified. It does not skip a first block in another language to search for a later Python answer. A final expression is converted into a print; an uncalled function is not automatically invoked. The program runs in a restricted Docker container. The default timeout is 10 seconds and combined captured-output limit is 65,536 bytes. Both models must use the same resolved container image and execution limits. There is no host-code-execution fallback.

Finite scalar output is compared with the finite reference using `math.isclose(observed, expected, rel_tol=1e-3, abs_tol=0.0)`. Zero requires exact numeric agreement. Missing code, program errors, timeouts, nonnumeric output, and mismatches are scored as failures. Infrastructure failures remain ungraded; `fyp.compare` refuses a final paired comparison with any ungraded row. Unexecuted inference results intentionally have accuracy `null`.

For a completed, fully graded run, accuracy is `number correct / number evaluated`. The comparison reports category counts, accuracy, the **percentage-point** change, and IDs that improved or regressed. It also rejects mismatched data, question order/content, prompt/template/tokenizer hashes, generation settings, numerical runtime, and scoring conditions.

This metric is end-to-end **numeric execution-output accuracy**. A correct printed number does not validate every explanation step, constitute a formal proof, or show that the model computed the answer without external Python.

## 5. What results can actually be checked

| Claim | Evidence and permitted interpretation |
|---|---|
| Arithmetic 14/20 → 17/20 (70% → 85%) | Reported in the November thesis. Four recovered notebooks record 17/20 on the first 20 arithmetic CSV rows, but select `qwen-3/transformers/4b/1`, not the project merged model or `4b-base/1`. The thesis comparison still lacks authenticated paired predictions and subset identities. |
| Reasoning 12/20 → 15/20 (60% → 75%) | The original notebooks contain aligned saved responses/flags, supported by recovered sequential execution logs. The new merged export omits full responses, so the earlier notebook remains necessary. Five questions overlap reconstructed training. |
| Two evaluation CSVs named base/merged | They are byte-identical and both match the **base** notebook. The file called `merged` does not supply an independent merged run. |
| Merged successful stdout | Not printed in the archived notebook for its 15 successful cases; those values remain unknown, rather than being filled with references. Five failed outputs are recovered. |
| New split accuracy | No completed full 4B training and paired new-split benchmark is supplied. It must be produced by running the new experiment. |
| Tiny-model smoke test | Actual small CPU training, merge, reload, and inference test software integration. A randomly initialized miniature Qwen3 does not measure 4B mathematical accuracy. |

The thesis reports **+15 percentage points** in each category; the arithmetic comparison remains unverified. The reasoning logs support a small-sample observation with the above provenance limits. There is no single-teacher ablation, multiple-seed result, formal-proof checker, or evidence that a specific component alone caused the improvement. A new run may obtain different scores, including no improvement; preserve the complete results either way.

The [recovered notebook review](RECOVERED_NOTEBOOKS.md) also preserves early base 0/20 and merged 2/20 runs in which both system prompts were cleared. Later runs restore the prompts, and the base result-processing code also changes. These are different evaluation settings, not additional same-condition trials. Four arithmetic exports repeat identical questions, answers and correctness flags; they represent 20 questions, not 80 independent samples.

## 6. Reproduce the evidence, then run a new experiment

Run commands from the repository root. Python alone is sufficient for the first block; it reads saved sources and never executes archived model-generated programs:

```bash
python scripts/audit_artifacts.py --output outputs/artifact-audit.json
python scripts/audit_data.py
python scripts/audit_recovered_distillation.py
python scripts/reconcile_evaluation_files.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

The reconciliation report identifies the duplicate CSVs and saved notebook scores; the recovered-data audit checks selection evidence and overlap. These commands verify artifacts and recorded observations, not a new model run. For complete historical responses and explicitly unknown stdout, inspect [`cot_historical_results_enriched.json`](../reports/cot_historical_results_enriched.json).

Install the pinned environment in [RUNNING.md](RUNNING.md) before tokenizer/model commands. The tokenizer checks download tokenizer assets only; use new output directories:

```bash
python -m fyp.prepare --protocol historical_candidates --output-dir outputs/historical-candidates
python -m fyp.tokenize --data-dir outputs/historical-candidates --output outputs/historical-token-filter.json

python -m fyp.prepare --output-dir outputs/new-prepared
python -m fyp.tokenize --data-dir outputs/new-prepared --output outputs/new-token-filter.json
python tests/smoke_tiny_model.py --report outputs/tiny-model-smoke.json
```

Expect **560 → 452** retained examples for historical candidates and **436 → 372** for the new split with the fixed tokenizer/template. These count checks can match without requiring original fine-tuned weights.

For an actual new 4B experiment, use an appropriately provisioned GPU environment and keep the immutable base revision:

```bash
CUDA_VISIBLE_DEVICES=0 python -m fyp.train \
  --data-dir outputs/new-prepared \
  --model Qwen/Qwen3-4B-Base \
  --model-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --output-dir outputs/new-sft

python -m fyp.merge --adapter outputs/new-sft/adapter \
  --output-dir outputs/new-merged --device cpu --dtype float16

python -m fyp.evaluate --data-dir outputs/new-prepared \
  --model Qwen/Qwen3-4B-Base \
  --model-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --output outputs/new-base.jsonl

python -m fyp.evaluate --data-dir outputs/new-prepared \
  --model outputs/new-merged --local-files-only \
  --output outputs/new-merged.jsonl
```

Then, on Linux or WSL2 with Docker:

```bash
docker build -f docker/Dockerfile.eval -t fyp-python-math .
python -m fyp.score --predictions outputs/new-base.jsonl \
  --output outputs/new-base-scored.jsonl --docker-image fyp-python-math
python -m fyp.score --predictions outputs/new-merged.jsonl \
  --output outputs/new-merged-scored.jsonl --docker-image fyp-python-math
python -m fyp.compare --base outputs/new-base-scored.jsonl \
  --merged outputs/new-merged-scored.jsonl --output outputs/new-comparison.json
```

When moving predictions between machines, include each `.jsonl` file and both `.jsonl.meta.json` and `.jsonl.summary.json` sidecars. Output paths must be new; the workflow refuses to overwrite an existing run. The full 4B GPU workflow is not established by the CPU smoke test; consult the current [reproduction status](REPRODUCTION_STATUS.md) and GitHub Actions logs for the scope actually executed.

For a result another person can verify, retain the exact repository commit, data manifests, token-filter membership, completed training manifest, adapter/tokenizer, merged checkpoint, complete paired predictions and their sidecars, scoring image identity, and final comparison. Publish an immutable model-artifact location when a real run exists. Matching historical counts or reading an old score is not a substitute for these new-run artifacts, and original historical accuracy cannot be guaranteed without the missing original checkpoints and evaluation evidence.
