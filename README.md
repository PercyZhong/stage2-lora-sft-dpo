# Stage 2: LoRA SFT / DPO

Lineage: Qwen/Qwen2.5-0.5B base → v1 LoRA-SFT → v2 LoRA-DPO. DPO's frozen reference is v1, loaded separately. v2 is trained on v1 merged into memory and stores only a new adapter; loading v2 repeats the v1 merge.

## Windows → GitHub → Linux

Edit, inspect, test, commit and push on Windows. Do not download models/data or install training packages here. Clone on a Linux Python 3.11 GPU host and install `requirements-linux.txt`.

On Linux with network access, `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE` and run `python scripts/fetch_assets.py`. This downloads a pinned base model snapshot, saves only the dataset's paired `train_prefs` split with `save_to_disk`, and writes ignored `config/local.json` with actual local paths and repository commit SHAs. The other dataset splits are not input to this experiment. The tracked template remains unchanged. After downloading, set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for training and evaluation. Then:

```bash
python -m experiment.cli check --config config/local.json
python -m experiment.cli prepare --config config/local.json --dry-run
python -m experiment.cli prepare --config config/local.json
python -m experiment.cli sft --config config/local.json --dry-run
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python -m experiment.cli sft --config config/local.json
python -m experiment.cli dpo --config config/local.json --dry-run
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python -m experiment.cli dpo --config config/local.json
CUDA_VISIBLE_DEVICES=0 python -m experiment.cli evaluate --config config/local.json --version base
CUDA_VISIBLE_DEVICES=0 python -m experiment.cli evaluate --config config/local.json --version v1
CUDA_VISIBLE_DEVICES=0 python -m experiment.cli evaluate --config config/local.json --version v2
python -m experiment.cli summarize --config config/local.json
```

On a shared server, replace `0` with the GPU assigned to you. These environment variables affect only the launched process; do not change drivers, CUDA, system services or other users' processes. Keeping one RTX 40-series GPU visible also avoids unsupported NCCL P2P/IB initialization across multiple consumer GPUs.

Input: a local `datasets.save_to_disk` directory or JSONL with `prompt_id`, `chosen`, `rejected`; responses are `[{role:"user",content:"..."},{role:"assistant",content:"..."}]`. Preparation repartitions **all** supplied source splits by prompt ID into distinct SFT, DPO, validation and final test partitions. Do not use an original source test split again. The manifest records source metadata/hash, counts, seed, ID lists and SHA256s. For alternate preference pairs with the same ID and prompt, it retains the pair with the lowest row SHA256 independent of source order and counts the resolved duplicates. Conflicting prompt text for one ID and shared prompts across IDs are rejected. All prepared data, adapters, checkpoints and run results stay local.
Rows with empty prompt/answer content or identical chosen and rejected answers are excluded and counted as `filtered_empty_content` or `filtered_identical_answers` in the dry-run output and manifest; malformed conversations still raise a validation error.

SFT uses only chosen completions and completion-only loss. DPO uses paired chosen/rejected answers for the same prompt and explicitly loads a frozen v1 reference. The final test prompts and deterministic generation parameters are identical for base/v1/v2. Summary length statistics are descriptive; no quality claim follows from them. Inspect held-out answers or use a separate blinded judge. Training preference accuracy is not final effect.

All model loads use local paths and `local_files_only=True`; Linux can also set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Run records include config, dependency versions, Git SHA, model revision, GPU, manifest and adapter paths. API implementation targets TRL 0.26.0 and PEFT 0.15+ based on [SFT](https://huggingface.co/docs/trl/v0.26.0/en/sft_trainer), [DPO](https://huggingface.co/docs/trl/v0.26.0/en/dpo_trainer) and [PEFT](https://huggingface.co/docs/peft/en/package_reference/peft_model) docs.
The training entry points follow Accelerate's launcher behavior for GPU setups without NCCL P2P/IB support: they set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` before initializing the trainer, and record both values in run metadata.

## Linux verification

Verified on 2026-10-08 with Python 3.11, one RTX 4090, PyTorch 2.5.1+cu121, Transformers 4.57.6, TRL 0.26.0, PEFT 0.21.2 and Accelerate 1.15.0. Both a 40-row smoke run and the configured 4,000-row experiment completed offline. The full split contained 1,600 SFT, 1,600 DPO, 400 validation and 400 final-test prompts. SFT completed 100 steps, DPO completed 200 steps, and base/v1/v2 evaluation used the same 400 test IDs with SHA256 `2937bfed632b4afe805acbb8f64f8a3570ea23c2bc4cc6c58c0e700000ccb344`.

The observed mean generated lengths were 112.89 tokens for base, 119.78 for v1 and 119.39 for v2. These are descriptive pipeline outputs, not model-quality scores. Local adapters, per-sample generations, run records and `summary.csv` remain ignored by Git.

## Supplemental quality evaluation

This workflow is read-only with respect to the prepared splits, original `runs/eval_*` files and v1/v2 adapters. It verifies their hashes before and after every stage and writes new artifacts only under ignored `runs/quality_eval/`. Run it on Linux with the existing offline model, data and adapters; do not retrain:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
python -m experiment.quality_eval check --config config/local.json
python -m experiment.quality_eval preference --config config/local.json
python -m experiment.quality_eval blind-pack --config config/local.json --seed 20261008 --sample-size 60 --max-new-tokens 512
```

On a shared host, replace GPU `0` with the single GPU assigned to you. A command refuses to overwrite its own existing output. After an interrupted run, repeat that command with `--resume`; resume succeeds only when the config and protected input hashes still match.

`preference_rows.csv` records chosen/rejected assistant-token counts, summed log probabilities, per-answer-token means, `raw_margin` and `mean_margin` for base/v1/v2. Prompt and padding tokens are excluded. Chat-template assistant terminators and EOS are included consistently. Inputs over the 1024-token training context are excluded as a whole prompt pair across all versions. `preference_summary.json` and `.csv` report positive, zero and mean margins with valid and excluded N. Summed margins have answer-length bias; normalized margins still measure preference-label likelihood rather than complete response quality.

The blind pack uniformly samples 60 IDs from the verified 400-ID test order with `random.Random(20261008).sample`, then regenerates each prompt for all three versions with deterministic decoding and `max_new_tokens=512`. Version outputs go to `generated_512/`. `blind_sheet.csv` exposes only randomized A/B/C responses; `blind_key.json` contains the separate mapping. Do not open the key before scoring.

For every row, fill `A_vs_B` with `A`, `B`, or `tie`; `A_vs_C` with `A`, `C`, or `tie`; and `B_vs_C` with `B`, `C`, or `tie`. Add a nonempty `reason` based on instruction following, correctness, coherence, repetition and unsupported claims. Do not edit IDs, prompts or answers. Incomplete or illegal scores are rejected. After all 60 rows are scored:

```bash
python -m experiment.quality_eval blind-summarize --config config/local.json --scores runs/quality_eval/blind_sheet.csv
python -m experiment.quality_eval report --config config/local.json
```

The blind summary reports base↔v1, v1↔v2 and base↔v2 wins/ties/losses plus a 95% Wilson interval for the first model's win fraction among non-tied paired judgments. This is one rater, one pass, 60 internally held-out prompts, and deterministic decoding; even 512 tokens can truncate. The test set comes from a prompt-disjoint internal split of UltraFeedback Binarized `train_prefs`, not an external benchmark. Preference accuracy, generated length and this small blind sample must not be presented as broad proof of model quality.

For a three-model-judge variant, keep `blind_key.json` hidden until ChatGPT Work, DeepSeek and Qwen have each completed all 60 anonymous rows. `scripts/api_blind_judge.py` produces validated DeepSeek or Qwen score sheets using API keys from environment variables and supports `--resume`. After all three sheets are fixed, copy the blind key locally and run `python scripts/summarize_multi_judge.py`. It reports every judge separately, two-of-three majority outcomes, no-consensus cases, pairwise judge agreement and Fleiss κ under `runs/quality_eval/multi_judge/`. These judges are not independent humans; they may share data and style biases, and the Qwen judge can have model-family bias.

## Stage 2 evidence archive

The tracked `scripts/archive_stage2.py` script is read-only with respect to experiment inputs. On Linux, after pulling this commit and ensuring the existing prepared test, original evaluation JSONL files, and quality-evaluation `state.json` are present, generate the final archive with:

```bash
python scripts/archive_stage2.py --root . --output stage2_phase2_evidence.zip
```

It verifies each included file against `state.json` before writing the ZIP. The archive contains `data/prepared/test.jsonl`, `runs/eval_base.jsonl`, `runs/eval_v1.jsonl`, `runs/eval_v2.jsonl`, `runs/quality_eval/state.json`, and `stage2_phase2_report.md`; it does not include model weights, the original complete dataset, or credentials. The report records the DeepSeek/Qwen metadata blind-sheet hash mismatch as an unresolved byte-level provenance difference when the original bytes cannot be found. Do not claim this command was run on Linux unless it was actually executed there.
