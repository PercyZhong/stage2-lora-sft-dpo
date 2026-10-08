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
python -m experiment.cli sft --config config/local.json
python -m experiment.cli dpo --config config/local.json --dry-run
python -m experiment.cli dpo --config config/local.json
python -m experiment.cli evaluate --config config/local.json --version base
python -m experiment.cli evaluate --config config/local.json --version v1
python -m experiment.cli evaluate --config config/local.json --version v2
python -m experiment.cli summarize --config config/local.json
```

Input: a local `datasets.save_to_disk` directory or JSONL with `prompt_id`, `chosen`, `rejected`; responses are `[{role:"user",content:"..."},{role:"assistant",content:"..."}]`. Preparation repartitions **all** supplied source splits by prompt ID into distinct SFT, DPO, validation and final test partitions. Do not use an original source test split again. The manifest records source metadata/hash, counts, seed, ID lists and SHA256s. Conflicting duplicate IDs and shared prompts across IDs are rejected. All prepared data, adapters, checkpoints and run results stay local.

SFT uses only chosen completions and completion-only loss. DPO uses paired chosen/rejected answers for the same prompt and explicitly loads a frozen v1 reference. The final test prompts and deterministic generation parameters are identical for base/v1/v2. Summary length statistics are descriptive; no quality claim follows from them. Inspect held-out answers or use a separate blinded judge. Training preference accuracy is not final effect.

All model loads use local paths and `local_files_only=True`; Linux can also set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Run records include config, dependency versions, Git SHA, model revision, GPU, manifest and adapter paths. API implementation targets TRL 0.26.0 and PEFT 0.15+ based on [SFT](https://huggingface.co/docs/trl/v0.26.0/en/sft_trainer), [DPO](https://huggingface.co/docs/trl/v0.26.0/en/dpo_trainer) and [PEFT](https://huggingface.co/docs/peft/en/package_reference/peft_model) docs. **待 Linux smoke test**: verify installed APIs, tokenizer template, GPU memory and one tiny train/eval run before the full experiment.
