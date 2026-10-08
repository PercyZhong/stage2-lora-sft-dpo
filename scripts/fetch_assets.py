"""Download pinned Hugging Face assets on Linux and write a local config.

Run from any directory with: python scripts/fetch_assets.py
Only the train_prefs split has the paired fields required by this experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B"
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-template",
        type=Path,
        default=ROOT / "config/experiment.json",
        help="tracked experiment template (default: config/experiment.json)",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=ROOT / "config/local.json",
        help="local config written with pinned revisions (default: config/local.json)",
    )
    args = parser.parse_args()

    if sys.platform != "linux":
        parser.error("download this model and dataset on Linux, not Windows")
    if any(os.environ.get(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        parser.error("unset HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE before downloading")
    template = args.config_template.resolve()
    output = args.output_config.resolve()
    if template == output:
        parser.error("output config must differ from the tracked template")
    if output.exists():
        parser.error(f"output config already exists: {output}")
    if not template.is_file():
        parser.error(f"config template not found: {template}")

    # Imports happen after argument and platform checks, so --help works on Windows
    # without installing training dependencies.
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    model_sha = api.model_info(MODEL_ID).sha
    dataset_sha = api.dataset_info(DATASET_ID).sha
    if not model_sha or not dataset_sha:
        raise RuntimeError("could not resolve immutable model and dataset revisions")

    model_dir = ROOT / "models/Qwen2.5-0.5B" / model_sha
    data_dir = ROOT / "data/raw/ultrafeedback_binarized_train_prefs" / dataset_sha
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, revision=model_sha, local_dir=str(model_dir))

    if data_dir.exists():
        dataset = load_from_disk(str(data_dir))
    else:
        dataset = load_dataset(DATASET_ID, revision=dataset_sha, split="train_prefs")
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(data_dir))
    required = {"prompt_id", "chosen", "rejected"}
    if not required <= set(dataset.column_names):
        raise ValueError(f"train_prefs is missing columns: {sorted(required - set(dataset.column_names))}")

    cfg = json.loads(template.read_text(encoding="utf-8"))
    cfg["model"].update(base_path=str(model_dir), source=MODEL_ID, revision=model_sha)
    cfg["data"].update(
        source_path=str(data_dir),
        source_name=DATASET_ID,
        source_revision=dataset_sha,
        source_split="train_prefs",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Model snapshot: {model_dir}\nModel revision: {model_sha}")
    print(f"Preference dataset: {data_dir}\nDataset revision: {dataset_sha}")
    print(f"Preference rows: {len(dataset)}\nLocal config: {output}")


if __name__ == "__main__":
    main()
