"""Standard-library configuration, data validation, splits, and provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import subprocess
from pathlib import Path


class UnusablePreference(ValueError):
    """A well-formed row that cannot express a chosen/rejected preference."""


class EmptyContent(UnusablePreference):
    """A preference row with an empty prompt or answer."""


def digest(value):
    if isinstance(value, Path):
        h = hashlib.sha256()
        with value.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def path(root, name):
    p = Path(name).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def config(file):
    file = Path(file).resolve()
    cfg = json.loads(file.read_text(encoding="utf-8"))
    root = file.parent.parent
    for section in ("model", "data", "lora", "sft", "dpo", "generation", "paths"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"missing section: {section}")
    if not isinstance(cfg.get("seed"), int) or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    fractions = cfg["data"]["split_fractions"]
    if set(fractions) != {"sft", "dpo", "validation", "test"} or any(not 0 < v < 1 for v in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-8:
        raise ValueError("split fractions must be positive and sum to one")
    for key in ("max_records",):
        if not isinstance(cfg["data"][key], int) or cfg["data"][key] < 4:
            raise ValueError(f"invalid {key}")
    for section in ("sft", "dpo"):
        x = cfg[section]
        if any(x[k] <= 0 for k in ("learning_rate", "epochs", "batch_size", "gradient_accumulation_steps", "max_length")):
            raise ValueError(f"invalid {section} training value")
    if cfg["dpo"]["beta"] <= 0 or cfg["lora"]["r"] <= 0 or cfg["lora"]["alpha"] <= 0 or not 0 <= cfg["lora"]["dropout"] < 1 or not cfg["lora"]["target_modules"]:
        raise ValueError("invalid LoRA/DPO value")
    if cfg["model"]["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("invalid dtype")
    if cfg["generation"]["max_new_tokens"] <= 0 or cfg["generation"]["do_sample"] is not False:
        raise ValueError("evaluation requires deterministic generation")
    for section, keys in {"model": ("base_path",), "data": ("source_path", "prepared_dir"), "paths": ("v1_adapter", "v2_adapter", "runs")}.items():
        for key in keys:
            cfg[section][key] = str(path(root, cfg[section][key]))
    for section, keys in {"model": ("source", "revision"), "data": ("source_name", "source_revision", "source_split")}.items():
        for key in keys:
            if not cfg[section].get(key):
                raise ValueError(f"missing {section}.{key}")
    if len({cfg["model"]["base_path"], cfg["paths"]["v1_adapter"], cfg["paths"]["v2_adapter"]}) != 3:
        raise ValueError("base, v1 and v2 paths must differ")
    return cfg


def validate_row(row):
    pid = row.get("prompt_id")
    if not isinstance(pid, str) or not pid.strip():
        raise ValueError("missing prompt_id")
    for key in ("chosen", "rejected"):
        messages = row.get(key)
        if not isinstance(messages, list) or len(messages) != 2 or [m.get("role") for m in messages if isinstance(m, dict)] != ["user", "assistant"]:
            raise ValueError(f"{pid}: {key} must be user/assistant messages")
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise ValueError(f"{pid}: non-string content")
        if any(not m["content"].strip() for m in messages):
            raise EmptyContent(f"{pid}: empty content")
    if row["chosen"][0]["content"] != row["rejected"][0]["content"]:
        raise ValueError(f"{pid}: chosen/rejected prompts differ")
    if row["chosen"][1]["content"].strip() == row["rejected"][1]["content"].strip():
        raise UnusablePreference(f"{pid}: identical answers")
    return {"prompt_id": pid, "prompt": row["chosen"][0]["content"], "chosen": row["chosen"][1]["content"], "rejected": row["rejected"][1]["content"]}


def source_rows(source):
    if source.is_file() and source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif source.is_dir():
        from datasets import DatasetDict, load_from_disk
        ds = load_from_disk(str(source))
        splits = ds.values() if isinstance(ds, DatasetDict) else [ds]
        for split in splits:
            yield from split
    else:
        raise ValueError("source must be local JSONL or datasets.save_to_disk directory")


def split_rows(rows, seed, max_records, fractions):
    unique, prompts = {}, {}
    counts = {"input": 0, "filtered_empty_content": 0, "filtered_identical_answers": 0, "duplicate_ids": 0, "duplicate_ids_conflicting": 0, "valid_unique": 0, "selected": 0}
    for raw in rows:
        counts["input"] += 1
        try:
            row = validate_row(raw)
        except EmptyContent:
            counts["filtered_empty_content"] += 1
            continue
        except UnusablePreference:
            counts["filtered_identical_answers"] += 1
            continue
        pid, prompt = row["prompt_id"], row["prompt"]
        if pid in unique:
            if unique[pid]["prompt"] != prompt:
                raise ValueError(f"conflicting prompt text for prompt_id: {pid}")
            counts["duplicate_ids"] += 1
            if unique[pid] != row:
                counts["duplicate_ids_conflicting"] += 1
                # Select one preference pair per ID without depending on source order.
                if digest(row) < digest(unique[pid]):
                    unique[pid] = row
            continue
        if prompt in prompts and prompts[prompt] != pid:
            raise ValueError(f"same prompt under multiple IDs: {pid}, {prompts[prompt]}")
        prompts[prompt], unique[pid] = pid, row
    counts["valid_unique"] = len(unique)
    ids = sorted(unique)
    random.Random(seed).shuffle(ids)
    ids = ids[:max_records]
    counts["selected"] = len(ids)
    if len(ids) < 4:
        raise ValueError("need at least four unique prompts")
    n = len(ids)
    cuts = [int(n * fractions[k]) for k in ("sft", "dpo", "validation")]
    cuts = [max(1, x) for x in cuts]
    if sum(cuts) >= n:
        raise ValueError("too few records for four splits")
    a, b, c = cuts
    partitions = {"sft": ids[:a], "dpo": ids[a:a+b], "validation": ids[a+b:a+b+c], "test": ids[a+b+c:]}
    assert sum(map(len, partitions.values())) == n
    return {k: [unique[i] for i in values] for k, values in partitions.items()}, counts


def write_json(pathname, value):
    pathname = Path(pathname)
    pathname.parent.mkdir(parents=True, exist_ok=True)
    pathname.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(pathname, rows):
    pathname = Path(pathname)
    pathname.parent.mkdir(parents=True, exist_ok=True)
    with pathname.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(pathname):
    with Path(pathname).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepared(cfg, split):
    directory = Path(cfg["data"]["prepared_dir"])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(directory / f"{split}.jsonl")
    if digest(rows) != manifest["splits"][split]["rows_sha256"] or [r["prompt_id"] for r in rows] != manifest["splits"][split]["prompt_ids"]:
        raise ValueError(f"prepared {split} data differs from manifest")
    ids = [set(item["prompt_ids"]) for item in manifest["splits"].values()]
    if sum(map(len, ids)) != len(set().union(*ids)):
        raise ValueError("split ID leakage")
    return rows, manifest


def record(cfg, stage, extra=None):
    versions = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "peft", "trl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    gpu = None
    try:
        import torch
        gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    out = {"stage": stage, "config": cfg, "git_commit": commit, "versions": versions, "gpu": gpu, "adapter_paths": {k: cfg["paths"][k] for k in ("v1_adapter", "v2_adapter")}}
    manifest = Path(cfg["data"]["prepared_dir"]) / "manifest.json"
    out["manifest_sha256"] = digest(manifest) if manifest.exists() else None
    out.update(extra or {})
    write_json(Path(cfg["paths"]["runs"]) / f"{stage}.json", out)
