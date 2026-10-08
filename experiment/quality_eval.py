"""Offline held-out preference scoring and blinded long-generation evaluation."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .cli import adapter_digest, local_guard, prompt_text, tokenizer_and_model
from .core import config, digest, prepared, read_jsonl, write_json


VERSIONS = ("base", "v1", "v2")
EXPECTED_TEST_IDS_SHA256 = "2937bfed632b4afe805acbb8f64f8a3570ea23c2bc4cc6c58c0e700000ccb344"
TRAINING_COMMIT = "4119b6ed667b0dba14d07d8c7db335d2a8bdb022"
CONTEXT_LIMIT = 1024
STATE_SCHEMA = 1


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def quality_dir(cfg):
    return Path(cfg["paths"]["runs"]) / "quality_eval"


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tree_digest(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    files = [(str(p.relative_to(folder)), digest(p)) for p in folder.rglob("*") if p.is_file()]
    return digest(sorted(files))


def relevant_config(cfg):
    return {
        "seed": cfg["seed"],
        "model": {k: cfg["model"][k] for k in ("source", "revision", "dtype")},
        "data": {
            k: cfg["data"][k]
            for k in ("source_name", "source_revision", "source_split", "max_records", "split_fractions")
        },
        "lora": cfg["lora"],
        "sft": cfg["sft"],
        "dpo": cfg["dpo"],
        "generation": cfg["generation"],
    }


def protected_paths(cfg):
    runs = Path(cfg["paths"]["runs"])
    prepared_dir = Path(cfg["data"]["prepared_dir"])
    paths = {
        "prepared_manifest": prepared_dir / "manifest.json",
        "prepared_test": prepared_dir / "test.jsonl",
        "sft_record": runs / "sft.json",
        "dpo_record": runs / "dpo.json",
        "old_summary": runs / "summary.csv",
    }
    for version in VERSIONS:
        paths[f"eval_{version}_record"] = runs / f"eval_{version}.json"
        paths[f"eval_{version}_rows"] = runs / f"eval_{version}.jsonl"
    return paths


def protected_hashes(cfg):
    result = {}
    for name, pathname in protected_paths(cfg).items():
        if not pathname.is_file():
            raise FileNotFoundError(f"missing required original input: {pathname}")
        result[name] = digest(pathname)
    result["v1_adapter_tree"] = tree_digest(cfg["paths"]["v1_adapter"])
    result["v2_adapter_tree"] = tree_digest(cfg["paths"]["v2_adapter"])
    result["base_model_tree"] = tree_digest(cfg["model"]["base_path"])
    return result


def _assert_record_config(record, cfg, stage):
    recorded = record.get("config")
    if not isinstance(recorded, dict):
        raise ValueError(f"runs/{stage}.json has no recorded config")
    checks = (
        (recorded.get("seed"), cfg["seed"], "seed"),
        (recorded.get("model", {}).get("source"), cfg["model"]["source"], "model source"),
        (recorded.get("model", {}).get("revision"), cfg["model"]["revision"], "model revision"),
        (recorded.get("data", {}).get("source_name"), cfg["data"]["source_name"], "data source"),
        (recorded.get("data", {}).get("source_revision"), cfg["data"]["source_revision"], "data revision"),
        (recorded.get(stage), cfg[stage], f"{stage} config"),
        (recorded.get("lora"), cfg["lora"], "LoRA config"),
        (record.get("adapter_paths"), {key: cfg["paths"][key] for key in ("v1_adapter", "v2_adapter")}, "adapter paths"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"{stage} record {label} differs from current config")


def validate_test_identity(test_rows, manifest, expected_hash=EXPECTED_TEST_IDS_SHA256):
    test_ids = [row["prompt_id"] for row in test_rows]
    if len(test_rows) != 400 or len(set(test_ids)) != 400:
        raise ValueError("quality evaluation requires exactly 400 unique test IDs")
    test_manifest = manifest.get("splits", {}).get("test", {})
    if test_manifest.get("count") != 400:
        raise ValueError("manifest test count is not 400")
    if test_manifest.get("ids_sha256") != expected_hash or digest(test_ids) != expected_hash:
        raise ValueError("test ID order/hash differs from the verified experiment")
    if test_manifest.get("prompt_ids") != test_ids or test_manifest.get("rows_sha256") != digest(test_rows):
        raise ValueError("test rows differ from the manifest")
    return test_ids


def validate_eval_rows(test_rows, eval_rows, version):
    if len(eval_rows) != len(test_rows):
        raise ValueError(f"eval_{version}.jsonl does not contain the expected rows")
    if [row.get("prompt_id") for row in eval_rows] != [row["prompt_id"] for row in test_rows]:
        raise ValueError(f"eval_{version} test ID/order mismatch")
    if [row.get("prompt") for row in eval_rows] != [row["prompt"] for row in test_rows]:
        raise ValueError(f"eval_{version} prompt mismatch")


def verify_inputs(cfg):
    """Validate all original data, generations, training records, and lineage."""
    local_guard(cfg, ("v1_adapter", "v2_adapter"))
    test_rows, manifest = prepared(cfg, "test")
    test_ids = validate_test_identity(test_rows, manifest)

    runs = Path(cfg["paths"]["runs"])
    expected_signature = None
    manifest_hash = digest(Path(cfg["data"]["prepared_dir"]) / "manifest.json")
    eval_hashes = {}
    old_limit_hits = {}
    for version in VERSIONS:
        meta_path = runs / f"eval_{version}.json"
        rows_path = runs / f"eval_{version}.jsonl"
        meta = json_file(meta_path)
        rows = read_jsonl(rows_path)
        validate_eval_rows(test_rows, rows, version)
        ids = [row.get("prompt_id") for row in rows]
        signature = (
            ids,
            meta.get("test_ids_sha256"),
            meta.get("chat_template_sha256"),
            meta.get("generation"),
            meta.get("manifest_sha256"),
        )
        if expected_signature is not None and signature != expected_signature:
            raise ValueError("base/v1/v2 evaluation metadata differs")
        expected_signature = signature
        if meta.get("test_ids_sha256") != EXPECTED_TEST_IDS_SHA256:
            raise ValueError(f"eval_{version} has an unexpected test hash")
        if meta.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"eval_{version} manifest hash mismatch")
        if meta.get("generation") != cfg["generation"]:
            raise ValueError(f"eval_{version} generation config mismatch")
        if meta.get("stage") != f"eval_{version}" or meta.get("samples") != 400:
            raise ValueError(f"eval_{version} completion record is invalid")
        eval_hashes[version] = {"record": digest(meta_path), "rows": digest(rows_path)}
        old_limit_hits[version] = sum(row.get("generated_tokens") == cfg["generation"]["max_new_tokens"] for row in rows)

    sft = json_file(runs / "sft.json")
    dpo = json_file(runs / "dpo.json")
    for stage, record in (("sft", sft), ("dpo", dpo)):
        if record.get("status") != "complete":
            raise ValueError(f"{stage} run is not complete")
        if record.get("stage") != stage or record.get("rows") != manifest["splits"][stage]["count"]:
            raise ValueError(f"{stage} completion record has invalid stage or row count")
        if record.get("git_commit") != TRAINING_COMMIT:
            raise ValueError(f"{stage} was not recorded at the expected training commit")
        if record.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"{stage} manifest hash mismatch")
        _assert_record_config(record, cfg, stage)
    if dpo.get("reference") != "frozen merged v1":
        raise ValueError("DPO lineage does not identify frozen merged v1 reference")
    current_v1_digest = adapter_digest(cfg["paths"]["v1_adapter"])
    if dpo.get("v1_adapter_sha256") != current_v1_digest:
        raise ValueError("current v1 adapter differs from the DPO reference source")

    return {
        "rows": test_rows,
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
        "test_ids_sha256": EXPECTED_TEST_IDS_SHA256,
        "chat_template_sha256": expected_signature[2],
        "old_generation": expected_signature[3],
        "eval_hashes": eval_hashes,
        "old_limit_hits": old_limit_hits,
        "v1_adapter_sha256": current_v1_digest,
        "protected_hashes": protected_hashes(cfg),
    }


def environment_record():
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name] = dist.version
    gpu = None
    torch_info = None
    try:
        import torch

        gpu = [
            {
                "visible_index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
        torch_info = {"version": torch.__version__, "cuda": torch.version.cuda}
    except ImportError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_info,
        "python_packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "gpu": gpu,
    }


def state_path(cfg):
    return quality_dir(cfg) / "state.json"


def load_state(cfg, verified=None):
    pathname = state_path(cfg)
    if not pathname.is_file():
        raise FileNotFoundError("run quality_eval check first")
    state = json_file(pathname)
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("unsupported quality evaluation state schema")
    verified = verified or verify_inputs(cfg)
    if state.get("config_sha256") != digest(cfg):
        raise ValueError("quality evaluation config hash changed")
    if state.get("protected_hashes") != verified["protected_hashes"]:
        raise ValueError("an original experiment input changed after quality check")
    return state


def update_state(cfg, state, outputs):
    state.setdefault("outputs", {}).update(outputs)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path(cfg), state)


def require_recorded_outputs(state, out, relative_paths):
    for relative in relative_paths:
        pathname = out / relative
        expected = state.get("outputs", {}).get(relative)
        if not pathname.is_file() or expected is None or digest(pathname) != expected:
            raise ValueError(f"quality evaluation artifact changed or is unrecorded: {relative}")


def local_readme():
    return """# Quality evaluation outputs

All files in this directory are derived evaluation artifacts and are ignored by Git.

- `preference_rows.csv` contains assistant-token-only chosen/rejected log probabilities. Raw margins use sums and therefore have length bias; mean margins normalize each answer by its scored token count.
- Template terminators and EOS tokens produced by the chat template are included in the assistant answer span. Prompt and padding tokens are excluded.
- Inputs longer than the 1024-token context limit are excluded as a complete pair for all three versions and listed explicitly.
- `generated_512/` contains deterministic 512-token-limit generations for the fixed 60-ID sample.
- Fill `blind_sheet.csv` without opening `blind_key.json`. Each comparison must contain the corresponding letter or `tie`; every row also needs a reason.
- Preference likelihood on this internal split and a single-rater 60-prompt blind comparison are exploratory evidence, not a complete measure of real answer quality.
"""


def run_check(cfg, resume=False):
    verified = verify_inputs(cfg)
    out = quality_dir(cfg)
    state_file = state_path(cfg)
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError("quality_eval already contains artifacts; use --resume after confirming provenance")
    if out.exists() and any(out.iterdir()) and resume and not state_file.exists():
        raise FileExistsError("cannot resume quality_eval without an existing state.json")
    if state_file.exists():
        state = load_state(cfg, verified)
        print(json.dumps({"status": "already checked", "test_rows": 400, "state": str(state_file)}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": STATE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "training_commit": TRAINING_COMMIT,
        "config_sha256": digest(cfg),
        "config": relevant_config(cfg),
        "model_revision": cfg["model"]["revision"],
        "data_revision": cfg["data"]["source_revision"],
        "seed": cfg["seed"],
        "manifest_sha256": verified["manifest_sha256"],
        "test_ids_sha256": verified["test_ids_sha256"],
        "chat_template_sha256": verified["chat_template_sha256"],
        "old_generation": verified["old_generation"],
        "old_limit_hits": verified["old_limit_hits"],
        "v1_adapter_sha256": verified["v1_adapter_sha256"],
        "protected_hashes": verified["protected_hashes"],
        "environment": environment_record(),
        "outputs": {},
    }
    write_json(state_file, state)
    readme_path = out / "README.md"
    readme_path.write_text(local_readme(), encoding="utf-8")
    update_state(cfg, state, {"README.md": digest(readme_path)})
    if protected_hashes(cfg) != verified["protected_hashes"]:
        raise AssertionError("original experiment inputs changed during check")
    print(json.dumps({"status": "checked", "test_rows": 400, "test_ids_sha256": verified["test_ids_sha256"]}, indent=2))


def tokenized_pair(tokenizer, row, context_limit=CONTEXT_LIMIT):
    prompt_messages = [{"role": "user", "content": row["prompt"]}]
    prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    lengths = {}
    sequences = {}
    for label in ("chosen", "rejected"):
        messages = prompt_messages + [{"role": "assistant", "content": row[label]}]
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        lengths[label] = len(full_ids)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            return None, {"reason": f"{label}_prefix_mismatch", "prompt_tokens": len(prompt_ids), **{f"{k}_total_tokens": v for k, v in lengths.items()}}
        answer_ids = full_ids[len(prompt_ids) :]
        if not answer_ids:
            return None, {"reason": f"{label}_has_no_answer_tokens", "prompt_tokens": len(prompt_ids), **{f"{k}_total_tokens": v for k, v in lengths.items()}}
        sequences[label] = {"input_ids": full_ids, "answer_start": len(prompt_ids), "answer_tokens": len(answer_ids)}
    if max(lengths.values()) > context_limit:
        return None, {
            "reason": "pair_exceeds_context_limit",
            "prompt_tokens": len(prompt_ids),
            "chosen_total_tokens": lengths["chosen"],
            "rejected_total_tokens": lengths["rejected"],
        }
    return sequences, None


def partition_tokenized_rows(tokenizer, rows, context_limit=CONTEXT_LIMIT):
    valid, exclusions = [], {}
    for row in rows:
        pair, excluded = tokenized_pair(tokenizer, row, context_limit)
        if excluded:
            exclusions[row["prompt_id"]] = excluded
        else:
            valid.append({**row, "sequences": pair})
    return valid, exclusions


def answer_token_logps(logits, input_ids, answer_start, attention_mask=None):
    """Pure-Python reference scorer used by CPU tests; logits[t-1] predicts token t."""
    if answer_start < 1 or answer_start >= len(input_ids):
        raise ValueError("invalid assistant answer boundary")
    if len(logits) < len(input_ids) - 1:
        raise ValueError("not enough shifted logits")
    mask = attention_mask or [1] * len(input_ids)
    values = []
    for token_index in range(answer_start, len(input_ids)):
        if not mask[token_index]:
            continue
        row = logits[token_index - 1]
        target = input_ids[token_index]
        maximum = max(row)
        log_denom = maximum + math.log(sum(math.exp(value - maximum) for value in row))
        values.append(row[target] - log_denom)
    if not values:
        raise ValueError("no unmasked assistant tokens")
    return values


def margin(chosen_sum, chosen_tokens, rejected_sum, rejected_tokens):
    if chosen_tokens <= 0 or rejected_tokens <= 0:
        raise ValueError("answer token counts must be positive")
    return {
        "raw_margin": chosen_sum - rejected_sum,
        "mean_margin": chosen_sum / chosen_tokens - rejected_sum / rejected_tokens,
    }


def score_sequence(model, sequence):
    import torch

    ids = torch.tensor([sequence["input_ids"]], dtype=torch.long, device=model.device)
    attention = torch.ones_like(ids)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1, :].float()
        targets = ids[:, 1:]
        start = sequence["answer_start"] - 1
        selected = torch.log_softmax(logits[:, start:, :], dim=-1).gather(
            -1, targets[:, start:].unsqueeze(-1)
        ).squeeze(-1)
    if selected.numel() != sequence["answer_tokens"]:
        raise AssertionError("assistant token/logit alignment failed")
    return float(selected.sum().item()), int(selected.numel())


def load_model(cfg, version):
    from peft import PeftModel

    tokenizer, model = tokenizer_and_model(cfg)
    if version == "v1":
        model = PeftModel.from_pretrained(model, cfg["paths"]["v1_adapter"], is_trainable=False)
    elif version == "v2":
        model = PeftModel.from_pretrained(model, cfg["paths"]["v1_adapter"], is_trainable=False).merge_and_unload()
        model = PeftModel.from_pretrained(model, cfg["paths"]["v2_adapter"], is_trainable=False)
    model.eval()
    model.to("cuda")
    return tokenizer, model


def load_tokenizer(cfg):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_path"], local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("local tokenizer has no chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def release_model(model=None, tokenizer=None):
    del model, tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def append_jsonl(pathname, row):
    pathname = Path(pathname)
    pathname.parent.mkdir(parents=True, exist_ok=True)
    with pathname.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def resumable_prefix(pathname, expected_rows, prompt_key=True):
    pathname = Path(pathname)
    existing = read_jsonl(pathname) if pathname.exists() else []
    if len(existing) > len(expected_rows):
        raise ValueError(f"partial output has too many rows: {pathname}")
    for index, row in enumerate(existing):
        expected = expected_rows[index]
        if row.get("prompt_id") != expected["prompt_id"]:
            raise ValueError(f"partial output ID/order mismatch: {pathname}")
        if prompt_key and row.get("prompt") != expected["prompt"]:
            raise ValueError(f"partial output prompt mismatch: {pathname}")
    return existing


def write_csv(pathname, rows, fieldnames):
    pathname = Path(pathname)
    pathname.parent.mkdir(parents=True, exist_ok=True)
    with pathname.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(values):
    positive = sum(value > 0 for value in values)
    zero = sum(value == 0 for value in values)
    return {
        "n": len(values),
        "positive_count": positive,
        "positive_fraction": positive / len(values) if values else None,
        "zero_count": zero,
        "zero_fraction": zero / len(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def run_preference(cfg, resume=False):
    verified = verify_inputs(cfg)
    state = load_state(cfg, verified)
    out = quality_dir(cfg)
    final_paths = [out / "preference_rows.csv", out / "preference_summary.json", out / "preference_summary.csv"]
    if any(path.exists() for path in final_paths):
        if resume and all(path.exists() for path in final_paths):
            print("preference evaluation already complete")
            return
        raise FileExistsError("preference output already exists; refusing to overwrite")

    tokenizer = load_tokenizer(cfg)
    rows = verified["rows"]
    tokenized, exclusions = partition_tokenized_rows(tokenizer, rows)
    tokenizer = None
    release_model()
    if not tokenized:
        raise ValueError("no preference rows fit the context and template requirements")

    parts = out / "preference_parts"
    for version in VERSIONS:
        part = parts / f"{version}.jsonl"
        if part.exists() and not resume:
            raise FileExistsError(f"partial preference output exists: {part}; use --resume")
        existing = resumable_prefix(part, tokenized)
        tokenizer = model = None
        try:
            tokenizer, model = load_model(cfg, version)
            for row in tokenized[len(existing) :]:
                chosen_sum, chosen_n = score_sequence(model, row["sequences"]["chosen"])
                rejected_sum, rejected_n = score_sequence(model, row["sequences"]["rejected"])
                margins = margin(chosen_sum, chosen_n, rejected_sum, rejected_n)
                append_jsonl(part, {
                    "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"],
                    "chosen_sum_logp": chosen_sum,
                    "chosen_tokens": chosen_n,
                    "chosen_mean_logp": chosen_sum / chosen_n,
                    "rejected_sum_logp": rejected_sum,
                    "rejected_tokens": rejected_n,
                    "rejected_mean_logp": rejected_sum / rejected_n,
                    **margins,
                })
        finally:
            model = None
            tokenizer = None
            release_model()

    scored = {version: read_jsonl(parts / f"{version}.jsonl") for version in VERSIONS}
    if any(len(scored[version]) != len(tokenized) for version in VERSIONS):
        raise RuntimeError("preference scoring incomplete")
    by_version = {version: {row["prompt_id"]: row for row in values} for version, values in scored.items()}
    output_rows = []
    for row in rows:
        pid = row["prompt_id"]
        output = {"prompt_id": pid, "status": "excluded" if pid in exclusions else "scored", "exclusion_reason": exclusions.get(pid, {}).get("reason", "")}
        if pid in exclusions:
            output.update(exclusions[pid])
        else:
            for version in VERSIONS:
                for key, value in by_version[version][pid].items():
                    if key not in ("prompt_id", "prompt"):
                        output[f"{version}_{key}"] = value
            output["v2_minus_v1_raw_margin"] = output["v2_raw_margin"] - output["v1_raw_margin"]
            output["v2_minus_v1_mean_margin"] = output["v2_mean_margin"] - output["v1_mean_margin"]
        output_rows.append(output)
    fields = sorted({key for row in output_rows for key in row}, key=lambda key: (key != "prompt_id", key))
    write_csv(final_paths[0], output_rows, fields)

    summary = {
        "context_limit": CONTEXT_LIMIT,
        "answer_span_rule": "prompt and padding excluded; chat-template assistant terminators/EOS included",
        "test_n": len(rows),
        "valid_n": len(tokenized),
        "excluded_n": len(exclusions),
        "exclusions": [{"prompt_id": pid, **detail} for pid, detail in exclusions.items()],
        "versions": {},
    }
    summary_rows = []
    for version in VERSIONS:
        raw = metric_summary([row["raw_margin"] for row in scored[version]])
        mean = metric_summary([row["mean_margin"] for row in scored[version]])
        summary["versions"][version] = {"raw_margin": raw, "mean_margin": mean}
        for metric_name, values in (("raw_margin", raw), ("mean_margin", mean)):
            summary_rows.append({"version": version, "metric": metric_name, "excluded_n": len(exclusions), **values})
    deltas = [row["v2_minus_v1_raw_margin"] for row in output_rows if row["status"] == "scored"]
    mean_deltas = [row["v2_minus_v1_mean_margin"] for row in output_rows if row["status"] == "scored"]
    summary["v2_minus_v1"] = {"raw_margin": metric_summary(deltas), "mean_margin": metric_summary(mean_deltas)}
    write_json(final_paths[1], summary)
    write_csv(final_paths[2], summary_rows, ["version", "metric", "n", "excluded_n", "positive_count", "positive_fraction", "zero_count", "zero_fraction", "mean"])
    outputs = {str(path.relative_to(out)): digest(path) for path in final_paths}
    outputs.update({str((parts / f"{version}.jsonl").relative_to(out)): digest(parts / f"{version}.jsonl") for version in VERSIONS})
    update_state(cfg, state, outputs)
    if protected_hashes(cfg) != verified["protected_hashes"]:
        raise AssertionError("original experiment inputs changed during preference evaluation")
    print(json.dumps(summary, indent=2))


def sample_plan(rows, seed, sample_size):
    if not 1 <= sample_size <= len(rows):
        raise ValueError("sample size must be between 1 and the test population")
    selected_ids = random.Random(seed).sample([row["prompt_id"] for row in rows], sample_size)
    lookup = {row["prompt_id"]: row for row in rows}
    selected = [lookup[pid] for pid in selected_ids]
    mapping_rng = random.Random(seed ^ 0x5A17)
    key = []
    for row in selected:
        versions = list(VERSIONS)
        mapping_rng.shuffle(versions)
        key.append({"prompt_id": row["prompt_id"], "A": versions[0], "B": versions[1], "C": versions[2]})
    return selected, key


def build_blind_rows(selected, generated, key):
    key_by_id = {row["prompt_id"]: row for row in key}
    generations = {version: {row["prompt_id"]: row for row in generated[version]} for version in VERSIONS}
    blind = []
    for source in selected:
        pid = source["prompt_id"]
        mapping = key_by_id[pid]
        row = {"prompt_id": pid, "prompt": source["prompt"]}
        for letter in "ABC":
            row[f"{letter}_response"] = generations[mapping[letter]][pid]["generated"]
        row.update({"A_vs_B": "", "A_vs_C": "", "B_vs_C": "", "reason": ""})
        blind.append(row)
    return blind


def run_blind_pack(cfg, seed, sample_size, max_new_tokens, resume=False):
    if max_new_tokens != 512:
        raise ValueError("blind comparison requires max-new-tokens=512")
    verified = verify_inputs(cfg)
    state = load_state(cfg, verified)
    out = quality_dir(cfg)
    generated_dir = out / "generated_512"
    ids_path, key_path, sheet_path = out / "sample_ids.json", out / "blind_key.json", out / "blind_sheet.csv"
    selected, key = sample_plan(verified["rows"], seed, sample_size)
    plan = {
        "seed": seed,
        "sample_size": sample_size,
        "max_new_tokens": max_new_tokens,
        "population_size": len(verified["rows"]),
        "algorithm": "random.Random(seed).sample(test_ids_in_verified_order, sample_size)",
        "prompt_ids": [row["prompt_id"] for row in selected],
    }
    plan["prompt_ids_sha256"] = digest(plan["prompt_ids"])
    if ids_path.exists():
        if not resume:
            raise FileExistsError("blind-pack artifacts already exist; use --resume")
        if json_file(ids_path) != plan:
            raise ValueError("blind-pack seed/sample/config differs from existing sample")
    else:
        write_json(ids_path, plan)

    generated = {}
    for version in VERSIONS:
        pathname = generated_dir / f"{version}.jsonl"
        if pathname.exists() and not resume:
            raise FileExistsError(f"generation output exists: {pathname}; use --resume")
        existing = resumable_prefix(pathname, selected)
        tokenizer = model = None
        try:
            tokenizer, model = load_model(cfg, version)
            for row in selected[len(existing) :]:
                import torch

                prompt = prompt_text(tokenizer, row["prompt"])
                encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    ids = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_ids = ids[0][encoded["input_ids"].shape[1] :]
                append_jsonl(pathname, {
                    "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"],
                    "generated": tokenizer.decode(new_ids, skip_special_tokens=True),
                    "generated_tokens": int(len(new_ids)),
                    "hit_max_new_tokens": len(new_ids) == max_new_tokens,
                    "generation": {"max_new_tokens": max_new_tokens, "do_sample": False},
                    "model": version,
                })
        finally:
            model = None
            tokenizer = None
            release_model()
        generated[version] = read_jsonl(pathname)
        if len(generated[version]) != sample_size:
            raise RuntimeError(f"{version} long generation is incomplete")

    if key_path.exists() or sheet_path.exists():
        if not resume:
            raise FileExistsError("blind key or sheet already exists; refusing to overwrite")
        if key_path.exists() and json_file(key_path) != {"seed": seed, "mapping": key}:
            raise ValueError("existing blind key differs")
        if sheet_path.exists():
            print("blind pack already complete; existing scoring sheet was preserved")
            return
    if not key_path.exists():
        write_json(key_path, {"seed": seed, "mapping": key})
    blind_rows = build_blind_rows(selected, generated, key)
    fields = ["prompt_id", "prompt", "A_response", "B_response", "C_response", "A_vs_B", "A_vs_C", "B_vs_C", "reason"]
    write_csv(sheet_path, blind_rows, fields)
    metadata_path = generated_dir / "metadata.json"
    metadata = {
        "seed": seed,
        "sample_size": sample_size,
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": False},
        "chat_template_sha256": verified["chat_template_sha256"],
        "model_revisions": {"base": cfg["model"]["revision"], "v1_adapter": verified["v1_adapter_sha256"], "v2_adapter_tree": verified["protected_hashes"]["v2_adapter_tree"]},
        "files": {version: {"sha256": digest(generated_dir / f"{version}.jsonl"), "hit_limit": sum(row["hit_max_new_tokens"] for row in generated[version])} for version in VERSIONS},
    }
    write_json(metadata_path, metadata)
    artifact_paths = [ids_path, key_path, sheet_path, metadata_path] + [generated_dir / f"{version}.jsonl" for version in VERSIONS]
    update_state(cfg, state, {str(path.relative_to(out)): digest(path) for path in artifact_paths})
    if protected_hashes(cfg) != verified["protected_hashes"]:
        raise AssertionError("original experiment inputs changed during blind generation")
    print(json.dumps({"status": "blind pack complete", "sample_size": sample_size, "sheet": str(sheet_path)}, indent=2))


def read_csv(pathname):
    with Path(pathname).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def validate_scores(rows, sample_ids):
    if len(rows) != len(sample_ids) or [row.get("prompt_id") for row in rows] != sample_ids:
        raise ValueError("score sheet must contain every sampled ID once and in order")
    rules = {"A_vs_B": {"A", "B", "tie"}, "A_vs_C": {"A", "C", "tie"}, "B_vs_C": {"B", "C", "tie"}}
    for index, row in enumerate(rows, 1):
        for field, allowed in rules.items():
            if row.get(field) not in allowed:
                raise ValueError(f"row {index}: {field} must be one of {sorted(allowed)}")
        if not row.get("reason", "").strip():
            raise ValueError(f"row {index}: reason is required")
    return rows


def validate_blind_content(score_rows, expected_rows):
    protected_fields = ("prompt_id", "prompt", "A_response", "B_response", "C_response")
    if len(score_rows) != len(expected_rows):
        raise ValueError("score sheet row count changed")
    for index, (actual, expected) in enumerate(zip(score_rows, expected_rows), 1):
        if any(actual.get(field) != expected.get(field) for field in protected_fields):
            raise ValueError(f"row {index}: prompt or anonymous response was modified")


def wilson_interval(wins, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_blind_scores(score_rows, key_rows):
    key_by_id = {row["prompt_id"]: row for row in key_rows}
    canonical = (("base", "v1"), ("v1", "v2"), ("base", "v2"))
    counts = {pair: {"wins": 0, "ties": 0, "losses": 0} for pair in canonical}
    merged = []
    for row in score_rows:
        mapping = key_by_id[row["prompt_id"]]
        resolved = []
        for field, left, right in (("A_vs_B", "A", "B"), ("A_vs_C", "A", "C"), ("B_vs_C", "B", "C")):
            first, second = mapping[left], mapping[right]
            selected = row[field]
            winner = "tie" if selected == "tie" else mapping[selected]
            pair = next(pair for pair in canonical if set(pair) == {first, second})
            if winner == "tie":
                counts[pair]["ties"] += 1
            elif winner == pair[0]:
                counts[pair]["wins"] += 1
            else:
                counts[pair]["losses"] += 1
            resolved.append(f"{pair[0]}_vs_{pair[1]}:{winner}")
        merged.append({**row, "resolved": ";".join(resolved)})
    summary = {}
    for pair, values in counts.items():
        decisive = values["wins"] + values["losses"]
        summary[f"{pair[0]}_vs_{pair[1]}"] = {
            "first_version": pair[0],
            "second_version": pair[1],
            **values,
            "n": sum(values.values()),
            "decisive_n": decisive,
            "first_version_win_fraction_among_decisive": values["wins"] / decisive if decisive else None,
            "wilson_95_ci_among_decisive": wilson_interval(values["wins"], decisive),
        }
    return summary, merged


def run_blind_summarize(cfg, scores_path, resume=False):
    verified = verify_inputs(cfg)
    state = load_state(cfg, verified)
    out = quality_dir(cfg)
    require_recorded_outputs(
        state,
        out,
        ["sample_ids.json", "blind_key.json", "generated_512/base.jsonl", "generated_512/v1.jsonl", "generated_512/v2.jsonl"],
    )
    plan = json_file(out / "sample_ids.json")
    key = json_file(out / "blind_key.json")["mapping"]
    score_rows = validate_scores(read_csv(scores_path), plan["prompt_ids"])
    if {row["prompt_id"] for row in key} != set(plan["prompt_ids"]):
        raise ValueError("blind key does not match sampled IDs")
    if any(sorted(row.get(letter) for letter in "ABC") != sorted(VERSIONS) for row in key):
        raise ValueError("blind key does not map A/B/C to all three versions")
    source_by_id = {row["prompt_id"]: row for row in verified["rows"]}
    selected = [source_by_id[pid] for pid in plan["prompt_ids"]]
    generated = {version: read_jsonl(out / "generated_512" / f"{version}.jsonl") for version in VERSIONS}
    expected_blind = build_blind_rows(selected, generated, key)
    validate_blind_content(score_rows, expected_blind)
    destinations = [out / "blind_summary.json", out / "blind_summary.csv", out / "blind_results.csv"]
    if any(path.exists() for path in destinations):
        if resume and all(path.exists() for path in destinations):
            print("blind summary already complete")
            return
        raise FileExistsError("blind summary output exists; refusing to overwrite")
    summary, merged = summarize_blind_scores(score_rows, key)
    payload = {
        "sample_size": len(score_rows),
        "rater_design": "one rater, one pass, randomized anonymous A/B/C order",
        "uncertainty": "95% Wilson interval for the first-version win fraction among non-tied paired judgments",
        "comparisons": summary,
    }
    write_json(destinations[0], payload)
    flat = []
    for name, values in summary.items():
        row = {"comparison": name, **values}
        row["wilson_95_low"], row["wilson_95_high"] = row.pop("wilson_95_ci_among_decisive")
        flat.append(row)
    write_csv(destinations[1], flat, list(flat[0]))
    write_csv(destinations[2], merged, list(merged[0]))
    update_state(cfg, state, {str(path.relative_to(out)): digest(path) for path in destinations})
    if protected_hashes(cfg) != verified["protected_hashes"]:
        raise AssertionError("original experiment inputs changed during blind summary")
    print(json.dumps(payload, indent=2))


def run_report(cfg, resume=False):
    verified = verify_inputs(cfg)
    state = load_state(cfg, verified)
    out = quality_dir(cfg)
    preference_path, blind_path = out / "preference_summary.json", out / "blind_summary.json"
    if not preference_path.is_file() or not blind_path.is_file():
        raise FileNotFoundError("complete preference and blind-summarize before report")
    require_recorded_outputs(
        state,
        out,
        ["preference_summary.json", "blind_summary.json", "generated_512/metadata.json"],
    )
    report_path = out / "report.md"
    if report_path.exists():
        if resume:
            print("report already exists")
            return
        raise FileExistsError("report already exists; refusing to overwrite")
    preference, blind = json_file(preference_path), json_file(blind_path)
    long_generation = json_file(out / "generated_512" / "metadata.json")
    old_summary = read_csv(Path(cfg["paths"]["runs"]) / "summary.csv")
    lines = [
        "# Stage 2 quality evaluation report",
        "",
        "## Provenance",
        "",
        f"- Quality-evaluation code commit: `{state['code_commit']}`",
        f"- Original training commit: `{TRAINING_COMMIT}`",
        f"- Base model revision: `{cfg['model']['revision']}`",
        f"- Dataset revision: `{cfg['data']['source_revision']}`",
        f"- Split seed: `{cfg['seed']}`",
        f"- Test IDs: 400, SHA256 `{verified['test_ids_sha256']}`",
        "- DPO reference lineage: frozen merged v1 SFT adapter (verified against the recorded adapter digest).",
        "",
        "## Training configuration",
        "",
        f"- LoRA: `{json.dumps(cfg['lora'], ensure_ascii=False, sort_keys=True)}`",
        f"- SFT: `{json.dumps(cfg['sft'], ensure_ascii=False, sort_keys=True)}`",
        f"- DPO: `{json.dumps(cfg['dpo'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Original deterministic generation",
        "",
        "| version | samples | mean generated tokens | reached 128 limit |",
        "|---|---:|---:|---:|",
    ]
    for row in old_summary:
        lines.append(f"| {row['version']} | {row['samples']} | {row['mean_generated_tokens']} | {verified['old_limit_hits'][row['version']]} |")
    lines += [
        "",
        "The original maximum was 128 new tokens, and many outputs reached that limit. Length differences do not establish answer quality.",
        "",
        "## Held-out preference likelihood",
        "",
        f"Valid N: {preference['valid_n']}; excluded N: {preference['excluded_n']}; context limit: {preference['context_limit']} tokens.",
        "Prompt and padding tokens are excluded. Chat-template assistant terminators/EOS are included consistently. Raw margins use summed log probability and are length-biased; mean margins normalize each answer independently by its answer-token count.",
        "",
        "| version | raw > 0 | raw mean | mean-token > 0 | mean-token mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for version in VERSIONS:
        raw = preference["versions"][version]["raw_margin"]
        mean = preference["versions"][version]["mean_margin"]
        lines.append(f"| {version} | {raw['positive_fraction']:.4f} | {raw['mean']:.4f} | {mean['positive_fraction']:.4f} | {mean['mean']:.4f} |")
    delta = preference["v2_minus_v1"]
    lines += [
        "",
        f"The mean v2−v1 raw-margin change was {delta['raw_margin']['mean']:.4f}; the mean normalized-margin change was {delta['mean_margin']['mean']:.4f}. These are changes relative to v1 on the same labels, not a universal three-model quality ranking. DPO training reward accuracy is not used as final quality evidence.",
        "",
        "## Anonymous 512-token comparison",
        "",
        "| version | sampled prompts | reached 512 limit |",
        "|---|---:|---:|",
    ]
    for version in VERSIONS:
        lines.append(f"| {version} | {long_generation['sample_size']} | {long_generation['files'][version]['hit_limit']} |")
    lines.append("")
    for name, values in blind["comparisons"].items():
        low, high = values["wilson_95_ci_among_decisive"]
        interval = "undefined" if low is None else f"[{low:.3f}, {high:.3f}]"
        lines.append(f"- {name}: {values['wins']} wins / {values['ties']} ties / {values['losses']} losses for the first version; decisive 95% Wilson interval {interval}.")
    lines += [
        "",
        "## Limitations",
        "",
        "The test set is an internal prompt-disjoint split of UltraFeedback Binarized `train_prefs`, not an external benchmark. Preference likelihood measures agreement with its chosen/rejected labels and is not sufficient evidence of real response quality. The long-output comparison covers 60 uniformly sampled prompts, uses deterministic decoding, and has one rater in one pass. A 512-token limit can still truncate outputs. This is an exploratory small-scale, one-epoch experiment; uncertainty intervals and observed differences should not be generalized beyond this setup.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    update_state(cfg, state, {str(report_path.relative_to(out)): digest(report_path)})
    if protected_hashes(cfg) != verified["protected_hashes"]:
        raise AssertionError("original experiment inputs changed during report generation")
    print(report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "preference", "blind-pack", "blind-summarize", "report"))
    parser.add_argument("--config", default="config/local.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--scores", default="runs/quality_eval/blind_sheet.csv")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.command == "check":
        run_check(cfg, args.resume)
    elif args.command == "preference":
        run_preference(cfg, args.resume)
    elif args.command == "blind-pack":
        run_blind_pack(cfg, args.seed, args.sample_size, args.max_new_tokens, args.resume)
    elif args.command == "blind-summarize":
        run_blind_summarize(cfg, args.scores, args.resume)
    else:
        run_report(cfg, args.resume)


if __name__ == "__main__":
    main()
