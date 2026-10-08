"""Offline experiment entry points; heavy dependencies imported only for real runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from .core import config, digest, prepared, read_jsonl, record, source_rows, split_rows, write_json, write_jsonl


def local_guard(cfg, adapters=()):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    base = Path(cfg["model"]["base_path"])
    if not (base / "config.json").is_file():
        raise FileNotFoundError(f"local base model missing config.json: {base}")
    for adapter in adapters:
        folder = Path(cfg["paths"][adapter])
        if not (folder / "adapter_config.json").is_file():
            raise FileNotFoundError(f"local adapter missing adapter_config.json: {folder}")


def configure_nccl(supports_p2p_ib=None):
    """Match Accelerate's launcher behavior on GPUs without P2P/IB support."""
    if supports_p2p_ib is None:
        from accelerate.utils import check_cuda_p2p_ib_support

        supports_p2p_ib = check_cuda_p2p_ib_support()

    if not supports_p2p_ib:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        print("Disabled NCCL P2P and IB for this GPU configuration")


def prepare(cfg, dry):
    if cfg["data"]["source_revision"].startswith("SET_") or cfg["model"]["revision"].startswith("SET_"):
        raise ValueError("set immutable model and dataset source revisions in config before preparation")
    source = Path(cfg["data"]["source_path"])
    if not source.exists():
        raise FileNotFoundError(source)
    partitions, counts = split_rows(source_rows(source), cfg["seed"], cfg["data"]["max_records"], cfg["data"]["split_fractions"])
    source_hash = digest(source) if source.is_file() else digest(sorted((str(p.relative_to(source)), p.stat().st_size, digest(p)) for p in source.rglob("*") if p.is_file()))
    manifest = {"source": {"path": str(source), "name": cfg["data"]["source_name"], "revision": cfg["data"]["source_revision"], "split": cfg["data"]["source_split"], "sha256": source_hash}, "seed": cfg["seed"], "counts": counts, "splits": {}}
    for name, rows in partitions.items():
        ids = [r["prompt_id"] for r in rows]
        manifest["splits"][name] = {"count": len(rows), "prompt_ids": ids, "ids_sha256": digest(ids), "rows_sha256": digest(rows)}
    if dry:
        print(json.dumps({"dry_run": True, "counts": counts, "split_counts": {k: len(v) for k, v in partitions.items()}}, indent=2))
        return
    directory = Path(cfg["data"]["prepared_dir"])
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"prepared directory is not empty: {directory}")
    for name, rows in partitions.items():
        write_jsonl(directory / f"{name}.jsonl", rows)
    write_json(directory / "manifest.json", manifest)
    record(cfg, "prepare", {"counts": counts})
    print(f"Prepared {counts['selected']} prompts in {directory}")


def training_data(cfg, split):
    rows, manifest = prepared(cfg, split)
    if not rows:
        raise ValueError(f"empty {split} split")
    return rows, manifest


def tokenizer_and_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = cfg["model"]["base_path"]
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if not tok.chat_template:
        raise ValueError("local tokenizer has no chat template")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, cfg["model"]["dtype"])
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=dtype)
    model.config.use_cache = False
    return tok, model


def prompt_text(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def completion(tok, prompt, answer):
    prefix = prompt_text(tok, prompt)
    full = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], tokenize=False, add_generation_prompt=False)
    if not full.startswith(prefix):
        raise ValueError("chat template does not preserve prompt prefix")
    return full[len(prefix):]


def lora_config(cfg):
    from peft import LoraConfig
    x = cfg["lora"]
    return LoraConfig(r=x["r"], lora_alpha=x["alpha"], lora_dropout=x["dropout"], target_modules=x["target_modules"], bias="none", task_type="CAUSAL_LM")


def train(cfg, stage, dry):
    if cfg["model"]["revision"].startswith("SET_") or cfg["data"]["source_revision"].startswith("SET_"):
        raise ValueError("set immutable source revisions in config")
    split = "sft" if stage == "sft" else "dpo"
    rows, manifest = training_data(cfg, split)
    adapters = () if stage == "sft" else ("v1_adapter",)
    local_guard(cfg, adapters)
    if dry:
        print(json.dumps({"dry_run": True, "stage": stage, "rows": len(rows), "manifest_sha256": digest(manifest)}, indent=2))
        return
    configure_nccl()
    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    tok, model = tokenizer_and_model(cfg)
    output = cfg["paths"]["v1_adapter" if stage == "sft" else "v2_adapter"]
    if Path(output).exists() and any(Path(output).iterdir()):
        raise FileExistsError(f"output already contains files: {output}")
    common = dict(output_dir=output, seed=cfg["seed"], data_seed=cfg["seed"], num_train_epochs=cfg[stage]["epochs"], per_device_train_batch_size=cfg[stage]["batch_size"], gradient_accumulation_steps=cfg[stage]["gradient_accumulation_steps"], learning_rate=cfg[stage]["learning_rate"], max_length=cfg[stage]["max_length"], report_to="none", save_strategy="no", gradient_checkpointing=True)
    prompt_rows = [{"prompt": prompt_text(tok, r["prompt"]), "chosen": completion(tok, r["prompt"], r["chosen"]), "rejected": completion(tok, r["prompt"], r["rejected"])} for r in rows]
    template_hash = digest(tok.chat_template)
    if stage == "sft":
        dataset = Dataset.from_list([{"prompt": r["prompt"], "completion": r["chosen"]} for r in prompt_rows])
        args = SFTConfig(**common, completion_only_loss=True, packing=False)
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tok, peft_config=lora_config(cfg))
    else:
        v1 = cfg["paths"]["v1_adapter"]
        adapter_hash_before = adapter_digest(v1)
        # Separate instances: the reference is exactly the merged v1 weights and is frozen.
        policy = PeftModel.from_pretrained(model, v1, is_trainable=False).merge_and_unload()
        _, ref_base = tokenizer_and_model(cfg)
        reference = PeftModel.from_pretrained(ref_base, v1, is_trainable=False).merge_and_unload()
        reference.eval()
        reference.requires_grad_(False)
        if any(p.requires_grad for p in reference.parameters()):
            raise AssertionError("reference is not frozen")
        if policy is reference or adapter_digest(v1) != adapter_hash_before:
            raise AssertionError("v1 reference provenance check failed")
        dataset = Dataset.from_list(prompt_rows)
        # TRL 0.26.0 otherwise rejects a simultaneous peft_config and ref_model.
        args = DPOConfig(**common, beta=cfg["dpo"]["beta"], sync_ref_model=False, force_use_ref_model=True)
        trainer = DPOTrainer(model=policy, ref_model=reference, args=args, train_dataset=dataset, processing_class=tok, peft_config=lora_config(cfg))
    record(cfg, stage, {"status": "started", "rows": len(rows), "chat_template_sha256": template_hash, "reference": ("frozen merged v1" if stage == "dpo" else None), "v1_adapter_sha256": adapter_digest(cfg["paths"]["v1_adapter"]) if stage == "dpo" else None})
    trainer.train()
    trainer.save_model(output)
    tok.save_pretrained(output)
    if stage == "dpo" and adapter_digest(cfg["paths"]["v1_adapter"]) != adapter_hash_before:
        raise AssertionError("v1 adapter changed during DPO")
    record(cfg, stage, {"status": "complete", "rows": len(rows), "chat_template_sha256": template_hash, "reference": ("frozen merged v1" if stage == "dpo" else None), "v1_adapter_sha256": adapter_digest(cfg["paths"]["v1_adapter"]) if stage == "dpo" else None})


def adapter_digest(folder):
    folder = Path(folder)
    return digest(sorted((p.name, digest(p)) for p in folder.iterdir() if p.name.startswith("adapter_") and p.is_file()))


def evaluate(cfg, version, dry):
    rows, manifest = training_data(cfg, "test")
    adapters = () if version == "base" else (("v1_adapter",) if version == "v1" else ("v1_adapter", "v2_adapter"))
    local_guard(cfg, adapters)
    if dry:
        print(json.dumps({"dry_run": True, "version": version, "test_rows": len(rows)}, indent=2))
        return
    import torch
    from peft import PeftModel
    tok, model = tokenizer_and_model(cfg)
    if version != "base":
        model = PeftModel.from_pretrained(model, cfg["paths"]["v1_adapter"], is_trainable=False)
        if version == "v2":
            model = model.merge_and_unload()
            model = PeftModel.from_pretrained(model, cfg["paths"]["v2_adapter"], is_trainable=False)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    output = []
    for row in rows:
        prompt = prompt_text(tok, row["prompt"])
        encoded = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            ids = model.generate(**encoded, max_new_tokens=cfg["generation"]["max_new_tokens"], do_sample=False, pad_token_id=tok.pad_token_id)
        generated = ids[0][encoded["input_ids"].shape[1]:]
        output.append({"prompt_id": row["prompt_id"], "prompt": row["prompt"], "generated": tok.decode(generated, skip_special_tokens=True), "generated_tokens": len(generated)})
    run_dir = Path(cfg["paths"]["runs"])
    write_jsonl(run_dir / f"eval_{version}.jsonl", output)
    record(cfg, f"eval_{version}", {"test_ids_sha256": manifest["splits"]["test"]["ids_sha256"], "chat_template_sha256": digest(tok.chat_template), "generation": cfg["generation"], "samples": len(output)})


def summarize(cfg):
    run_dir = Path(cfg["paths"]["runs"])
    summary = []
    expected = None
    for version in ("base", "v1", "v2"):
        rows = read_jsonl(run_dir / f"eval_{version}.jsonl")
        meta = json.loads((run_dir / f"eval_{version}.json").read_text(encoding="utf-8"))
        ids = [r["prompt_id"] for r in rows]
        signature = (ids, meta["test_ids_sha256"], meta["chat_template_sha256"], meta["generation"])
        if expected is not None and signature != expected:
            raise ValueError("evaluation prompt/template/generation mismatch")
        expected = signature
        summary.append({"version": version, "samples": len(rows), "mean_generated_tokens": round(sum(r["generated_tokens"] for r in rows) / len(rows), 2), "test_ids_sha256": meta["test_ids_sha256"]})
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "prepare", "sft", "dpo", "evaluate", "summarize"))
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--version", choices=("base", "v1", "v2"))
    args = parser.parse_args()
    cfg = config(args.config)
    if args.command == "check":
        print("configuration valid")
    elif args.command == "prepare":
        prepare(cfg, args.dry_run)
    elif args.command in ("sft", "dpo"):
        train(cfg, args.command, args.dry_run)
    elif args.command == "evaluate":
        if args.version is None:
            parser.error("evaluate requires --version")
        evaluate(cfg, args.version, args.dry_run)
    else:
        summarize(cfg)


if __name__ == "__main__":
    main()
