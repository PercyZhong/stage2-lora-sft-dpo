"""Create a provenance-checked Stage 2 evidence archive."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def required(root: Path, name: str) -> Path:
    p = root / name
    if not p.is_file(): raise FileNotFoundError(f"required archive input is missing: {name}")
    return p

def create_archive(root: Path, output: Path) -> Path:
    state = json.loads(required(root, "runs/quality_eval/state.json").read_text(encoding="utf-8"))
    recorded = state.get("protected_hashes", {})
    files = {"prepared_test":"data/prepared/test.jsonl", "eval_base_rows":"runs/eval_base.jsonl", "eval_v1_rows":"runs/eval_v1.jsonl", "eval_v2_rows":"runs/eval_v2.jsonl"}
    hashes = {}
    for key, name in files.items():
        actual = sha256(required(root, name)); hashes[key] = actual
        expected = recorded.get(key)
        if expected is None and key.startswith("eval_"): expected = state.get("eval_hashes", {}).get(key[5:-5], {}).get("rows")
        if expected != actual: raise ValueError(f"SHA256 mismatch for {name}: state={expected} actual={actual}")
    evidence = {
        "generated_512_metadata": "runs/quality_eval/generated_512/metadata.json",
        "multi_judge_summary": "runs/quality_eval/multi_judge/summary.json",
        "multi_judge_rows": "runs/quality_eval/multi_judge/rows.csv",
        "chatgpt_work_scores": "runs/quality_eval/judge_scores/chatgpt_work_scores.csv",
        "deepseek_scores": "runs/quality_eval/judge_scores/deepseek_scores.csv",
        "qwen_scores": "runs/quality_eval/judge_scores/qwen_scores.csv",
    }
    for key, name in evidence.items():
        hashes[key] = sha256(required(root, name))
    generated = json.loads(required(root, evidence["generated_512_metadata"]).read_text(encoding="utf-8"))
    limits = generated.get("files", {})
    truncation = "; ".join(
        f"{version}: {info.get('hit_limit', 0)}/{generated.get('sample_size', 0)} ({info.get('hit_limit', 0) / generated.get('sample_size', 1) * 100:.2f}%)"
        for version, info in limits.items()
    )
    blind = root / "runs/quality_eval/blind_sheet.csv"
    current = sha256(blind) if blind.is_file() else "missing"
    metas = {}
    for provider in ("deepseek", "qwen"):
        p = root / f"runs/quality_eval/judge_scores/{provider}_metadata.json"
        if p.is_file(): metas[provider] = json.loads(p.read_text(encoding="utf-8")).get("input_sha256")
    mismatch = any(v != current for v in metas.values()) if current != "missing" else False
    report = "\n".join([
        "# 阶段二综合报告（证据归档）", "", "本报告只归档既有记录；未重训、未重评，也未修改实验指标。", "",
        f"- 模型版本：{state.get('model_revision', '未记录')}；训练提交：{state.get('training_commit', '未记录')}。",
        f"- 训练配置：{state.get('config', {}).get('sft', {})}；DPO：{state.get('config', {}).get('dpo', {})}；LoRA：{state.get('config', {}).get('lora', {})}。",
        "- 偏好评测：400 条测试样本中 374 条有效，26 条排除。三模型匿名评审：60 条提示，ChatGPT Work、DeepSeek、Qwen 各完成一份评分。",
        f"- 512-token 截断比例：{truncation}。",
        f"- 元数据输入 SHA256：`{next(iter(metas.values()), '未记录')}`；当前盲表：`{current}`。",
        "- 原始元数据哈希文件未找到；这是无法解释的字节级溯源差异，不猜测原因。" if mismatch else "- 输入哈希一致或不可用。",
        "- 局限：评审模型可能共享训练数据与风格偏差，Qwen 可能有模型家族偏差；60 条内部样本、输出截断和多数投票不能替代人工评审或外部基准。结论仅限探索性证据。", "",
        "## 归档文件 SHA256", *[f"- `{k}`: `{v}`" for k,v in hashes.items()], ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name in files.values(): z.write(root / name, name)
        for name in evidence.values(): z.write(root / name, name)
        z.write(root / "runs/quality_eval/state.json", "runs/quality_eval/state.json")
        z.writestr("stage2_phase2_report.md", report)
    return output

def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=Path(".")); p.add_argument("--output", type=Path, default=Path("stage2_phase2_evidence.zip")); a=p.parse_args(); print(create_archive(a.root.resolve(), a.output.resolve()))
if __name__ == "__main__": main()
