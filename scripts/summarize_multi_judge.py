"""Unblind and summarize three completed anonymous judge score sheets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


VERSIONS = ("base", "v1", "v2")
CANONICAL_PAIRS = (("base", "v1"), ("v1", "v2"), ("base", "v2"))
PAIR_FIELDS = (("A_vs_B", "A", "B"), ("A_vs_C", "A", "C"), ("B_vs_C", "B", "C"))
PROTECTED_FIELDS = ("prompt_id", "prompt", "A_response", "B_response", "C_response")
RULES = {
    "A_vs_B": {"A", "B", "tie"},
    "A_vs_C": {"A", "C", "tie"},
    "B_vs_C": {"B", "C", "tie"},
}
OUTCOMES = ("first", "tie", "second")


def read_json(pathname):
    return json.loads(Path(pathname).read_text(encoding="utf-8"))


def read_csv(pathname):
    with Path(pathname).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest_bytes(pathname):
    return hashlib.sha256(Path(pathname).read_bytes()).hexdigest()


def digest_value(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_inputs(sample, key_payload, score_files):
    ids = sample.get("prompt_ids")
    if sample.get("sample_size") != 60 or not isinstance(ids, list) or len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError("sample_ids.json must contain 60 unique prompt IDs")
    if sample.get("prompt_ids_sha256") != digest_value(ids):
        raise ValueError("sample ID hash mismatch")
    key_rows = key_payload.get("mapping")
    if key_payload.get("seed") != sample.get("seed") or not isinstance(key_rows, list):
        raise ValueError("blind key seed or mapping is invalid")
    if [row.get("prompt_id") for row in key_rows] != ids:
        raise ValueError("blind key ID order differs from the sample plan")
    for row in key_rows:
        if sorted(row.get(letter) for letter in "ABC") != sorted(VERSIONS):
            raise ValueError(f"invalid blind mapping for {row.get('prompt_id')}")

    judges = {}
    reference = None
    for judge, pathname in score_files.items():
        rows = read_csv(pathname)
        if len(rows) != 60 or [row.get("prompt_id") for row in rows] != ids:
            raise ValueError(f"{judge}: score rows or ID order differs from sample plan")
        if len({row["prompt_id"] for row in rows}) != 60:
            raise ValueError(f"{judge}: duplicate prompt IDs")
        for index, row in enumerate(rows, 1):
            if any(row.get(field) not in allowed for field, allowed in RULES.items()):
                raise ValueError(f"{judge} row {index}: illegal score")
            if not row.get("reason", "").strip():
                raise ValueError(f"{judge} row {index}: empty reason")
        protected = [{field: row.get(field) for field in PROTECTED_FIELDS} for row in rows]
        if reference is None:
            reference = protected
        elif protected != reference:
            raise ValueError(f"{judge}: prompt or anonymous response content differs")
        judges[judge] = rows
    if len(judges) != 3:
        raise ValueError("exactly three judge files are required")
    return ids, key_rows, judges


def canonical_pair(left, right):
    return next(pair for pair in CANONICAL_PAIRS if set(pair) == {left, right})


def resolve_judge_row(score_row, mapping):
    resolved = {}
    for field, left_letter, right_letter in PAIR_FIELDS:
        left, right = mapping[left_letter], mapping[right_letter]
        pair = canonical_pair(left, right)
        selected = score_row[field]
        winner = "tie" if selected == "tie" else mapping[selected]
        outcome = "tie" if winner == "tie" else ("first" if winner == pair[0] else "second")
        resolved[pair] = outcome
    if set(resolved) != set(CANONICAL_PAIRS):
        raise AssertionError("one resolved vote per canonical model pair is required")
    return resolved


def majority(votes):
    counts = {outcome: votes.count(outcome) for outcome in OUTCOMES}
    best = max(counts.values())
    winners = [outcome for outcome, count in counts.items() if count == best]
    return (winners[0] if best >= 2 and len(winners) == 1 else "no_consensus"), counts


def wilson_interval(wins, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def agreement_summary(items, judge_names):
    unanimous = sum(len(set(item["votes"])) == 1 for item in items)
    no_consensus = sum(item["consensus"] == "no_consensus" for item in items)
    pairwise = {}
    for first, second in itertools.combinations(judge_names, 2):
        agree = sum(item["by_judge"][first] == item["by_judge"][second] for item in items)
        pairwise[f"{first}_vs_{second}"] = {"agree": agree, "n": len(items), "fraction": agree / len(items)}

    n_raters = len(judge_names)
    observed = []
    category_totals = {outcome: 0 for outcome in OUTCOMES}
    for item in items:
        counts = [item["votes"].count(outcome) for outcome in OUTCOMES]
        observed.append((sum(count * count for count in counts) - n_raters) / (n_raters * (n_raters - 1)))
        for outcome, count in zip(OUTCOMES, counts):
            category_totals[outcome] += count
    p_bar = sum(observed) / len(observed)
    total_ratings = len(items) * n_raters
    p_e = sum((count / total_ratings) ** 2 for count in category_totals.values())
    kappa = (p_bar - p_e) / (1 - p_e) if p_e < 1 else None
    return {
        "items": len(items),
        "unanimous": unanimous,
        "unanimous_fraction": unanimous / len(items),
        "two_of_three_majority": len(items) - unanimous - no_consensus,
        "no_consensus": no_consensus,
        "pairwise": pairwise,
        "fleiss_kappa_three_categories": kappa,
        "fleiss_categories": list(OUTCOMES),
    }


def summarize(sample, key_rows, judges):
    judge_names = tuple(judges)
    key_by_id = {row["prompt_id"]: row for row in key_rows}
    rows_by_judge = {judge: {row["prompt_id"]: row for row in rows} for judge, rows in judges.items()}
    items = []
    detail_rows = []
    for pid in sample["prompt_ids"]:
        mapping = key_by_id[pid]
        resolved = {judge: resolve_judge_row(rows_by_judge[judge][pid], mapping) for judge in judge_names}
        for pair in CANONICAL_PAIRS:
            votes = [resolved[judge][pair] for judge in judge_names]
            consensus, counts = majority(votes)
            item = {
                "prompt_id": pid,
                "pair": pair,
                "comparison": f"{pair[0]}_vs_{pair[1]}",
                "votes": votes,
                "by_judge": {judge: resolved[judge][pair] for judge in judge_names},
                "consensus": consensus,
                "counts": counts,
            }
            items.append(item)
            detail = {
                "prompt_id": pid,
                "comparison": item["comparison"],
                **{f"{judge}_outcome": item["by_judge"][judge] for judge in judge_names},
                "first_votes": counts["first"],
                "tie_votes": counts["tie"],
                "second_votes": counts["second"],
                "consensus": consensus,
            }
            for judge in judge_names:
                detail[f"{judge}_reason"] = rows_by_judge[judge][pid]["reason"]
            detail_rows.append(detail)

    individual = {}
    for judge in judge_names:
        individual[judge] = {}
        for pair in CANONICAL_PAIRS:
            values = [item["by_judge"][judge] for item in items if item["pair"] == pair]
            individual[judge][f"{pair[0]}_vs_{pair[1]}"] = {
                "first_wins": values.count("first"),
                "ties": values.count("tie"),
                "first_losses": values.count("second"),
                "n": len(values),
            }

    consensus_summary = {}
    for pair in CANONICAL_PAIRS:
        values = [item["consensus"] for item in items if item["pair"] == pair]
        wins, ties, losses = values.count("first"), values.count("tie"), values.count("second")
        decisive = wins + losses
        consensus_summary[f"{pair[0]}_vs_{pair[1]}"] = {
            "first_wins": wins,
            "ties": ties,
            "first_losses": losses,
            "no_consensus": values.count("no_consensus"),
            "n": len(values),
            "decisive_n": decisive,
            "first_win_fraction_among_decisive": wins / decisive if decisive else None,
            "wilson_95_ci_among_decisive": wilson_interval(wins, decisive),
        }

    per_pair_agreement = {}
    for pair in CANONICAL_PAIRS:
        pair_items = [item for item in items if item["pair"] == pair]
        per_pair_agreement[f"{pair[0]}_vs_{pair[1]}"] = agreement_summary(pair_items, judge_names)
    return {
        "sample_size": 60,
        "judges": list(judge_names),
        "individual": individual,
        "majority_consensus": consensus_summary,
        "agreement_overall": agreement_summary(items, judge_names),
        "agreement_by_pair": per_pair_agreement,
    }, detail_rows


def write_csv(pathname, rows):
    with Path(pathname).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(summary, provenance):
    lines = [
        "# Three-judge anonymous comparison",
        "",
        "Three separately produced score sheets were unblinded only after all 60 rows were complete. Each prompt contributes one judgment per model pair from each judge. Majority consensus requires at least two matching votes; one first-win, one tie and one second-win is recorded as no consensus.",
        "",
        "## Majority consensus",
        "",
        "| comparison | first wins | ties | first losses | no consensus | decisive 95% Wilson CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, values in summary["majority_consensus"].items():
        low, high = values["wilson_95_ci_among_decisive"]
        interval = "undefined" if low is None else f"[{low:.3f}, {high:.3f}]"
        lines.append(f"| {name} | {values['first_wins']} | {values['ties']} | {values['first_losses']} | {values['no_consensus']} | {interval} |")
    lines += ["", "## Individual judges", ""]
    for judge, comparisons in summary["individual"].items():
        lines.append(f"### {judge}")
        lines.append("")
        lines.append("| comparison | first wins | ties | first losses |")
        lines.append("|---|---:|---:|---:|")
        for name, values in comparisons.items():
            lines.append(f"| {name} | {values['first_wins']} | {values['ties']} | {values['first_losses']} |")
        lines.append("")
    agreement = summary["agreement_overall"]
    lines += [
        "## Agreement",
        "",
        f"Across 180 prompt-pair items, all three judges agreed on {agreement['unanimous']} ({agreement['unanimous_fraction']:.1%}); {agreement['no_consensus']} had one vote in each category. Fleiss κ over first/tie/second was {agreement['fleiss_kappa_three_categories']:.3f}.",
        "",
    ]
    for name, values in agreement["pairwise"].items():
        lines.append(f"- {name}: {values['agree']}/{values['n']} agreement ({values['fraction']:.1%}).")
    lines += [
        "",
        "## Limits",
        "",
        "The judges are language models rather than independent human raters and can share training-data and stylistic biases. The Qwen judge also belongs to the broader model family being evaluated, which can introduce family-specific bias. Randomized A/B/C order reduces fixed position bias but cannot remove judge bias. The comparison uses 60 uniformly sampled prompts, deterministic generations and a 512-token cap; some outputs may still be truncated. Majority voting reduces single-judge idiosyncrasy but does not establish factual correctness. Results remain exploratory and apply to this internal UltraFeedback-derived test sample.",
        "",
        "## Input hashes",
        "",
    ]
    for name, value in provenance.items():
        lines.append(f"- {name}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ids", default="runs/quality_eval/sample_ids.json")
    parser.add_argument("--blind-key", default="runs/quality_eval/blind_key.json")
    parser.add_argument(
        "--chatgpt-work",
        default="runs/quality_eval/judge_scores/chatgpt_work_scores.csv",
    )
    parser.add_argument("--deepseek", default="runs/quality_eval/judge_scores/deepseek_scores.csv")
    parser.add_argument("--qwen", default="runs/quality_eval/judge_scores/qwen_scores.csv")
    parser.add_argument("--output-dir", default="runs/quality_eval/multi_judge")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = {
        "sample_ids": Path(args.sample_ids),
        "blind_key": Path(args.blind_key),
        "chatgpt_work": Path(args.chatgpt_work),
        "deepseek": Path(args.deepseek),
        "qwen": Path(args.qwen),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise FileNotFoundError(f"missing inputs: {missing}")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    sample, key = read_json(paths["sample_ids"]), read_json(paths["blind_key"])
    score_files = {name: paths[name] for name in ("chatgpt_work", "deepseek", "qwen")}
    _, key_rows, judges = validate_inputs(sample, key, score_files)
    summary, details = summarize(sample, key_rows, judges)
    provenance = {name: digest_bytes(path) for name, path in paths.items()}
    payload = {"provenance": provenance, **summary}
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "rows.csv", details)
    (output / "report.md").write_text(report_text(summary, provenance), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
