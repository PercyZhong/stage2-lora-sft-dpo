import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment.core import digest
from experiment.quality_eval import (
    answer_token_logps,
    build_blind_rows,
    margin,
    partition_tokenized_rows,
    run_check,
    sample_plan,
    summarize_blind_scores,
    validate_blind_content,
    validate_eval_rows,
    validate_scores,
    validate_test_identity,
)


def preference_row(index):
    return {
        "prompt_id": f"id-{index:03d}",
        "prompt": f"prompt {index}",
        "chosen": "chosen",
        "rejected": "rejected",
    }


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.assert_tokenize(tokenize)
        if len(messages) == 1:
            return [10, 11]
        answer = messages[1]["content"]
        answer_ids = [20] * (6 if answer == "long" else 1)
        return [10, 11] + answer_ids + [21]

    @staticmethod
    def assert_tokenize(value):
        if value is not True:
            raise AssertionError("test tokenizer expects token IDs")


class IdentityTests(unittest.TestCase):
    def test_test_id_order_and_hash_mismatch_are_rejected(self):
        rows = [preference_row(i) for i in range(400)]
        ids = [row["prompt_id"] for row in rows]
        expected_hash = digest(ids)
        manifest = {
            "splits": {
                "test": {
                    "count": 400,
                    "prompt_ids": ids,
                    "ids_sha256": expected_hash,
                    "rows_sha256": digest(rows),
                }
            }
        }
        self.assertEqual(validate_test_identity(rows, manifest, expected_hash), ids)
        reordered = rows[:]
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(ValueError, "order/hash"):
            validate_test_identity(reordered, manifest, expected_hash)
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            validate_eval_rows(rows, reordered, "v1")
        wrong_prompt = [dict(row) for row in rows]
        wrong_prompt[2]["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "prompt mismatch"):
            validate_eval_rows(rows, wrong_prompt, "v2")


class ScoringTests(unittest.TestCase):
    def test_assistant_only_mask_uses_shifted_logits(self):
        input_ids = [0, 1, 2, 3]
        logits = [
            [20.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 9.0],
        ]
        values = answer_token_logps(logits, input_ids, answer_start=2, attention_mask=[1, 1, 1, 1])
        expected_first = 8.0 - math.log(math.exp(8.0) + 3)
        expected_second = 9.0 - math.log(math.exp(9.0) + 3)
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], expected_first)
        self.assertAlmostEqual(values[1], expected_second)

    def test_equal_and_unequal_length_margins(self):
        self.assertEqual(margin(-2.0, 2, -2.0, 2), {"raw_margin": 0.0, "mean_margin": 0.0})
        result = margin(-2.0, 1, -3.0, 3)
        self.assertEqual(result["raw_margin"], 1.0)
        self.assertEqual(result["mean_margin"], -1.0)

    def test_overlong_pair_is_excluded_once_for_all_versions(self):
        rows = [
            {"prompt_id": "ok", "prompt": "p", "chosen": "short", "rejected": "short"},
            {"prompt_id": "long", "prompt": "p2", "chosen": "long", "rejected": "short"},
        ]
        valid, exclusions = partition_tokenized_rows(FakeTokenizer(), rows, context_limit=5)
        self.assertEqual([row["prompt_id"] for row in valid], ["ok"])
        self.assertEqual(set(exclusions), {"long"})
        self.assertEqual(exclusions["long"]["reason"], "pair_exceeds_context_limit")


class BlindTests(unittest.TestCase):
    def test_anonymous_mapping_restores_versions_without_sheet_leakage(self):
        rows = [preference_row(i) for i in range(6)]
        selected, key = sample_plan(rows, seed=17, sample_size=6)
        generated = {
            version: [
                {"prompt_id": row["prompt_id"], "generated": f"anonymous response {index}"}
                for index, row in enumerate(selected)
            ]
            for version in ("base", "v1", "v2")
        }
        blind = build_blind_rows(selected, generated, key)
        self.assertFalse(any(name in column.lower() for column in blind[0] for name in ("base", "v1", "v2")))
        key_by_id = {row["prompt_id"]: row for row in key}
        for row in blind:
            mapping = key_by_id[row["prompt_id"]]
            for field, left, right in (("A_vs_B", "A", "B"), ("A_vs_C", "A", "C"), ("B_vs_C", "B", "C")):
                pair = {mapping[left], mapping[right]}
                canonical_first = "base" if "base" in pair else "v1"
                row[field] = next(letter for letter in (left, right) if mapping[letter] == canonical_first)
            row["reason"] = "manual reason"
        summary, _ = summarize_blind_scores(blind, key)
        self.assertEqual(summary["base_vs_v1"]["wins"], 6)
        self.assertEqual(summary["base_vs_v2"]["wins"], 6)
        self.assertEqual(summary["v1_vs_v2"]["wins"], 6)

    def test_missing_illegal_scores_and_modified_answers_are_rejected(self):
        rows = [{
            "prompt_id": "x", "prompt": "p", "A_response": "a", "B_response": "b", "C_response": "c",
            "A_vs_B": "", "A_vs_C": "A", "B_vs_C": "tie", "reason": "why",
        }]
        with self.assertRaisesRegex(ValueError, "A_vs_B"):
            validate_scores(rows, ["x"])
        rows[0]["A_vs_B"] = "A"
        rows[0]["reason"] = ""
        with self.assertRaisesRegex(ValueError, "reason"):
            validate_scores(rows, ["x"])
        rows[0]["reason"] = "why"
        expected = [dict(rows[0])]
        rows[0]["B_response"] = "edited"
        with self.assertRaisesRegex(ValueError, "modified"):
            validate_blind_content(rows, expected)


class OutputProtectionTests(unittest.TestCase):
    def test_check_refuses_to_overwrite_without_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            template = json.loads((Path(__file__).parent.parent / "config/experiment.json").read_text(encoding="utf-8"))
            template["paths"]["runs"] = str(root / "runs")
            verified = {
                "manifest_sha256": "manifest",
                "test_ids_sha256": "ids",
                "chat_template_sha256": "template",
                "old_generation": {"max_new_tokens": 128, "do_sample": False},
                "old_limit_hits": {"base": 1, "v1": 1, "v2": 1},
                "v1_adapter_sha256": "v1",
                "protected_hashes": {"old": "hash"},
            }
            with patch("experiment.quality_eval.verify_inputs", return_value=verified), patch(
                "experiment.quality_eval.protected_hashes", return_value=verified["protected_hashes"]
            ), patch("experiment.quality_eval.environment_record", return_value={}):
                run_check(template)
                state_before = (root / "runs" / "quality_eval" / "state.json").read_bytes()
                with self.assertRaisesRegex(FileExistsError, "--resume"):
                    run_check(template)
                self.assertEqual(state_before, (root / "runs" / "quality_eval" / "state.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
