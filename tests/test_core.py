import tempfile
import unittest
from pathlib import Path

from experiment.core import config, digest, split_rows, validate_row


def row(pid, prompt=None):
    prompt = prompt or pid
    return {"prompt_id": pid, "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "yes"}], "rejected": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "no"}]}


class DataTests(unittest.TestCase):
    def test_disjoint_reproducible_splits(self):
        data = [row(str(i)) for i in range(40)] + [row("0")]
        fractions = {"sft": .4, "dpo": .4, "validation": .1, "test": .1}
        a, counts = split_rows(data, 42, 40, fractions)
        b, _ = split_rows(data, 42, 40, fractions)
        self.assertEqual(digest(a), digest(b))
        self.assertEqual(counts["duplicate_ids"], 1)
        ids = [[r["prompt_id"] for r in rows] for rows in a.values()]
        self.assertEqual(sum(map(len, ids)), len(set().union(*map(set, ids))))

    def test_reject_conflicting_duplicate_and_shared_prompt(self):
        fractions = {"sft": .4, "dpo": .4, "validation": .1, "test": .1}
        with self.assertRaisesRegex(ValueError, "conflicting prompt text"):
            split_rows([row("x"), row("x", "other")], 1, 10, fractions)
        with self.assertRaisesRegex(ValueError, "same prompt"):
            split_rows([row("x", "same"), row("y", "same")], 1, 10, fractions)

    def test_alternative_pairs_same_id_are_deduplicated_independent_of_order(self):
        data = [row(str(i)) for i in range(20)]
        alternative = row("0")
        alternative["chosen"][1]["content"] = "another valid answer"
        data.append(alternative)
        fractions = {"sft": .4, "dpo": .4, "validation": .1, "test": .1}
        forward, counts = split_rows(data, 42, 20, fractions)
        reversed_input, reversed_counts = split_rows(reversed(data), 42, 20, fractions)
        self.assertEqual(digest(forward), digest(reversed_input))
        self.assertEqual(counts, reversed_counts)
        self.assertEqual(counts["duplicate_ids"], 1)
        self.assertEqual(counts["duplicate_ids_conflicting"], 1)
        self.assertEqual(counts["valid_unique"], 20)

    def test_reject_mismatched_pair(self):
        bad = row("x")
        bad["rejected"][0]["content"] = "different"
        with self.assertRaisesRegex(ValueError, "prompts differ"):
            validate_row(bad)

    def test_identical_answers_are_counted_and_excluded(self):
        data = [row(str(i)) for i in range(20)]
        bad = row("bad")
        bad["rejected"][1]["content"] = " yes "
        data.insert(0, bad)
        fractions = {"sft": .4, "dpo": .4, "validation": .1, "test": .1}
        partitions, counts = split_rows(data, 42, 20, fractions)
        self.assertEqual(counts["input"], 21)
        self.assertEqual(counts["filtered_identical_answers"], 1)
        self.assertEqual(counts["valid_unique"], 20)
        self.assertNotIn("bad", {r["prompt_id"] for part in partitions.values() for r in part})

    def test_empty_content_is_counted_and_excluded(self):
        data = [row(str(i)) for i in range(20)]
        empty_answer = row("empty_answer")
        empty_answer["chosen"][1]["content"] = "  "
        empty_prompt = row("empty_prompt")
        empty_prompt["chosen"][0]["content"] = " "
        data.extend((empty_answer, empty_prompt))
        fractions = {"sft": .4, "dpo": .4, "validation": .1, "test": .1}
        partitions, counts = split_rows(data, 42, 20, fractions)
        self.assertEqual(counts["input"], 22)
        self.assertEqual(counts["filtered_empty_content"], 2)
        self.assertEqual(counts["valid_unique"], 20)
        self.assertEqual(sum(map(len, partitions.values())), 20)

    def test_non_string_content_still_fails(self):
        bad = row("bad")
        bad["chosen"][1]["content"] = None
        with self.assertRaisesRegex(ValueError, "non-string content"):
            validate_row(bad)

    def test_config_rejects_bad_fractions(self):
        import json
        original = json.loads((Path(__file__).parent.parent / "config/experiment.json").read_text(encoding="utf-8"))
        original["data"]["split_fractions"]["test"] = .2
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "config.json"
            file.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split fractions"):
                config(file)


if __name__ == "__main__":
    unittest.main()
