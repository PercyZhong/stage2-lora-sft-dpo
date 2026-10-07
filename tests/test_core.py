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
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            split_rows([row("x"), row("x", "other")], 1, 10, fractions)
        with self.assertRaisesRegex(ValueError, "same prompt"):
            split_rows([row("x", "same"), row("y", "same")], 1, 10, fractions)

    def test_reject_mismatched_pair(self):
        bad = row("x")
        bad["rejected"][0]["content"] = "different"
        with self.assertRaisesRegex(ValueError, "prompts differ"):
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
