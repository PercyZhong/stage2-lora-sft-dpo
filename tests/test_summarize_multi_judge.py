import unittest

from scripts.summarize_multi_judge import (
    agreement_summary,
    majority,
    resolve_judge_row,
    summarize,
)


class MultiJudgeTests(unittest.TestCase):
    def test_resolve_anonymous_choices_to_canonical_pairs(self):
        mapping = {"A": "v2", "B": "base", "C": "v1"}
        scores = {"A_vs_B": "A", "A_vs_C": "tie", "B_vs_C": "C"}
        resolved = resolve_judge_row(scores, mapping)
        self.assertEqual(resolved[("base", "v2")], "second")
        self.assertEqual(resolved[("v1", "v2")], "tie")
        self.assertEqual(resolved[("base", "v1")], "second")

    def test_majority_and_no_consensus(self):
        self.assertEqual(majority(["first", "first", "second"])[0], "first")
        self.assertEqual(majority(["tie", "tie", "first"])[0], "tie")
        self.assertEqual(majority(["first", "tie", "second"])[0], "no_consensus")

    def test_agreement_counts_and_kappa(self):
        judges = ("a", "b", "c")
        items = [
            {"votes": ["first", "first", "first"], "consensus": "first", "by_judge": {"a": "first", "b": "first", "c": "first"}},
            {"votes": ["first", "tie", "second"], "consensus": "no_consensus", "by_judge": {"a": "first", "b": "tie", "c": "second"}},
        ]
        result = agreement_summary(items, judges)
        self.assertEqual(result["unanimous"], 1)
        self.assertEqual(result["no_consensus"], 1)
        self.assertEqual(result["pairwise"]["a_vs_b"]["agree"], 1)
        self.assertIsInstance(result["fleiss_kappa_three_categories"], float)

    def test_summary_keeps_each_judge_and_majority_separate(self):
        ids = ["x"]
        sample = {"prompt_ids": ids}
        key = [{"prompt_id": "x", "A": "base", "B": "v1", "C": "v2"}]
        base_row = {"prompt_id": "x", "A_vs_B": "A", "A_vs_C": "A", "B_vs_C": "B", "reason": "r"}
        judges = {
            "j1": [base_row],
            "j2": [base_row],
            "j3": [{**base_row, "A_vs_B": "B"}],
        }
        summary, rows = summarize(sample, key, judges)
        self.assertEqual(summary["majority_consensus"]["base_vs_v1"]["first_wins"], 1)
        self.assertEqual(summary["individual"]["j3"]["base_vs_v1"]["first_losses"], 1)
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
