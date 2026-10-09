import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.api_blind_judge import (
    ALL_FIELDS,
    canonical_input_hash,
    endpoint,
    extract_json,
    read_sheet,
    validate_resume,
    validate_score,
    write_scores,
)


def rows(count=60):
    return [
        {
            "prompt_id": f"id-{index}",
            "prompt": f"prompt {index}",
            "A_response": "answer A",
            "B_response": "answer B",
            "C_response": "answer C",
            "A_vs_B": "",
            "A_vs_C": "",
            "B_vs_C": "",
            "reason": "",
        }
        for index in range(count)
    ]


def write_sheet(pathname, values):
    with Path(pathname).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=ALL_FIELDS)
        writer.writeheader()
        writer.writerows(values)


class ApiBlindJudgeTests(unittest.TestCase):
    def test_sheet_requires_60_blank_anonymous_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            pathname = Path(folder) / "blind.csv"
            write_sheet(pathname, rows())
            self.assertEqual(len(read_sheet(pathname)), 60)
            invalid = rows()
            invalid[0]["A_vs_B"] = "A"
            write_sheet(pathname, invalid)
            with self.assertRaisesRegex(ValueError, "must be blank"):
                read_sheet(pathname)

    def test_json_score_validation(self):
        raw = '```json\n{"A_vs_B":"A","A_vs_C":"tie","B_vs_C":"C","reason":"A is clearer."}\n```'
        score = validate_score(extract_json(raw))
        self.assertEqual(score["A_vs_B"], "A")
        with self.assertRaisesRegex(ValueError, "invalid B_vs_C"):
            validate_score({**score, "B_vs_C": "A"})
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_score({**score, "extra": 1})

    def test_resume_is_bound_to_provider_model_hash_and_order(self):
        source = rows()
        input_hash = canonical_input_hash(source)
        score = {"A_vs_B": "A", "A_vs_C": "tie", "B_vs_C": "B", "reason": "reason"}
        records = [{"provider": "qwen", "model": "qwen-plus", "prompt_id": source[0]["prompt_id"], "input_sha256": input_hash, "score": score}]
        self.assertEqual(validate_resume(records, source, "qwen", "qwen-plus", input_hash), [score])
        records[0]["prompt_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_resume(records, source, "qwen", "qwen-plus", input_hash)

    def test_output_preserves_anonymous_content(self):
        with tempfile.TemporaryDirectory() as folder:
            pathname = Path(folder) / "scores.csv"
            source = rows()
            scores = [{"A_vs_B": "A", "A_vs_C": "C", "B_vs_C": "tie", "reason": "reason"}] * 60
            write_scores(pathname, source, scores)
            with pathname.open(newline="", encoding="utf-8-sig") as stream:
                output = list(csv.DictReader(stream))
            self.assertEqual(output[0]["prompt"], source[0]["prompt"])
            self.assertEqual(output[0]["C_response"], source[0]["C_response"])
            self.assertEqual(output[0]["A_vs_C"], "C")

    def test_endpoint_accepts_base_or_full_url(self):
        base = "https://example.test/compatible-mode/v1"
        self.assertEqual(endpoint(base), base + "/chat/completions")
        self.assertEqual(endpoint(base + "/chat/completions"), base + "/chat/completions")


if __name__ == "__main__":
    unittest.main()
