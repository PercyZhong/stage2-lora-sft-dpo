import json, tempfile, unittest, zipfile
from pathlib import Path
from scripts.archive_stage2 import create_archive, sha256

class ArchiveTests(unittest.TestCase):
    def test_archive_checks_hashes_and_contains_only_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); names=("data/prepared/test.jsonl","runs/eval_base.jsonl","runs/eval_v1.jsonl","runs/eval_v2.jsonl")
            for n in names:
                p=root/n; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(n,encoding="utf-8")
            (root/"runs/quality_eval").mkdir(parents=True); (root/"runs/quality_eval/blind_sheet.csv").write_text("x",encoding="utf-8")
            protected={"prepared_test":sha256(root/names[0]), **{f"eval_{v}_rows":sha256(root/f"runs/eval_{v}.jsonl") for v in ("base","v1","v2")}}
            (root/"runs/quality_eval/state.json").write_text(json.dumps({"protected_hashes":protected}),encoding="utf-8")
            out=create_archive(root,root/"a.zip")
            with zipfile.ZipFile(out) as z: self.assertIn("stage2_phase2_report.md",z.namelist()); self.assertNotIn("models/weights.bin",z.namelist())
if __name__ == "__main__": unittest.main()
