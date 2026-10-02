"""Portable source paths for copied teacher-study results."""

import json
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path

from eval.hint_compare_cache import digest
from eval.teacher_cohort import load_teacher_cohort, resolve_cohort_dir


class CohortPathTests(unittest.TestCase):
    def test_recorded_results_paths_are_project_relative(self):
        relative = Path("results/demo_gain/solution/Qwen3-1.7B")
        self.assertEqual(resolve_cohort_dir(Path("/old/machine/project") / relative), relative)
        self.assertEqual(resolve_cohort_dir(relative), relative)

    def test_explicit_overrides_and_external_paths_are_preserved(self):
        recorded = "/old/project/results/demo_gain/solution/Qwen3-1.7B"
        for override in ("copied/cohort", "/external/results/cohort"):
            self.assertEqual(resolve_cohort_dir(recorded, override), Path(override))
        self.assertEqual(resolve_cohort_dir("/external/cohort"), Path("/external/cohort"))

    def test_load_relocated_cohort_and_validate_checksum(self):
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            cohort = Path("results/demo_gain/solution/test")
            study = Path("results/teacher_uncertainty/default_hint/test")
            cohort.mkdir(parents=True)
            study.mkdir(parents=True)
            rows = [{"question_id": "q", "question_idx": 3, "question": "Question",
                     "final_answer": "42", "hint": "Hint", "solution": "Solution"}]
            identity = {"cohort_hash": digest(rows), "tokenizer_hash": "tokenizer"}
            (cohort / "manifest.json").write_text(json.dumps({
                "model": "test", "dataset": "deepmath", **identity,
            }))
            (cohort / "cohort.jsonl").write_text(json.dumps(rows[0]) + "\n")
            meta = {"teacher_model": "test", "n_problems": 1, "source": {
                "problem_model": "test", "question_ids": ["q"], "question_indices": [3],
                "cohort": {"cohort_dir": "/old/machine/project/" + str(cohort), **identity},
            }}
            (study / "teacher_uncertainty_run_meta.json").write_text(json.dumps(meta))
            selected, returned_meta = load_teacher_cohort(study)
            self.assertEqual(selected, [{**rows[0], "dataset": "deepmath"}])
            self.assertEqual(returned_meta, meta)  # Preserve original provenance.
            (cohort / "cohort.jsonl").write_text(json.dumps({**rows[0], "hint": "Changed"}) + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_teacher_cohort(study)


if __name__ == "__main__":
    unittest.main()
