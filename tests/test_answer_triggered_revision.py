"""Selection, evidence, and denominator regressions for answer-triggered revision."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval import answer_triggered_revision as ar


TEXT = "I calculate 5. The supplied answer is 7, so my result conflicts. I replace the linear model with a quadratic one."


def positive(**changes):
    return {
        "status": "yes", "candidate_answer": "5",
        "candidate_quote": "I calculate 5.",
        "mismatch_quote": "The supplied answer is 7, so my result conflicts.",
        "revision_quote": "I replace the linear model with a quadratic one.",
        "revision_type": "approach_change", "rationale": "Explicit mismatch leads to a new approach.",
        **changes,
    }


class EvidenceTests(unittest.TestCase):
    def test_exact_ordered_evidence(self):
        result = ar.parse_judgment(json.dumps(positive()), TEXT)
        for field, span in result["evidence_spans"].items():
            self.assertEqual(TEXT[span["char_start"]:span["char_end"]], result[field])

    def test_invented_reordered_missing_or_truncated_evidence_is_unresolved(self):
        for change in [
            {"candidate_quote": "I compute five."},
            {"revision_quote": "I calculate 5."},
            {"mismatch_quote": ""}, {"candidate_answer": ""},
            {"revision_type": "none"}, {"status": "no"},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ar.parse_judgment(json.dumps(positive(**change)), TEXT)
        with self.assertRaisesRegex(ValueError, "truncated"):
            ar.parse_judgment(json.dumps(positive()), TEXT, "length")

    def test_negative_and_uncertain_are_distinct(self):
        for status in ["no", "uncertain"]:
            result = ar.parse_judgment(json.dumps(positive(
                status=status, candidate_answer="", candidate_quote="", mismatch_quote="",
                revision_quote="", revision_type="none",
            )), TEXT)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["evidence_spans"], {})


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.selected = [{"trajectory_id": str(i), "question_id": str(i // 2),
                          "truncated": i == 3} for i in range(4)]

    def test_failures_never_become_negatives(self):
        judgments = [{"trajectory_id": str(i), "status": status,
                      "revision_type": "approach_change"} for i, status in
                     enumerate(["yes", "no", "uncertain", "invalid"])]
        report = ar.summarize(self.selected, judgments, 100, 42)
        self.assertEqual(report["n_selected"], 4)
        self.assertEqual(report["n_resolved"], 2)
        self.assertEqual(report["fraction_among_resolved"], .5)
        self.assertIsNone(report["fraction_of_selected"])
        self.assertEqual(report["selected_fraction_bounds"], [.25, .75])
        self.assertEqual(report["n_truncated_solver_responses"], 1)

    def test_complete_fraction_is_trajectory_weighted_and_deterministic(self):
        judgments = [{"trajectory_id": str(i), "status": "yes" if i == 0 else "no",
                      "revision_type": "local_correction"} for i in range(4)]
        report = ar.summarize(self.selected, judgments, 100, 42)
        self.assertEqual(report["fraction_of_selected"], .25)
        self.assertEqual(report, ar.summarize(self.selected, judgments, 100, 42))
        self.assertEqual(report["earliest_event_revision_types"], {"local_correction": 1})

    def test_pending_and_empty(self):
        report = ar.summarize(self.selected, [], 10)
        self.assertEqual(report["n_pending"], 4)
        self.assertEqual(report["selected_fraction_bounds"], [0, 1])
        self.assertIsNone(report["fraction_among_resolved"])
        self.assertIsNone(ar.summarize([], [], 10)["selected_fraction_bounds"])

    def test_duplicate_or_unknown_judgments_rejected(self):
        for judgments in [
            [{"trajectory_id": "x", "status": "no"}],
            [{"trajectory_id": "0", "status": "no"}] * 2,
        ]:
            with self.assertRaises(ValueError):
                ar.summarize(self.selected, judgments)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.study, self.behaviors = self.root / "study", self.root / "behaviors"
        self.problems = [{"question_id": "q0", "question_idx": 0,
                          "question": "Question", "final_answer": "7"}]
        self.meta = {"teacher_model": "base", "n_samples": 4}
        self.sources = [{
            "question_idx": 0, "question_id": "q0", "sample_idx": i,
            "text": TEXT, "n_tokens": 50, "correct": False,
            "truncated": False, "unclosed": False, "e_think": 0, "e_total": 0,
        } for i in range(4)]
        self.chunks = [{
            "question_idx": 0, "sample_idx": i, "chunk_idx": 0,
            "char_start": 0, "char_end": len(TEXT), "n_classifier_tokens": 40,
            "parse_failed": i == 2, "backtracking": [1, 0, 2, 3][i],
            "verification": 0, "subgoal_setting": 0, "backward_chaining": 0,
        } for i in range(4)]
        self.behavior_meta = {
            "status": "complete", "pi_mode": "answer", "n_trajectories": 4, "n_chunks": 4,
            "config": {"teacher_model": "base", "samples_per_problem": 4,
                       "source_fingerprint": ar.digest(self.sources)},
        }
        ar.write_rows(self.study / "completions_answer.jsonl", self.sources)
        ar.write_rows(self.behaviors / "behaviors_answer.jsonl", self.chunks)
        ar.write_json(self.behaviors / "behaviors_meta_answer.json", self.behavior_meta)
        patcher = patch.object(ar, "load_teacher_cohort", return_value=(self.problems, self.meta))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_only_successfully_judged_answer_backtracking(self):
        selected, provenance = ar.select_trajectories(self.study, self.behaviors)
        self.assertEqual([r["sample_idx"] for r in selected], [0, 3])
        self.assertEqual(provenance["n_successfully_judged"], 3)
        self.assertEqual(provenance["n_original_judge_failures"], 1)
        self.assertEqual(provenance["n_backtracking_trajectories"], 2)
        self.assertEqual(selected[0]["text"], TEXT)
        self.assertEqual(selected[0]["supplied_answer"], "7")
        user = json.loads(ar.messages(selected[0])[1]["content"])
        self.assertEqual(user["response"], TEXT)
        self.assertNotIn("backtracking_count", user)

    def test_changed_completion_wrong_pi_and_missing_chunks_rejected(self):
        self.sources[0]["text"] += " changed"
        ar.write_rows(self.study / "completions_answer.jsonl", self.sources)
        with self.assertRaisesRegex(ValueError, "do not match"):
            ar.select_trajectories(self.study, self.behaviors)
        self.sources[0]["text"] = TEXT
        ar.write_rows(self.study / "completions_answer.jsonl", self.sources)
        self.behavior_meta["pi_mode"] = "full"
        ar.write_json(self.behaviors / "behaviors_meta_answer.json", self.behavior_meta)
        with self.assertRaisesRegex(ValueError, "answer-PI"):
            ar.select_trajectories(self.study, self.behaviors)
        self.behavior_meta["pi_mode"] = "answer"
        ar.write_json(self.behaviors / "behaviors_meta_answer.json", self.behavior_meta)
        self.chunks[0]["char_end"] -= 10
        ar.write_rows(self.behaviors / "behaviors_answer.jsonl", self.chunks)
        with self.assertRaisesRegex(ValueError, "omit the end"):
            ar.select_trajectories(self.study, self.behaviors)

    def runtime(self, status="yes", prompt_length=100):
        selected, identity = ar.select_trajectories(self.study, self.behaviors)
        args = ar.build_parser().parse_args([])
        args.bootstrap_samples = 20
        args.max_model_len = 2000
        root = self.root / "output"
        tokenizer = SimpleNamespace(
            chat_template="template", get_vocab=lambda: {"a": 0},
            apply_chat_template=lambda *a, **kw: {"input_ids": [0] * prompt_length},
        )

        def chat(conversations, *a, **kw):
            for conversation in conversations:
                self.assertEqual(json.loads(conversation[1]["content"])["response"], TEXT)
            output = json.dumps(positive()) if status == "yes" else "broken JSON"
            return [SimpleNamespace(outputs=[SimpleNamespace(text=output, finish_reason="stop")])
                    for _ in conversations]

        engine = SimpleNamespace(chat=chat)
        return args, [(root, selected, identity)], tokenizer, engine

    def test_complete_runtime_and_resume_without_inference(self):
        args, jobs, tokenizer, engine = self.runtime()
        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(_commit_hash="revision")), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch.object(ar, "create_judge", return_value=(engine, None)) as loader:
            ar.classify(args, jobs)
            self.assertEqual(loader.call_count, 1)
            report = ar.read_json(jobs[0][0] / "summary.json")
            self.assertEqual(report["n_yes"], 2)
            self.assertEqual(report["fraction_of_selected"], 1)
            loader.side_effect = AssertionError("Completed cache must skip inference")
            ar.classify(args, jobs)
            self.assertEqual(ar.read_json(jobs[0][0] / "summary.json"), report)
            args.max_output_tokens += 1
            with self.assertRaisesRegex(ValueError, "Judge settings changed"):
                ar.classify(args, jobs)

    def test_runtime_preserves_invalid_labels(self):
        args, jobs, tokenizer, engine = self.runtime(status="invalid")
        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(_commit_hash="revision")), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch.object(ar, "create_judge", return_value=(engine, None)):
            ar.classify(args, jobs)
        report = ar.read_json(jobs[0][0] / "summary.json")
        self.assertEqual(report["n_invalid"], 2)
        self.assertEqual(report["n_no"], 0)
        self.assertIsNone(report["fraction_of_selected"])
        self.assertEqual(report["selected_fraction_bounds"], [0, 1])

    def test_overflow_does_not_truncate_or_load_judge(self):
        args, jobs, tokenizer, engine = self.runtime(prompt_length=3000)
        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(_commit_hash="revision")), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch.object(ar, "create_judge", side_effect=AssertionError("No fitting requests")):
            ar.classify(args, jobs)
        report = ar.read_json(jobs[0][0] / "summary.json")
        self.assertEqual(report["n_context_overflow"], 2)
        self.assertEqual(report["n_no"], 0)


if __name__ == "__main__":
    unittest.main()
