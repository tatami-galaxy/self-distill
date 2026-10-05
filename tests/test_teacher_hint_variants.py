"""Cohort, estimator, provenance, and interrupted-run checks without GPU inference."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval import teacher_hint_variant_behaviors as cognitive
from eval import teacher_hint_variants as study
from eval.teacher_hint_common import (
    LEVELS,
    digest,
    load_rows,
    paired_metrics,
    save_rows,
    validate_completions,
    write_json,
)
from utils import gen_teacher_hint_variants as hints


class Tokenizer:
    chat_template = "test-template"
    is_fast = False

    def get_vocab(self):
        return {"test": 0}

    def __call__(self, texts, **kwargs):
        def ids(text):
            return list(range(max(1, len(text.split()))))

        return {
            "input_ids": [ids(t) for t in texts]
            if isinstance(texts, list)
            else ids(texts)
        }

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        text = json.dumps(messages)
        return list(range(50000 if "OVERFLOW" in text else 5)) if tokenize else text


def source_rows():
    return [
        {
            "question_id": f"q{i}",
            "question_idx": i,
            "question": f"Problem {i}",
            "final_answer": "2",
            "solution": "private reasoning</think>worked answer",
            "demo_hash": f"demo{i}",
            "dataset": "deepmath",
        }
        for i in (8, 2, 6)
    ]


def hint_rows(problems):
    return [
        {
            "question_id": p["question_id"],
            "question_idx": p["question_idx"],
            "demo_hash": p["demo_hash"],
            "condition": level,
            "sample_idx": 0,
            "hint": "Try factoring.",
            "invalid_reason": "",
            "truncated": False,
            "n_tokens": 4,
        }
        for p in problems
        for level in LEVELS
    ]


class FakeSolver:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.requests = 0
        self.fail_after = fail_after

    def generate(self, prompts, params):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("interrupted")
        self.calls += 1
        self.requests += len(prompts)
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="Check the factors. \\boxed{2}",
                        token_ids=[1, 2, 3, 4],
                        finish_reason="stop",
                    )
                    for _ in range(param.n)
                ]
            )
            for param in params
        ]


class FakeJudge:
    def chat(self, conversations, params, **kwargs):
        text = json.dumps(
            {
                "verification": 2,
                "backtracking": 1,
                "subgoal_setting": 0,
                "backward_chaining": 0,
            }
        )
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for _ in conversations
        ]


class CohortTests(unittest.TestCase):
    def test_shared_intersection_preserves_order_and_rejects_bad_sources(self):
        problems = source_rows()
        rows = hint_rows(problems)
        rows[3]["truncated"] = True
        selected, excluded = study.select_problems(problems, rows)
        self.assertEqual([p["question_idx"] for p in selected], [8, 6])
        self.assertEqual(excluded[0]["question_id"], "q2")
        with self.assertRaisesRegex(ValueError, "Missing"):
            study.select_problems(problems, rows[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            study.select_problems(problems, rows + rows[:1])
        rows[0]["demo_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            study.select_problems(problems, rows)

    def test_only_hint_text_changes_and_reference_does_not_enter_teacher_prompt(self):
        problem = study.select_problems(source_rows(), hint_rows(source_rows()))[0][0]
        for level in LEVELS:
            messages = study.messages_for(problem, level)
            self.assertNotIn("private reasoning", str(messages))
            self.assertIn("Try factoring.", messages[-1]["content"])
        self.assertNotIn("Try factoring.", str(study.messages_for(problem, "none")))

    def test_target_does_not_truncate_hint_text_and_diagnostics_keep_invalid(self):
        messages = hints.build_messages(source_rows()[0], "hint_short", 32)
        self.assertIn("ONE short sentence", messages[-1]["content"])
        rows = hint_rows(source_rows())
        rows[0]["invalid_reason"] = "answer_leak"
        rows[1]["truncated"] = True
        report = hints.diagnostics(rows)
        self.assertEqual(report["n_questions"], 3)
        self.assertEqual(report["n_valid_triplets"], 2)


class StatisticsTests(unittest.TestCase):
    def test_passk_uses_multiple_responses_not_any_correct_as_pass1(self):
        rows = [{"question_idx": 0, "correct": c} for c in [True, False, False, True]]
        metrics = study.accuracy_rows(rows, [1, 2, 4])[0]
        self.assertAlmostEqual(metrics["pass@1"], 0.5)
        self.assertAlmostEqual(metrics["pass@2"], 5 / 6)
        self.assertEqual(metrics["pass@4"], 1)
        with self.assertRaises(ValueError):
            study.accuracy_rows(rows, [8])

    def test_pooled_rates_paired_zero_and_complete_question_filter(self):
        rows = [
            {"question_idx": q, "sample_idx": i, "count": count, "tokens": tokens}
            for q, count, tokens in [(0, 1, 10), (1, 9, 100)]
            for i in range(2)
        ]
        metrics = {"rate": (lambda r: r["count"], lambda r: r["tokens"])}
        estimate = paired_metrics(rows, metrics, 100)
        self.assertAlmostEqual(estimate["metrics"]["rate"]["mean"], 10 / 110)
        paired = paired_metrics(rows, metrics, 100, baseline=rows, complete_n=2)
        self.assertEqual(paired["metrics"]["rate"], {"delta": 0.0, "ci95": [0.0, 0.0]})
        paired = paired_metrics(rows[:-1], metrics, 100, baseline=rows, complete_n=2)
        self.assertEqual(paired["question_indices"], [0])

    def test_missing_or_duplicated_chunks_are_not_partial_trajectories(self):
        chunks = [{"question_idx": 8, "sample_idx": 0, "chunk_idx": 0}]
        with self.assertRaises(ValueError):
            cognitive.validate_chunks(chunks, {"8:0": 2})
        with self.assertRaises(ValueError):
            cognitive.validate_chunks(chunks * 2, {"8:0": 1})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "study"
        self.hint_root = Path(self.tmp.name) / "hints"
        from eval.demo_gain import tokenizer_hash

        self.hint_config = {
            "model": "test",
            "revision": "test-rev",
            "tokenizer_hash": tokenizer_hash(Tokenizer()),
            "source_meta_hash": "source-hash",
            "problems": source_rows(),
        }
        write_json(self.hint_root / "experiment.json", self.hint_config)
        save_rows(self.hint_root, "hints", hint_rows(source_rows()), self.hint_config)
        self.args = study.build_parser().parse_args(
            [
                "--hints-dir",
                str(self.hint_root),
                "--output-dir",
                str(self.root),
                "--n",
                "2",
                "--batch-size",
                "1",
                "--bootstrap-samples",
                "50",
            ]
        )
        self.patch_tokenizer = patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
        )
        self.patch_tokenizer.start()
        self.addCleanup(self.patch_tokenizer.stop)

    def prepared(self):
        return study.prepare(self.args)

    def generated(self):
        experiment = self.prepared()
        with patch("vllm.LLM", return_value=FakeSolver()):
            study.generate(self.args, experiment)
        return experiment

    def test_dry_run_no_writes_context_exclusion_and_configuration_freeze(self):
        self.args.dry_run = True
        study.prepare(self.args)
        self.assertFalse(self.root.exists())
        self.args.dry_run = False
        self.prepared()
        self.args.n = 4
        with self.assertRaisesRegex(ValueError, "Configuration"):
            self.prepared()

    def test_context_overflow_drops_question_from_every_arm(self):
        rows = hint_rows(source_rows())
        rows[0]["hint"] = "OVERFLOW"
        save_rows(self.hint_root, "hints", rows, self.hint_config)
        experiment = self.prepared()
        self.assertEqual(len(experiment["problems"]), 2)
        self.assertEqual(experiment["excluded"][0]["question_id"], "q8")

    def test_review_catches_false_negatives_and_requires_complete_exact_text(self):
        rows = hint_rows(source_rows())
        rows[0]["invalid_reason"] = "answer_leak"
        save_rows(self.hint_root, "hints", rows, self.hint_config)
        review = {
            "version": 1,
            "hints_hash": digest(rows),
            "reviews": [
                {
                    "question_id": r["question_id"],
                    "condition": r["condition"],
                    "hint_hash": digest(r["hint"]),
                    "decision": "reject" if r["question_id"] == "q2" else "accept",
                    "reason": "reviewed explicit answer"
                    if r["question_id"] == "q2"
                    else "guidance without final answer",
                }
                for r in rows
            ],
        }
        path = Path(self.tmp.name) / "review.json"
        write_json(path, review)
        decisions, _ = study.read_hint_reviews(path, rows, digest(rows))
        selected, _ = study.select_problems(source_rows(), rows, decisions)
        self.assertEqual([p["question_idx"] for p in selected], [8, 6])
        self.args.hint_review = str(path)
        experiment = self.prepared()
        self.assertEqual(experiment["hint_validation"], "reviewed")
        review["reviews"][0]["reason"] = "changed"
        write_json(path, review)
        with self.assertRaisesRegex(ValueError, "review changed"):
            study.load_experiment(self.root)
        review["reviews"][0]["hint_hash"] = "wrong-text"
        write_json(path, review)
        with self.assertRaisesRegex(ValueError, "changed hint"):
            study.read_hint_reviews(path, rows, digest(rows))
        review["reviews"].pop(0)
        write_json(path, review)
        with self.assertRaisesRegex(ValueError, "every hint"):
            study.read_hint_reviews(path, rows, digest(rows))

    def test_hint_generation_resumes_without_loading_weights(self):
        root = Path(self.tmp.name) / "generated_hints"
        args = hints.build_parser().parse_args(
            ["--teacher-study-dir", "source", "--output-dir", str(root)]
        )
        source = {
            "teacher_model": "test",
            "generation": {
                "revision": "test-rev",
                "tokenizer_hash": self.hint_config["tokenizer_hash"],
            },
        }
        solver = FakeSolver()
        with (
            patch.object(
                hints, "load_teacher_cohort", return_value=(source_rows(), source)
            ),
            patch("vllm.LLM", return_value=solver),
        ):
            hints.generate(args)
        self.assertEqual(solver.requests, 9)
        with (
            patch.object(
                hints, "load_teacher_cohort", return_value=(source_rows(), source)
            ),
            patch("vllm.LLM", side_effect=AssertionError("cached hints")),
        ):
            hints.generate(args)
        rows, _ = load_rows(root, "hints")
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["invalid_reason"] == "answer_leak" for row in rows))

    def test_generation_resumes_per_request_and_summary_never_loads_model(self):
        experiment = self.prepared()
        with (
            patch("vllm.LLM", return_value=FakeSolver(fail_after=1)),
            self.assertRaisesRegex(RuntimeError, "interrupted"),
        ):
            study.generate(self.args, experiment)
        solver = FakeSolver()
        with patch("vllm.LLM", return_value=solver):
            study.generate(self.args, experiment)
        self.assertEqual(solver.requests, 11)
        with patch("vllm.LLM", side_effect=AssertionError("must reuse cache")):
            study.generate(self.args, experiment)
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("CPU summary"),
        ):
            result = study.summarize(self.args, experiment)
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(
            result["arms"]["none"]["accuracy"]["metrics"]["pass@1"]["mean"], 1.0
        )
        rows, _ = study.load_completions(self.root, experiment, "none")
        with self.assertRaises(ValueError):
            validate_completions(rows[:-1], experiment["problems"], 2)

    def test_artifact_corruption_and_changed_hint_source_rejected(self):
        experiment = self.generated()
        path = self.root / "none/completions.jsonl"
        path.write_text(path.read_text().replace("Check", "Changed"))
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            study.load_completions(self.root, experiment, "none")
        rows = hint_rows(source_rows())
        rows[0]["hint"] = "new hint"
        save_rows(self.hint_root, "hints", rows, self.hint_config)
        with self.assertRaisesRegex(ValueError, "source changed"):
            study.load_experiment(self.root)

    def test_judging_resume_failure_denominators_and_cpu_plot(self):
        experiment = self.generated()
        args = cognitive.build_parser().parse_args(
            ["--output-dir", str(self.root), "--bootstrap-samples", "50"]
        )
        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(_commit_hash="judge-rev"),
            ),
            patch(
                "eval.teacher_behaviors.create_classifier",
                return_value=(FakeJudge(), None),
            ),
        ):
            cognitive.classify(args, experiment)
        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(_commit_hash="judge-rev"),
            ),
            patch(
                "eval.teacher_behaviors.create_classifier",
                side_effect=AssertionError("must reuse"),
            ),
        ):
            cognitive.classify(args, experiment)
        chunks, meta = load_rows(self.root / "hint_short", "behaviors")
        chunks[0]["parse_failed"] = True
        save_rows(self.root / "hint_short", "behaviors", chunks, meta["config"])
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("CPU summary"),
        ):
            result = cognitive.summarize(args, experiment)
            ordinary = study.summarize(self.args, experiment)
            from eval.viz.teacher_hint_variants import plot

            plot(self.root)
        self.assertEqual(
            result["arms"]["hint_short"]["cognitive"]["n_trajectories_dropped"], 1
        )
        self.assertEqual(result["contrasts"]["hint_short-minus-none"]["n_questions"], 2)
        self.assertEqual(
            ordinary["arms"]["hint_short"]["uncertainty"]["n_completions"], 6
        )
        self.assertTrue((self.root / "comparison.png").is_file())
        self.assertTrue((self.root / "comparison.csv").is_file())
        source, source_meta = load_rows(self.root / "hint_short", "completions")
        source[0]["text"] = "Changed after judgment"
        save_rows(
            self.root / "hint_short", "completions", source, source_meta["config"]
        )
        with self.assertRaisesRegex(ValueError, "source"):
            cognitive.summarize(args, experiment)


if __name__ == "__main__":
    unittest.main()
