"""Cached student analysis: cohort matching, provenance, paired statistics, and resume."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datasets import Dataset

from eval import student_behaviors as sb
from eval import teacher_behaviors as tb
from tests.test_teacher_behaviors import WordTokenizer


def rollout(
    question=1, sample=0, text="Check the result.\n\nNow use a different approach."
):
    return {
        "question_id": f"q{question}",
        "question_idx": question,
        "sample_idx": sample,
        "question": f"problem {question}",
        "final_answer": "42",
        "completion_text": text,
        "completion_ids": [1, 2, 3],
        "n_tokens": 3,
        "reward": 1.0,
        "truncated": False,
        "finish_reason": "stop",
        "rollout_id": f"r{question}-{sample}",
    }


def trajectory(q, s, count, tokens=1000):
    return {
        "question_idx": q,
        "sample_idx": s,
        "n_tokens": tokens,
        "correct": True,
        "truncated": False,
        **{name: count for name in tb.BEHAVIORS},
    }


class SourceTest(unittest.TestCase):
    def test_selection_keeps_same_questions_and_first_samples(self):
        rows = [rollout(q, s) for q in range(3) for s in range(3)]
        selected = sb.adapt_rollouts(rows, {"q1": (1, "problem 1", "42")}, 2)
        self.assertEqual(
            [(r["question_idx"], r["sample_idx"]) for r in selected], [(1, 0), (1, 1)]
        )
        self.assertTrue(selected[0]["unclosed"])
        self.assertEqual(selected[0]["e_total"], 1)

    def test_missing_duplicate_and_misaligned_rollouts_fail(self):
        questions = {"q1": (1, "problem 1", "42")}
        for rows in (
            [],
            [rollout(), rollout()],
            [{**rollout(), "question": "different"}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                sb.adapt_rollouts(rows, questions, 1)

    def test_content_changes_invalidate_cache_even_with_same_ids(self):
        args = sb.build_parser().parse_args([])
        args.max_output_tokens = 1024
        job = {
            "model": "model",
            "arm": "hint",
            "step": 20,
            "source": Path("cache"),
            "generation": {},
            "questions": {"q1": (1, "problem 1", "42")},
        }
        old = sb.adapt_rollouts([rollout()], job["questions"], 1)
        new = sb.adapt_rollouts(
            [rollout(text="Entirely different reasoning.")], job["questions"], 1
        )
        before = sb.classification_config(args, job, old)
        after = sb.classification_config(args, job, new)
        self.assertNotEqual(before["source_fingerprint"], after["source_fingerprint"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"verification": 1}]
            sb.write_jsonl(root / "behaviors.jsonl", rows)
            tb.write_json_atomic(
                root / "meta.json",
                {
                    "status": "complete",
                    "config": before,
                    "chunks_fingerprint": sb.fingerprint(rows),
                },
            )
            self.assertEqual(sb.cached_classification(root, before, False), rows)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                sb.cached_classification(root, after, False)
            self.assertIsNone(sb.cached_classification(root, after, True))
            sb.write_jsonl(root / "behaviors.jsonl", [{"verification": 999}])
            with self.assertRaisesRegex(ValueError, "content mismatch"):
                sb.cached_classification(root, before, False)


class PairedTest(unittest.TestCase):
    def test_pooled_rate_not_mean_of_question_rates(self):
        base = [trajectory(0, 0, 0, 1000), trajectory(1, 0, 0, 9000)]
        candidate = [trajectory(0, 0, 10, 1000), trajectory(1, 0, 0, 9000)]
        result = sb.paired_differences(candidate, base, 100, 42, 1)
        self.assertEqual(result["metrics"]["verification/rate_per_1k"]["delta"], 1.0)
        self.assertEqual(result["metrics"]["verification/prevalence"]["delta"], 0.5)

    def test_identical_conditions_have_exactly_zero_paired_uncertainty(self):
        rows = [trajectory(q, s, q + s) for q in range(4) for s in range(2)]
        result = sb.paired_differences(rows, rows, 100, 42, 2)
        for metric in result["metrics"].values():
            self.assertEqual(metric, {"delta": 0.0, "ci95": [0.0, 0.0]})

    def test_failed_samples_drop_question_from_paired_estimate(self):
        base = [trajectory(q, s, 1) for q in range(2) for s in range(2)]
        result = sb.paired_differences(base[:-1], base, 20, 42, 2)
        self.assertEqual(result["question_indices"], [0])
        self.assertEqual(result["n_questions"], 1)

    def test_no_usable_questions_is_reported(self):
        result = sb.paired_differences([], [trajectory(1, 0, 1)], 20, 42, 1)
        self.assertEqual(result["metrics"], {})
        self.assertEqual(result["n_questions"], 0)


class DriverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "input"
        self.output = self.root / "output"
        self.model = "Qwen3-1.7B"
        for arm, questions in (("hint", [0, 1]), ("full", [1, 2])):
            run = self.input / self.model / f"deepmath_{arm}"
            cohort = [
                {
                    "question_id": f"q{q}",
                    "question_idx": q,
                    "question": f"problem {q}",
                    "final_answer": "42",
                }
                for q in questions
            ]
            Dataset.from_list(cohort).save_to_disk(str(run / "cohort"))
            ids = [r["question_id"] for r in cohort]
            cohort_fp = tb.fingerprint_ids(ids)
            tb.write_json_atomic(
                run / "cohort_meta.json",
                {
                    "status": "complete",
                    "question_ids": ids,
                    "cohort_fingerprint": cohort_fp,
                },
            )
            for step in (0, 20):
                path = run / f"step-{step:06d}"
                Dataset.from_list(
                    [rollout(q, s) for q in questions for s in range(2)]
                ).save_to_disk(str(path / "rollouts"))
                config = {
                    "base_model": f"Qwen/{self.model}",
                    "dataset": "deepmath",
                    "step": step,
                    "student_model": f"model-{step}",
                    "cohort_fingerprint": cohort_fp,
                    "n": 2,
                    "seed": 42,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "max_completion_length": 8192,
                }
                tb.write_json_atomic(
                    path / "rollout_meta.json", {"status": "complete", "config": config}
                )
        self.argv = [
            "student_behaviors",
            "--rollout-root",
            str(self.input),
            "--output-root",
            str(self.output),
            "--models",
            self.model,
            "--arms",
            "hint",
            "full",
            "--samples-per-problem",
            "2",
            "--bootstrap-samples",
            "16",
            "--no-plots",
        ]

    def test_intersection_and_automatic_base_step(self):
        args = sb.build_parser().parse_args(self.argv[1:] + ["--steps", "20"])
        jobs, manifest = sb.prepare_model(args, self.model)
        self.assertEqual(manifest["question_ids"], ["q1"])
        self.assertEqual(
            [(j["arm"], j["step"]) for j in jobs],
            [("hint", 0), ("hint", 20), ("full", 0), ("full", 20)],
        )

    def test_mismatched_generation_settings_fail(self):
        path = self.input / self.model / "deepmath_full/step-000020/rollout_meta.json"
        meta = json.loads(path.read_text())
        meta["config"]["max_completion_length"] = 16384
        path.write_text(json.dumps(meta))
        args = sb.build_parser().parse_args(self.argv[1:])
        with self.assertRaisesRegex(ValueError, "Generation settings differ"):
            sb.prepare_model(args, self.model)

    def test_end_to_end_resume_and_cpu_summary(self):
        def classify(llm, sampling, plan, prompt, evidence):
            return [
                {
                    **{
                        key: row[key]
                        for key in (
                            "question_idx",
                            "sample_idx",
                            "chunk_idx",
                            "char_start",
                            "char_end",
                            "n_classifier_tokens",
                        )
                    },
                    "parse_failed": False,
                    **{name: 1 for name in tb.BEHAVIORS},
                }
                for row in plan
            ]

        with (
            mock.patch.object(sys, "argv", self.argv),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=WordTokenizer(),
            ),
            mock.patch.object(
                tb, "create_classifier", return_value=(object(), object())
            ) as factory,
            mock.patch.object(
                tb, "classify_chunks", side_effect=classify
            ) as classify_mock,
        ):
            sb.main()
            factory.assert_called_once()
            self.assertEqual(classify_mock.call_count, 4)
        for extra in ([], ["--phase", "summarize"]):
            with (
                mock.patch.object(sys, "argv", self.argv + extra),
                mock.patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    side_effect=AssertionError("tokenizer loaded"),
                ),
                mock.patch.object(
                    tb, "create_classifier", side_effect=AssertionError("judge loaded")
                ),
            ):
                sb.main()
        summary = json.loads((self.output / self.model / "summary.json").read_text())
        self.assertEqual(summary["n_questions"], 1)
        self.assertEqual(set(summary["arms"]), {"hint", "full"})
        self.assertEqual(
            summary["arms"]["hint"]["20"]["paired_vs_step_zero"]["n_questions"], 1
        )
        sb.plot_curves(summary, self.output / self.model)
        self.assertTrue((self.output / self.model / "rate_per_1k.png").is_file())

    def test_dry_run_never_loads_judge_or_writes_results(self):
        with (
            mock.patch.object(sys, "argv", self.argv + ["--dry-run"]),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=WordTokenizer(),
            ),
            mock.patch.object(
                tb, "create_classifier", side_effect=AssertionError("judge loaded")
            ),
        ):
            sb.main()
        self.assertFalse(self.output.exists())


class SharedJudgeTest(unittest.TestCase):
    def test_student_and_teacher_share_judge_defaults(self):
        teacher = tb.build_parser().parse_args([])
        student = sb.build_parser().parse_args([])
        self.assertEqual(
            {k: getattr(teacher, k) for k in sb.JUDGE_KEYS},
            {k: getattr(student, k) for k in sb.JUDGE_KEYS},
        )

    def test_shared_chunk_context_is_not_counted_twice(self):
        source = [{"question_idx": 1, "sample_idx": 0, "text": "a b\n\nc d\n\ne f"}]
        plan = tb.build_chunk_plan(source, WordTokenizer(), 2, 1)
        self.assertEqual(len(plan), 3)
        self.assertNotIn("context", plan[0])
        self.assertEqual(plan[1]["context"], "a b")
        self.assertEqual(plan[2]["text"], "e f")
        self.assertEqual(plan[2]["context"], "c d")


if __name__ == "__main__":
    unittest.main()
