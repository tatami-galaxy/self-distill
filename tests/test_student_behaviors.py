"""Student generation, checkpoint selection, pairing and resume without downloads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval import student_behaviors as sb
from eval import teacher_behaviors as tb
from tests.test_teacher_behaviors import WordTokenizer


class ChatTokenizer(WordTokenizer):
    chat_template = "test-template"

    def get_vocab(self):
        return {"Check": 1, "result": 2}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] and kwargs["add_generation_prompt"]
        return "\n".join(m["content"] for m in messages) + "\nassistant"


def completion(p, sample=0):
    return {
        "question_id": p["question_id"],
        "question_idx": p["question_idx"],
        "sample_idx": sample,
        "text": "Check result.",
        "completion_ids": [1, 2],
        "n_tokens": 2,
        "correct": True,
        "truncated": False,
        "unclosed": True,
        "finish_reason": "stop",
        "e_total": 1,
        "e_think": 1,
        "e_post": 0,
        "e_by_marker": {m: int(m == "check") for m in tb.EPISTEMIC_MARKERS},
    }


def judged(plan, fail=False):
    return [
        {
            **{
                k: r[k]
                for k in (
                    "question_idx",
                    "sample_idx",
                    "chunk_idx",
                    "char_start",
                    "char_end",
                    "n_classifier_tokens",
                )
            },
            "parse_failed": fail,
            **{b: 1 for b in tb.BEHAVIORS},
        }
        for r in plan
    ]


class Fixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.study, self.cohort = self.root / "study", self.root / "cohort"
        self.output, self.run, self.benchmark = (
            self.root / "output",
            self.root / "run",
            self.root / "bench",
        )
        self.rows = [
            {
                "question_id": f"q{i}",
                "question_idx": 10 + i,
                "question": f"Question {i}",
                "final_answer": "42",
                "hint": "DO NOT INCLUDE HINT",
                "solution": "DO NOT INCLUDE SOLUTION",
            }
            for i in range(3)
        ]
        tok_hash = sb.digest(
            {
                "vocab": ChatTokenizer().get_vocab(),
                "template": ChatTokenizer.chat_template,
            }
        )
        sb.write_rows(self.cohort / "cohort.jsonl", self.rows)
        sb.write_json(
            self.cohort / "manifest.json",
            {
                "model": "Qwen/test",
                "dataset": "deepmath",
                "cohort_hash": sb.digest(self.rows),
                "tokenizer_hash": tok_hash,
            },
        )
        self.teacher = {
            "teacher_model": "Qwen/test",
            "n_problems": 2,
            "n_samples": 2,
            "source": {
                "problem_model": "Qwen/test",
                "question_ids": ["q2", "q0"],
                "question_indices": [12, 10],
                "cohort": {
                    "cohort_dir": str(self.cohort),
                    "cohort_hash": sb.digest(self.rows),
                    "tokenizer_hash": tok_hash,
                },
            },
            "generation": {
                "revision": "rev",
                "tokenizer_hash": tok_hash,
                "enable_thinking": True,
                "dtype": "bfloat16",
                "max_tokens": 100,
                "max_model_len": 1000,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "seed": 42,
            },
        }
        sb.write_json(self.study / "teacher_uncertainty_run_meta.json", self.teacher)
        self.meta = {"model": "Qwen/test", "dataset": "deepmath", "pi_mode": "hint"}
        sb.write_json(self.run / "run_meta.json", self.meta)
        for step in (20, 40):
            path = self.run / f"checkpoint-{step}"
            path.mkdir()
            (path / "model.safetensors").write_bytes(b"test weights")
        self.argv = [
            "--teacher-study-dir",
            str(self.study),
            "--run",
            f"hint={self.run}",
            "--output-dir",
            str(self.output),
            "--samples-per-problem",
            "2",
            "--bootstrap-samples",
            "20",
            "--no-plots",
        ]

    def args(self, *extra):
        args = sb.build_parser().parse_args(self.argv + list(extra))
        args.judge_max_tokens = 1024
        return args

    def result(self, step, correct, pass16=1.0, **overrides):
        path = self.benchmark / f"checkpoint-{step}" / "summary.json"
        summary = {
            "model": str(self.run / f"checkpoint-{step}"),
            "arm": {
                "algo": "sdft",
                "model": "test",
                "train_dataset": "deepmath",
                "variant": "hint",
                "run": "run-1",
                "step": f"checkpoint-{step}",
            },
            "n_samples": 16,
            "dataset_size": 2,
            "pass_at_k": {"pass@1": correct / 16, "pass@16": pass16},
            "eval_config": {
                "n": 16,
                "eval_dataset": "aime24",
                "max_tokens": 32000,
                "seed": 42,
                "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
            },
        }
        summary.update(overrides)
        sb.write_json(path, summary)
        sb.write_json(
            path.parent / "results.json",
            [
                {
                    "problem": f"Benchmark {i}",
                    "answer": "42",
                    "n_samples": 16,
                    "n_correct": correct,
                    "samples": [{"correct": j < correct} for j in range(16)],
                }
                for i in range(2)
            ],
        )
        return path

    def best_args(self, *extra):
        return self.args(
            "--checkpoint-selection",
            "best",
            "--selection-benchmark",
            "aime24",
            "--benchmark-results",
            f"hint={self.benchmark}",
            *extra,
        )

    def prepared(self):
        args = self.args()
        experiment, plan = sb.make_plan(args)
        for job in plan["jobs"]:
            rows = [completion(p, i) for p in experiment["problems"] for i in range(2)]
            sb.save_artifact(
                self.output / job["relative_dir"],
                "completions",
                sb.generation_config(experiment, job),
                rows,
            )
        return args, experiment, plan


class CohortTests(Fixture):
    def test_exact_retained_order_and_no_pi(self):
        rows, _ = sb.load_teacher_cohort(self.study)
        self.assertEqual([r["question_id"] for r in rows], ["q2", "q0"])
        self.assertTrue(
            all(
                "DO NOT INCLUDE" not in p
                for p in sb.render_prompts(rows, ChatTokenizer(), 1000, 100)
            )
        )
        self.assertNotIn("hint", rows[0])

    def test_checksum_and_index_mismatch(self):
        sb.write_rows(self.cohort / "cohort.jsonl", self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "checksum"):
            sb.load_teacher_cohort(self.study)
        sb.write_rows(self.cohort / "cohort.jsonl", self.rows)
        self.teacher["source"]["question_indices"][0] = 999
        sb.write_json(self.study / "teacher_uncertainty_run_meta.json", self.teacher)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            sb.load_teacher_cohort(self.study)

    def test_no_silent_context_filtering(self):
        rows, _ = sb.load_teacher_cohort(self.study)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            sb.render_prompts(rows, ChatTokenizer(), 101, 100)

    def test_bad_completion_groups(self):
        problems, _ = sb.load_teacher_cohort(self.study)
        rows = [completion(p) for p in problems]
        for invalid in (
            rows[:-1],
            rows + [rows[0]],
            [{**rows[0], "n_tokens": 3}, rows[1]],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sb.validate_rows(invalid, problems, 1)


class SelectionTests(Fixture):
    def test_legacy_results_require_explicit_option_and_record_missing_provenance(self):
        for step in (20, 40):
            path = self.result(step, 8 if step == 20 else 10)
            data = sb.read_json(path)
            data.pop("eval_config")
            data.pop("arm")
            data["max_tokens"] = 32000
            sb.write_json(path, data)
        with self.assertRaisesRegex(ValueError, "Legacy result lacks eval_config"):
            sb.make_plan(self.best_args())
        _, plan = sb.make_plan(self.best_args("--allow-legacy-benchmark-results"))
        report = plan["selection"]["hint"]
        self.assertEqual(report["selected_steps"], [40])
        self.assertFalse(report["generation_protocol_verified"])
        self.assertTrue(report["candidates"][0]["provenance_gaps"])

    def test_avg16_not_pass16_and_earliest_tie(self):
        self.result(20, 8, pass16=1.0)
        self.result(40, 10, pass16=0.5)
        _, plan = sb.make_plan(self.best_args())
        self.assertEqual([j["step"] for j in plan["jobs"]], [0, 40])
        self.result(20, 10)
        _, plan = sb.make_plan(self.best_args())
        self.assertEqual(plan["jobs"][1]["step"], 20)

    def test_ineligible_counts_excluded(self):
        self.result(20, 8)
        self.result(40, 16, n_samples=8)
        _, plan = sb.make_plan(self.best_args())
        report = plan["selection"]["hint"]
        self.assertEqual(report["selected_steps"], [20])
        self.assertEqual(report["without_eligible_results"], [40])

    def test_missing_winner_never_falls_back(self):
        self.result(20, 8)
        self.result(60, 12)
        with self.assertRaisesRegex(ValueError, "weights unavailable"):
            sb.make_plan(self.best_args())

    def test_mixed_questions_or_sampling(self):
        self.result(20, 8)
        path = self.result(40, 10)
        rows = sb.read_json(path.parent / "results.json")
        rows[0]["problem"] = "Different question"
        sb.write_json(path.parent / "results.json", rows)
        with self.assertRaisesRegex(ValueError, "questions or generation"):
            sb.make_plan(self.best_args())
        path = self.result(40, 10)
        data = sb.read_json(path)
        data["eval_config"]["sampling"]["temperature"] = 0.6
        sb.write_json(path, data)
        with self.assertRaisesRegex(ValueError, "questions or generation"):
            sb.make_plan(self.best_args())

    def test_inconsistent_score_or_other_run(self):
        path = self.result(20, 8)
        data = sb.read_json(path)
        data["pass_at_k"]["pass@1"] = 0.99
        sb.write_json(path, data)
        with self.assertRaisesRegex(ValueError, "does not equal avg@16"):
            sb.make_plan(self.best_args())
        self.result(20, 8, model=str(self.root / "other" / "checkpoint-20"))
        with self.assertRaisesRegex(ValueError, "path differs"):
            sb.make_plan(self.best_args())

    def test_all_numeric_single_base_solution_supported_rollout_excluded(self):
        (self.run / "final").mkdir()
        _, plan = sb.make_plan(self.args())
        self.assertEqual([j["step"] for j in plan["jobs"]], [0, 20, 40])
        self.meta["pi_mode"] = "solution"
        sb.write_json(self.run / "run_meta.json", self.meta)
        _, plan = sb.make_plan(self.args())
        self.assertEqual(plan["jobs"][1]["arm"], "solution")
        self.meta["pi_mode"] = "rollout"
        sb.write_json(self.run / "run_meta.json", self.meta)
        with self.assertRaisesRegex(ValueError, "rollout is excluded"):
            sb.make_plan(self.args())

    def test_frozen_selection_and_shared_paths(self):
        self.result(20, 8)
        self.result(40, 10)
        args = self.best_args()
        experiment, plan = sb.make_plan(args)
        sb.write_json(self.output / "experiment.json", experiment)
        sb.write_json(sb.selection_path(args), plan)
        self.result(20, 16)
        _, frozen = sb.make_plan(args)
        self.assertEqual(frozen["jobs"][1]["step"], 40)
        _, refreshed = sb.make_plan(self.best_args("--refresh-selection"))
        self.assertEqual(refreshed["jobs"][1]["step"], 20)
        _, all_plan = sb.make_plan(self.args())
        self.assertEqual(plan["jobs"][1], all_plan["jobs"][2])

    def test_dry_run_no_writes_or_solver(self):
        with mock.patch.object(sb, "generate_phase") as generate:
            sb.main(self.argv + ["--dry-run"])
        generate.assert_not_called()
        self.assertFalse(self.output.exists())


class PipelineTests(Fixture):
    def test_teacher_reference_alignment_and_judge_provenance(self):
        args, experiment, _plan = self.prepared()
        rows = sb.read_rows(self.output / "base" / "completions.jsonl")
        sb.write_rows(self.study / "completions_hint.jsonl", rows)
        behavior_dir = self.root / "teacher-behaviors"
        args.teacher_behaviors_dir = str(behavior_dir)
        judge = {**vars(sb.judge_args(args)), "rubric": tb.rubric_fingerprint()}
        config = {
            **vars(sb.judge_args(args)),
            "rubric_fingerprint": tb.rubric_fingerprint(),
            "samples_per_problem": 2,
            "limit": None,
            "source_fingerprint": tb.source_fingerprint(rows),
        }
        sb.write_json(
            behavior_dir / "behaviors_meta_hint.json",
            {"status": "complete", "config": config},
        )
        chunks = tb.build_chunk_plan(rows, ChatTokenizer(), 1000, 0)
        sb.write_rows(behavior_dir / "behaviors_hint.jsonl", judged(chunks))
        result = sb.teacher_references(experiment, args, judge)
        self.assertTrue(result["same_generation_protocol"])
        self.assertEqual(result["arms"]["hint"]["cognitive_status"], "available")
        self.assertFalse(result["arms"]["hint"]["judge_revision_verified"])
        judge["temperature"] = 0.1
        result = sb.teacher_references(experiment, args, judge)
        self.assertEqual(
            result["arms"]["hint"]["cognitive_status"], "judge_configuration_mismatch"
        )
        sb.write_rows(self.study / "completions_hint.jsonl", rows[:-1])
        with self.assertRaisesRegex(ValueError, "identities differ"):
            sb.teacher_references(experiment, args, judge)

    def test_generation_partial_resume_and_completed_skip(self):
        from eval import teacher_uncertainty as tu

        args = self.args("--batch-size", "1")
        experiment, plan = sb.make_plan(args)
        job = plan["jobs"][1]
        response = SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    text="Check result.", token_ids=[1, 2], finish_reason="stop"
                )
                for _ in range(2)
            ]
        )
        with (
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=ChatTokenizer(),
            ),
            mock.patch("vllm.LLM") as engine,
            mock.patch.object(tu, "grade", return_value=("42", True)),
        ):
            engine.return_value.generate.side_effect = [
                [response],
                RuntimeError("interrupted"),
            ]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                sb.generate_job(experiment, job, vars(args))
            self.assertFalse(
                (self.output / job["relative_dir"] / "completions_meta.json").exists()
            )
            engine.reset_mock()
            engine.return_value.generate.side_effect = None
            engine.return_value.generate.return_value = [response]
            sb.generate_job(experiment, job, vars(args))
            self.assertEqual(engine.return_value.generate.call_count, 1)
            self.assertNotIn(
                "DO NOT INCLUDE", engine.return_value.generate.call_args.args[0][0]
            )
        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("loaded"),
        ):
            sb.generate_job(experiment, job, vars(args))
        self.assertEqual(
            len(sb.load_artifact(self.output / job["relative_dir"], "completions")), 4
        )

    def test_judge_resume_cpu_summary_and_generation_denominators(self):
        args, experiment, plan = self.prepared()
        args.judge_batch_size = 2
        calls = []

        def classify(llm, sampling, chunks, prompt, evidence):
            calls.append(chunks)
            if len(calls) == 2:
                raise RuntimeError("judge interrupted")
            return judged(chunks, fail=len(calls) == 1)

        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(_commit_hash="judge-rev"),
            ),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=ChatTokenizer(),
            ),
            mock.patch.object(
                tb, "create_classifier", return_value=(object(), object())
            ) as create,
            mock.patch.object(tb, "classify_chunks", side_effect=classify),
        ):
            with self.assertRaisesRegex(RuntimeError, "judge interrupted"):
                sb.classify_phase(experiment, plan, args)
            self.assertEqual(
                len(
                    list(
                        (self.output / "base" / "score_cache" / "classification").glob(
                            "*.json"
                        )
                    )
                ),
                2,
            )
            sb.classify_phase(experiment, plan, args)
            self.assertEqual(create.call_count, 2)
            create.reset_mock()
            sb.classify_phase(experiment, plan, args)
            create.assert_not_called()
        with (
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                side_effect=AssertionError("loaded"),
            ),
            mock.patch.object(
                tb, "create_classifier", side_effect=AssertionError("loaded")
            ),
        ):
            result = sb.summarize_phase(experiment, plan, args)
        base = result["jobs"]["base"]
        self.assertEqual(base["uncertainty"]["n_completions"], 4)
        self.assertEqual(base["cognitive"]["n_trajectories"], 2)
        self.assertEqual(base["cognitive"]["n_trajectories_dropped"], 2)
        self.assertEqual(
            result["jobs"]["hint/checkpoint-20"]["cognitive_vs_base"]["n_questions"], 1
        )
        self.assertFalse(result["missing_classifications"])
        sb.plot_results(result, self.output / "test.png")
        self.assertTrue((self.output / "test.png").is_file())

    def test_corrupted_artifacts_and_changed_settings(self):
        _, experiment, plan = self.prepared()
        root = self.output / "base"
        config = sb.generation_config(experiment, plan["jobs"][0])
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            sb.load_artifact(root, "completions", {**config, "revision": "different"})
        rows = sb.read_rows(root / "completions.jsonl")
        rows[0]["text"] = "changed"
        sb.write_rows(root / "completions.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "checksum"):
            sb.load_artifact(root, "completions", config)

    def test_uncertainty_before_judging(self):
        args, experiment, plan = self.prepared()
        result = sb.summarize_phase(experiment, plan, args)
        self.assertEqual(len(result["missing_classifications"]), 3)
        self.assertEqual(
            result["jobs"]["base"]["uncertainty"]["e_per_1k_tokens"], 500.0
        )


class StatisticsTests(unittest.TestCase):
    def rows(self):
        return [
            {
                **completion({"question_id": f"q{q}", "question_idx": q}, s),
                "n_tokens": 1000 if q == 0 else 9000,
                **{b: 10 if q == 0 else 0 for b in sb.BEHAVIORS},
            }
            for q in range(2)
            for s in range(2)
        ]

    def test_pooled_rate_and_identical_paired_zero(self):
        rows = self.rows()
        result = sb.bootstrap_metrics(rows, 50, 42, cognitive=True)
        self.assertEqual(result["metrics"]["verification/rate_per_1k"]["mean"], 1.0)
        result = sb.bootstrap_metrics(rows, 50, 42, baseline=rows, n=2, cognitive=True)
        for value in result["metrics"].values():
            self.assertEqual(value, {"delta": 0.0, "ci95": [0.0, 0.0]})

    def test_incomplete_samples_drop_question_from_paired_estimate(self):
        rows = self.rows()
        result = sb.bootstrap_metrics(
            rows[:-1], 50, 42, baseline=rows, n=2, cognitive=True
        )
        self.assertEqual(result["question_indices"], [0])


if __name__ == "__main__":
    unittest.main()
