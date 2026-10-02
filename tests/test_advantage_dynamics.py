"""Unit tests for the checkpoint-level SDFT advantage evaluator."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval import advantage_dynamics_sdft
from eval import student_behaviors
from eval.hint_compare_cache import digest
from utils import format_prompt_math


class FakeTokenizer:
    def __init__(self, vocab, special_ids=(0, 1), template="template"):
        self._vocab = vocab
        self.all_special_ids = list(special_ids)
        self.chat_template = template
        self.name_or_path = "fake"

    def get_vocab(self):
        return self._vocab

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        if not tokenize:
            return "\n".join(m["content"] for m in messages)
        return {"input_ids": [
            list(range(len("\n".join(m["content"] for m in conversation))))
            for conversation in messages
        ]}


def compatible_training_args(**overrides):
    values = {
        "distillation_mode": "sampled_token",
        "distillation_alpha": 1.0,
        "teacher_model_kind": "base",
        "generate_from_teacher": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": None,
        "repetition_penalty": 1.0,
        "gradient_accumulation_steps": 16,
        "steps_per_generation": 16,
        "num_iterations": 1,
        "distillation_is_clip": 2.0,
        "num_loss_tokens_to_skip": 0,
        "max_completion_length": 8192,
        "max_prompt_length": 8192,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class AdvantageDefinitionTest(unittest.TestCase):
    def test_advantage_is_teacher_minus_student_log_probability(self):
        teacher = [-1.0, -3.0, -2.0]
        student = [-2.0, -2.0, -2.5]
        advantages = advantage_dynamics_sdft.compute_advantages(teacher, student)
        self.assertEqual(advantages, [1.0, -1.0, 0.5])
        sampled_reverse_kl = sum(s - t for s, t in zip(student, teacher)) / 3
        self.assertAlmostEqual(sum(advantages) / 3, -sampled_reverse_kl)

    def test_misaligned_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            advantage_dynamics_sdft.compute_advantages([-1.0], [-1.0, -2.0])

    def test_summary_is_token_weighted_and_honors_skipped_tokens(self):
        rows = [
            {
                "eval_pi_mode": "hint",
                "question_id": "q1",
                "reward": 1.0,
                "truncated": False,
                "loss_token_start": 1,
                "advantages": [99.0, 3.0],
            },
            {
                "eval_pi_mode": "hint",
                "question_id": "q2",
                "reward": 0.0,
                "truncated": True,
                "loss_token_start": 0,
                "advantages": [1.0, 1.0, 1.0],
            },
        ]
        summary = advantage_dynamics_sdft.summarize_score_rows(rows, bootstrap_samples=20, seed=7)
        hint = summary["hint"]
        self.assertAlmostEqual(hint["mean_advantage_per_token"], 1.5)
        self.assertAlmostEqual(hint["mean_rollout_advantage"], 2.0)
        self.assertEqual(hint["num_tokens"], 4)
        self.assertEqual(hint["by_outcome"]["correct"]["num_tokens"], 1)
        self.assertEqual(hint["by_outcome"]["truncated"]["num_tokens"], 3)

    def test_question_bootstrap_is_deterministic(self):
        totals = {"a": (2.0, 2), "b": (-1.0, 1)}
        first = advantage_dynamics_sdft.question_cluster_bootstrap_ci(totals, 50, 3)
        second = advantage_dynamics_sdft.question_cluster_bootstrap_ci(totals, 50, 3)
        self.assertEqual(first, second)

    def test_dynamics_reuses_already_computed_step_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = {
                "output_dir": temporary,
                "run_dir": "run",
                "base_model": "base",
                "dataset": "deepmath",
                "training_pi_mode": "hint",
            }
            summaries = [{"step": 20, "advantages": {}}]
            with mock.patch.object(
                advantage_dynamics_sdft,
                "aggregate_step",
                side_effect=AssertionError("aggregate_step should not be called"),
            ):
                dynamics = advantage_dynamics_sdft.aggregate_dynamics(
                    SimpleNamespace(), run, [20], summaries
                )
            self.assertEqual(dynamics["steps"], summaries)
            self.assertTrue((Path(temporary) / "dynamics.json").is_file())


class ProvenanceTest(unittest.TestCase):
    def test_tokenizer_mapping_mismatch_is_rejected(self):
        student = FakeTokenizer({"a": 1, "b": 2})
        compatible = FakeTokenizer({"b": 2, "a": 1}, template="different")
        student_meta, teacher_meta = advantage_dynamics_sdft.verify_tokenizer_compatibility(
            student, compatible
        )
        self.assertEqual(student_meta["vocab_hash"], teacher_meta["vocab_hash"])
        with self.assertRaisesRegex(ValueError, "token-to-ID"):
            advantage_dynamics_sdft.verify_tokenizer_compatibility(
                student, FakeTokenizer({"a": 2, "b": 1})
            )

    def test_checkpoint_discovery_is_numeric_and_includes_base_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint-20").mkdir()
            (root / "checkpoint-3").mkdir()
            (root / "checkpoint-final").mkdir()
            (root / "checkpoint-4").write_text("not a directory")
            checkpoints = advantage_dynamics_sdft.discover_checkpoints(root)
            self.assertEqual(list(checkpoints), [0, 3, 20])
            self.assertEqual(checkpoints[0], "")

    def test_training_configuration_accepts_current_runs(self):
        config = advantage_dynamics_sdft.validate_training_configuration(
            compatible_training_args()
        )
        self.assertFalse(config["importance_correction_active"])
        self.assertEqual(config["distillation_is_clip"], 2.0)

    def test_training_configuration_rejects_non_base_teacher(self):
        with self.assertRaisesRegex(ValueError, "teacher_model_kind"):
            advantage_dynamics_sdft.validate_training_configuration(
                compatible_training_args(teacher_model_kind="ema")
            )

    def test_training_configuration_rejects_active_importance_correction(self):
        with self.assertRaisesRegex(ValueError, "importance correction"):
            advantage_dynamics_sdft.validate_training_configuration(
                compatible_training_args(gradient_accumulation_steps=15)
            )

    def test_ids_are_stable_and_distinguish_samples(self):
        question_id = advantage_dynamics_sdft.stable_question_id("q", "a")
        first = advantage_dynamics_sdft.stable_rollout_id("checkpoint", question_id, 0, [1, 2], 42)
        self.assertEqual(
            first,
            advantage_dynamics_sdft.stable_rollout_id("checkpoint", question_id, 0, [1, 2], 42),
        )
        self.assertNotEqual(
            first,
            advantage_dynamics_sdft.stable_rollout_id("checkpoint", question_id, 1, [1, 2], 42),
        )


class PromptAndCliTest(unittest.TestCase):
    def test_none_is_student_prompt_and_pi_is_inserted(self):
        problem = {
            "question": "What is 1+1?",
            "final_answer": "2",
            "hint": "Add the units.",
            "solution": "Private reasoning.</think>One plus one is two.",
        }
        self.assertEqual(
            advantage_dynamics_sdft.build_teacher_messages(problem, "none"),
            format_prompt_math(problem["question"]),
        )
        for pi_mode, text in (
            ("answer", "2"),
            ("hint", "Add the units."),
            ("full", "One plus one is two."),
            ("solution", "One plus one is two."),
        ):
            with self.subTest(pi_mode=pi_mode):
                messages = advantage_dynamics_sdft.build_teacher_messages(problem, pi_mode)
                self.assertIn(text, messages[-1]["content"])
                if pi_mode == "solution":
                    self.assertNotIn("Private reasoning", messages[-1]["content"])
                    self.assertNotIn("</think>", messages[-1]["content"])

    def test_solution_requires_a_well_formed_reference(self):
        with self.assertRaisesRegex(ValueError, "malformed"):
            advantage_dynamics_sdft.privileged_context({"solution": "no boundary"}, "solution")

    def test_parser_requires_training_pi(self):
        with self.assertRaises(SystemExit):
            advantage_dynamics_sdft.build_parser().parse_args(["--run-dir", "run"])

    def test_parser_accepts_one_training_pi(self):
        args = advantage_dynamics_sdft.build_parser().parse_args(
            ["--run-dir", "run", "--pi-mode", "hint", "--teacher-study-dir", "study"]
        )
        self.assertEqual(args.phase, "sweep")
        self.assertEqual(args.pi_mode, "hint")
        self.assertIsNone(args.steps)
        self.assertEqual(args.num_problems, 0)
        self.assertEqual(args.n, 2)
        self.assertEqual(args.output_root, "results/advantage_dynamics/teacher_cohort")

    def test_solution_replaces_rollout_and_source_is_required(self):
        parser = advantage_dynamics_sdft.build_parser()
        common = ["--run-dir", "run", "--teacher-study-dir", "study", "--pi-mode"]
        self.assertEqual(parser.parse_args(common + ["solution"]).pi_mode, "solution")
        with self.assertRaises(SystemExit):
            parser.parse_args(common + ["rollout"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--run-dir", "run", "--pi-mode", "hint"])

    def test_run_pi_mismatch_is_rejected_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "run_meta.json").write_text(
                '{"model":"base","dataset":"deepmath","pi_mode":"hint"}'
            )
            args = advantage_dynamics_sdft.build_parser().parse_args(
                ["--run-dir", temporary, "--pi-mode", "full", "--teacher-study-dir", "study"]
            )
            with self.assertRaisesRegex(ValueError, "does not match training PI"):
                advantage_dynamics_sdft.prepare_run(args)


class TeacherCohortTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.study, self.source = self.root / "study", self.root / "source"
        self.source.mkdir()
        self.tokenizer = FakeTokenizer({"a": 0, "b": 1})
        self.rows = [
            {
                "question_id": f"teacher-id-{i}", "question_idx": 10 + i,
                "question": f"Question {i}", "final_answer": "42",
                "hint": f"Original hint {i}",
                "solution": "Private reasoning.</think>Worked solution, answer 42.",
            }
            for i in range(4)
        ]
        self.meta = {
            "teacher_model": "base", "n_problems": 3,
            "source": {
                "problem_model": "base",
                "question_ids": ["teacher-id-2", "teacher-id-0", "teacher-id-1"],
                "question_indices": [12, 10, 11],
                "cohort": {"cohort_dir": str(self.source)},
            },
        }
        self.save_sources()
        self.args = advantage_dynamics_sdft.build_parser().parse_args([
            "--run-dir", "run", "--pi-mode", "hint",
            "--teacher-study-dir", str(self.study),
            "--max-completion-length", "100", "--max-model-len", "1100",
        ])
        self.run = {
            "base_model": "base", "dataset": "deepmath", "training_pi_mode": "hint",
            "eval_pi_modes": ["none", "hint"], "training_config": {"max_prompt_length": 1000},
            "output_dir": str(self.root / "out"), "checkpoints": {0: "base"},
        }
        patch = mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer)
        self.loader = patch.start()
        self.addCleanup(patch.stop)

    def save_sources(self):
        tokenizer_hash = digest({"vocab": self.tokenizer.get_vocab(), "template": self.tokenizer.chat_template})
        identity = {"cohort_hash": digest(self.rows), "tokenizer_hash": tokenizer_hash}
        (self.source / "cohort.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        advantage_dynamics_sdft.write_json_atomic(self.source / "manifest.json", {
            "model": "base", "dataset": "deepmath", **identity,
        })
        self.meta["source"]["cohort"].update(identity)
        advantage_dynamics_sdft.write_json_atomic(self.study / "teacher_uncertainty_run_meta.json", self.meta)

    def mode(self, name):
        self.run.update(training_pi_mode=name, eval_pi_modes=["none", name])

    def test_exact_student_cohort_ids_order_and_original_pi(self):
        cohort, meta = advantage_dynamics_sdft.build_cohort(self.args, self.run)
        students, _ = student_behaviors.load_teacher_cohort(self.study)
        self.assertEqual(list(cohort["question_id"]), [r["question_id"] for r in students])
        self.assertEqual(list(cohort["question_idx"]), [12, 10, 11])
        self.assertEqual(cohort[0]["hint"], "Original hint 2")
        self.assertEqual(cohort[0]["solution"], self.rows[2]["solution"])
        self.assertNotIn("Original hint", cohort[0]["student_prompt_text"])
        self.assertEqual(cohort[0]["teacher_prompt_ids_none"], cohort[0]["student_prompt_ids"])
        self.assertEqual(meta["num_questions"], 3)
        self.assertEqual(meta["excluded_questions"], [])
        self.args.seed = 999
        other, _ = advantage_dynamics_sdft.build_cohort(self.args, self.run)
        self.assertEqual(cohort.to_list(), other.to_list())

    def test_full_filter_drops_without_replacement_and_freezes_order(self):
        self.rows[2]["solution"] = "x" * 1500 + "</think>Worked solution."
        self.save_sources()
        self.mode("full")
        cohort, meta = advantage_dynamics_sdft.ensure_cohort(self.args, self.run)
        self.assertEqual(list(cohort["question_id"]), ["teacher-id-0", "teacher-id-1"])
        self.assertEqual(meta["num_source_questions"], 3)
        self.assertEqual(meta["excluded_questions"][0]["question_id"], "teacher-id-2")
        self.assertEqual(meta["excluded_questions"][0]["reason"], "full_prompt_too_long")
        self.loader.side_effect = AssertionError("Cached cohort should not load tokenizer")
        cached, _ = advantage_dynamics_sdft.ensure_cohort(self.args, self.run)
        self.assertEqual(cohort.to_list(), cached.to_list())

    def test_prefix_limit_applies_before_full_filtering(self):
        self.rows[2]["solution"] = "x" * 1500 + "</think>Worked solution."
        self.save_sources()
        self.mode("full")
        self.args.num_problems = 2
        cohort, _ = advantage_dynamics_sdft.build_cohort(self.args, self.run)
        self.assertEqual(list(cohort["question_id"]), ["teacher-id-0"])
        self.args.num_problems = 1
        with self.assertRaisesRegex(ValueError, "No teacher-study questions"):
            advantage_dynamics_sdft.build_cohort(self.args, self.run)

    def test_solution_fits_after_removing_reference_thinking(self):
        self.rows[2]["solution"] = "x" * 1500 + "</think>Worked solution."
        self.save_sources()
        self.mode("solution")
        cohort, meta = advantage_dynamics_sdft.build_cohort(self.args, self.run)
        self.assertEqual(meta["num_questions"], 3)
        prompt = advantage_dynamics_sdft.build_teacher_messages(cohort[0], "solution")[-1]["content"]
        self.assertIn("Worked solution.", prompt)
        self.assertNotIn("xxx", prompt)

    def test_non_full_overflows_are_errors(self):
        for field, mode, message in [
            ("hint", "hint", "hint prompt does not fit"),
            ("question", "full", "Student prompt does not fit"),
            ("solution", "solution", "solution prompt does not fit"),
        ]:
            with self.subTest(field=field):
                original = self.rows[2][field]
                self.rows[2][field] = ("trace</think>" if field == "solution" else "") + "x" * 1500
                self.save_sources()
                self.mode(mode)
                with self.assertRaisesRegex(ValueError, message):
                    advantage_dynamics_sdft.build_cohort(self.args, self.run)
                self.rows[2][field] = original

    def test_source_model_dataset_tokenizer_and_missing_hint_rejected(self):
        for key, value, message in [
            ("base_model", "different", "model differs"),
            ("dataset", "different", "dataset differs"),
        ]:
            run = {**self.run, key: value}
            with self.assertRaisesRegex(ValueError, message):
                advantage_dynamics_sdft.build_cohort(self.args, run)
        self.tokenizer.chat_template = "changed"
        with self.assertRaisesRegex(ValueError, "Tokenizer differs"):
            advantage_dynamics_sdft.build_cohort(self.args, self.run)
        self.tokenizer.chat_template = "template"
        self.rows[2]["hint"] = ""
        self.save_sources()
        with self.assertRaisesRegex(ValueError, "Missing source hint"):
            advantage_dynamics_sdft.build_cohort(self.args, self.run)

    def test_changed_source_and_corrupted_cache_rejected(self):
        cohort, _ = advantage_dynamics_sdft.ensure_cohort(self.args, self.run)
        self.rows[2]["hint"] = "Changed hint"
        with (self.source / "cohort.jsonl").open("w") as handle:
            handle.write("".join(json.dumps(r) + "\n" for r in self.rows))
        with self.assertRaisesRegex(ValueError, "checksum"):
            advantage_dynamics_sdft.ensure_cohort(self.args, self.run)
        self.save_sources()
        with self.assertRaisesRegex(ValueError, "different provenance"):
            advantage_dynamics_sdft.ensure_cohort(self.args, self.run)
        self.args.force = True
        cohort, _ = advantage_dynamics_sdft.ensure_cohort(self.args, self.run)
        self.args.force = False
        damaged = cohort.map(lambda r: {"hint": "Tampered derived hint"})
        advantage_dynamics_sdft.save_dataset_atomic(damaged, Path(self.run["output_dir"]) / "cohort")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            advantage_dynamics_sdft.ensure_cohort(self.args, self.run)

    def test_source_provenance_reaches_generation_and_score_cache_keys(self):
        _, meta = advantage_dynamics_sdft.build_cohort(self.args, self.run)
        self.args.tensor_parallel_size = 1
        self.run["training_config"]["num_loss_tokens_to_skip"] = 0
        rollout = advantage_dynamics_sdft._rollout_config(self.args, self.run, 0, meta)
        score = advantage_dynamics_sdft._score_config(
            self.args, self.run, 0, meta, {"rollout_fingerprint": "samples"},
        )
        for config in (rollout, score):
            self.assertEqual(config["cohort_config"]["source"]["teacher_meta_hash"], digest(self.meta))
            self.assertEqual(config["cohort_content_hash"], meta["cohort_content_hash"])
        self.assertEqual(rollout["temperature"], 1.0)
        self.assertEqual(rollout["top_p"], 1.0)
        self.assertEqual(rollout["top_k"], 0)

    def test_aggregate_rejects_old_scores_after_cohort_replacement(self):
        _, meta = advantage_dynamics_sdft.ensure_cohort(self.args, self.run)
        out = advantage_dynamics_sdft.step_dir(self.run, 0)
        (out / "scores").mkdir(parents=True)
        advantage_dynamics_sdft.write_json_atomic(out / "rollout_meta.json", {
            "config": advantage_dynamics_sdft._rollout_config(self.args, self.run, 0, meta),
            "rollout_fingerprint": "samples",
        })
        advantage_dynamics_sdft.write_json_atomic(out / "score_meta.json", {"config": {"old": True}})
        self.run["training_config"]["num_loss_tokens_to_skip"] = 0
        with self.assertRaisesRegex(ValueError, "Score provenance mismatch"):
            advantage_dynamics_sdft.aggregate_step(self.args, self.run, 0)


if __name__ == "__main__":
    unittest.main()
