"""CPU checks for cohort-backed teacher uncertainty and the standard hint arm."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval import teacher_uncertainty as tu
from eval.teacher_uncertainty import DEFAULT_PI_MODES


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, **kwargs):
        assert kwargs["enable_thinking"] is True
        if tokenize:
            text = str(messages)
            return {"input_ids": [list(range(100 if "TOO_LONG" in text else 5))]}
        return str(messages)


def problems():
    return [
        {
            "question_idx": 3,
            "question_id": "qid3",
            "question": "Compute x",
            "answer": "42",
            "solution": "Reference trace</think>42",
            "rollout": "Attempt",
            "hint": "standard advice",
        }
    ]


class CohortAlignmentTest(unittest.TestCase):
    def args(self, *extra):
        return tu.build_parser().parse_args(
            [
                "--teacher-model",
                "student",
                "--cohort-dir",
                "cohort",
                "--pi-modes",
                *DEFAULT_PI_MODES,
                "--num-problems",
                "0",
                *extra,
            ]
        )

    def test_preparation_uses_standard_cohort_hint_without_variant_artifacts(self):
        manifest = {
            "model": "student",
            "model_identity": {"model": "student"},
            "revision": "rev",
            "cohort_hash": "cohort",
            "tokenizer_hash": "tok",
            "rollout_pi_root": "rollouts",
            "rollout_pi_sample_idx": 0,
        }
        cohort = [dict(problems()[0], final_answer="42")]
        with (
            mock.patch("eval.passk_pi.load_cohort", return_value=(manifest, cohort)),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ),
            mock.patch.object(tu, "tokenizer_hash", return_value="tok"),
        ):
            rows, meta = tu.prepare_problems(self.args())
        self.assertEqual(rows[0]["hint"], "standard advice")
        self.assertEqual(meta["cohort"]["hint_source"], "cohort.hint")
        self.assertEqual(meta["question_ids"], ["qid3"])
        self.assertNotIn("hint_sample_idx", meta["cohort"])

    def test_self_and_strong_use_identical_source_filters(self):
        metadata = {"revision": "rev", "tokenizer_hash": "tok"}
        with (
            mock.patch.object(
                tu, "load_demo_problems", return_value=(problems(), metadata)
            ) as loader,
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ),
            mock.patch.object(tu, "tokenizer_hash", return_value="tok"),
        ):
            self_rows, self_meta = tu.prepare_problems(self.args())
            strong = self.args(
                "--teacher-model",
                "strong",
                "--problem-model",
                "student",
                "--pi-modes",
                "none",
                "--align-pi-modes",
                *DEFAULT_PI_MODES,
            )
            strong_rows, strong_meta = tu.prepare_problems(strong)
            self.assertEqual(self_rows, strong_rows)
            self.assertEqual(self_meta, strong_meta)
            self.assertEqual(loader.call_args.args[1], "student")
            self.assertEqual(set(loader.call_args.args[2]), set(DEFAULT_PI_MODES))

    def test_long_hint_is_checked_even_when_only_aligning_strong_baseline(self):
        rows = problems()
        rows[0]["hint"] = "TOO_LONG"
        with (
            mock.patch.object(
                tu,
                "load_demo_problems",
                return_value=(rows, {"revision": "rev", "tokenizer_hash": "tok"}),
            ),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ),
            mock.patch.object(tu, "tokenizer_hash", return_value="tok"),
            self.assertRaisesRegex(ValueError, "No common"),
        ):
            tu.prepare_problems(
                self.args(
                    "--teacher-model",
                    "strong",
                    "--problem-model",
                    "student",
                    "--pi-modes",
                    "none",
                    "--align-pi-modes",
                    "hint",
                    "--max-model-len",
                    "50",
                    "--max-tokens",
                    "10",
                )
            )


class GenerationTest(unittest.TestCase):
    def test_hint_generation_preserves_ids_and_measures_actual_response(self):
        llm = mock.Mock()
        llm.generate.return_value = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="Wait, check. </think>42",
                        token_ids=[1, 2, 3],
                        finish_reason="length",
                    )
                ]
            )
        ]
        with mock.patch.object(tu, "grade", return_value=("42", True)):
            records = tu.generate_arm(
                llm, Tokenizer(), problems(), "hint", SimpleNamespace(n=1)
            )
        self.assertIn("standard advice", llm.generate.call_args.args[0][0])
        self.assertNotIn("Reference trace", llm.generate.call_args.args[0][0])
        self.assertEqual(records[0]["question_id"], "qid3")
        self.assertEqual(records[0]["question_idx"], 3)
        self.assertEqual(records[0]["sample_idx"], 0)
        self.assertEqual(records[0]["e_total"], 2)
        self.assertTrue(records[0]["truncated"])
        self.assertFalse(records[0]["unclosed"])

    def test_main_preflight_then_all_conditions_without_real_model(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                tu,
                "prepare_problems",
                return_value=(
                    problems(),
                    {
                        "cohort": {"revision": "rev"},
                        "rollout_pi": None,
                        "question_ids": ["qid3"],
                        "problem_model": "student",
                    },
                ),
            ),
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(_commit_hash="rev"),
            ),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ),
            mock.patch.object(tu, "tokenizer_hash", return_value="tok"),
            mock.patch.object(tu, "grade", return_value=("42", True)),
            mock.patch.object(tu, "LLM") as llm,
        ):
            llm.return_value.generate.return_value = [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="Maybe check.</think>42",
                            token_ids=[1, 2],
                            finish_reason="stop",
                        )
                    ]
                )
            ]
            argv = [
                "teacher_uncertainty",
                "--teacher-model",
                "student",
                "--cohort-dir",
                "cohort",
                "--n",
                "1",
                "--output-dir",
                directory,
            ]
            with mock.patch.object(sys, "argv", argv + ["--prepare-only"]):
                tu.main()
            llm.assert_not_called()
            with mock.patch.object(sys, "argv", argv):
                tu.main()
            summary = json.loads(
                (
                    Path(directory) / "student/teacher_uncertainty_summary.json"
                ).read_text()
            )
            self.assertEqual(set(summary["behavior"]), set(DEFAULT_PI_MODES))
            for mode in DEFAULT_PI_MODES:
                rows = (
                    (Path(directory) / f"student/completions_{mode}.jsonl")
                    .read_text()
                    .splitlines()
                )
                self.assertEqual(json.loads(rows[0])["question_id"], "qid3")
            self.assertEqual(
                llm.return_value.generate.call_count, len(DEFAULT_PI_MODES)
            )

    def test_removed_hint_options_rejected_before_preparation(self):
        for extra in (
            ["--pi-modes", "none", "hint_short"],
            ["--pi-modes", "none", "hint_medium"],
            ["--pi-modes", "none", "hint_detailed"],
            ["--align-pi-modes", "hint_short"],
            ["--align-pi-modes", "hint_medium"],
            ["--align-pi-modes", "hint_detailed"],
            ["--hint-sample-idx", "0"],
        ):
            with (
                self.subTest(extra=extra),
                mock.patch.object(
                    sys,
                    "argv",
                    ["teacher_uncertainty", "--cohort-dir", "cohort", *extra],
                ),
                mock.patch.object(sys, "stderr"),
                mock.patch.object(tu, "prepare_problems") as prepare,
                self.assertRaises(SystemExit) as raised,
            ):
                tu.main()
            self.assertEqual(raised.exception.code, 2)
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
