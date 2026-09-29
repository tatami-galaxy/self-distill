"""CPU-only checks for diversity statistics and persisted sweep results."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval import hint_diversity as hd


def hint(text, sample=0, question="q", **kwargs):
    return {
        "question_id": question,
        "sample_idx": sample,
        "hint": text,
        "n_tokens": 5,
        "invalid_reason": "",
        "truncated": False,
        **kwargs,
    }


class DiversityTest(unittest.TestCase):
    def test_duplicates_and_distinct_mathematics(self):
        rows = [
            hint("Use the quadratic formula."),
            hint(" USE the quadratic  formula. ", 1),
        ]
        result = hd.diversity(rows)
        self.assertEqual(result["n_unique_normalized"], 1)
        self.assertEqual(result["duplicate_pair_fraction"], 1)
        self.assertEqual(result["mean_pairwise_trigram_jaccard"], 1)
        self.assertNotEqual(hd.lexical_tokens("x = 2"), hd.lexical_tokens("x = 3"))
        self.assertEqual(
            hd.diversity([hint("a b c"), hint("d e f", 1)])[
                "mean_pairwise_trigram_jaccard"
            ],
            0,
        )

    def test_no_pairs_are_missing_not_maximally_diverse(self):
        for rows in ([], [hint("single")], [hint("x"), hint("y", 1)]):
            self.assertIsNone(hd.diversity(rows)["mean_pairwise_trigram_jaccard"])
        self.assertIsNone(hd.diversity([])["unique_fraction"])

    def test_metrics_are_within_question_and_valid_subset_has_support(self):
        rows = [
            hint("a b c", question="a"),
            hint("a b c", 1, "a"),
            hint("a b c", question="b"),
            hint("d e f", 1, "b", truncated=True),
        ]
        summary = hd.summarize(rows)
        self.assertEqual(summary["all"]["mean_pairwise_trigram_jaccard"], 0.5)
        self.assertEqual(summary["valid_complete"]["n_hints"], 3)
        self.assertEqual(summary["valid_complete"]["n_questions_with_pairs"], 1)
        self.assertEqual(summary["truncation_fraction"], 0.25)


class ArtifactTest(unittest.TestCase):
    def test_sweep_keeps_prompts_fixed_and_reuses_completed_temperatures(self):
        cohort = [
            {
                "question_id": "q",
                "question": "Problem",
                "final_answer": "42",
                "solution": "Demo",
                "prompt_ids": [1, 2, 3],
            }
        ]
        outputs = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(text=text, token_ids=[4, 5], finish_reason=finish)
                    for text, finish in [
                        ("Use an invariant", "stop"),
                        ("Incomplete", "length"),
                    ]
                ]
            )
        ]
        engine = mock.Mock()
        engine.generate.return_value = outputs
        llm = mock.Mock(return_value=engine)
        fake_modules = {
            "utils": mock.Mock(),
            "utils.gen_hints": SimpleNamespace(
                build_messages=lambda q, s, d: [q, s],
                HINT_VALIDATION_VERSION=2,
                leaks_answer=lambda *args: False,
            ),
            "utils.model_adapters": SimpleNamespace(
                vllm_model_and_adapter=lambda model: ({"model": model}, None, None),
            ),
            "vllm": SimpleNamespace(LLM=llm, SamplingParams=SimpleNamespace),
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict("sys.modules", fake_modules),
            mock.patch.object(
                hd,
                "prepare",
                side_effect=lambda args, config: {
                    "config": config,
                    "cohort": cohort,
                },
            ),
            mock.patch(
                "sys.argv",
                [
                    "hint_diversity",
                    "--output-dir",
                    tmp,
                    "--num-questions",
                    "1",
                    "--hints-per-question",
                    "2",
                    "--temperatures",
                    "1.0",
                    "1.2",
                ],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            hd.main()
            hd.main()
            self.assertEqual(llm.call_count, 1)
            self.assertEqual(engine.generate.call_count, 2)
            for call, temperature in zip(engine.generate.call_args_list, [1.0, 1.2]):
                self.assertEqual(call.args[0], [{"prompt_token_ids": [1, 2, 3]}])
                self.assertEqual(call.args[1].n, 2)
                self.assertEqual(call.args[1].temperature, temperature)
                self.assertEqual(call.args[1].top_k, -1)
            self.assertEqual(
                hd.read_json(Path(tmp) / "summary.json")["temperatures"]["1.2"][
                    "valid_complete"
                ]["n_hints"],
                1,
            )

    def test_report_and_corrupt_or_incomplete_cache(self):
        manifest = {
            "config": {"temperatures": [1.0], "hints_per_question": 2},
            "cohort": [
                {
                    "question_id": "q",
                    "question": "x < 2?",
                    "final_answer": "1",
                    "solution": "Reference",
                }
            ],
        }
        rows = [
            hint("<script>alert(1)</script>", invalid_reason="answer_leak"),
            hint("A second hint", 1, truncated=True),
        ]
        data = {
            "manifest_fingerprint": hd.digest(manifest),
            "temperature": 1.0,
            "hints": rows,
            "hints_fingerprint": hd.digest(rows),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = hd.temperature_path(root, 1.0)
            hd.write_json(path, data)
            with contextlib.redirect_stdout(io.StringIO()):
                hd.report(root, manifest)
            report = (root / "report.html").read_text()
            self.assertIn("&lt;script&gt;", report)
            self.assertNotIn("<script>", report)
            self.assertEqual(
                hd.read_json(root / "summary.json")["temperatures"]["1.0"][
                    "valid_complete"
                ]["n_hints"],
                0,
            )
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                hd.load_samples(path, {**manifest, "changed": True}, 1.0)
            rows[0]["hint"] = "Changed text"
            hd.write_json(path, data)
            with self.assertRaisesRegex(ValueError, "changed samples"):
                hd.load_samples(path, manifest, 1.0)
            data["hints"] = rows[:1]
            data["hints_fingerprint"] = hd.digest(rows[:1])
            hd.write_json(path, data)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                hd.load_samples(path, manifest, 1.0)


if __name__ == "__main__":
    unittest.main()
