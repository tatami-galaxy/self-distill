"""Demo-gain alignment, real causal scoring, statistics, and incremental caches."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from eval import demo_gain as dg
from eval.hint_compare_cache import ConditionCache, digest, model_identity
from utils import gen_hint_variants as hv


class CharacterTokenizer:
    chat_template = "test"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text))}

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


class AlignmentTest(unittest.TestCase):
    def test_same_solution_target_every_pi_without_trace_and_with_terminator(self):
        row = {
            "question": "Compute x",
            "solution": "Reasoning.</think>Answer: 42",
            "final_answer": "42",
            "hint": "Use algebra",
            "rollout": "An unverified attempt",
        }
        expected = None
        for condition in ("none",) + dg.CONDITIONS:
            prompt, target = dg.render_target(
                CharacterTokenizer(),
                dg.messages_for(row, condition, "A hint"),
                row["solution"],
            )
            text = "".join(map(chr, target))
            self.assertEqual(text, "Answer: 42<end>")
            if condition != "full":
                self.assertNotIn("Reasoning.", "".join(map(chr, prompt)))
            if expected is not None:
                self.assertEqual(expected, target)
            expected = target
        self.assertIn(row["solution"], dg.messages_for(row, "full")[-1]["content"])

    def test_bad_demo_and_template_fail(self):
        with self.assertRaisesRegex(ValueError, "malformed"):
            dg.render_target(CharacterTokenizer(), [], "No closing tag")
        tok = CharacterTokenizer()
        with (
            mock.patch.object(
                tok, "apply_chat_template", side_effect=["prefix", "different"]
            ),
            self.assertRaisesRegex(ValueError, "prefix_mismatch"),
        ):
            dg.render_target(tok, [], "Trace</think>Answer")

    def test_cached_qwen_template_if_available(self):
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-1.7B", local_files_only=True
            )
        except OSError:
            self.skipTest("Qwen tokenizer not cached locally")
        solution = "Compute 2+2=4.</think>\nThe answer is \\boxed{4}."
        row = {
            "question": "What is 2+2?",
            "solution": solution,
            "final_answer": "4",
            "hint": "Add",
            "rollout": "Maybe 4",
        }
        targets = [
            dg.render_target(tok, dg.messages_for(row, c, "Add"), solution)[1]
            for c in ("none",) + dg.CONDITIONS
        ]
        self.assertTrue(all(t == targets[0] for t in targets))
        decoded = tok.decode(targets[0])
        self.assertEqual(decoded, "The answer is \\boxed{4}.<|im_end|>\n")
        prompt, target = dg.render_target(tok, dg.messages_for(row, "none"), solution)
        self.assertTrue(tok.decode(prompt).endswith("<think>\n\n</think>\n\n"))
        self.assertNotIn("Compute 2+2=4.", tok.decode(prompt))
        # Changing hidden reasoning cannot affect either scoring input. This
        # would fail if we merely sliced the loss after a teacher-forced trace.
        alternative = (
            "Entirely different hidden reasoning.</think>\nThe answer is \\boxed{4}."
        )
        self.assertEqual(
            (prompt, target),
            dg.render_target(tok, dg.messages_for(row, "none"), alternative),
        )

    def test_missing_empty_or_ambiguous_solution_is_rejected(self):
        for source, message in [
            ("No delimiter", "malformed_thinking_trace"),
            ("Trace</think>   ", "empty_final_solution"),
            ("Trace</think>Answer</think>Extra", "malformed_thinking_trace"),
        ]:
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ValueError, message),
            ):
                dg.render_target(CharacterTokenizer(), [], source)

    def test_template_that_leaves_thinking_in_target_is_rejected(self):
        tok = CharacterTokenizer()
        with (
            mock.patch.object(
                tok,
                "apply_chat_template",
                side_effect=[
                    "header",
                    "header<think></think>Final solution",
                ],
            ),
            self.assertRaisesRegex(
                ValueError, "solution_target_contains_thinking_tags"
            ),
        ):
            dg.render_target(tok, [], "Private reasoning</think>Final solution")


class RealScoringTest(unittest.TestCase):
    def test_block_scoring_matches_full_sequence_cross_entropy(self):
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                attn_implementation="eager",
            )
        ).eval()
        prompt, target = [1, 2, 3], [4, 5, 6, 7, 8]
        with torch.inference_mode():
            logits = (
                model(torch.tensor([prompt + target]))
                .logits[0, len(prompt) - 1 : -1]
                .float()
            )
            expected = (
                torch.log_softmax(logits, -1)
                .gather(-1, torch.tensor(target)[:, None])
                .squeeze(-1)
                .numpy()
            )
        for block in (1, 2, 100):
            actual = dg.score_tokens(model, prompt, target, "cpu", block)
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        none = dg.score_tokens(model, prompt, target, "cpu", 2)
        np.testing.assert_array_equal(np.asarray(none) - np.asarray(none), np.zeros(5))


class MetricsTest(unittest.TestCase):
    def test_total_normalized_and_position_conserve_gain(self):
        result = dg.token_metrics([1.0, -2.0, 3.0], 20, 5)
        self.assertEqual(result["total_gain"], 2.0)
        self.assertAlmostEqual(result["normalized_gain"], 2 / 3)
        self.assertEqual(result["cumulative_gain"][-1], 2.0)
        self.assertAlmostEqual(np.mean(result["position_gain"]), 2 / 3)
        self.assertEqual(result["early_gain"], [1.0, -2.0, 3.0, None, None])

    def test_bootstrap_with_no_supported_draws_returns_null_interval(self):
        result = dg.estimate([[None], [4.0]], np.array([[0, 0]]))
        self.assertEqual(result["mean"], [4.0])
        self.assertEqual(result["ci95"], [[None, None]])

    def test_support_and_question_balancing(self):
        draws = np.array([[0, 1], [0, 0], [1, 1]])
        result = dg.estimate([[1.0, None], [3.0, 4.0]], draws)
        self.assertEqual(result["mean"], [2.0, 4.0])
        self.assertEqual(result["n_questions"], [2, 1])


class PipelineTest(unittest.TestCase):
    def fixture(self, root):
        rows = [
            {
                "question_id": "q",
                "question_idx": 0,
                "question": "Q",
                "final_answer": "4",
                "solution": "Reason</think>4",
                "demo_hash": digest("Reason</think>4"),
                "target_tokens": 3,
                "target_ids": [4, 5, 6],
                "prompt_ids": {"none": [1], "answer": [2], "full": [3]},
            }
        ]
        manifest = {
            "version": dg.VERSION,
            "dataset": "deepmath",
            "target_format": dg.TARGET_FORMAT,
            "model": "test",
            "model_identity": model_identity("test"),
            "revision": "revision",
            "tokenizer_hash": "tokenizer",
            "cohort_hash": digest(rows),
            "conditions": ["answer", "full"],
            "max_model_len": 100,
        }
        dg.write_rows(root / "cohort.jsonl", rows)
        dg.write_json(root / "manifest.json", manifest)
        return dg.build_parser().parse_args(
            [
                "--output-dir",
                str(root),
                "--phase",
                "score",
                "--bootstrap-samples",
                "20",
                "--early-tokens",
                "5",
            ]
        )

    def test_incremental_conditions_reuse_baseline_and_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)

            def fake_score(model, prompt, target, device, block):
                return [-3.0 + prompt[0] / 2] * len(target)

            with (
                mock.patch(
                    "transformers.AutoModelForCausalLM.from_pretrained"
                ) as loader,
                mock.patch.object(
                    dg, "score_tokens", side_effect=fake_score
                ) as forward,
            ):
                args.conditions = ["answer"]
                dg.score(args)
                self.assertEqual(forward.call_count, 2)
                args.conditions = ["answer", "full"]
                dg.score(args)
                self.assertEqual(forward.call_count, 3)
                index = dg.read_json(root / "score_index.json")
                self.assertEqual(index["cache_stats"], {"hits": 2, "misses": 1})
                dg.score(args)
                self.assertEqual(
                    loader.call_count, 2
                )  # fully cached run loads no GPU model
            summary = dg.aggregate(args)
            self.assertEqual(summary["conditions"]["answer"]["total_gain"]["mean"], 1.5)
            self.assertEqual(
                summary["conditions"]["full"]["normalized_gain"]["mean"], 1.0
            )
            self.assertEqual(
                summary["paired_differences"]["answer_minus_full"]["total_gain"][
                    "mean"
                ],
                -1.5,
            )
            from eval.viz.demo_gain import plot

            plot(root)
            self.assertTrue((root / "figures" / "gain.png").exists())
            record = index["records"][0]["scores"]["full"][0]
            path = root / "score_cache" / "demo_logps" / (record["key"] + ".json")
            saved = dg.read_json(path)
            saved["result"]["logps"][0] = 100
            dg.write_json(path, saved)
            with self.assertRaisesRegex(ValueError, "Invalid condition cache"):
                dg.aggregate(args)

    def test_cohort_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            rows = dg.read_rows(root / "cohort.jsonl")
            rows[0]["target_ids"][0] += 1
            dg.write_rows(root / "cohort.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "contents changed"):
                dg.load_cohort(root)

    def test_whole_trace_cohorts_are_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            manifest = dg.read_json(root / "manifest.json")
            manifest["version"] = 1
            dg.write_json(root / "manifest.json", manifest)
            with (
                mock.patch(
                    "transformers.AutoModelForCausalLM.from_pretrained"
                ) as loader,
                self.assertRaisesRegex(
                    ValueError, "prepare a new solution-only cohort"
                ),
            ):
                dg.score(args)
            loader.assert_not_called()

    def test_cache_keys_cover_exact_prompt_target_and_numerics(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ConditionCache(Path(directory), "x", {"dtype": "float32"})
            self.assertNotEqual(
                cache.key({"prompt_ids": [1], "target_ids": [2]}),
                cache.key({"prompt_ids": [1], "target_ids": [3]}),
            )
            self.assertNotEqual(
                cache.key({"prompt_ids": [1], "target_ids": [2]}),
                cache.key({"prompt_ids": [3], "target_ids": [2]}),
            )


class HintTest(unittest.TestCase):
    def test_levels_share_source_but_have_distinct_instructions(self):
        row = {"question": "A problem", "solution": "A solution"}
        texts = [
            hv.build_messages(row, c, b)[-1]["content"]
            for c, (b, _) in hv.LEVELS.items()
        ]
        self.assertEqual(len(set(texts)), 3)
        self.assertTrue(all("A problem" in t and "A solution" in t for t in texts))

    def test_invalid_outputs_are_labelled(self):
        self.assertEqual(hv.classify_hint("", "17"), "empty")
        self.assertEqual(hv.classify_hint("<think>guess", "17"), "thinking")
        self.assertEqual(hv.classify_hint("The answer is 17", "17"), "answer_leak")
        self.assertEqual(hv.classify_hint("Use symmetry", "17"), "")


class PreparationTest(unittest.TestCase):
    def test_context_filter_counts_full_pi_but_only_solution_target(self):
        import types

        hints = [
            {"question": str(i), "final_answer": "42", "hint": "Use algebra"}
            for i in range(5)
        ]
        solutions = [
            {**hints[0], "solution": "Reasoning</think>42"},
            {**hints[1], "solution": "x" * 1200 + "</think>42"},
            {**hints[2], "solution": "No thinking boundary"},
            {**hints[3], "solution": "First</think>42"},
            {**hints[3], "solution": "Different</think>42"},
            {**hints[3], "solution": "First</think>42"},
            {**hints[4], "solution": "x" * 600 + "</think>42"},
        ]
        tok = CharacterTokenizer()
        tok.get_vocab = lambda: {chr(i): i for i in range(128)}
        with tempfile.TemporaryDirectory() as directory:
            args = dg.build_parser().parse_args(
                [
                    "--output-dir",
                    directory,
                    "--model",
                    "test",
                    "--conditions",
                    "answer",
                    "full",
                    "hint",
                    "--num-problems",
                    "0",
                    "--max-model-len",
                    "1000",
                ]
            )
            with (
                mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    return_value=types.SimpleNamespace(
                        max_position_embeddings=2000, _commit_hash="rev"
                    ),
                ),
                mock.patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=tok
                ),
                mock.patch.object(dg, "load_hint_cache", return_value=hints),
                mock.patch.object(dg, "load_train_dataset", return_value=solutions),
            ):
                dg.prepare(args)
            manifest, cohort = dg.load_cohort(directory)
            self.assertEqual(len(cohort), 2)
            self.assertEqual(manifest["target_format"], dg.TARGET_FORMAT)
            self.assertTrue(
                all(r["target_ids"] == list(map(ord, "42<end>")) for r in cohort)
            )
            self.assertEqual(manifest["ambiguous_source_question_answers"], 1)
            self.assertEqual(
                manifest["exclusions_in_scanned_candidates"]["ambiguous_demo"], 1
            )
            self.assertEqual(
                manifest["exclusions_in_scanned_candidates"]["over_context_full"], 1
            )
            self.assertEqual(
                manifest["exclusions_in_scanned_candidates"][
                    "malformed_thinking_trace"
                ],
                1,
            )


class VariantPipelineTest(unittest.TestCase):
    def test_generation_resume_new_samples_and_failure_retention(self):
        import sys
        import types

        row = {
            "question_id": "q",
            "question": "question",
            "final_answer": "42",
            "solution": "reason</think>42",
            "demo_hash": "demo",
        }
        manifest = {"model": "test", "revision": "rev", "cohort_hash": "cohort"}
        tok = types.SimpleNamespace(
            chat_template="test",
            get_vocab=lambda: {"x": 1},
            apply_chat_template=lambda *a, **k: {"input_ids": [[1, 2, 3]]},
        )
        requests = []

        def chat(messages, params, **kwargs):
            outputs = []
            for p in params:
                requests.append(p.seed)
                text = "The answer is 42" if p.max_tokens == 512 else "Use algebra"
                reason = "length" if p.max_tokens == 32 else "stop"
                outputs.append(
                    types.SimpleNamespace(
                        outputs=[
                            types.SimpleNamespace(
                                text=text, token_ids=[1, 2], finish_reason=reason
                            )
                        ]
                    )
                )
            return outputs

        engine = mock.Mock()
        engine.chat.side_effect = chat
        fake = types.SimpleNamespace(
            LLM=mock.Mock(return_value=engine),
            SamplingParams=lambda **kw: types.SimpleNamespace(**kw),
        )
        with tempfile.TemporaryDirectory() as directory:
            args = hv.build_parser().parse_args(
                [
                    "--cohort-dir",
                    directory,
                    "--output-dir",
                    str(Path(directory) / "hints"),
                ]
            )
            with (
                mock.patch.object(hv, "load_cohort", return_value=(manifest, [row])),
                mock.patch.object(
                    hv,
                    "vllm_model_and_adapter",
                    return_value=(
                        {"model": "test"},
                        None,
                        types.SimpleNamespace(base_model="test"),
                    ),
                ),
                mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    return_value=types.SimpleNamespace(_commit_hash="rev"),
                ),
                mock.patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=tok
                ),
                mock.patch.dict(sys.modules, {"vllm": fake}),
            ):
                hv.generate(args)
                first = dg.read_rows(Path(args.output_dir) / "hints.jsonl")
                self.assertEqual(len(requests), 3)
                hv.generate(args)
                self.assertEqual(len(requests), 3)
                self.assertEqual(fake.LLM.call_count, 1)
                args.samples_per_level = 2
                hv.generate(args)
                self.assertEqual(len(requests), 6)
                updated = dg.read_rows(Path(args.output_dir) / "hints.jsonl")
                self.assertEqual([r for r in updated if r["sample_idx"] == 0], first)
                self.assertEqual(
                    next(r for r in first if r["condition"] == "hint_detailed")[
                        "invalid_reason"
                    ],
                    "answer_leak",
                )
                self.assertTrue(
                    next(r for r in first if r["condition"] == "hint_short")[
                        "truncated"
                    ]
                )

    def test_invalid_hint_excludes_question_from_every_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = PipelineTest().fixture(root)
            manifest, rows = dg.load_cohort(root)
            original = rows[0]
            rows.append({**original, "question_id": "q2"})
            manifest["cohort_hash"] = digest(rows)
            dg.write_rows(root / "cohort.jsonl", rows)
            dg.write_json(root / "manifest.json", manifest)
            variants = [
                {
                    "question_id": r["question_id"],
                    "demo_hash": r["demo_hash"],
                    "condition": "hint_short",
                    "sample_idx": 0,
                    "hint": "algebra",
                    "n_tokens": 2,
                    "invalid_reason": "answer_leak" if r["question_id"] == "q2" else "",
                    "truncated": False,
                }
                for r in rows
            ]
            dg.write_rows(root / "hints" / "hints.jsonl", variants)
            dg.write_json(
                root / "hints" / "manifest.json",
                {"hints_hash": digest(variants), "config": {"samples_per_level": 1}},
            )
            args.conditions = ["answer", "hint_short"]
            with (
                mock.patch("transformers.AutoTokenizer.from_pretrained"),
                mock.patch.object(dg, "tokenizer_hash", return_value="tokenizer"),
                mock.patch.object(
                    dg, "render_target", return_value=([7], original["target_ids"])
                ),
                mock.patch("transformers.AutoModelForCausalLM.from_pretrained"),
                mock.patch.object(dg, "score_tokens", return_value=[-1.0] * 3),
            ):
                dg.score(args)
            index = dg.read_json(root / "score_index.json")
            self.assertEqual([r["question_id"] for r in index["records"]], ["q"])
            self.assertEqual(
                index["exclusions"], {"invalid_or_truncated_hint_short": 1}
            )
            summary = dg.aggregate(args)
            self.assertEqual(summary["n_questions"], 1)
            variants[0]["hint"] = "another hint"
            dg.write_rows(root / "hints" / "hints.jsonl", variants)
            dg.write_json(
                root / "hints" / "manifest.json",
                {"hints_hash": digest(variants), "config": {"samples_per_level": 1}},
            )
            with self.assertRaisesRegex(ValueError, "Hint artifacts changed"):
                dg.aggregate(args)


if __name__ == "__main__":
    unittest.main()
