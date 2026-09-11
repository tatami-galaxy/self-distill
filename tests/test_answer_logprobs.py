"""Extraction, causal scoring, statistics, and resume checks without model downloads."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from datasets import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM

from eval import answer_logprobs as experiment


class CharacterTokenizer:
    chat_template = "test"
    all_special_ids = ()

    def decode(self, ids, **kwargs):
        return "".join("\\boxed{" if i == 1000 else chr(i) for i in ids)

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(map(ord, text)),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def apply_chat_template(self, conversations, **kwargs):
        return {
            "input_ids": [
                list(map(ord, "\n".join(m["content"] for m in messages) + "\n"))
                for messages in conversations
            ]
        }

    def get_vocab(self):
        return {chr(i): i for i in range(128)}


class BoxExtractionTest(unittest.TestCase):
    def test_last_box_handles_nested_fractions_and_escaped_braces(self):
        text = r"First \boxed{8}, finally \boxed{ \frac{1}{2} + \{3\} }."
        box, error = experiment.final_box(text)
        self.assertIsNone(error)
        self.assertEqual(box["answer"], r"\frac{1}{2} + \{3\}")
        self.assertEqual(box["box_count"], 2)
        self.assertEqual(text[slice(*box["answer_chars"])], box["answer"])

    def test_bad_final_box_does_not_fall_back_to_an_earlier_box(self):
        for text, reason in (
            (r"\boxed{7} then \boxed{8", "malformed_final_box"),
            ("no answer", "no_box"),
            (r"\boxed{ }", "empty_box"),
        ):
            with self.subTest(text=text):
                self.assertEqual(experiment.final_box(text), (None, reason))

    def test_noncanonical_original_ids_are_preserved(self):
        tokenizer = CharacterTokenizer()
        ids = list(map(ord, "Thus ")) + [1000] + list(map(ord, "12}."))
        text = tokenizer.decode(ids)
        box, _ = experiment.final_box(text)
        box_span, answer_span = experiment.token_spans(
            tokenizer, ids, text, [box["box_chars"], box["answer_chars"]]
        )
        self.assertEqual(
            ids[slice(*box_span["tokens"])], [1000, ord("1"), ord("2"), ord("}")]
        )
        self.assertEqual(ids[slice(*answer_span["tokens"])], [ord("1"), ord("2")])
        self.assertEqual(answer_span["left_overlap_chars"], 0)

    def test_boundary_overlap_is_recorded(self):
        class JoinedTokenizer(CharacterTokenizer):
            def decode(self, ids, **kwargs):
                return "".join("7}" if i == 1001 else chr(i) for i in ids)

        tokenizer = JoinedTokenizer()
        ids = list(map(ord, r"\boxed{")) + [1001]
        text = tokenizer.decode(ids)
        box, _ = experiment.final_box(text)
        _, answer_span = experiment.token_spans(
            tokenizer, ids, text, [box["box_chars"], box["answer_chars"]]
        )
        self.assertEqual(ids[slice(*answer_span["tokens"])], [1001])
        self.assertEqual(answer_span["right_overlap_chars"], 1)

    def test_pi_changes_prompt_only_and_none_is_exact_control(self):
        problem = {"question": "Q", "final_answer": "7", "hint": "H", "solution": "S"}
        self.assertEqual(
            experiment.teacher_messages(problem, "none"),
            experiment.format_prompt_math("Q"),
        )
        for mode in ("answer", "hint", "full"):
            messages = experiment.teacher_messages(problem, mode)
            self.assertEqual(len(messages), 2)
            self.assertTrue(messages[-1]["content"].startswith("Q\n\n"))
        self.assertNotIn("rollout", experiment.PI_MODES)


class CausalAnswerScoringTest(unittest.TestCase):
    def test_scores_match_full_forward_and_ignore_text_after_box(self):
        torch.manual_seed(10)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
                attention_dropout=0.0,
            )
        ).eval()
        tokenizer = CharacterTokenizer()
        text = r"Work gives \boxed{17}. trailing text"
        ids = list(map(ord, text))
        box, _ = experiment.final_box(text)
        box_span, answer_span = experiment.token_spans(
            tokenizer, ids, text, [box["box_chars"], box["answer_chars"]]
        )
        record = {
            "completion_ids": ids,
            "box_span": box_span,
            "answer_span": answer_span,
        }
        prompt = list(map(ord, "PI: "))
        actual = experiment.score_box(model, prompt, record, "cpu")
        start, end = box_span["tokens"]
        with torch.inference_mode():
            logits = model(
                input_ids=torch.tensor([prompt + ids[:end]]), use_cache=False
            ).logits
            selected = (
                logits[0, len(prompt) + start - 1 : len(prompt) + end - 1]
                .float()
                .log_softmax(-1)
            )
            expected = selected.gather(
                -1, torch.tensor(ids[start:end]).unsqueeze(-1)
            ).squeeze(-1)
        torch.testing.assert_close(torch.tensor(actual), expected)
        changed = {**record, "completion_ids": ids[:end] + [1, 2, 3]}
        self.assertEqual(actual, experiment.score_box(model, prompt, changed, "cpu"))
        reduced = experiment.reduce_box_logps(actual, record)
        a, b = answer_span["tokens"]
        self.assertAlmostEqual(
            reduced["answer_sum"], sum(actual[a - start : b - start])
        )
        self.assertEqual(reduced["answer_token_count"], 2)


class CorrectnessComparisonTest(unittest.TestCase):
    def test_all_correct_and_all_incorrect_questions_contribute_to_pooled_metrics(self):
        labels = np.array([1, 1, 0, 0], dtype=bool)
        scores = np.array([-0.1, -0.2, -2.0, -3.0])
        groups = experiment.question_groups(["easy", "easy", "hard", "hard"])
        metrics = experiment.comparison_metrics(labels, scores, groups)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertGreater(metrics["point_biserial"], 0)
        self.assertEqual(metrics["mixed_questions"], 0)
        self.assertIsNone(metrics["within_question_pair_accuracy"])

    def test_within_question_comparison_does_not_compare_different_questions(self):
        labels = np.array([1, 0, 1, 0], dtype=bool)
        scores = np.array([-11.0, -10.0, -1.0, -0.5])
        groups = experiment.question_groups(["a", "a", "b", "b"])
        metrics = experiment.comparison_metrics(labels, scores, groups)
        self.assertEqual(metrics["within_question_pair_accuracy"], 0.0)
        self.assertEqual(metrics["roc_auc"], 0.25)
        self.assertEqual(experiment.roc_auc(labels, np.ones(4)), 0.5)
        self.assertIsNone(experiment.roc_auc(np.ones(4, dtype=bool), scores))

    def test_bootstrap_preserves_questions_and_paired_baseline(self):
        labels = np.array([1, 0, 1, 0], dtype=bool)
        scores = np.array([-1.0, -2.0, -0.1, -0.2])
        groups = experiment.question_groups(["a", "a", "b", "b"])
        result = experiment.bootstrap_comparison(
            labels, scores, groups, samples=30, seed=7, baseline=scores
        )
        self.assertEqual(result["auc_difference_vs_none"]["ci95"], [0.0, 0.0])
        self.assertEqual(result["within_question_pair_accuracy"]["ci95"], [1.0, 1.0])
        self.assertEqual(
            result,
            experiment.bootstrap_comparison(
                labels, scores, groups, samples=30, seed=7, baseline=scores
            ),
        )

    def test_context_join_uses_question_and_gold_and_rejects_conflicts(self):
        source = [
            {"question": "q", "final_answer": "1", "hint": "a"},
            {"question": "q", "final_answer": "2", "hint": "b"},
            {"question": "q", "final_answer": "1", "hint": "c"},
        ]
        result = experiment.context_index(source, "hint", {("q", "1"), ("q", "2")})
        self.assertEqual(result, {("q", "1"): "", ("q", "2"): "b"})


class SourceStampTest(unittest.TestCase):
    def test_legacy_manifest_round_trip_preserves_stamp_and_score_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.arrow"
            path.write_bytes(b"original cache")
            stamp = experiment.source_stamp(tmp)
            # Existing manifests serialized tuple entries as JSON arrays.
            legacy = {"source_stamp": [tuple(entry) for entry in stamp]}
            loaded = json.loads(json.dumps(legacy))
            self.assertEqual(loaded["source_stamp"], experiment.source_stamp(tmp))
            self.assertEqual(
                experiment.fingerprint(legacy), experiment.fingerprint(loaded)
            )
            path.write_bytes(b"a changed cache with a different size")
            self.assertNotEqual(loaded["source_stamp"], experiment.source_stamp(tmp))


class FullCachePipelineTest(unittest.TestCase):
    def test_prepare_score_resume_and_all_question_aggregation(self):
        tokenizer = CharacterTokenizer()
        texts = [
            ("mixed", r"\boxed{7}", 1),
            ("mixed", r"\boxed{8}", 0),
            ("correct", r"\boxed{7}", 1),
            ("correct", r"\boxed{7}", 1),
            ("incorrect", r"\boxed{8}", 0),
            ("incorrect", r"\boxed{8}", 0),
            ("missing", "no box", 0),
            ("malformed", r"\boxed{7", 0),
            ("disagrees", r"\boxed{8}", 1),
            ("no_hint", r"\boxed{7}", 1),
        ]
        rows = [
            {
                "question": q,
                "final_answer": "7",
                "completion_ids": list(map(ord, text)),
                "completion_text": text,
                "reward": reward,
                "gen_model": "fake",
                "dataset": "deepmath",
            }
            for q, text, reward in texts
        ]
        source = [
            {"question": q, "final_answer": "7", "hint": "h", "solution": "worked"}
            for q in dict.fromkeys(q for q, _, _ in texts)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "rollouts"
            cache.mkdir()
            cache_file = cache / "data.arrow"
            cache_file.write_bytes(b"nonempty cache fixture")
            args = experiment.build_parser().parse_args(
                [
                    "--model",
                    "fake",
                    "--rollout-cache",
                    str(cache),
                    "--output-dir",
                    str(Path(tmp) / "out"),
                    "--bootstrap-samples",
                    "10",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                ]
            )
            self.assertFalse(hasattr(args, "num_problems"))
            self.assertEqual(args.pi_modes, ["none", "answer", "hint", "full"])
            with (
                mock.patch(
                    "datasets.load_from_disk", return_value=Dataset.from_list(rows)
                ),
                mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    return_value=types.SimpleNamespace(
                        _commit_hash="fixed", max_position_embeddings=4096
                    ),
                ),
                mock.patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
                ),
                mock.patch.object(
                    experiment,
                    "load_hint_cache",
                    return_value=[r for r in source if r["question"] != "no_hint"],
                ),
                mock.patch.object(
                    experiment, "load_train_dataset", return_value=source
                ),
            ):
                experiment.prepare(args)
            out = Path(args.output_dir)
            manifest = experiment.read_json(out / "manifest.json")
            self.assertEqual(manifest["source_rollouts"], 10)
            self.assertEqual(manifest["scorable_rollouts"], 7)
            self.assertEqual(manifest["scorable_questions"], 4)
            self.assertEqual(manifest["scorable_mixed_questions"], 1)
            self.assertEqual(manifest["diagnostics"]["label_disagreements"], 1)
            self.assertEqual(
                manifest["exclusions"],
                {"no_box": 1, "malformed_final_box": 1, "missing_or_ambiguous_pi": 1},
            )

            def fake_score(model, prompt, record, device):
                start, end = record["box_span"]["tokens"]
                return [-0.1 if record["correct"] else -2.0] * (end - start)

            with (
                mock.patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=torch.nn.Linear(1, 1),
                ) as load,
                mock.patch.object(
                    experiment, "score_box", side_effect=fake_score
                ) as score,
            ):
                experiment.score(args)
                self.assertEqual(score.call_count, 28)
                self.assertEqual(load.call_count, 1)
                score.reset_mock()
                load.reset_mock()
                with (
                    mock.patch.object(
                        experiment,
                        "summarize",
                        side_effect=ModuleNotFoundError("scipy"),
                    ),
                    self.assertRaises(ModuleNotFoundError),
                ):
                    experiment.aggregate(args)
                manifest_before = (out / "manifest.json").read_bytes()
                experiment.prepare(args)
                experiment.score(args)
                score.assert_not_called()
                load.assert_not_called()
                self.assertEqual((out / "manifest.json").read_bytes(), manifest_before)
            experiment.aggregate(args)
            summary = experiment.read_json(out / "summary.json")
            self.assertEqual(summary["questions"], 4)
            self.assertEqual(summary["rollouts"], 7)
            self.assertEqual(summary["correct"], 3)
            self.assertEqual(summary["incorrect"], 4)
            for mode in args.pi_modes:
                self.assertEqual(
                    summary["conditions"][mode]["answer_mean"]["roc_auc"], 1.0
                )
            self.assertTrue((out / "answer_score_distributions.png").exists())
            self.assertEqual(len(list(experiment.jsonl_rows(out / "answers.jsonl"))), 7)
            cache_file.write_bytes(b"changed rollout cache")
            with self.assertRaisesRegex(ValueError, "Rollout cache changed"):
                experiment.prepare(args)
            score_file = next((out / "scores").glob("*.json"))
            saved = experiment.read_json(score_file)
            saved["signature"] = "wrong"
            experiment.write_json(score_file, saved)
            with self.assertRaisesRegex(ValueError, "provenance"):
                experiment.aggregate(args)


if __name__ == "__main__":
    unittest.main()
