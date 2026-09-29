"""Selection math, cache separation, and a tiny CPU-only pipeline with fake inference."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset, load_from_disk

from eval import hint_selection as hs


def hint(i, tokens, valid=True):
    return {"hint_id": str(i), "sample_idx": i, "hint": f"hint {i}",
            "n_tokens": tokens, "valid": valid, "question_id": "q"}


class SelectionTest(unittest.TestCase):
    def test_epsilon_boundary_and_zero_change_winner(self):
        hints = [hint(0, 80), hint(1, 20), hint(2, 1)]
        scores = {"0": {"sufficiency": 1.0, "raw_transfer": .01},
                  "1": {"sufficiency": 7 / 8, "raw_transfer": .01},
                  "2": {"sufficiency": 6 / 8, "raw_transfer": .01}}
        loose = hs.choose_hint(hints, scores, 1 / 8, 6, 128)
        strict = hs.choose_hint(hints, scores, 0, 6, 128)
        self.assertEqual(loose["selected"]["hint_id"], "1")
        self.assertEqual(loose["threshold"], 7 / 8)
        self.assertEqual(loose["n_eligible"], 2)
        self.assertEqual(strict["selected"]["hint_id"], "0")

    def test_transfer_cost_can_outweigh_length_and_raw_score_is_preserved(self):
        hints = [hint(0, 10), hint(1, 30)]
        scores = {"0": {"sufficiency": .5, "raw_transfer": .1},
                  "1": {"sufficiency": .5, "raw_transfer": -.01}}
        self.assertEqual(hs.choose_hint(hints, scores, 0, 0, 128)["selected"]["hint_id"], "0")
        selected = hs.choose_hint(hints, scores, 0, 6, 128)["selected"]
        self.assertEqual(selected["hint_id"], "1")
        self.assertEqual(selected["raw_transfer"], -.01)
        self.assertEqual(selected["transfer_cost"], 0)
        self.assertAlmostEqual(selected["objective"], 30 / 128)

    def test_invalid_hint_cannot_set_best_or_win(self):
        hints = [hint(0, 1, False), hint(1, 40)]
        result = hs.choose_hint(hints, {"1": {"sufficiency": .25, "raw_transfer": .01}}, .125, 6, 128)
        self.assertEqual(result["best_sufficiency"], .25)
        self.assertEqual(result["selected"]["hint_id"], "1")

    def test_zero_success_and_missing_candidates_are_explicit(self):
        result = hs.choose_hint([hint(0, 20)], {"0": {"sufficiency": 0, "raw_transfer": .02}}, .125, 6, 128)
        self.assertEqual(result["status"], "selected_zero_success")
        self.assertEqual(result["threshold"], 0)
        self.assertIsNotNone(result["selected"])
        self.assertEqual(hs.choose_hint([hint(0, 20, False)], {}, .125, 6, 128)["status"], "no_valid_hints")

    def test_ties_are_stable_and_bad_scores_rejected(self):
        hints = [hint(1, 20), hint(0, 20)]
        scores = {str(i): {"sufficiency": .5, "raw_transfer": .02} for i in range(2)}
        self.assertEqual(hs.choose_hint(hints, scores, 0, 6, 128)["selected"]["hint_id"], "0")
        scores["1"]["raw_transfer"] = float("nan")
        with self.assertRaises(ValueError):
            hs.choose_hint(hints, scores, 0, 6, 128)

    def test_student_samples_support_legacy_row_order_and_reject_duplicates(self):
        legacy = Dataset.from_dict({"question": ["a", "b", "a"]})
        index, source = hs.index_student_samples(legacy)
        self.assertEqual(index["a"], {0: 0, 1: 2})
        self.assertEqual(source, "legacy_row_order")
        modern = legacy.add_column("sample_idx", [4, 0, 2])
        index, source = hs.index_student_samples(modern)
        self.assertEqual(sorted(index["a"]), [2, 4])
        self.assertEqual(source, "stored_sample_idx")
        with self.assertRaises(ValueError):
            hs.index_student_samples(legacy.add_column("sample_idx", [0, 0, 0]))

    def test_protected_cache_roots_and_checksum(self):
        for path in ("data/pi/hint", "data/pi/hint/deepmath/new", "data/pi", "data/rollouts/new"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                hs.protect_output(Path(path))
        hs.protect_output(Path("data/pi/hint_selection/pilot"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hs.save_artifact(root, "test", "id", [1])
            self.assertEqual(hs.load_artifact(root, "test", "id"), [1])
            with self.assertRaises(ValueError):
                hs.load_artifact(root, "test", "different")
            data = hs.read_json(root / "test.json")
            data["rows"] = [2]
            hs.write_json(root / "test.json", data)
            with self.assertRaises(ValueError):
                hs.load_artifact(root, "test", "id")


class PipelineTest(unittest.TestCase):
    def test_resumable_scoring_and_independent_epsilon_exports(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp) / "experiment"
            args = hs.build_parser().parse_args(["--num-questions", "1", "--num-hints", "3",
                                                 "--transfer-rollouts", "2"])
            c = hs.prepare_config(args)
            q = {"question_id": "q", "question": "A toy question", "final_answer": "42",
                 "solution": "Demo", "hint_prompt_ids": [1, 2], "student_prompt_ids": [3],
                 "student_completion_ids": [[4, 5], [6]], "student_sample_indices": [0, 1]}
            manifest = {"config": c, "cohort_fingerprint": hs.digest([q])}
            hs.save_artifact(root, "cohort", hs.digest(c), [q])
            hs.write_json(root / "manifest.json", manifest)
            engine = mock.Mock()
            engine.generate.return_value = [SimpleNamespace(outputs=[
                SimpleNamespace(text="Use a substitution", token_ids=[7] * 80, finish_reason="stop"),
                SimpleNamespace(text="Use an invariant", token_ids=[8] * 20, finish_reason="stop"),
                SimpleNamespace(text="Partial hint", token_ids=[9] * 128, finish_reason="length"),
            ])]
            fake_vllm = SimpleNamespace(SamplingParams=SimpleNamespace)
            with mock.patch.dict("sys.modules", {"vllm": fake_vllm}), mock.patch.object(hs, "new_engine", return_value=engine) as create:
                hs.generate(args, root)
                hs.generate(args, root)
                self.assertEqual(create.call_count, 1)
                self.assertEqual(engine.generate.call_args.args[1].temperature, 1.4)
                rows = hs.candidates(root, manifest, [q])
                self.assertEqual([r["valid"] for r in rows], [True, True, False])
                # Independent per-sample outcomes are stored, including capped solutions.
                engine.generate.reset_mock()
                engine.generate.return_value = [SimpleNamespace(outputs=[
                    SimpleNamespace(text="correct" if i < successes else "wrong", token_ids=[1, 2], finish_reason="length" if i == 7 else "stop")
                    for i in range(8)
                ]) for successes in (8, 7)]
                with mock.patch.object(hs, "render", return_value=[3, 8]), mock.patch("utils.grade", side_effect=lambda text, *_: (None, text == "correct")):
                    hs.sufficiency(args, root)
                    hs.sufficiency(args, root)
                self.assertEqual(engine.generate.call_count, 1)
                self.assertEqual(len(engine.generate.call_args.args[0]), 2)
                self.assertEqual(engine.generate.call_args.args[1].n, 8)
            fake_model = mock.Mock()
            fake_model.eval.return_value.to.return_value = fake_model
            # Opposite per-rollout differences cancel before clamping; no per-token
            # weighting across differently sized rollouts is allowed.
            def score(_model, prompt, ids):
                if prompt == [3]:
                    return [-1.0] * len(ids)
                return ([-.9] * len(ids)) if len(ids) == 2 else [-1.1]
            with mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model) as load_model, \
                    mock.patch("transformers.AutoTokenizer.from_pretrained"), \
                    mock.patch.object(hs, "render", return_value=[3, 8]), \
                    mock.patch.object(hs, "score_completion", side_effect=score) as forward:
                hs.transfer(args, root)
                self.assertEqual(forward.call_count, 6)  # 2 shared student + 2x2 teacher.
                hs.transfer(args, root)
                self.assertEqual(load_model.call_count, 1)
                self.assertEqual(forward.call_count, 6)
            hs.select(args, root)
            hs.select(args, root)  # Verify committed export integrity on reuse.
            args.epsilon = 0
            hs.select(args, root)
            selections = sorted((root / "selections").iterdir())
            self.assertEqual(len(selections), 2)
            winners = set()
            for out in selections:
                exported = load_from_disk(str(out / "hints"))
                self.assertEqual(len(exported), 1)
                self.assertEqual(exported[0]["gen_model"], args.model)
                winners.add(exported[0]["hint_sample_idx"])
                self.assertIn(exported[0]["n_tokens"], (20, 80))
            self.assertEqual(winners, {0, 1})
            # Downstream SDFT can consume the new cache via its explicit override.
            from train.opsd.train_sdft import build_hint_dataset
            ds = build_hint_dataset(args.model, args.dataset, 1, hint_cache=str(selections[0] / "hints"))
            self.assertEqual(len(ds), 1)
            self.assertIn("privileged_context", ds.column_names)


if __name__ == "__main__":
    unittest.main()
