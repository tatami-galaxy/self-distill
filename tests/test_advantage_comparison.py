"""Correctness tests for fixed-rollout distillation / Vine MC comparisons."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from eval import advantage_comparison as ac


class PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **kwargs):
        pieces = [self.pieces[i] for i in ids]
        if pieces and isinstance(pieces[0], bytes):
            return b"".join(pieces).decode("utf-8", errors="replace")
        return "".join(pieces)


def rollout(rid="r"):
    return {"rollout_id": rid, "question_id": "q", "student_prompt_ids": [7, 8],
            "completion_ids": [1, 2, 3, 9], "final_answer": "answer",
            "reward": 1.0, "truncated": False}


class SelectionTest(unittest.TestCase):
    def test_delimiters_snap_right_without_retokenizing(self):
        tokenizer = PieceTokenizer({1: "a\n", 2: "\nb", 3: "\n\n\n\n", 4: "c"})
        self.assertEqual(ac.delimiter_boundaries(tokenizer, [1, 2, 3, 4]), [2, 3])

    def test_partial_unicode_does_not_shift_boundary(self):
        tokenizer = PieceTokenizer({1: b"\xe2", 2: b"\x82", 3: b"\xac\n", 4: b"\nx"})
        self.assertEqual(ac.delimiter_boundaries(tokenizer, [1, 2, 3, 4]), [4])

    def test_segmentation_preserves_every_token_and_bounds(self):
        for length in range(1, 65):
            spans = ac.make_segments(length, [3, 7, 17, 18, 40, 60], 5, 11)
            self.assertEqual([i for a, b in spans for i in range(a, b)], list(range(length)))
            self.assertTrue(all(5 <= b - a <= 11 for a, b in spans[:-1]))
            self.assertTrue(1 <= spans[-1][1] - spans[-1][0] <= 11)
        self.assertEqual(ac.make_segments(20, [2, 4, 8, 16], 5, 10), [(0, 8), (8, 16), (16, 20)])

    def test_uniform_is_reproducible_and_clamps_short_rows(self):
        a = ac.uniform_positions(100, 12, 42, "r")
        self.assertEqual(a, ac.uniform_positions(100, 12, 42, "r"))
        self.assertEqual(len(set(a)), 12)
        self.assertNotEqual(a, ac.uniform_positions(100, 12, 42, "other"))
        self.assertEqual(ac.uniform_positions(3, 12, 42, "r"), [0, 1, 2])

    def test_plan_reuses_prefixes_and_never_branches_terminal(self):
        tokenizer = PieceTokenizer({1: "a", 2: "\n\n", 3: "b", 9: "EOS"})
        plan = ac.make_plan([rollout(), rollout("other")], tokenizer, ["uniform", "steps"], 9, 1, 3, 42)
        self.assertEqual(len(plan["prefixes"]), 4)
        self.assertEqual([len(p["prefix_ids"]) for p in plan["prefixes"]], [0, 1, 2, 3])
        self.assertIsNone(plan["rollouts"][0]["prefix_keys"]["4"])
        self.assertEqual(plan["rollouts"][0]["prefix_keys"], plan["rollouts"][1]["prefix_keys"])


class MCTest(unittest.TestCase):
    def test_interior_and_terminal_advantages(self):
        before, after = {"successes": 2, "samples": 4}, {"successes": 3, "samples": 4}
        self.assertEqual(ac.mc_advantage(before, after)["advantage"], 0.25)
        terminal = ac.mc_advantage(before, None, terminal_reward=1)
        self.assertEqual(terminal["advantage"], 0.5)
        self.assertEqual(terminal["value_after"], 0)
        self.assertEqual(terminal["transition_reward"], 1)

    def test_equal_mc_values_do_not_claim_certainty(self):
        for successes in (0, 16):
            counts = {"successes": successes, "samples": 16}
            result = ac.mc_advantage(counts, counts)
            self.assertEqual(result["advantage"], 0)
            self.assertLess(result["ci95"][0], 0)
            self.assertGreater(result["ci95"][1], 0)

    def test_eos_and_budget_validation(self):
        ac.validate_completion([1, 9], 5, [9], "stop")
        ac.validate_completion([1, 2], 2, [9], "length")
        for ids, budget in (([9, 1], 2), ([1], 2), ([], 2), ([1, 2, 3], 2)):
            with self.assertRaises(ValueError):
                ac.validate_completion(ids, budget, [9], "stop")

    def test_mc_exact_prefix_remaining_budget_and_extending_k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = {"key": "key", "question_id": "q", "student_prompt_ids": [7, 8],
                      "prefix_ids": [1, 2], "final_answer": "answer"}
            tokenizer = PieceTokenizer({1: "prefix", 2: " text", 9: " suffix"})
            ac.write_json(root / "cohort.json", {"eos_ids": [9], "tokenizers": {"student": {}}})
            ac.write_json(root / "plan.json", {"prefixes": [prefix]})
            calls = []

            def generate(prompts, params, **kwargs):
                calls.extend(zip(prompts, params))
                return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[9], finish_reason="stop")])
                        for _ in prompts]

            llm = SimpleNamespace(get_tokenizer=lambda: tokenizer, generate=generate)
            args = SimpleNamespace(output_dir=directory, mc_samples=2, max_completion_length=6,
                                   seed=42, mc_batch_size=3, save_mc_completions=True)
            with mock.patch.object(ac, "load_llm", return_value=llm), \
                 mock.patch.object(ac, "tokenizer_identity", return_value={}), \
                 mock.patch.object(ac, "sampling_params", side_effect=lambda a, remaining, seed, eos: (remaining, seed)), \
                 mock.patch("utils.grade", return_value=("answer", True)) as grade:
                ac.mc(args)
                first = ac.read_json(root / "mc/key.json")
                args.mc_samples = 4
                ac.mc(args)
                final = ac.read_json(root / "mc/key.json")
                ac.mc(args)
                self.assertEqual(len(calls), 4)
                self.assertEqual(final["draws"][:2], first["draws"])
                self.assertEqual(final["successes"], 4)
                self.assertTrue(all(prompt["prompt_token_ids"] == [7, 8, 1, 2] for prompt, _ in calls))
                self.assertTrue(all(param[0] == 4 for _, param in calls))
                self.assertEqual(len({param[1] for _, param in calls}), 4)
                grade.assert_called_with("prefix text suffix", "answer")


class ScoringTest(unittest.TestCase):
    def test_streamed_alignment_matches_full_tiny_qwen(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=19, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=8, max_position_embeddings=128, attention_dropout=0.0)).eval()
        for prompt, completion in (([1], [2]), ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12])):
            with torch.inference_mode():
                full = model(torch.tensor([prompt + completion]), use_cache=False).logits
            expected = full[:, len(prompt) - 1:-1]
            for size in (1, 3, 8):
                chunks = list(ac.causal_logit_chunks(model, prompt, completion, size))
                actual = torch.cat([value for _, value in chunks], dim=1)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_fp32_raw_and_exact_centered_signal(self):
        student = torch.tensor([[[1.5, -0.2, 0.6]]], dtype=torch.bfloat16)
        teacher = torch.tensor([[[-0.1, 0.8, 1.1]]], dtype=torch.bfloat16)
        p, q = student.float().log_softmax(-1), teacher.float().log_softmax(-1)
        expectation = 0.0
        for token in range(3):
            scores = ac.distillation_scores(student, teacher, [token])
            self.assertAlmostEqual(scores["raw"][0], float(q[0, 0, token] - p[0, 0, token]), places=6)
            self.assertAlmostEqual(scores["reverse_kl"][0], float((p.exp() * (p - q)).sum()), places=6)
            expectation += float(p.exp()[0, 0, token]) * scores["centered"][0]
        self.assertAlmostEqual(expectation, 0.0, places=6)
        self.assertEqual(ac.distillation_scores(student, student, [0])["centered"], [0.0])


class JoinTest(unittest.TestCase):
    def test_matched_tokens_and_explicit_step_credit(self):
        row = rollout()
        tokenizer = PieceTokenizer({1: "a", 2: "\n\n", 3: "b", 9: "EOS"})
        plan = ac.make_plan([row], tokenizer, ["uniform", "steps"], 9, 1, 3, 42)
        counts = {p["key"]: {"successes": 2, "samples": 4} for p in plan["prefixes"]}
        scores = {name: {"rollout_id": "r", "student_logps": [-1] * 4,
                        "teacher_logps": [-0.9] * 4, "raw": [0.1, 0.2, 0.3, 0.4],
                        "centered": [0.2, 0.3, 0.4, 0.5], "reverse_kl": [0.1] * 4}
                  for name in ("opd", "opsd_answer")}
        joined = ac.join_comparisons(row, plan["rollouts"][0], scores, counts)
        for item in joined:
            self.assertEqual(item["token_ids"], row["completion_ids"][item["start"]:item["end"]])
            self.assertEqual(item["distillation"]["opd"]["raw"], item["distillation"]["opsd_answer"]["raw"])
            if item["selection"] == "steps":
                self.assertEqual(item["vine"]["credit"], "segment_broadcast")
        self.assertEqual({s["unit"] for s in ac.summarize(joined, 0, 42)}, {"token", "segment_mean"})
        scores["opd"]["raw"].pop()
        with self.assertRaisesRegex(ValueError, "aligned"):
            ac.join_comparisons(row, plan["rollouts"][0], scores, counts)

    def test_cli_controls(self):
        args = ac.build_parser().parse_args(["--student", "s", "--opd-teacher", "t", "--output-dir", "out",
                                             "--num-token-samples", "7", "--K", "23"])
        self.assertEqual(args.pi_modes, ["answer", "full", "hint"])
        self.assertEqual(args.selection_modes, ["uniform", "steps"])
        self.assertEqual(args.num_token_samples, 7)
        self.assertEqual(args.mc_samples, 23)



class SourceTest(unittest.TestCase):
    def test_default_dataset_joins_hints_by_question_and_answer(self):
        args = SimpleNamespace(cohort=None, dataset="deepmath", pi_modes=["answer", "full", "hint"],
                               opsd_teacher="teacher")
        rows = [{"question": "q1", "final_answer": "1", "solution": "s1"},
                {"question": "q2", "final_answer": "2", "solution": "s2"}]
        hints = [{"question": "q2", "final_answer": "2", "hint": "h2"},
                 {"question": "q1", "final_answer": "wrong", "hint": "wrong hint"}]
        with mock.patch("utils.load_train_dataset", return_value=rows) as dataset, \
             mock.patch("utils.load_hint_cache", return_value=hints) as cache:
            source = ac.load_source(args)
        dataset.assert_called_once_with("deepmath", require_solution=True)
        cache.assert_called_once_with("teacher", "deepmath")
        self.assertIsNone(source[0]["hint"])
        self.assertEqual(source[1]["hint"], "h2")
        self.assertEqual(source[1]["solution"], "s2")

    def test_custom_cohort_keeps_its_own_hints(self):
        args = SimpleNamespace(cohort="custom.json")
        rows = [{"question": "q", "final_answer": "a", "hint": "custom"}]
        with mock.patch.object(ac, "load_rows", return_value=rows), \
             mock.patch("utils.load_hint_cache", side_effect=AssertionError("Unexpected cache load")):
            self.assertEqual(ac.load_source(args), rows)


class PipelineTest(unittest.TestCase):
    def test_artifact_pipeline_and_resume_with_local_tiny_models(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "one": 2, "two": 3}, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
            tokenizer.chat_template = "{{ messages | map(attribute='content') | join(' ') }}"
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=4, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                max_position_embeddings=256, eos_token_id=1))
            student_path, teacher_path = root / "student", root / "teacher"
            for path in (student_path, teacher_path):
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
            source = root / "source.json"
            ac.write_json(source, [
                {"question": "one", "final_answer": "one", "solution": "one"},
                {"question": "two", "final_answer": "two", "solution": "two"}])
            args = ac.build_parser().parse_args([
                "--student", str(student_path), "--opd-teacher", str(teacher_path),
                "--cohort", str(source), "--output-dir", str(root / "out"),
                "--num-problems", "2", "--n", "1", "--num-token-samples", "3",
                "--mc-samples", "2", "--min-segment-tokens", "1", "--max-segment-tokens", "2",
                "--max-completion-length", "8", "--max-model-len", "256",
                "--student-device", "cpu", "--teacher-device", "cpu", "--dtype", "float32",
                "--score-chunk-size", "8", "--bootstrap-samples", "3", "--pi-modes", "answer"])
            args.opsd_teacher = args.student
            ac.ensure_manifest(args)
            ac.prepare(args)
            llm = SimpleNamespace(get_tokenizer=lambda: tokenizer, generate=lambda prompts, *a, **kw: [
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[2, 3, 1], finish_reason="stop")])
                for _ in prompts])
            with mock.patch.object(ac, "load_llm", return_value=llm), \
                 mock.patch.object(ac, "sampling_params", return_value=None), \
                 mock.patch("utils.grade", return_value=("one", True)):
                ac.generate(args)
                ac.plan(args)
                ac.plan(args)  # JSON round-trip must not invalidate the segment plan.
                ac.mc(args)
                ac.score_condition(args, "opd", args.opd_teacher)
                ac.score_condition(args, "opsd_answer", args.opsd_teacher)
                ac.aggregate(args)
                summary = ac.read_json(root / "out/k-2/summary.json")
                self.assertEqual(summary["num_rollouts"], 2)
                self.assertEqual(summary["pass_at_1"], 1)
                self.assertTrue(summary["comparisons"])
                with mock.patch.object(ac, "load_hf", side_effect=AssertionError("cache missed")):
                    ac.score_condition(args, "opd", args.opd_teacher)
                args.mc_samples = 3
                ac.ensure_manifest(args)
                ac.mc(args)
                ac.aggregate(args)
                self.assertTrue((root / "out/k-3/comparisons.jsonl").exists())
                args.num_token_samples += 1
                with self.assertRaisesRegex(ValueError, "different experiment"):
                    ac.ensure_manifest(args)


if __name__ == "__main__":
    unittest.main()
