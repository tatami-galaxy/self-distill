"""Evidence, two-stage aggregation, provenance, and inference-order regressions."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval import matching_answer_revision as mr


TEXT = ("I divide by zero. Therefore the answer is 7. "
        "But dividing by zero is invalid. I replace that division with a limit.")


def candidate(**changes):
    return {"candidate_answer": "7", "conclusion_quote": "Therefore the answer is 7.",
            "criticism_quote": "But dividing by zero is invalid.",
            "reconsideration_quote": "I replace that division with a limit.",
            "sequence_status": "complete_candidate", "linkage_reason": "The division is reconsidered.",
            **changes}


def validation(**changes):
    return {"matching_conclusion": "yes", "identified_flaw": "yes", "actual_revision": "yes",
            "revision_linked_to_flaw": "yes", "flaw_assessment": "confirmed_error",
            "revision_type": "approach_change", "flaw_explanation": "Division by zero is undefined.",
            "conclusion_quote": "Therefore the answer is 7.", "reasoning_quote": "I divide by zero.",
            "criticism_quote": "But dividing by zero is invalid.",
            "revision_quote": "I replace that division with a limit.",
            "rationale": "The solver replaces the invalid operation after identifying it.", **changes}


def parse(stage, value, text=TEXT):
    return mr.parse_judgment(stage, json.dumps(value), text)


def row(identity="0:0"):
    return {"trajectory_id": identity, "question_id": identity.split(":")[0],
            "question": "Find the answer.", "supplied_answer": "7", "text": TEXT,
            "truncated": False, "backtracking_count": 2}


class EvidenceTests(unittest.TestCase):
    def test_discovery_accepts_incomplete_and_uncertain_without_invented_continuation(self):
        for state in ("incomplete", "uncertain"):
            result = parse("discovery", {"episodes": [candidate(sequence_status=state, reconsideration_quote="")]})
            self.assertEqual(result["episodes"][0]["sequence_status"], state)
        self.assertEqual(parse("discovery", {"episodes": []})["status"], "valid")

    def test_bad_discovery_evidence_and_truncation_are_invalid(self):
        for change in ({"conclusion_quote": "I got seven."}, {"criticism_quote": "I divide by zero."},
                       {"reconsideration_quote": ""}, {"candidate_answer": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse("discovery", {"episodes": [candidate(**change)]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse("discovery", {"episodes": [candidate(), candidate()]})
        with self.assertRaisesRegex(ValueError, "truncated"):
            mr.parse_judgment("discovery", '{"episodes": []}', TEXT, "length")

    def test_behavior_and_confirmed_error_are_separate(self):
        for flaw, status in (("confirmed_error", "yes"), ("no_demonstrated_error", "no"), ("uncertain", "uncertain")):
            result = parse("validation", validation(flaw_assessment=flaw))
            self.assertEqual(result["behavioral_status"], "yes")
            self.assertEqual(result["strict_status"], status)
            for field, span in result["evidence_spans"].items():
                self.assertEqual(TEXT[span["char_start"]:span["char_end"]], result[field])

    def test_revision_requires_evidence_and_consistent_labels(self):
        for change in ({"revision_quote": ""}, {"reasoning_quote": ""}, {"revision_type": "none"},
                       {"revision_quote": "I divide by zero."}, {"actual_revision": "no"},
                       {"identified_flaw": "uncertain"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse("validation", validation(**change))
        result = parse("validation", validation(actual_revision="no", revision_quote="",
                        revision_type="none", revision_linked_to_flaw="no"))
        self.assertEqual(result["behavioral_status"], "no")
        self.assertEqual(result["strict_status"], "no")

    def test_inputs_exclude_original_backtracking_labels(self):
        episode = candidate()
        payload = json.loads(mr.messages("validation", {**row(), "episode": episode})[1]["content"])
        self.assertNotIn("backtracking_count", payload)
        self.assertEqual(payload["response"], TEXT)
        self.assertEqual(payload["proposed_episode"], episode)


class AggregationTests(unittest.TestCase):
    def test_later_positive_wins_even_if_earlier_episode_failed(self):
        selected = [row()]
        discoveries = [{"trajectory_id": "0:0", "status": "valid", "episodes": [candidate(), candidate()]}]
        validations = [{"request_id": "0:0/episode-0", "status": "invalid"},
                       {"request_id": "0:0/episode-1", **parse("validation", validation())}]
        labels = mr.trajectory_labels(selected, discoveries, validations, "strict_status")
        self.assertEqual(labels[0]["status"], "yes")
        self.assertEqual(labels[0]["event_request_id"], "0:0/episode-1")

    def test_failures_remain_unresolved_and_empty_discovery_is_negative(self):
        selected = [row(f"{i}:0") for i in range(5)]
        discoveries = [{"trajectory_id": r["trajectory_id"], "status": "valid", "episodes": [candidate()]}
                       for r in selected]
        discoveries[0]["episodes"] = []
        discoveries[1].update(status="invalid")
        validations = [{"request_id": "2:0/episode-0", "status": "context_overflow"},
                       {"request_id": "3:0/episode-0", **parse("validation", validation(flaw_assessment="uncertain"))},
                       {"request_id": "4:0/episode-0", **parse("validation", validation(actual_revision="no",
                            revision_quote="", revision_type="none", revision_linked_to_flaw="no"))}]
        labels = mr.trajectory_labels(selected, discoveries, validations, "strict_status")
        self.assertEqual([l["status"] for l in labels], ["no", "invalid", "context_overflow", "uncertain", "no"])
        report = mr.source.summarize(selected, labels, 30)
        self.assertEqual(report["n_resolved"], 2)
        self.assertEqual(report["selected_fraction_bounds"], [0, .6])

    def test_incomplete_candidates_are_sent_for_independent_validation(self):
        discoveries = [{"trajectory_id": "0:0", **parse("discovery", {"episodes": [candidate(
            sequence_status="incomplete", reconsideration_quote="")]})}]
        requests = mr.episode_rows([row()], discoveries)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["episode_idx"], 0)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.args = mr.build_parser().parse_args([])
        self.args.bootstrap_samples = 30
        self.jobs = [(self.root / model, [row()], {"model": model}) for model in ("small", "large")]
        self.tokenizer = SimpleNamespace(chat_template="template", get_vocab=lambda: {"x": 0},
            apply_chat_template=lambda *a, **kw: {"input_ids": [0] * 100})
        self.calls = []
        self.discovery_output = {"episodes": [candidate()]}
        self.validation_output = validation()

        def chat(conversations, sampling, **kwargs):
            self.calls.append(sampling)
            value = self.discovery_output if sampling == "discovery" else self.validation_output
            return [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(value), finish_reason="stop")])
                    for _ in conversations]

        self.engine = SimpleNamespace(chat=chat)
        patches = [patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(_commit_hash="rev")),
                   patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer),
                   patch.object(mr, "create_judge", return_value=(self.engine, {s: s for s in ("discovery", "validation")}))]
        self.loader = None
        for p in patches:
            self.loader = p.start()
            self.addCleanup(p.stop)

    def test_model_order_resume_and_changed_settings(self):
        mr.classify(self.args, self.jobs)
        self.assertEqual(self.calls, ["discovery", "validation", "discovery", "validation"])
        self.assertEqual(self.loader.call_count, 1)
        for root, _, _ in self.jobs:
            report = mr.read_json(root / "summary.json")
            self.assertEqual(report["strict"]["fraction_of_selected"], 1)
            self.assertEqual(report["n_candidate_episodes"], 1)
        self.calls.clear()
        self.loader.side_effect = AssertionError("Resume must not load the GPU judge")
        mr.classify(self.args, self.jobs)
        self.assertEqual(self.calls, [])
        self.args.max_output_tokens += 1
        with self.assertRaisesRegex(ValueError, "settings"):
            mr.classify(self.args, self.jobs)

    def test_reclassify_invalidates_dependent_validation(self):
        mr.classify(self.args, self.jobs[:1])
        self.discovery_output["episodes"][0]["linkage_reason"] = "Changed interpretation."
        self.args.reclassify = True
        mr.classify(self.args, self.jobs[:1])
        self.assertEqual(self.calls, ["discovery", "validation"] * 2)
        root = self.jobs[0][0]
        meta = mr.read_json(root / "validation_meta.json")
        self.assertEqual(meta["config"]["discovery_hash"], mr.digest(mr.read_rows(root / "discovery.jsonl")))

    def test_no_candidates_skip_validation_inference(self):
        self.discovery_output = {"episodes": []}
        mr.classify(self.args, self.jobs[:1])
        self.assertEqual(self.calls, ["discovery"])
        report = mr.read_json(self.jobs[0][0] / "summary.json")
        self.assertEqual(report["strict"]["n_no"], 1)
        self.assertEqual(report["n_candidate_episodes"], 0)

    def test_overflow_never_truncates_or_loads_judge(self):
        self.tokenizer.apply_chat_template = lambda *a, **kw: {"input_ids": [0] * 40000}
        mr.classify(self.args, self.jobs[:1])
        self.loader.assert_not_called()
        report = mr.read_json(self.jobs[0][0] / "summary.json")
        self.assertEqual(report["strict"]["n_context_overflow"], 1)
        self.assertEqual(report["strict"]["selected_fraction_bounds"], [0, 1])

    def test_invalid_discovery_never_becomes_negative(self):
        self.discovery_output = {"episodes": [candidate(criticism_quote="fabricated")]}
        mr.classify(self.args, self.jobs[:1])
        report = mr.read_json(self.jobs[0][0] / "summary.json")
        self.assertEqual(self.calls, ["discovery"])
        self.assertEqual(report["strict"]["n_invalid"], 1)

    def test_summary_rejects_altered_derived_judgment_even_with_updated_hash(self):
        mr.classify(self.args, self.jobs[:1])
        root, selected, identity = self.jobs[0]
        results = mr.read_rows(root / "validation.jsonl")
        results[0]["strict_status"] = "no"
        mr.write_rows(root / "validation.jsonl", results)
        meta = mr.read_json(root / "validation_meta.json")
        meta["results_hash"] = mr.digest(results)
        mr.write_json(root / "validation_meta.json", meta)
        with self.assertRaisesRegex(ValueError, "raw response"):
            mr.save_summary(self.args, root, selected, identity)


if __name__ == "__main__":
    unittest.main()
