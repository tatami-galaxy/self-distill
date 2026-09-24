"""AM ingestion, standalone workflow, PI generation, and leak regressions."""

import json
import types
from unittest import mock

import numpy as np
import pytest

from eval import am_demo_gain as am
from eval import demo_gain as dg
from eval.hint_compare_cache import digest, model_identity
from utils import gen_am_demo_pi as gp
from utils.am_hint_validation import validation_flags


def raw_record(question="Find the value.", answer="42"):
    thinking = "Compute carefully."
    final = f"The answer is \\boxed{{{answer}}}."
    return {
        "system": "Use <think> and <answer> tags.",
        "conversations": [
            {
                "from": "human",
                "value": question,
                "info": {"category": "math", "ground_truth": answer, "source": "test"},
            },
            {
                "from": "assistant",
                "value": f"<think>{thinking}</think><answer>{final}</answer>",
                "info": {
                    "think_content": thinking,
                    "answer_content": final,
                    "verify_score": 1.0,
                },
            },
        ],
    }


def fixture(root):
    rows = [am.normalize_record(raw_record(f"Question {i}"), i) for i in range(2)]
    for row in rows:
        row.update(
            prompt_ids={"none": [1], "answer": [2], "full": [3]},
            target_ids=[4, 5, 6],
            target_tokens=3,
            thinking_end=2,
        )
    manifest = {
        "version": dg.VERSION,
        "dataset": am.DATASET,
        "model": "test",
        "revision": "rev",
        "model_identity": model_identity("test"),
        "tokenizer_hash": "tok",
        "source": {"repo": am.REPO},
        "cohort_hash": digest(rows),
        "max_model_len": 10000,
        "conditions": list(dg.CONDITIONS),
    }
    dg.write_rows(root / "cohort.jsonl", rows)
    dg.write_json(root / "manifest.json", manifest)
    return manifest, rows


def artifact(root, manifest, rows, kind="hints", invalid=None):
    conditions = am.HINTS if kind == "hints" else ["rollout"]
    samples = [
        {
            "question_id": q["question_id"],
            "demo_hash": q["demo_hash"],
            "condition": c,
            "sample_idx": 0,
            "text": "Use algebra",
            "n_tokens": 2,
            "invalid_reason": "answer_leak" if (i, c) == invalid else "",
            "truncated": False,
        }
        for i, q in enumerate(rows)
        for c in conditions
    ]
    dg.write_rows(root / kind / "samples.jsonl", samples)
    dg.write_json(
        root / kind / "manifest.json",
        {
            "dataset": am.DATASET,
            "cohort_hash": manifest["cohort_hash"],
            "samples_hash": digest(samples),
            "conditions": list(conditions),
            "config": {"model": "test", "samples_per_level": 1},
            "diagnostics": {},
        },
    )


def test_normalization_preserves_prompt_trace_and_multiple_choice():
    q = r"Choose. $\textbf{(A)}\ 12 \qquad \textbf{(B)}\ 14 \qquad \textbf{(C)}\ 26$"
    raw = raw_record(q, "B")
    row = am.normalize_record(raw, 4)
    assert row["answer_aliases"] == ["14"]
    assert row["answer_pi"] == "B (14)"
    assert row["base_messages"][0]["content"] == raw["system"]
    assert row["solution"] == raw["conversations"][1]["value"]
    assert "B (14)" in dg.messages_for(row, "answer")[-1]["content"]
    raw["conversations"][1]["info"]["verify_score"] = 0
    with pytest.raises(ValueError, match="not_verified"):
        am.normalize_record(raw, 4)


def test_malformed_segments_are_not_silently_reconstructed():
    raw = raw_record()
    raw["conversations"][1]["info"]["think_content"] = "Different reasoning"
    with pytest.raises(ValueError, match="inconsistent_trace_segments"):
        am.normalize_record(raw, 0)
    with pytest.raises(ValueError, match="unresolved_multiple_choice"):
        am.normalize_record(raw_record("Choose without options", "C"), 0)


def test_streaming_sampling_deduplicates_and_rejects_conflicting_gold(tmp_path):
    path = tmp_path / "math.jsonl"
    records = [
        raw_record("same", "42"),
        raw_record("same", "42"),
        raw_record("conflict", "12"),
        raw_record("conflict", "13"),
        raw_record("third", "8"),
    ]
    dg.write_rows(path, records)
    candidates, exclusions, checksum, count = am.scan_candidates(path, 42)
    assert count == 5 and len(candidates) == 2
    assert exclusions["duplicate_question"] == 2
    assert exclusions["conflicting_gold_questions"] == 1
    assert len(checksum) == 64
    assert am.scan_candidates(path, 42)[0] == candidates
    with path.open("rb") as f:
        for _, offset, number in candidates:
            f.seek(offset)
            assert json.loads(f.readline()) == records[number]


@pytest.mark.parametrize(
    "gold,aliases,text",
    [
        (r"\dfrac{1}{4}", [], r"The derivative is $\frac{1}{4}$."),
        (r"1 + \sqrt{2}", [], r"The maximum is $\sqrt{2}+1$."),
        ("0", [], "The integral is zero due to symmetry."),
        ("C", ["The origin is a saddle point"], "The origin is a saddle point."),
        ("B", ["14"], "She has 14 dollars remaining."),
        ("C", ["Something"], "The correct option is C."),
    ],
)
def test_equivalent_and_multiple_choice_leaks(gold, aliases, text):
    assert validation_flags(text, {"final_answer": gold, "answer_aliases": aliases})


def test_safe_hint_and_incidental_single_digit():
    assert not validation_flags("Apply the chain rule at x = 0.", {"final_answer": "0"})
    assert not validation_flags("Use symmetry.", {"final_answer": "42"})


def test_regions_conserve_total_even_short_traces():
    g = np.array([1.0, -2.0, 3.0])
    r = am.region_metrics(g, 2)
    assert r["thinking_total_gain"] + r["final_total_gain"] == g.sum()
    assert r["first_5pct_total_gain"] + r["remaining_95pct_total_gain"] == g.sum()


def test_standalone_cli_conditions_cache_and_root_report(tmp_path):
    manifest, rows = fixture(tmp_path)
    artifact(tmp_path, manifest, rows, invalid=(1, "hint_short"))
    artifact(tmp_path, manifest, rows, "rollouts")
    argv = [
        "am_demo_gain",
        "--output-dir",
        str(tmp_path),
        "--bootstrap-samples",
        "20",
        "--early-tokens",
        "4",
    ]

    def run(phase, conditions=None):
        command = argv + ["--phase", phase]
        if conditions:
            command += ["--conditions", *conditions]
        with mock.patch("sys.argv", command):
            am.main()

    with (
        mock.patch("transformers.AutoTokenizer.from_pretrained"),
        mock.patch.object(dg, "tokenizer_hash", return_value="tok"),
        mock.patch.object(dg, "render_target", return_value=([7], [4, 5, 6])),
        mock.patch("transformers.AutoModelForCausalLM.from_pretrained") as loader,
        mock.patch.object(
            dg,
            "score_tokens",
            side_effect=lambda model, prompt, *a: [-4.0 + prompt[0] / 4] * 3,
        ),
    ):
        # Every pass replaces the single root report. Omitting --conditions
        # restores the manifest's selection, including the invalid short hint.
        for conditions, n in [
            (["answer", "full"], 2),
            (["answer", "full", "rollout"], 2),
            (None, 1),
        ]:
            run("score", conditions)
            run("aggregate")
            summary = dg.read_json(tmp_path / "summary.json")
            assert summary["n_questions"] == n
            assert "thinking_normalized_gain" in summary["conditions"]["full"]
            assert set(summary["conditions"]) == set(conditions or dg.CONDITIONS)
        loader.reset_mock()
        run("score")
        loader.assert_not_called()
        run("aggregate")
    assert not (tmp_path / "analyses").exists()
    from eval.viz.demo_gain import plot

    plot(tmp_path)
    assert (tmp_path / "figures/gain.png").exists()
    assert (tmp_path / "figures/region_gain.png").exists()
    # Replacing a generated artifact invalidates its dependent report.
    artifact(tmp_path, manifest, rows)
    with pytest.raises(ValueError, match="PI artifacts changed"):
        run("aggregate")


def test_missing_samples_and_other_cohorts_fail(tmp_path):
    manifest, rows = fixture(tmp_path)
    artifact(tmp_path, manifest, rows)
    bad = [dict(rows[0], demo_hash="different"), rows[1]]
    with pytest.raises(ValueError, match="another dataset or cohort"):
        am.load_artifact(tmp_path / "hints", bad)
    samples = dg.read_rows(tmp_path / "hints/samples.jsonl")[:-1]
    dg.write_rows(tmp_path / "hints/samples.jsonl", samples)
    meta = dg.read_json(tmp_path / "hints/manifest.json")
    meta["samples_hash"] = digest(samples)
    dg.write_json(tmp_path / "hints/manifest.json", meta)
    args = am.build_parser().parse_args(
        ["--output-dir", str(tmp_path), "--conditions", "hint_short"]
    )
    with (
        mock.patch("transformers.AutoTokenizer.from_pretrained"),
        mock.patch.object(dg, "tokenizer_hash", return_value="tok"),
        mock.patch.object(dg, "render_target", return_value=([7], [4, 5, 6])),
        pytest.raises(ValueError, match="Missing hint_short"),
    ):
        am.preflight_am(args, manifest, rows, ["hint_short"])


def test_generation_cache_and_length_ceiling_are_independent(tmp_path):
    fixture(tmp_path)
    args = gp.build_parser().parse_args(
        ["--cohort-dir", str(tmp_path), "--kind", "hints", "--levels", "hint_short"]
    )
    tok = types.SimpleNamespace(
        apply_chat_template=lambda *a, **k: "prompt", encode=lambda *a, **k: [1, 2]
    )
    engine = mock.Mock()
    engine.chat.side_effect = lambda messages, params, **kw: [
        types.SimpleNamespace(
            outputs=[
                types.SimpleNamespace(
                    text="Use symmetry.", token_ids=[1] * 40, finish_reason="stop"
                )
            ]
        )
        for p in params
    ]
    fake = types.SimpleNamespace(
        LLM=mock.Mock(return_value=engine),
        SamplingParams=lambda **kw: types.SimpleNamespace(**kw),
    )
    with (
        mock.patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=types.SimpleNamespace(
                _commit_hash="rev", max_position_embeddings=10000
            ),
        ),
        mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tok),
        mock.patch.object(dg, "tokenizer_hash", return_value="tok"),
        mock.patch.dict("sys.modules", {"vllm": fake}),
    ):
        gp.generate(args)
        first = dg.read_rows(tmp_path / "hints/samples.jsonl")
        assert all(
            r["length_noncompliant"] and not r["truncated"] and not r["invalid_reason"]
            for r in first
        )
        params = engine.chat.call_args.args[1]
        assert all(p.max_tokens == 1024 for p in params)
        gp.generate(args)
        assert fake.LLM.call_count == 1
        args.samples_per_level = 2
        gp.generate(args)
        expanded = dg.read_rows(tmp_path / "hints/samples.jsonl")
        assert [r for r in expanded if r["sample_idx"] == 0] == first
        assert len(expanded) == 4
        args.kind = "rollouts"
        args.samples_per_level = 1
        gp.generate(args)
        messages = engine.chat.call_args.args[0]
        assert all(
            len(m) == 2 and "Worked solution" not in m[-1]["content"] for m in messages
        )
        assert engine.chat.call_args.kwargs["chat_template_kwargs"]["enable_thinking"]


def test_real_am_target_has_identical_tokens_and_answer_wrapper():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    row = am.normalize_record(raw_record(), 0)
    row["rollout"] = "An independent attempt."
    targets = [
        dg.render_target(tok, dg.messages_for(row, c, "Use algebra"), row["solution"])[
            1
        ]
        for c in ("none", *dg.CONDITIONS)
    ]
    assert all(t == targets[0] for t in targets)
    text = tok.decode(targets[0])
    assert text.count("<think>") == 1 and "<answer>" in text and "</answer>" in text
    assert "<|im_end|>" in text


def test_prepare_cli_reads_local_am_records(tmp_path):
    from tests.test_demo_gain import CharacterTokenizer

    source = tmp_path / "math.jsonl"
    dg.write_rows(source, [raw_record("First"), raw_record("Second")])
    output = tmp_path / "run"
    tok = CharacterTokenizer()
    tok.encode = lambda text, **kwargs: list(map(ord, text))
    tok.get_vocab = lambda: {"a": 97}
    with (
        mock.patch(
            "sys.argv",
            [
                "am_demo_gain",
                "--phase",
                "prepare",
                "--output-dir",
                str(output),
                "--model",
                "test",
                "--num-problems",
                "1",
                "--data-file",
                str(source),
                "--conditions",
                "answer",
                "full",
                "hint_short",
            ],
        ),
        mock.patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=types.SimpleNamespace(
                max_position_embeddings=10000, _commit_hash="rev"
            ),
        ),
        mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tok),
        mock.patch("huggingface_hub.hf_hub_download") as download,
    ):
        am.main()
    download.assert_not_called()
    manifest, rows = dg.load_cohort(output)
    assert manifest["dataset"] == am.DATASET and len(rows) == 1
    assert manifest["source"]["revision"] is None
    assert manifest["conditions"] == ["answer", "full", "hint_short"]
    assert 0 < rows[0]["thinking_end"] < rows[0]["target_tokens"]


def test_am_cli_rejects_other_dataset_before_scoring(tmp_path):
    dg.write_json(tmp_path / "manifest.json", {"dataset": "deepmath"})
    with (
        mock.patch(
            "sys.argv",
            ["am_demo_gain", "--phase", "score", "--output-dir", str(tmp_path)],
        ),
        mock.patch.object(dg, "score") as score,
        pytest.raises(ValueError, match="requires an AM-Qwen3 cohort"),
    ):
        am.main()
    score.assert_not_called()


def test_default_generation_runs_requested_pi_in_separate_processes(tmp_path):
    manifest, _ = fixture(tmp_path)
    args = gp.build_parser().parse_args(
        [
            "--cohort-dir",
            str(tmp_path),
            "--model",
            "hint-adapter",
            "--revision",
            "hint-revision",
            "--samples-per-level",
            "3",
            "--max-new-tokens",
            "768",
            "--rollout-max-new-tokens",
            "4096",
        ]
    )
    context = mock.Mock()
    context.Process.return_value.exitcode = 0
    with mock.patch("multiprocessing.get_context", return_value=context) as factory:
        gp.run_generation(args)
    factory.assert_called_once_with("spawn")
    children = [call.kwargs["args"][0] for call in context.Process.call_args_list]
    hints, rollout = children
    assert [c.kind for c in children] == ["hints", "rollouts"]
    assert hints.levels == list(am.HINTS) and hints.samples_per_level == 3
    assert hints.model == "hint-adapter" and hints.max_new_tokens == 768
    assert rollout.model is None and rollout.revision is None
    assert rollout.samples_per_level == 1 and rollout.max_new_tokens == 4096
    assert hints.output_dir == str(tmp_path / "hints")
    assert rollout.output_dir == str(tmp_path / "rollouts")
    assert [c[0] for c in context.mock_calls] == [
        "Process",
        "Process().start",
        "Process().join",
        "Process",
        "Process().start",
        "Process().join",
    ]
    # A hints-only cohort requires no rollout generation.
    manifest["conditions"] = ["answer", "full", "hint_short"]
    dg.write_json(tmp_path / "manifest.json", manifest)
    context.reset_mock()
    with mock.patch("multiprocessing.get_context", return_value=context):
        gp.run_generation(args)
    context.Process.assert_called_once()
    assert context.Process.call_args.kwargs["args"][0].levels == ["hint_short"]
    # Stop the pipeline if generation fails, instead of starting the next kind.
    manifest["conditions"] = list(dg.CONDITIONS)
    dg.write_json(tmp_path / "manifest.json", manifest)
    context.reset_mock()
    context.Process.return_value.exitcode = 1
    with (
        mock.patch("multiprocessing.get_context", return_value=context),
        pytest.raises(RuntimeError, match="hints generation failed"),
    ):
        gp.run_generation(args)
    context.Process.assert_called_once()
