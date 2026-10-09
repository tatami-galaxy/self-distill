"""CPU integration: cached scores -> matched-arm statistics -> figures."""

import copy
import json

import pytest
from datasets import Dataset

from eval import hint_gen_compare as compare
from eval.viz import hint_gen_ablations as viz


@pytest.fixture
def ablation_summary(tmp_path, monkeypatch):
    args = compare.build_parser().parse_args([
        "--model", "base", "--output-dir", str(tmp_path / "eval"),
        "--bootstrap-samples", "20", "--k", "1",
    ])
    variants = [("fresh_base", "base")]
    for arm, cost, gamma in [
        ("full", 1, 6), ("no_transfer", 1, 0), ("no_length", 0, 6),
        ("sufficiency_only", 0, 0),
    ]:
        run = tmp_path / arm
        run.mkdir()
        (run / "run_meta.json").write_text(json.dumps({
            "method": "constrained_hint_gen_grpo", "model": "base", "dataset": "deepmath",
            "tau": .7, "hint_budget": 128, "seed": 42, "dual_init": 1,
            "length_cost_coef": cost, "gamma": gamma,
        }))
        for step in (20, 40):
            checkpoint = run / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "constrained_reward_state.json").write_text(json.dumps({"dual_lambda": 2}))
            variants.append((f"{arm}-step{step}", str(checkpoint)))
            if arm != "full":
                args.compare.append(f"{arm}-step{step}=full-step{step}")
    for step in (20, 40):
        for arm in ("full", "no_transfer", "no_length"):
            args.compare.append(f"{arm}-step{step}=sufficiency_only-step{step}")
    cohort = Dataset.from_list([{"question_id": q, "difficulty": "hard"} for q in ("q1", "q2")])
    hints, sufficiency, transfer = [], [], []
    for label, _ in variants:
        for qid in ("q1", "q2"):
            hint_id = f"{label}-{qid}"
            invalid = label.startswith("no_length") and qid == "q1"
            hints.append({
                "generator_id": label, "question_id": qid, "hint_id": hint_id,
                "hint_sample_idx": 0, "hint": "Use parity.", "hint_token_ids": [1, 2],
                "num_hint_tokens": 2, "truncated": False,
                "invalid_reason": "answer_leak" if invalid else "", "validity_observable": True,
            })
            sufficiency.append({"hint_id": hint_id, "generator_id": label,
                                "question_id": qid, "n_samples": 4, "n_correct": 4,
                                "n_truncated": 0})
            transfer.append({"hint_id": hint_id, "raw_transfer": .1})
    sufficiency.extend({"generator_id": "no_hint", "question_id": q,
                        "n_samples": 4, "n_correct": 2} for q in ("q1", "q2"))
    score_meta = {"config": {"cohort_fingerprint": "cohort", "hint_fingerprint": compare.hint_fingerprint(hints)}}
    monkeypatch.setattr(compare, "generator_variants", lambda _: variants)
    monkeypatch.setattr(compare, "load_cohort", lambda _: (cohort, {"cohort_fingerprint": "cohort"}))
    monkeypatch.setattr(compare, "load_all_hints", lambda _: (hints, {}))
    monkeypatch.setattr(compare, "_load_score_dataset", lambda _, kind: (
        sufficiency if kind == "sufficiency" else transfer, score_meta,
    ))
    return compare.summarize_phase(args)


def test_joint_summary_keeps_controls_and_compares_ablations(ablation_summary):
    summary = ablation_summary
    assert summary["no_hint_self_teacher"]["pass@1"] == .5
    arm = summary["generators"]["no_length-step20"]
    assert arm["all_outputs"]["sufficiency"]["pass@1"]["value"] == 1
    assert arm["metric_estimates"]["training_aligned_sufficiency"]["mean"] == .5
    assert arm["training_aligned_lift_over_no_hint"]["mean_delta_candidate_minus_reference"] == 0
    contrast = summary["paired_comparisons"]["no_length-step20_minus_full-step20"]["metrics"]
    assert contrast["training_aligned_sufficiency"]["mean_delta_candidate_minus_reference"] == -.5
    assert contrast["training_aligned_sufficiency"]["n_questions"] == 2
    assert "no_length-step20_minus_fresh_base" in summary["paired_comparisons"]


def test_matched_plot_writes_figures_and_rejects_unmatched_inputs(ablation_summary, tmp_path):
    specs = viz.default_arm_specs(ablation_summary)
    assert specs[-1] == "sufficiency_only=sufficiency_only"
    arms = viz.collect_series(ablation_summary, specs)
    out = tmp_path / "plots"
    viz.plot(ablation_summary, arms, "full", out)
    for name in ("attributes.png", "attributes.pdf", "attributes.csv", "differences.png", "differences.pdf"):
        assert (out / name).stat().st_size > 100
    with pytest.raises(ValueError, match="same evaluated steps"):
        changed = copy.deepcopy(ablation_summary)
        changed["generator_order"].remove("no_length-step40")
        viz.collect_series(changed, specs)
    with pytest.raises(ValueError, match="seed"):
        changed = copy.deepcopy(ablation_summary)
        changed["generator_runs"]["no_length-step40"]["run_meta"]["seed"] = 99
        viz.collect_series(changed, specs)
    with pytest.raises(ValueError, match="Missing.*--compare"):
        changed = copy.deepcopy(ablation_summary)
        del changed["paired_comparisons"]["no_length-step20_minus_full-step20"]
        viz.contrast_series(changed, arms, "full")


def test_sufficiency_reference_and_legacy_three_arm_detection(ablation_summary):
    arms = viz.collect_series(ablation_summary, viz.default_arm_specs(ablation_summary))
    contrasts = viz.contrast_series(ablation_summary, arms, "sufficiency_only")
    assert set(contrasts) == {"full", "no_transfer", "no_length"}
    assert contrasts["no_length"][0][1]["training_aligned_sufficiency"][
        "mean_delta_candidate_minus_reference"
    ] == -.5
    old = copy.deepcopy(ablation_summary)
    old["generator_order"] = [label for label in old["generator_order"]
                              if not label.startswith("sufficiency_only-")]
    assert len(viz.default_arm_specs(old)) == 3
    assert len(viz.collect_series(old, viz.default_arm_specs(old))) == 3
