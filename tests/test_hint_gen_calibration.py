"""Reward calibration and CPU command construction for the matched rerun."""

from pathlib import Path
import shlex
import subprocess

import pytest

from eval.calibrate_hint_gen_penalties import calibrate, load_gamma


ROOT = Path(__file__).resolve().parents[1]


def example():
    hints = [dict(generator_id="fresh_base", hint_id=f"h{i}", question_id="q",
                  num_hint_tokens=n, validity_observable=True, invalid_reason="")
             for i, n in enumerate((10, 30))]
    # Rollout means -0.1 and 0.1 -> clamp to 0 and 0.1.
    scores = [dict(generator_id="fresh_base", hint_id=f"h{i}", rollout_position=j,
                   raw_transfer=value)
              for i, values in enumerate(((-.4, .2), (0, .2)))
              for j, value in enumerate(values)]
    return hints, scores


def test_calibration_clamps_after_rollout_mean_and_ignores_other_arms():
    hints, scores = example()
    scores.append(dict(generator_id="trained", hint_id="h0"))
    result = calibrate(hints, scores, 100, 2)
    assert result["gamma"] == pytest.approx(2)
    assert result["mean_within_question_std_cost"] == pytest.approx(.1)
    assert result["mean_within_question_std_transfer"] == pytest.approx(.05)
    assert result["eligible_questions"] == 1
    assert result["gamma"] * result["mean_within_question_std_transfer"] == pytest.approx(
        result["mean_within_question_std_cost"])


def test_calibration_excludes_invalid_hints_and_insufficient_groups():
    hints, scores = example()
    hints.extend([
        dict(hints[0], hint_id="bad", num_hint_tokens=100, invalid_reason="answer_leak"),
        dict(hints[0], hint_id="single", question_id="q2"),
    ])
    scores.extend(dict(row, hint_id="single") for row in scores[:2])
    result = calibrate(hints, scores, 100, 2)
    assert result["gamma"] == pytest.approx(2)
    assert result["invalid_hints"] == 1
    assert result["excluded_questions"] == 1


def test_calibration_weights_questions_equally_despite_different_group_sizes():
    hints, scores = example()
    # q has std(C)=.1, std(T)=.05; q2 has .2 and .05 despite twice as many hints.
    for i, tokens in enumerate((20, 60, 20, 60)):
        hid = f"extra{i}"
        hints.append(dict(hints[0], hint_id=hid, question_id="q2", num_hint_tokens=tokens))
        scores.extend(dict(generator_id="fresh_base", hint_id=hid, rollout_position=j,
                           raw_transfer=.1 * (i % 2)) for j in range(2))
    assert calibrate(hints, scores, 100, 2)["gamma"] == pytest.approx(3)


def test_calibration_rejects_incomplete_or_degenerate_data():
    hints, scores = example()
    with pytest.raises(ValueError, match="rollouts"):
        calibrate(hints, scores[:-1], 100, 2)
    with pytest.raises(ValueError, match="Duplicate"):
        calibrate(hints, scores + scores[:1], 100, 2)
    for row in scores:
        row["raw_transfer"] = -1
    with pytest.raises(ValueError, match="positive"):
        calibrate(hints, scores, 100, 2)
    hints[0]["generator_id"] = "trained"
    with pytest.raises(ValueError, match="fresh_base"):
        calibrate(hints, scores, 100, 2)


def run_script(script, *args):
    return subprocess.run(["bash", str(ROOT / script), *args], cwd=ROOT,
                          capture_output=True, text=True)


@pytest.mark.parametrize("model", ["Qwen3-1.7B", "Qwen3-4B"])
def test_training_and_evaluation_use_same_isolated_run_paths(model, tmp_path):
    path = ROOT / f"train/opsd/train_hint_gen/calibrations/{model}.json"
    gamma = load_gamma(path, f"Qwen/{model}")
    root = str(tmp_path / "runs with spaces")
    train = run_script("train/opsd/train_hint_gen/run_signal_matched.sh", model,
                       "--run-root", root, "--dry-run")
    assert train.returncode == 0, train.stderr
    commands = [shlex.split(line) for line in train.stdout.splitlines()]
    assert len(commands) == 4
    for command, arm, cost, transfer in zip(commands,
            ["full", "no_transfer", "no_length", "sufficiency_only"],
            [1, 1, 0, 0], [gamma, 0, gamma, 0], strict=True):
        assert command[command.index("--output-dir") + 1] == f"{root}/{arm}"
        assert float(command[command.index("--length-cost-coef") + 1]) == cost
        assert float(command[command.index("--gamma") + 1]) == transfer
        assert "--resume-from-checkpoint" not in command
    evaluate = run_script("eval/run_hint_gen_ablations.sh", model,
                          "--signal-matched", "--run-root", root, "--steps", "20", "--dry-run")
    assert evaluate.returncode == 0, evaluate.stderr
    command = shlex.split(evaluate.stdout.splitlines()[1])
    for arm in ["full", "no_transfer", "no_length", "sufficiency_only"]:
        assert f"{arm}-step20={root}/{arm}/checkpoint-20" in command
    assert "no_length-step20=no_transfer-step20" in command
    assert command[command.index("--output-dir") + 1].endswith(f"{model}/signal_matched_seed42")
    assert not Path(root).exists()


def test_frozen_calibration_rejects_wrong_model_and_existing_runs(tmp_path):
    path = ROOT / "train/opsd/train_hint_gen/calibrations/Qwen3-4B.json"
    with pytest.raises(ValueError, match="model"):
        load_gamma(path, "Qwen/Qwen3-1.7B")
    occupied = tmp_path / "full"
    occupied.mkdir()
    (occupied / "run_meta.json").write_text("{}")
    result = run_script("train/opsd/train_hint_gen/run_signal_matched.sh", "Qwen3-4B",
                        "--run-root", str(tmp_path))
    assert result.returncode != 0
    assert "Run already exists" in result.stderr


def test_original_evaluation_paths_unchanged():
    result = run_script("eval/run_hint_gen_ablations.sh", "--steps", "20", "--dry-run")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[1])
    assert any("deepmath_t0.7_g6_c0_lora_r16/checkpoint-20" in arg for arg in command)
    assert not any("signal_matched" in arg for arg in command)
    assert not any("sufficiency_only-step20=" in arg for arg in command)
