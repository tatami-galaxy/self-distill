"""CodeIO schema, reward semantics, split isolation, and pipeline integration."""

import json
from unittest.mock import patch

import pytest
from datasets import Dataset

from utils import (
    DATASET_REGISTRY_EVAL,
    DATASET_REGISTRY_TRAIN,
    accuracy_reward_for_dataset,
    answer_context,
    format_prompt,
    format_prompt_math,
    grade,
    load_train_dataset,
)
from utils.codeio import (
    DATASET_ID,
    DATASET_REVISION,
    OUTPUT_INSTRUCTION,
    REFERENCE_MARKER,
    extract_output,
    function_id,
    load_codeio,
    normalize_row,
    outputs_equal,
)


def raw_row(code="def main_solution(x):\n    return x + 1", value=2, input_value=1):
    return {
        "prompt": f"Given the following input:\n{{'x': {input_value}}}\n\n{OUTPUT_INSTRUCTION}\n\n{REFERENCE_MARKER} Use it for reasoning.\n\n{code}",
        "turn_1": "Add one to the input.\n```json\n"
        + json.dumps({"output": value})
        + "\n```",
        "feedback_1": "Correct output!",
        "turn_2": None,
        "feedback_2": None,
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        True,
        0,
        "",
        "a } brace",
        [],
        {},
        [1, [False, None]],
        {"b": [2], "a": 1},
    ],
)
def test_json_values_survive_loader_and_grading(value):
    row = normalize_row(raw_row(value=value))
    assert json.loads(row["final_answer"]) == value
    assert grade(row["solution"], row["final_answer"], "codeio")[1]


@pytest.mark.parametrize(
    "response",
    [
        '{"output": 2} then {"output": 3}',
        '{"output": 2} then {"output":',
        '{"output": 2, "output": 3}',
        '{"bad": {"output": 2}',
        '{"bad": 0, "bad": {"output": 2}}',
        '{"output": NaN}',
        '{"output": Infinity}',
        '{"output": 1e999}',
        '{"output": 2, "extra": 0}',
        '{"input": 2}',
        '__import__("os").system("false")',
        '<think>{"output": 2}</think> No answer.',
    ],
)
def test_rejects_wrong_or_malformed_final_answers(response):
    assert not grade(response, "2", "codeio")[1]


def test_json_types_order_and_numeric_tolerance():
    assert outputs_equal({"b": 2, "a": [1]}, {"a": [1], "b": 2})
    assert not outputs_equal([1, 2], [2, 1])
    assert not outputs_equal(True, 1)
    assert not outputs_equal("2", 2)
    assert outputs_equal(2.501, 2.5)
    assert not outputs_equal(2.51, 2.5)
    assert not outputs_equal(1.9999, 2.0)  # upstream also requires equal integer parts
    assert not outputs_equal(float("nan"), float("nan"))
    assert not outputs_equal(float("inf"), float("inf"))
    assert extract_output('Reasoning {"example": 0}\n{"output": {"a": "}"}}') == {
        "output": {"a": "}"}
    }


def test_only_verified_first_turn_output_predictions_supply_targets():
    for feedback in [
        None,
        "",
        "Feasible input!",
        "[Mismatch] Your output is not correct!",
    ]:
        row = raw_row()
        row.update(
            feedback_1=feedback, feedback_2="Correct output!", turn_2='{"output": 2}'
        )
        assert normalize_row(row) is None
    row = raw_row()
    row["prompt"] = row["prompt"].replace(
        OUTPUT_INSTRUCTION, "Can you predict a feasible input without writing any code?"
    )
    assert normalize_row(row) is None
    row = raw_row()
    row["turn_1"] = "No parseable answer"
    assert normalize_row(row) is None


def test_function_split_keeps_different_inputs_together_and_filters_before_limit():
    candidates = [
        raw_row(code=f"def main_solution(x):\n    return x + {i}", value=i)
        for i in range(120)
    ]
    source = []
    for row in candidates:
        source.extend([dict(row, feedback_1="Wrong"), row, dict(row)])

    # Exercise streaming iteration and selection without network or HF disk caching.
    def materialize(generator, features):
        return Dataset.from_list(list(generator()), features=features)

    with (
        patch("datasets.load_dataset", return_value=source) as remote,
        patch("datasets.Dataset.from_generator", side_effect=materialize),
    ):
        train = load_codeio(10)
        heldout = load_codeio(split="test")
        all_train = load_codeio()
        empty = load_codeio(0)
    assert len(train) == 10 and len(heldout) > 0 and len(empty) == 0
    assert list(train) == list(all_train)[:10]
    assert len(all_train) + len(heldout) == len(candidates)
    train_ids = {function_id(row["question"]) for row in all_train}
    test_ids = {function_id(row["question"]) for row in heldout}
    assert train_ids.isdisjoint(test_ids)
    remote.assert_called_with(
        DATASET_ID, split="train", revision=DATASET_REVISION, streaming=True
    )
    assert function_id(raw_row(input_value=1)["prompt"]) == function_id(
        raw_row(input_value=99)["prompt"]
    )


def test_task_dispatch_preserves_math_and_routes_codeio():
    assert "codeio" in DATASET_REGISTRY_TRAIN and "codeio" in DATASET_REGISTRY_EVAL
    assert format_prompt("q") == format_prompt_math("q")
    assert "math" not in format_prompt("q", "codeio")[0]["content"]
    assert "boxed" not in answer_context("false", "codeio")
    from trl.rewards import accuracy_reward

    assert accuracy_reward_for_dataset("deepmath") is accuracy_reward
    reward = accuracy_reward_for_dataset("codeio")
    assert reward(
        ['{"output": false}', [{"role": "assistant", "content": '{"output": 1}'}]],
        ["false", "true"],
    ) == [1.0, 0.0]
    with patch.dict(
        DATASET_REGISTRY_TRAIN, {"codeio": lambda max_samples: max_samples}
    ):
        assert load_train_dataset("codeio", max_samples=3, require_solution=True) == 3


def test_training_builders_use_same_codeio_prompt_and_structured_target():
    from train.grpo.train_grpo import build_grpo_dataset
    from train.opd.train_gold import build_gold_dataset
    from train.opsd.train_sdft import build_sdft_dataset
    from train.ppo.train_ppo_pi import build_ppo_pi_dataset
    from train.sft.train_sft import build_sft_dataset

    row = normalize_row(raw_row(value={"items": [1, 2]}))
    row.pop("function_id")
    ds = Dataset.from_list([row])
    prompt = format_prompt(row["question"], "codeio")
    with patch("train.grpo.train_grpo.load_train_dataset", return_value=ds):
        grpo = build_grpo_dataset("codeio")[0]
        assert grpo["prompt"] == prompt
        assert grade('{"output": {"items": [1, 2]}}', grpo["solution"], "codeio")[1]
    with patch("train.opd.train_gold.load_train_dataset", return_value=ds):
        gold = build_gold_dataset("codeio")[0]["messages"]
        assert gold[:-1] == prompt
        assert grade(gold[-1]["content"], row["final_answer"], "codeio")[1]
    with patch("train.opsd.train_sdft.load_train_dataset", return_value=ds):
        for mode in ["answer", "full"]:
            sdft = build_sdft_dataset(
                mode, dataset="codeio", include_reward_solution=True
            )[0]
            assert sdft["prompt"] == prompt
            assert sdft["solution"] == row["final_answer"]
            assert "boxed" not in sdft["privileged_context"]
    with patch("train.ppo.train_ppo_pi.load_train_dataset", return_value=ds):
        for mode in ["answer", "full"]:
            ppo = build_ppo_pi_dataset(
                dataset="codeio",
                pi_mode=mode,
                model="fake",
                max_value_prompt_length=None,
            )[0]
            assert ppo["prompt"] == prompt and ppo["solution"] == row["final_answer"]
    with patch("train.sft.train_sft.load_train_dataset", return_value=ds):
        sft = build_sft_dataset(dataset="codeio")[0]
        assert sft["prompt"] == prompt
        assert sft["completion"][0]["content"] == row["solution"]


def test_hint_generation_and_teacher_analysis_share_task_prompt():
    from utils.gen_hints import build_messages, leaks_answer
    from train.opsd.train_hint_gen.lib import (
        invalid_hint_reason,
        FrozenHintTeacher,
        HintRewardConfig,
    )
    from eval.hint_gen_compare import hinted_teacher_messages
    from eval.passk_pi import build_teacher_messages
    from eval.teacher_uncertainty import measure_completion
    from train.ppo.train_ppo_val import compose_value_messages
    from eval.advantage_dynamics_sdft import build_teacher_messages as dynamics_messages

    problem = {
        "question": "q",
        "answer": "false",
        "final_answer": "false",
        "hint": "Trace the loop.",
        "dataset": "codeio",
    }
    prompt = format_prompt("q", "codeio")
    assert "math" not in build_messages("q", "response", "codeio")[0]["content"]
    assert leaks_answer('{"output": false}', "false", "codeio")
    assert invalid_hint_reason('{"output": false}', "false", "codeio") == "answer_leak"
    assert invalid_hint_reason("Trace the loop.", "false", "codeio") is None
    assert build_teacher_messages(problem, "none") == prompt
    assert dynamics_messages(problem, "none") == prompt
    backend = object.__new__(FrozenHintTeacher)
    backend.config = HintRewardConfig(model="fake", dataset="codeio")
    assert backend._hinted_messages("q", "Trace the loop.") == hinted_teacher_messages(
        "q", "Trace the loop.", "codeio"
    )
    assert measure_completion('{"output": false}', 5, "stop", "false", "codeio")[
        "correct"
    ]
    assert "mathematics" not in compose_value_messages(prompt, "codeio")[0]["content"]


def test_run_eval_passes_dataset_to_both_prompt_and_grader():
    import types
    from eval.run_eval import evaluate_model

    prompts = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            prompts.append(messages)
            return "prompt"

    class LLM:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, *args):
            return [
                types.SimpleNamespace(
                    outputs=[
                        types.SimpleNamespace(text='{"output": false}', token_ids=[1])
                    ]
                )
            ]

    with patch("eval.run_eval.LLM", LLM):
        result = evaluate_model(
            "fake",
            [
                {
                    "problem": "q",
                    "answer": "false",
                    "dataset": "codeio",
                    "unique_id": "id",
                }
            ],
        )
    assert prompts == [format_prompt("q", "codeio")]
    assert result["results"][0]["n_correct"] == 1
