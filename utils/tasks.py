"""Task-aware prompts and rewards; math behavior remains the default."""

from utils.codeio import SYSTEM_PROMPT, grade_output
from utils.pi import PI_ANSWER


def format_prompt(problem: str, dataset: str = "deepmath") -> list[dict]:
    if dataset == "codeio":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
    from utils.utils import format_prompt_math

    return format_prompt_math(problem)


def reward_solution(answer: str, dataset: str = "deepmath") -> str:
    return answer if dataset == "codeio" else "\\boxed{" + str(answer) + "}"


def answer_context(answer: str, dataset: str = "deepmath") -> str:
    if dataset == "codeio":
        return (
            'The correct output is: {"output": '
            + answer
            + "}. Derive it with your own reasoning."
        )
    return PI_ANSWER.format(answer=answer)


def codeio_accuracy_reward(completions, solution, **kwargs) -> list[float]:
    def text(completion):
        if isinstance(completion, str):
            return completion
        return next(
            (m["content"] for m in reversed(completion) if m["role"] == "assistant"), ""
        )

    return [
        float(grade_output(text(c), s)[1])
        for c, s in zip(completions, solution, strict=True)
    ]


def accuracy_reward_for_dataset(dataset: str):
    if dataset == "codeio":
        return codeio_accuracy_reward
    from trl.rewards import accuracy_reward

    return accuracy_reward


def dataset_provenance(dataset: str) -> dict:
    """Stamp CodeIO's target derivation and split policy into run metadata."""
    if dataset != "codeio":
        return {}
    from utils.codeio import DATASET_ID, DATASET_REVISION

    return {
        "codeio": {
            "source": DATASET_ID,
            "revision": DATASET_REVISION,
            "task": "output_prediction",
            "targets": "verified_first_turn_strict_json_v1",
            "split": "sha256_reference_code_mod100_test_lt5_v1",
            "grader": "typed_json_upstream_numeric_tolerance_1e-3_v1",
        }
    }
