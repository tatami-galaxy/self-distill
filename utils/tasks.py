"""Shared math prompts, reward targets, and answer-only privileged context."""

from utils.pi import PI_ANSWER


def format_prompt(problem: str, dataset: str = "deepmath") -> list[dict]:
    from utils.utils import format_prompt_math

    return format_prompt_math(problem)


def reward_solution(answer: str, dataset: str = "deepmath") -> str:
    return "\\boxed{" + str(answer) + "}"


def answer_context(answer: str, dataset: str = "deepmath") -> str:
    return PI_ANSWER.format(answer=answer)
