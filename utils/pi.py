"""Canonical privileged-information text templates shared by training algorithms."""

PI_FULL = (
    "This is an example of a correct, worked solution to the question above:\n\n"
    "{demo}\n\n"
    "Now write a complete solution of your own, including the reasoning."
)
PI_ANSWER = (
    "Hint: the correct final answer to the question above is \\boxed{{{answer}}}. "
    "Reach it with your own complete reasoning."
)
PI_HINT = (
    "Here are some useful concepts for the question above:\n\n"
    "{hint}\n\n"
    "Use them for your own complete solution if needed."
)
PI_ROLLOUT = (
    "Here is an attempted solution to the question above. It may or may not be correct:\n\n"
    "{attempt}\n\n"
    "Now write a complete solution of your own, including the reasoning."
)


def extract_final_solution(solution: str) -> str:
    """Return the complete response after one well-formed </think> boundary.

    Accept the R1 convention with no opening tag. Match reference-target validation
    without importing training or evaluation modules into shared PI utilities.
    """
    text = (solution or "").strip()
    n_open = text.count("<think>")
    if text.count("</think>") != 1 or not (
        n_open == 0 or (n_open == 1 and text.startswith("<think>"))
    ):
        raise ValueError("malformed_thinking_trace")
    final = text.split("</think>", 1)[1]
    if not final.strip():
        raise ValueError("empty_final_solution")
    return final


__all__ = ["PI_ANSWER", "PI_FULL", "PI_HINT", "PI_ROLLOUT", "extract_final_solution"]
