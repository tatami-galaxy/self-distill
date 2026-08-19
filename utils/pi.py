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

__all__ = ["PI_ANSWER", "PI_FULL", "PI_HINT", "PI_ROLLOUT"]
