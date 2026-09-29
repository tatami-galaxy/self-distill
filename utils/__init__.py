"""Shared helpers: prompts, answer grading, the dataset registries, resume validation, and the
privileged-context plumbing.

The implementation lives in `utils/utils.py`. This module re-exports its public surface so every
caller writes `from utils import grade` rather than `from utils.utils import grade` -- the names
below ARE the supported API, and anything not listed here is internal to that file.

`utils/gen_hints.py` is deliberately not imported: it is a CLI that loads vLLM, and importing it
here would drag an inference engine into every `import utils`. Run it as `python -m
utils.gen_hints`.
"""

from utils.pi import PI_ANSWER, PI_FULL, PI_HINT, PI_ROLLOUT, extract_final_solution
from utils.tasks import answer_context, format_prompt, reward_solution
from utils.utils import (
    # dataset registries and loaders
    DATASET_REGISTRY_EVAL,
    DATASET_REGISTRY_TRAIN,
    # prompts
    MATH_SYSTEM_PROMPT,
    # privileged context (PI)
    TEACHER_PROMPT_TEMPLATE,
    compose_pi_messages,
    # answer extraction / grading
    extract_answer,
    format_prompt_math,
    grade,
    grade_answer,
    has_solution,
    hint_path,
    load_hint_cache,
    load_train_dataset,
    register_dataset_eval,
    register_dataset_train,
    rollout_path,
    # resume validation
    validate_resume,
)

__all__ = [
    "DATASET_REGISTRY_EVAL",
    "DATASET_REGISTRY_TRAIN",
    "MATH_SYSTEM_PROMPT",
    "PI_ANSWER",
    "PI_FULL",
    "PI_HINT",
    "PI_ROLLOUT",
    "TEACHER_PROMPT_TEMPLATE",
    "answer_context",
    "compose_pi_messages",
    "extract_answer",
    "extract_final_solution",
    "format_prompt",
    "format_prompt_math",
    "grade",
    "grade_answer",
    "has_solution",
    "hint_path",
    "load_hint_cache",
    "load_train_dataset",
    "register_dataset_eval",
    "register_dataset_train",
    "reward_solution",
    "rollout_path",
    "validate_resume",
]
