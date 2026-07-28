"""Shared helpers: prompts, answer grading, the dataset registries, resume validation, and the
privileged-context plumbing.

The implementation lives in `utils/utils.py`. This module re-exports its public surface so every
caller writes `from utils import grade` rather than `from utils.utils import grade` -- the names
below ARE the supported API, and anything not listed here is internal to that file.

`utils/gen_hints.py` is deliberately not imported: it is a CLI that loads vLLM, and importing it
here would drag an inference engine into every `import utils`. Run it as `python -m
utils.gen_hints`.
"""

from utils.utils import (
    # prompts
    MATH_SYSTEM_PROMPT,
    format_prompt_math,
    # resume validation
    validate_resume,
    # privileged context (PI)
    TEACHER_PROMPT_TEMPLATE,
    compose_pi_messages,
    hint_path,
    load_hint_cache,
    # answer extraction / grading
    extract_answer,
    grade,
    grade_answer,
    # dataset registries and loaders
    DATASET_REGISTRY_EVAL,
    DATASET_REGISTRY_TRAIN,
    register_dataset_eval,
    register_dataset_train,
    has_solution,
    load_train_dataset,
)

__all__ = [
    "MATH_SYSTEM_PROMPT",
    "format_prompt_math",
    "validate_resume",
    "TEACHER_PROMPT_TEMPLATE",
    "compose_pi_messages",
    "hint_path",
    "load_hint_cache",
    "extract_answer",
    "grade",
    "grade_answer",
    "DATASET_REGISTRY_EVAL",
    "DATASET_REGISTRY_TRAIN",
    "register_dataset_eval",
    "register_dataset_train",
    "has_solution",
    "load_train_dataset",
]
