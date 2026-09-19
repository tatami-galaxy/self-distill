"""CodeIO output prediction from the upstream-verified public PyEdu release.

No reference code is executed. The public release lacks execution targets, so
only first-turn responses marked ``Correct output!`` supply reference outputs.
This is a verified subset, not a reproduction of CodeIO's unfiltered SFT corpus.
"""

import hashlib
import json
import math
import re

DATASET_ID = "hkust-nlp/CodeIO-PyEdu-Reasoning"
DATASET_REVISION = "6f45f6f4091ecd089054574ab46936994f420b24"
OUTPUT_INSTRUCTION = "Can you predict the output without writing any code?"
REFERENCE_MARKER = "Tip: Here is a reference code snippet for this question."
SYSTEM_PROMPT = (
    "Reason through the code and given input step by step without executing code. "
    'End with a JSON object {"output": <predicted output>}, preserving JSON types.'
)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _decoder():
    return json.JSONDecoder(
        parse_constant=_reject_constant, object_pairs_hook=_unique_keys
    )


def extract_output(text: str) -> dict | None:
    """Read the last top-level JSON object; never execute generated text.

    Objects in a closed thinking trace are ignored. Nested objects and braces
    inside strings are consumed by the JSON decoder, not by a brace regex.
    A later malformed answer must not fall back to an earlier correct example.
    """
    text = text.rsplit("</think>", 1)[-1]
    last = None
    start = None
    depth = 0
    in_string = escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start, depth = index, 1
                in_string = escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    last = _decoder().decode(text[start : index + 1])
                except (ValueError, RecursionError):
                    last = None
                start = None
    if start is not None:
        return None
    if not isinstance(last, dict) or set(last) != {"output"}:
        return None
    try:
        json.dumps(last, allow_nan=False)
    except (ValueError, RecursionError):
        return None
    return last


def outputs_equal(pred, target) -> bool:
    """Recursive upstream numeric tolerance (1e-3 and equal integer parts).

    JSON booleans are kept distinct from numbers, unlike Python's True == 1.
    Dictionary order is irrelevant; array order and string contents are exact.
    """
    if isinstance(pred, bool) or isinstance(target, bool):
        return type(pred) is type(target) and pred == target
    if isinstance(pred, dict) and isinstance(target, dict):
        return pred.keys() == target.keys() and all(
            outputs_equal(pred[key], target[key]) for key in pred
        )
    if isinstance(pred, list) and isinstance(target, list):
        return len(pred) == len(target) and all(
            outputs_equal(p, t) for p, t in zip(pred, target, strict=True)
        )
    if isinstance(pred, (int, float)) and isinstance(target, (int, float)):
        if isinstance(pred, int) and isinstance(target, int):
            return pred == target
        try:
            return (
                math.isfinite(pred)
                and math.isfinite(target)
                and abs(pred - target) <= 0.001 * abs(target)
                and int(pred) == int(target)
            )
        except (OverflowError, ValueError):
            return False
    return type(pred) is type(target) and pred == target


def grade_output(response: str, gold: str) -> tuple[str | None, bool]:
    prediction = extract_output(response)
    if prediction is None:
        return None, False
    serialized = json.dumps(prediction["output"], ensure_ascii=False, allow_nan=False)
    try:
        target = json.loads(
            gold, parse_constant=_reject_constant, object_pairs_hook=_unique_keys
        )
        return serialized, outputs_equal(prediction["output"], target)
    except (ValueError, RecursionError):
        return serialized, False


def function_id(prompt: str) -> str:
    """Group all I/O instances of identical reference code into the same split."""
    if REFERENCE_MARKER not in prompt:
        raise ValueError("CodeIO prompt has no reference-code boundary")
    code = prompt.split(REFERENCE_MARKER, 1)[1]
    # Skip the remaining fixed instruction before the reference code.
    code = code.split("\n\n", 1)[1]
    normalized = "\n".join(line.rstrip() for line in code.strip().splitlines())
    return hashlib.sha256(normalized.encode()).hexdigest()


def normalize_row(row: dict) -> dict | None:
    """Use verified first turns only; revisions rely on omitted conversation."""
    prompt = row.get("prompt") or ""
    instruction = prompt.split(REFERENCE_MARKER, 1)[0]
    if (
        OUTPUT_INSTRUCTION not in instruction
        or (row.get("feedback_1") or "").strip() != "Correct output!"
    ):
        return None
    solution = row.get("turn_1") or ""
    answer = extract_output(solution)
    if answer is None:
        return None
    try:
        group = function_id(prompt)
    except (ValueError, IndexError):
        return None
    return {
        "question": prompt,
        "final_answer": json.dumps(
            answer["output"], ensure_ascii=False, allow_nan=False
        ),
        "solution": solution,
        "function_id": group,
    }


def load_codeio(max_samples: int | None = None, *, split: str = "train"):
    """Stream the pinned release; reserve 5% of reference-code hashes for eval.

    Limits apply after verification and split selection, preserving prefix order.
    The returned schema matches the other training datasets exactly.
    """
    from datasets import Dataset, Features, Value, load_dataset

    if split not in {"train", "test"}:
        raise ValueError("CodeIO split must be train or test")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be nonnegative")

    def rows():
        if max_samples == 0:
            return
        source = load_dataset(
            DATASET_ID, split="train", revision=DATASET_REVISION, streaming=True
        )
        count = 0
        seen = set()
        for raw in source:
            row = normalize_row(raw)
            if row is None:
                continue
            held_out = int(row.pop("function_id"), 16) % 100 < 5
            if held_out != (split == "test"):
                continue
            identity = hashlib.sha256(row["question"].encode()).digest()
            if identity in seen:
                continue
            seen.add(identity)
            yield row
            count += 1
            if max_samples is not None and count >= max_samples:
                break

    features = Features(
        {name: Value("string") for name in ("question", "final_answer", "solution")}
    )
    if max_samples == 0:
        return Dataset.from_dict({name: [] for name in features}, features=features)
    return Dataset.from_generator(rows, features=features)


def leaks_output(hint: str, gold: str) -> bool:
    """Detect explicit output answers and distinctive literal output values.

    Like the math filter this is a heuristic, not a semantic leakage guarantee.
    Short scalar values can appear incidentally in useful hints.
    """
    if re.search(r'["\']output["\']\s*:', hint) or "\\boxed" in hint:
        return True
    if grade_output(hint, gold)[1]:
        return True
    try:
        value = json.loads(gold)
    except ValueError:
        return False
    literal = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, dict)):
        return re.sub(r"\s+", "", literal) in re.sub(r"\s+", "", hint)
    return (
        len(literal) >= 2
        and re.search(rf"(?<![\w.]){re.escape(literal)}(?![\w.])", hint) is not None
    )
