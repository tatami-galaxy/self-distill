"""Conservative AM hint checks, including math equivalence and explicit choices.

These are auditable heuristics, not a semantic guarantee. Ambiguous intermediate
values and paraphrases still require a manual pilot audit.
"""

from __future__ import annotations

import re
from functools import lru_cache

from utils.gen_hints import leaks_answer

VERSION = 1


def canonical_text(text):
    text = re.sub(r"\\(?:dfrac|tfrac)\b", r"\\frac", str(text))
    text = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" $\n\t.,;").lower()


@lru_cache(maxsize=8192)
def parse_math(text):
    from math_verify import parse

    return parse("\\boxed{" + text + "}", parsing_timeout=2)


def equivalent(a, b):
    from math_verify import verify

    try:
        gold, candidate = parse_math(a), parse_math(b)
        return bool(gold and candidate and verify(gold, candidate, timeout_seconds=2))
    except (ValueError, TypeError, TimeoutError, NotImplementedError):
        return False


def validation_flags(text, row):
    flags = []
    if not text.strip():
        return ["empty"]
    if "<think>" in text or "</think>" in text:
        flags.append("thinking")
    gold = row["final_answer"]
    if leaks_answer(text, gold):
        flags.append("answer_leak_lexical")
    normalized = canonical_text(text)
    answers = [gold, *row.get("answer_aliases", [])]
    # MC labels are not numerical expressions. Detect explicit label statements;
    # option content is checked below even when the label never appears.
    if gold.strip().strip("() ").upper() in tuple("ABCDE"):
        label = re.escape(gold.strip().strip("() "))
        if re.search(
            rf"\b(?:option|choice|answer)\s*(?:is|:|=)?\s*\(?{label}\)?\b",
            text,
            re.IGNORECASE,
        ):
            flags.append("answer_leak_choice")
    spans = re.findall(r"\$\$?(.*?)\$\$?|\\\((.*?)\\\)|\\\[(.*?)\\\]", text, re.DOTALL)
    expressions = []
    for parts in spans:
        expression = next((p for p in parts if p), "").strip()
        if expression:
            expressions.append(expression)
            if "=" in expression:
                expressions.append(expression.rsplit("=", 1)[-1].strip())
    expressions.extend(re.findall(r"\\(?:dfrac|tfrac|frac)\{[^{}]+\}\{[^{}]+\}", text))
    conclusion = re.search(
        r"\b(?:answer|result|value|integral|probability|maximum|minimum|derivative|entropy|flux)\b"
        r"[^.!?\n]{0,100}\b(?:is|equals|equal to|=)\s+",
        normalized,
    )
    words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    for answer in answers:
        value = canonical_text(answer)
        if not value or value.upper() in tuple("ABCDE"):
            continue
        if value in words:
            if conclusion and re.search(
                rf"\b(?:{re.escape(value)}|{words[value]})\b",
                normalized[conclusion.end() :],
            ):
                flags.append("answer_leak_explicit_value")
            # Single digits are common intermediate quantities; don't reject every occurrence.
            continue
        if len(value) > 1 and re.search(
            rf"(?<!\w){re.escape(value)}(?!\w)", normalized
        ):
            flags.append("answer_leak_equivalent_text")
        # Option descriptions often wrap the mathematical conclusion in prose.
        for phrase in ("saddle point", "local maximum", "local minimum"):
            if phrase in value and re.search(
                rf"\b(?:is|has|exhibits)\s+(?:a\s+)?{phrase}\b", normalized
            ):
                flags.append("answer_leak_choice_content")
        for expression in dict.fromkeys(expressions):
            if (
                len(expression) <= 512
                and len(value) <= 512
                and equivalent(answer, expression)
            ):
                flags.append("answer_leak_math_equivalence")
                break
    return sorted(set(flags))
