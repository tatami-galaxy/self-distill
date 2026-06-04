"""Deterministic segmentation of reasoning traces into indexable steps.

A "step" is a contiguous slice of the original trace with verbatim char offsets:
``trace[step.char_start:step.char_end] == step.text`` holds exactly. Keeping the
offsets lets us map any step (or a quoted snippet located inside it) back to a
character span for overlap/IoU scoring.

Strategy (see plan): split on blank lines, merge tiny fragments into the previous
step, then split over-long paragraphs at sentence boundaries (math-aware).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Step:
    index: int
    text: str
    char_start: int
    char_end: int  # exclusive: trace[char_start:char_end] == text


_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
_SENTENCE_END = re.compile(r"[.!?]+(\s+)")


def _strip_span(trace: str, start: int, end: int) -> tuple[int, int]:
    """Trim leading/trailing whitespace from a [start, end) span."""
    while start < end and trace[start].isspace():
        start += 1
    while end > start and trace[end - 1].isspace():
        end -= 1
    return start, end


def _raw_paragraphs(trace: str) -> list[tuple[int, int]]:
    """Blank-line-separated content spans, each stripped of surrounding whitespace."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BLANK_LINE.finditer(trace):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(trace)))

    out: list[tuple[int, int]] = []
    for start, end in spans:
        start, end = _strip_span(trace, start, end)
        if end > start:
            out.append((start, end))
    return out


def _merge_short(spans: list[tuple[int, int]], min_chars: int) -> list[tuple[int, int]]:
    """Merge fragments shorter than ``min_chars`` into the previous step.

    Merging extends the previous span's end to the fragment's end, so the merged
    text keeps the original separator(s) verbatim. A short leading fragment with no
    predecessor is merged forward into the next step instead.
    """
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and (end - start) < min_chars:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Leading fragment(s) shorter than min_chars: merge forward.
    while len(merged) >= 2 and (merged[0][1] - merged[0][0]) < min_chars:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged


def _sentence_starts(block: str) -> list[int]:
    """Local offsets where a new sentence begins, skipping math and decimals.

    Requiring whitespace after the terminal punctuation already excludes decimals
    like ``3.14`` (no space follows the dot).
    """
    math = [(m.start(), m.end()) for m in _MATH_SPAN.finditer(block)]

    def in_math(pos: int) -> bool:
        return any(a <= pos < b for a, b in math)

    starts: list[int] = []
    for m in _SENTENCE_END.finditer(block):
        if in_math(m.start()):
            continue
        starts.append(m.end())
    return starts


def _split_long(
    trace: str, start: int, end: int, max_chars: int
) -> list[tuple[int, int]]:
    """Split an over-long span at sentence boundaries, greedily filling ``max_chars``."""
    block = trace[start:end]
    cuts = sorted({0, *_sentence_starts(block), len(block)})
    sentences = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    chunks: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    for s, e in sentences:
        if cur_start is None:
            cur_start, cur_end = s, e
        elif (e - cur_start) <= max_chars:
            cur_end = e
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        chunks.append((cur_start, cur_end))

    out: list[tuple[int, int]] = []
    for s, e in chunks:
        gs, ge = _strip_span(trace, start + s, start + e)
        if ge > gs:
            out.append((gs, ge))
    return out


def segment_trace(trace: str, min_chars: int = 40, max_chars: int = 600) -> list[Step]:
    """Segment ``trace`` into normalized, verbatim-offset steps."""
    paragraphs = _raw_paragraphs(trace)
    merged = _merge_short(paragraphs, min_chars)

    spans: list[tuple[int, int]] = []
    for start, end in merged:
        if (end - start) > max_chars:
            spans.extend(_split_long(trace, start, end, max_chars))
        else:
            spans.append((start, end))

    return [
        Step(index=i, text=trace[s:e], char_start=s, char_end=e)
        for i, (s, e) in enumerate(spans)
    ]


def render_numbered(steps: list[Step]) -> str:
    """Render steps as ``[i] text`` lines for prompting."""
    return "\n".join(f"[{step.index}] {step.text}" for step in steps)
