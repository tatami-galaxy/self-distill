"""Deterministic paragraph segmentation of a rollout into numbered steps.

The step is the unit at which (a) a judge later labels the first uncorrected
error and (b) we aggregate per-token credit (A_t / Abar_t / reweight_t). It must
therefore be a deterministic function of the *stored* token stream so that a step
index maps back to an exact token range — no LLM renumbering, no quote matching.

Segmentation (see design.md, "how to segment"):
  1. Hard boundary at the end of the thinking block (``</think>``) so the final
     answer never merges into the reasoning.
  2. Primary split on blank lines (``\n\n``) — paragraphs.
  3. Over-long paragraphs (> ``max_tokens``) are sub-split at the latest available
     single-newline / sentence boundary before the limit, hard-cutting only if no
     such boundary exists.
  4. Short paragraphs (< ``min_tokens``) are merged into a neighbour (never across
     the think/answer boundary).
Steps 3+4 bound every step to roughly ``[min_tokens, max_tokens]`` tokens.

Text is reconstructed from the per-token ``token_str`` surfaces (the same string
whose char offsets line up with the credit signal); boundaries are snapped to
token indices, so a step is always a contiguous ``[tok_start, tok_end)`` range.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from pathlib import Path

# Blank line(s): a newline followed by one or more further (possibly indented)
# newlines. ``end()`` is the start of the next paragraph, so trailing newlines
# attach to the *preceding* step.
_PARA_RE = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")
# Finer fallbacks for splitting an over-long paragraph (allowed break points).
_NEWLINE_RE = re.compile(r"\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CLOSE_THINK_RE = re.compile(r"</think>")


def _char_offsets(tokens: list[dict]) -> tuple[str, list[int]]:
    """Reconstructed text + per-token *start* char offset (len == len(tokens))."""
    starts: list[int] = []
    pos = 0
    parts: list[str] = []
    for t in tokens:
        s = t["token_str"]
        starts.append(pos)
        parts.append(s)
        pos += len(s)
    return "".join(parts), starts


def _split_long(
    s: int, e: int, allowed: list[int], max_tokens: int
) -> list[tuple[int, int]]:
    """Greedily cut ``[s, e)`` into <= ``max_tokens`` chunks at ``allowed`` points.

    Prefer the latest allowed break point within the limit; hard-cut at the limit
    only when no allowed point falls inside the current window.
    """
    inside = [a for a in allowed if s < a < e]
    out: list[tuple[int, int]] = []
    start = s
    while e - start > max_tokens:
        limit = start + max_tokens
        cands = [a for a in inside if start < a <= limit]
        bp = cands[-1] if cands else limit
        out.append((start, bp))
        start = bp
    out.append((start, e))
    return out


def _merge_short(
    segs: list[tuple[int, int]], min_tokens: int, barriers: set[int]
) -> list[tuple[int, int]]:
    """Merge any < ``min_tokens`` step into a neighbour, never across a barrier.

    A step boundary in ``barriers`` (the think/answer split) is never dissolved, so
    a step isolated by barriers on both sides may stay below ``min_tokens``.
    """
    out = list(segs)
    changed = True
    while changed:
        changed = False
        for i, (s, e) in enumerate(out):
            if e - s >= min_tokens:
                continue
            if i > 0 and s not in barriers:  # merge into previous
                out[i - 1] = (out[i - 1][0], e)
                out.pop(i)
            elif i < len(out) - 1 and e not in barriers:  # merge into next
                out[i + 1] = (s, out[i + 1][1])
                out.pop(i)
            else:
                continue  # boxed in by barriers; leave it
            changed = True
            break
    return out


def segment_rollout(
    tokens: list[dict], min_tokens: int = 20, max_tokens: int = 200
) -> list[dict]:
    """Segment one rollout's ``tokens`` into numbered steps.

    Returns a list of ``{idx, tok_start, tok_end, n_tokens, region, text}`` where
    ``[tok_start, tok_end)`` indexes ``tokens`` and ``region`` is ``"think"`` or
    ``"answer"`` (relative to the ``</think>`` boundary, or all ``"think"`` if the
    trace has none).
    """
    n = len(tokens)
    if n == 0:
        return []
    text, starts = _char_offsets(tokens)

    def snap(c: int) -> int:  # first token index whose start char >= c
        return bisect.bisect_left(starts, c)

    cuts: set[int] = {0, n}
    barriers: set[int] = set()

    # think/answer boundary: end of the last </think>
    close = list(_CLOSE_THINK_RE.finditer(text))
    answer_start = n  # token index where the answer region begins (n => no answer)
    if close:
        answer_start = snap(close[-1].end())
        cuts.add(answer_start)
        barriers.add(answer_start)

    for m in _PARA_RE.finditer(text):
        cuts.add(snap(m.end()))

    ordered = sorted(c for c in cuts if 0 <= c <= n)
    segs = [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)
            if ordered[i] < ordered[i + 1]]

    # over-long split (allowed = single-newline + sentence boundaries)
    allowed = sorted({snap(m.end()) for m in _NEWLINE_RE.finditer(text)}
                     | {snap(m.end()) for m in _SENTENCE_RE.finditer(text)})
    split: list[tuple[int, int]] = []
    for s, e in segs:
        split.extend(_split_long(s, e, allowed, max_tokens) if e - s > max_tokens
                     else [(s, e)])

    merged = _merge_short(split, min_tokens, barriers)

    steps: list[dict] = []
    for idx, (s, e) in enumerate(merged):
        c0 = starts[s]
        c1 = starts[e] if e < n else len(text)
        steps.append({
            "idx": idx,
            "tok_start": s,
            "tok_end": e,
            "n_tokens": e - s,
            "region": "answer" if s >= answer_start else "think",
            "text": text[c0:c1],
        })
    return steps


# --------------------------------------------------------------------------- CLI


def _stream(path: Path, only: str, limit: int):
    """Yield up to ``limit`` records matching ``only`` without loading the file."""
    want = None if only == "any" else (only == "correct")
    found = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if want is not None and bool(rec.get("correct")) != want:
                continue
            yield rec
            found += 1
            if found >= limit:
                return


# main only for testing
def main() -> None:
    p = argparse.ArgumentParser(description="Eyeball rollout segmentation.")
    p.add_argument("--rollouts", required=True, help="JSONL from generate_rollouts.py")
    p.add_argument("--n", type=int, default=3, help="how many rollouts to print")
    p.add_argument("--only", choices=["correct", "incorrect", "any"], default="incorrect")
    p.add_argument("--min-tokens", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--snippet", type=int, default=100, help="chars of each step to show")
    args = p.parse_args()

    for rec in _stream(Path(args.rollouts), args.only, args.n):
        steps = segment_rollout(rec["tokens"], args.min_tokens, args.max_tokens)
        sizes = [s["n_tokens"] for s in steps]
        print("=" * 100)
        print(f"problem_id={rec['problem_id']}  correct={rec.get('correct')}  "
              f"completion_tokens={len(rec['tokens'])}  "
              f"steps={len(steps)}  token/step min/median/max="
              f"{min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}")
        for s in steps:
            snip = s["text"].replace("\n", "↵")[: args.snippet]
            print(f"  [{s['idx']:>3}] {s['region']:>6} "
                  f"toks {s['tok_start']:>5}-{s['tok_end']:<5} (n={s['n_tokens']:>3})  {snip}")


if __name__ == "__main__":
    main()
