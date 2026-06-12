"""Shared helpers for OPD/OPSD token-level credit assignment.

The per-token credit is the sampled-token log-ratio

    A_t = log pi_teacher(y_t | context) - log pi_student(y_t | x, y_<t)

For OPD the teacher is a separate model; only the teacher swaps for OPSD. Both
log-probs are read off vLLM ``prompt_logprobs`` (a teacher-forcing prefill at
temperature 1) so the student and teacher are scored apples-to-apples at T=1,
regardless of the sampling temperature used to draw the rollout. See design.md.
"""

from __future__ import annotations

import math
from typing import Any

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Same prompt the eval/collection pipelines use (eval/run_eval.py).
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


def build_prompt_ids(
    tokenizer: Any,
    problem: str,
    enable_thinking: bool = True,
    privileged_info: str | None = None,
) -> list[int]:
    """Render system+user chat prompt to token ids (with generation prompt).

    ``enable_thinking`` is forwarded to the chat template when supported (Qwen3);
    templates that don't accept it are called without it.

    ``privileged_info`` (OPSD only) is appended to the user turn so the model
    scores the *same* completion under privileged context f. The unprivileged
    prompt (``privileged_info=None``) reproduces exactly what generate_rollouts
    used, so the OPSD baseline log-probs (``student_lp``) carry over unchanged and
    only the privileged pass is recomputed. The generation prompt is identical in
    both cases, so the teacher-forced completion attaches at the same boundary.
    """
    user_content = problem if privileged_info is None else f"{problem}\n\n{privileged_info}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    # transformers 5.x returns a BatchEncoding (dict) when tokenize=True; older
    # versions return a plain list. return_dict=False normalizes to a list, and we
    # still guard for the dict form below.
    kwargs = {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
    try:
        ids = tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    if isinstance(ids, dict):  # BatchEncoding
        ids = ids["input_ids"]
    return [int(t) for t in ids]


def token_strings(tokenizer: Any, token_ids: list[int]) -> list[str]:
    """Per-token surface strings for display (single-token decode).

    Keyed to ``token_id``; the numeric signal is exact regardless. Rare
    multi-byte chars split across tokens may render imperfectly — acceptable for
    visualization only.
    """
    return tokenizer.batch_decode([[t] for t in token_ids])


def reconstruct_trace(tokens: list[dict]) -> str:
    """Canonical completion text = concatenation of per-token surfaces.

    Built from the stored ``token_str`` (single-token decode) so it is exactly the
    string whose char offsets line up with the per-token credit signal. We show
    *this* string to the error judge, so any verbatim quote it copies maps back to
    a contiguous token range by construction — no dependence on the judge's
    tokenizer, and no expensive O(n^2) re-decode. (Equals ``completion_text`` except
    at the rare multi-byte char split across tokens; immaterial for ASCII/LaTeX math.)
    """
    return "".join(t["token_str"] for t in tokens)


def token_char_offsets(tokens: list[dict]) -> list[tuple[int, int]]:
    """Per-token ``[char_start, char_end)`` into :func:`reconstruct_trace` output."""
    offsets: list[tuple[int, int]] = []
    pos = 0
    for t in tokens:
        n = len(t["token_str"])
        offsets.append((pos, pos + n))
        pos += n
    return offsets


def faithful_trace_and_offsets(
    tokenizer: Any, token_ids: list[int], window: int = 8
) -> tuple[str, list[tuple[int, int]]]:
    """Faithful completion text + per-token ``[char_start, char_end)`` offsets.

    Unlike :func:`reconstruct_trace` (single-token decode, which renders a multi-byte
    char split across tokens as ``�``), this decodes with the *student* tokenizer so
    math unicode (``√``, ``∞``, ...) is intact — important for both the judge's
    comprehension and verbatim quote matching. Per-token surface lengths come from a
    windowed incremental decode (``len(decode(ids[lo:i+1])) - len(decode(ids[lo:i]))``,
    ``lo = i-window+1``), which is O(n·window) instead of O(n²). Offsets index the
    returned faithful text; at the rare codepoint that straddles a token boundary the
    boundary may be off by ~1 char (immaterial for token-span overlap), and the final
    offset is clamped so the spans cover the full text exactly.
    """
    full = tokenizer.decode(token_ids)
    offsets: list[tuple[int, int]] = []
    pos = 0
    for i in range(len(token_ids)):
        lo = max(0, i - window + 1)
        a = len(tokenizer.decode(token_ids[lo : i + 1]))
        b = len(tokenizer.decode(token_ids[lo:i])) if i > lo else 0
        delta = max(0, a - b)
        offsets.append((pos, pos + delta))
        pos += delta
    if offsets and pos != len(full):  # absorb accumulated drift into the last token
        last_start = offsets[-1][0]
        offsets[-1] = (min(last_start, len(full)), len(full))
    return full, offsets


def _span_to_token_range(
    offsets: list[tuple[int, int]], c0: int, c1: int
) -> tuple[int, int]:
    """Half-open token range [lo, hi) whose char spans overlap ``[c0, c1)``."""
    lo = next((i for i, (_s, e) in enumerate(offsets) if e > c0), len(offsets))
    hi = next((i for i in range(len(offsets) - 1, -1, -1) if offsets[i][0] < c1), -1)
    return lo, (hi + 1 if hi >= lo else lo)


def quote_to_token_span(
    trace: str,
    offsets: list[tuple[int, int]],
    quote: str,
    fuzzy_cutoff: float = 60.0,
) -> dict:
    """Locate a judge's ``quote`` in ``trace`` -> token span, via a matching ladder.

    Returns ``{status, score, char_span, token_span}``. ``status`` (the tier that
    matched) is one of:
      - ``exact``      : the quote occurs exactly once (score 100).
      - ``ambiguous``  : occurs verbatim more than once; first occurrence (score 100).
      - ``normalized`` : found after collapsing runs of whitespace (score 100).
      - ``fuzzy``      : best edit-distance-aligned substring of ~quote length, when
                         the judge paraphrased; ``score`` is rapidfuzz partial_ratio
                         (0-100) and the match is kept only if ``>= fuzzy_cutoff``.
      - ``not_found``  : nothing at/above cutoff; spans are None.
    rapidfuzz's ``partial_ratio_alignment`` is the "slide a quote-sized window, take
    the most similar" step — it returns the aligned substring's char offsets directly.
    Heavy paraphrase (low score) is left for a later semantic-embedding tier.
    """
    import re

    if not quote:
        return {"status": "not_found", "score": None, "char_span": None, "token_span": None}

    first = trace.find(quote)
    if first >= 0:
        second = trace.find(quote, first + 1)
        status = "exact" if second < 0 else "ambiguous"
        c0, c1, score = first, first + len(quote), 100.0
    else:
        # Whitespace-flexible retry: the judge may re-flow spaces/newlines.
        pattern = r"\s+".join(re.escape(tok) for tok in quote.split())
        m = re.search(pattern, trace) if pattern else None
        if m is not None:
            status, c0, c1, score = "normalized", m.start(), m.end(), 100.0
        else:
            from rapidfuzz import fuzz

            ali = fuzz.partial_ratio_alignment(quote, trace, score_cutoff=fuzzy_cutoff)
            if ali is None:  # below cutoff
                return {"status": "not_found", "score": None, "char_span": None, "token_span": None}
            status, c0, c1, score = "fuzzy", ali.dest_start, ali.dest_end, float(ali.score)

    lo, hi = _span_to_token_range(offsets, c0, c1)
    return {"status": status, "score": score, "char_span": [c0, c1], "token_span": [lo, hi]}


def read_generation_logprobs(output: Any, topk: int) -> list[dict]:
    """Per-token ``{chosen_lp, topk}`` from a vLLM *generation* output.
    """
    comp = output.outputs[0]
    token_ids = comp.token_ids
    pos_logprobs = comp.logprobs  # list[dict[int, Logprob]], len == len(token_ids)
    per_token: list[dict] = []
    for tid, dist in zip(token_ids, pos_logprobs, strict=True):
        chosen_lp = float(dist[tid].logprob)
        # vLLM returns the top-k PLUS the sampled token when it falls outside the
        # top-k (dict size topk+1), and dict order isn't guaranteed.
        ranked = sorted(
            ((int(k), float(v.logprob)) for k, v in dist.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )[:topk]
        per_token.append({"chosen_lp": chosen_lp, "topk": [[k, lp] for k, lp in ranked]})
    return per_token


def expected_advantage(
    student_topk: list[list], teacher_topk: list[list]
) -> float | None:
    """Top-k approximation of the student-weighted expected advantage

        Abar_t = sum_v pi_student(v) * (log pi_teacher(v) - log pi_student(v))
               = -KL(pi_student || pi_teacher)_t        (<= 0)

    This is the per-position quantity the dense OPD/OPSD gradient responds to
    (the sampled-token A_t is its unbiased 1-sample estimate). Summed over the
    student's top-k support (where pi_student concentrates its mass). A
    student-top-k token absent from the teacher's top-k has its teacher log-prob
    floored at the teacher's k-th (smallest) value — an upper bound on the true
    value, so this slightly *under*-estimates the KL magnitude. The student tail
    outside top-k is dropped (small mass). Returns None if either top-k is empty.
    """
    if not student_topk or not teacher_topk:
        return None
    teacher_lp = {int(tid): lp for tid, lp in teacher_topk}
    teacher_floor = min(lp for _, lp in teacher_topk)
    abar = 0.0
    for tid, s_lp in student_topk:
        p = math.exp(s_lp)
        t_lp = teacher_lp.get(int(tid), teacher_floor)
        abar += p * (t_lp - s_lp)
    return abar


def score_prompt_logprobs(
    llm: LLM,
    sequences: list[tuple[list[int], int]],
    topk: int,
) -> list[list[dict]]:
    """Teacher-force score a batch of token sequences at T=1.

    Each item in ``sequences`` is ``(full_ids, start)`` where ``full_ids`` is
    ``prompt_ids + completion_ids`` and ``start = len(prompt_ids)``. Returns, per
    sequence, a list (length ``len(full_ids) - start``) of per-position dicts:

        {"chosen_lp": float, "topk": [[token_id, logprob], ...]}

    ``chosen_lp`` is the log-prob of the actually-present token at that position
    (vLLM always includes it in ``prompt_logprobs`` even if outside the top-k).
    """
    sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=topk)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _ in sequences]
    outputs = llm.generate(prompts, sampling)

    results: list[list[dict]] = []
    for (full_ids, start), output in zip(sequences, outputs, strict=True):
        plps = output.prompt_logprobs  # len == len(full_ids); plps[0] is None
        per_token: list[dict] = []
        for pos in range(start, len(full_ids)):
            dist = plps[pos]
            tid = full_ids[pos]
            chosen_lp = float(dist[tid].logprob)
            ranked = sorted(
                ((int(k), float(v.logprob)) for k, v in dist.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )[:topk]
            per_token.append({"chosen_lp": chosen_lp, "topk": [[k, lp] for k, lp in ranked]})
        results.append(per_token)
    return results
