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

    This is the per-position quantity the *dense* OPD/OPSD gradient responds to
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
