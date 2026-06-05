"""Ask a model to locate the first error in its own incorrect traces, vs a judge.

1. Load incorrect traces (from collect_incorrect_traces.py).
2. Skip truncated traces (no closing </think>).
3. Segment each trace into numbered steps (error_detect.segment).
4. Self-critique: the same small model is shown the numbered trace under a blind
   "may or may not contain an error" framing and asked for the first wrong-and-not-
   corrected step (or "no error").
5. Judge: a stronger model on a local vLLM OpenAI-compatible server gets the same
   numbered trace and framing. If the judge is a reasoning model, 
   parse_critique tolerates <think> / reasoning output.
6. Score overlap: exact step match, within-k, and char-IoU of quoted snippets.

Serve any strong instruct/reasoning model with vLLM, e.g.:
    vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 --port 8004
Run:
    python -m error_detect.identify_errors --judge-port 8004 \
        --judge-model Qwen/Qwen3.6-27B
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

from error_detect.segment import Step, render_numbered, segment_trace


DEFAULT_INPUT = "data/incorrect_traces/deepmath_qwen_incorrect.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


SYSTEM_PROMPT = (
    "You are a careful math grader. You are given a problem and a step-by-step "
    "solution attempt whose steps are numbered [0], [1], [2], .... The solution may "
    "or may not contain an error. Find the FIRST step that is mathematically wrong "
    "and is NOT corrected by a later step. If every step is correct, report no error."
)

USER_TEMPLATE = """Problem:
{question}

Numbered solution attempt:
{numbered}

Respond with ONLY a JSON object of this exact form, and nothing after it:
{{"has_error": true or false, "step": integer step index or null, "snippet": "a short verbatim quote copied from that step, or null", "reason": "one short sentence"}}"""


@dataclass
class Critique:
    has_error: bool | None
    step: int | None
    snippet: str | None
    reason: str | None
    parse_ok: bool


@dataclass
class Counters:
    seen: int = 0
    skipped_truncated: int = 0
    skipped_too_long: int = 0
    processed: int = 0
    self_parse_failed: int = 0
    judge_parse_failed: int = 0
    both_no_error: int = 0
    both_error: int = 0
    has_error_agreement: int = 0
    exact_match: int = 0  # among both_error
    within_k: int = 0  # among both_error
    iou_sum: float = 0.0  # among both_error; divide by both_error for mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate first errors in incorrect traces (self-model vs judge)."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Self-critique model.")
    parser.add_argument(
        "--output",
        default="data/identified_errors/deepmath_qwen_errors.jsonl",
    )
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--tolerance-k", type=int, default=2)
    parser.add_argument("--min-step-chars", type=int, default=40)
    parser.add_argument("--max-step-chars", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    # Judge: stronger model on a local vLLM OpenAI-compatible server.
    parser.add_argument("--judge-port", type=int, required=True)
    parser.add_argument("--judge-model", required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prompting and parsing
# ---------------------------------------------------------------------------


def build_messages(question: str, numbered: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, numbered=numbered)},
    ]


def _iter_json_objects(text: str) -> list[str]:
    """Return all balanced top-level ``{...}`` substrings, in order."""
    objects: list[str] = []
    depth = 0
    start = -1
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objects.append(text[start : idx + 1])
    return objects


def parse_critique(text: str, n_steps: int) -> Critique:
    """Take the last balanced JSON object carrying ``has_error`` and validate it.

    A thinking model may emit reasoning (and stray braces) before the answer, so we
    scan all balanced objects and use the last one that parses with a ``has_error`` key.
    """
    for candidate in reversed(_iter_json_objects(text)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "has_error" not in obj:
            continue

        has_error = bool(obj.get("has_error"))
        reason = obj.get("reason")
        reason = str(reason) if reason is not None else None

        if not has_error:
            return Critique(False, None, None, reason, parse_ok=True)

        step = obj.get("step")
        if not isinstance(step, int) or not (0 <= step < n_steps):
            # has_error claimed but step is missing/out of range -> unusable.
            return Critique(None, None, None, reason, parse_ok=False)
        snippet = obj.get("snippet")
        snippet = str(snippet) if snippet is not None else None
        return Critique(True, step, snippet, reason, parse_ok=True)

    return Critique(None, None, None, None, parse_ok=False)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def locate_span(trace: str, step: Step, snippet: str | None) -> tuple[int, int]:
    """Char span for a critique: the quoted snippet if found, else the whole step."""
    if snippet:
        idx = trace.find(snippet, step.char_start, step.char_end)
        if idx < 0:
            idx = trace.find(snippet)
        if idx >= 0:
            return idx, idx + len(snippet)
    return step.char_start, step.char_end


def span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def self_critique_batch(
    llm: LLM, prompts: list[str], args: argparse.Namespace
) -> list[str]:
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]


def make_judge(args: argparse.Namespace):
    from openai import OpenAI

    client = OpenAI(
        base_url=f"http://localhost:{args.judge_port}/v1",
        api_key="EMPTY",
    )
    model = args.judge_model
    print(f"Judge model: {model}", file=sys.stderr)

    def judge(messages: list[dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        # If the judge is a reasoning model, chain-of-thought lands in
        # message.reasoning_content (with a vLLM reasoning parser) or inline as
        # <think>...</think> in content; parse_critique tolerates both.
        return resp.choices[0].message.content or ""

    return judge


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


@dataclass
class Prepared:
    record: dict[str, Any]
    steps: list[Step]
    numbered: str
    messages: list[dict[str, str]]
    self_prompt: str


def resolve_context_len(llm: LLM, tokenizer: Any) -> int:
    """Effective context window of the self model (we never set max_model_len ourselves).

    Prefer the tokenizer's ``model_max_length``; HF uses a huge sentinel when it is
    unset, so fall back to the length vLLM actually resolved from the model config.
    """
    n = getattr(tokenizer, "model_max_length", None)
    if isinstance(n, int) and 0 < n < 10_000_000:
        return n
    return llm.llm_engine.model_config.max_model_len


def prepare(
    records: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    counters: Counters,
    context_len: int,
) -> list[Prepared]:
    prepared: list[Prepared] = []
    budget = context_len - args.max_new_tokens
    for record in records:
        counters.seen += 1
        trace = record["trace"]
        if "</think>" not in trace:
            counters.skipped_truncated += 1
            continue

        steps = segment_trace(trace, args.min_step_chars, args.max_step_chars)
        numbered = render_numbered(steps)
        messages = build_messages(str(record["question"]), numbered)
        self_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if len(tokenizer.encode(self_prompt)) > budget:
            counters.skipped_too_long += 1
            continue

        prepared.append(Prepared(record, steps, numbered, messages, self_prompt))
    return prepared


def score_and_record(
    item: Prepared,
    self_crit: Critique,
    judge_crit: Critique,
    args: argparse.Namespace,
    counters: Counters,
) -> dict[str, Any]:
    record = item.record
    trace = record["trace"]
    counters.processed += 1
    if not self_crit.parse_ok:
        counters.self_parse_failed += 1
    if not judge_crit.parse_ok:
        counters.judge_parse_failed += 1

    exact: bool | None = None
    within_k: bool | None = None
    char_iou: float | None = None

    if self_crit.parse_ok and judge_crit.parse_ok:
        if self_crit.has_error == judge_crit.has_error:
            counters.has_error_agreement += 1
        if not self_crit.has_error and not judge_crit.has_error:
            counters.both_no_error += 1
        elif self_crit.has_error and judge_crit.has_error:
            counters.both_error += 1
            exact = self_crit.step == judge_crit.step
            within_k = abs(self_crit.step - judge_crit.step) <= args.tolerance_k
            self_span = locate_span(trace, item.steps[self_crit.step], self_crit.snippet)
            judge_span = locate_span(trace, item.steps[judge_crit.step], judge_crit.snippet)
            char_iou = span_iou(self_span, judge_span)
            counters.exact_match += int(exact)
            counters.within_k += int(within_k)
            counters.iou_sum += char_iou

    return {
        "problem_id": record.get("problem_id"),
        "row_index": record.get("row_index"),
        "n_steps": len(item.steps),
        "self_has_error": self_crit.has_error,
        "self_step": self_crit.step,
        "self_snippet": self_crit.snippet,
        "self_reason": self_crit.reason,
        "self_parse_ok": self_crit.parse_ok,
        "judge_has_error": judge_crit.has_error,
        "judge_step": judge_crit.step,
        "judge_snippet": judge_crit.snippet,
        "judge_reason": judge_crit.reason,
        "judge_parse_ok": judge_crit.parse_ok,
        "exact": exact,
        "within_k": within_k,
        "char_iou": char_iou,
    }


def print_summary(counters: Counters) -> None:
    data = asdict(counters)
    if counters.both_error:
        data["mean_iou_both_error"] = counters.iou_sum / counters.both_error
        data["exact_rate_both_error"] = counters.exact_match / counters.both_error
        data["within_k_rate_both_error"] = counters.within_k / counters.both_error
    print(json.dumps(data, indent=2), file=sys.stderr)


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(args.input), args.max_traces)

    llm = LLM(
        args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    context_len = resolve_context_len(llm, tokenizer)
    print(f"Self model context length: {context_len}", file=sys.stderr)
    judge = make_judge(args)

    counters = Counters()
    prepared = prepare(records, tokenizer, args, counters, context_len)

    with output_path.open("w", encoding="utf-8") as out:
        for start in range(0, len(prepared), args.batch_size):
            batch = prepared[start : start + args.batch_size]

            self_texts = self_critique_batch(
                llm, [item.self_prompt for item in batch], args
            )
            for item, text in zip(batch, self_texts, strict=True):
                self_crit = parse_critique(text, len(item.steps))
                judge_text = judge(item.messages)
                judge_crit = parse_critique(judge_text, len(item.steps))
                record = score_and_record(item, self_crit, judge_crit, args, counters)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

            print_summary(counters)

    print_summary(counters)


if __name__ == "__main__":
    main()
