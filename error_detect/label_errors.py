"""Judge-label the first uncorrected error in each (incorrect) rollout, by step.

The rollout is segmented deterministically (:mod:`credit_assignment.segment`) into
numbered steps, the steps are shown to a strong judge, and the judge returns the
*index* of the first step containing a mathematical/logical error that is not
corrected later. Because the judge emits an index into our pre-computed steps —
not a quote — its tokenizer is irrelevant and the label maps back to an exact
token range with zero string matching. This is what lets us use a different-family
judge (Qwen3.6-27B, vocab 248044) that could never serve as a same-tokenizer
teacher.

The judge is privileged: it sees the gold final answer and DeepMath's
``r1_solution_1`` reference solution (rejoined by problem_id), so its error
localization is as reliable as possible — the "ground-truth" anchor we later
compare OPD/OPSD per-token credit against.

The judge runs as a separate vLLM server (``vllm serve``); this script is a thin
client that segments the rollouts, builds the judge prompts, and fires them all at
the server's OpenAI-compatible endpoint, letting vLLM batch them. Decoupling the
27B judge from this process makes iterating on the prompt / parsing far quicker.

Run from the repo root, e.g.:
    # 1. start the judge server (separate terminal; keep it up)
    PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1 \
    vllm serve Qwen/Qwen3.6-27B --tensor-parallel-size 2 --port 8000 \
        --max-model-len 40000

    # 2. send the rollouts to it
    python -m credit_assignment.label_errors \
        --rollouts data/credit_assignment/rollouts_Qwen_Qwen3-1.7B_deepmath.jsonl \
        --judge-model Qwen/Qwen3.6-27B --port 8000 --max-model-len 40000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer

from error_detect.score_opsd import build_solution_map
from error_detect.segment import segment_rollout

JUDGE_SYSTEM = (
    "You are an expert mathematician grading a student's step-by-step solution "
    "attempt. You are given the problem, the correct final answer, a correct "
    "reference solution, and the student's attempt split into numbered steps. Your "
    "job is to find the FIRST step that contains a mathematical or logical error "
    "that the student does NOT correct in a later step. A step where the student "
    "explores a wrong idea but later fixes it is NOT the first uncorrected error. "
    "If the whole attempt is mathematically sound (even if messy), report -1."
)

JUDGE_USER_TEMPLATE = """Problem:
{problem}

Correct final answer: {gold_answer}

Correct reference solution:
{solution}

Student attempt, split into numbered steps:
{steps_block}

Find the FIRST step whose reasoning is mathematically or logically wrong and is
not corrected by any later step. Then respond with ONLY a JSON object on the last
line, no extra text:
{{"first_error_step": <int, or -1 if no uncorrected error>, "error_type": "<calculation|algebra|logic|misread|other>", "severity": "<low|medium|high>", "reason": "<one short sentence>"}}"""


def stream_incorrect(path: Path, limit: int | None):
    """Yield up to ``limit`` incorrect rollouts without loading the whole file."""
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("correct"):
                continue
            yield rec
            n += 1
            if limit is not None and n >= limit:
                return


def faithful_step_texts(
    student_tok, comp_ids: list[int], steps: list[dict]
) -> list[str]:
    """Decode each step's token span with the *student* tokenizer (intact unicode)."""
    return [student_tok.decode(comp_ids[s["tok_start"] : s["tok_end"]]) for s in steps]


def build_steps_block(step_texts: list[str]) -> str:
    return "\n\n".join(
        f"### Step {i}\n{txt.strip()}" for i, txt in enumerate(step_texts)
    )


def extract_label(text: str) -> dict | None:
    """Parse the trailing JSON object from the judge output (after any thinking)."""
    start, end = text.rfind("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge first-uncorrected-error per rollout (by step).")
    p.add_argument("--rollouts", required=True, help="JSONL from generate_rollouts.py")
    p.add_argument("--judge-model", default="Qwen/Qwen3.6-27B",
                   help="Model id expected on the server (verified before sending).")
    p.add_argument("--host", default="127.0.0.1", help="vLLM server host")
    p.add_argument("--port", type=int, required=True, help="vLLM server port (from vllm serve)")
    p.add_argument("--api-key", default="EMPTY", help="API key sent to the server")
    p.add_argument("--max-concurrency", type=int, default=256,
                   help="max in-flight requests; the server batches them")
    p.add_argument("--request-timeout", type=float, default=3600.0,
                   help="per-request timeout in seconds (judge thinking can be long)")
    p.add_argument("--max-traces", type=int, default=None, help="cap on incorrect rollouts")
    p.add_argument("--min-tokens", type=int, default=20, help="segmentation min step size")
    p.add_argument("--max-tokens", type=int, default=200, help="segmentation max step size")
    p.add_argument("--max-gen-tokens", type=int, default=32000, help="judge thinking+JSON budget")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="skip prompts whose token_count + max-gen-tokens exceeds this "
                        "(match the server's --max-model-len)")
    p.add_argument("--output", default=None,
                   help="JSONL (default errors_<judge>_<rolloutstem>.jsonl beside rollouts)")
    return p.parse_args()


def default_output(args: argparse.Namespace, rollouts_path: Path) -> Path:
    judge_slug = args.judge_model.replace("/", "_")
    stem = rollouts_path.stem.replace("rollouts_", "")
    return rollouts_path.parent / f"errors_{judge_slug}_{stem}.jsonl"


def build_judge_prompt_ids(judge_tok, problem, gold_answer, solution, steps_block):
    user = JUDGE_USER_TEMPLATE.format(
        problem=problem, gold_answer=gold_answer, solution=solution,
        steps_block=steps_block,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    # transformers 5.x returns a BatchEncoding (dict) when tokenize=True;
    # return_dict=False normalizes to a list, and we still guard for the dict form.
    kwargs = {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
    try:
        ids = judge_tok.apply_chat_template(messages, enable_thinking=True, **kwargs)
    except TypeError:
        ids = judge_tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
    if isinstance(ids, dict):  # BatchEncoding
        ids = ids["input_ids"]
    return [int(t) for t in ids]


def verify_server_model(base_url: str, api_key: str, judge_model: str) -> None:
    """Fail fast unless the vLLM server at ``base_url`` is serving ``judge_model``."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as e:  # refused connection, wrong port, server still loading
        raise SystemExit(f"Could not reach a vLLM server at {base_url}: {e}")
    if judge_model not in served:
        raise SystemExit(
            f"Server at {base_url} is serving {served}, not --judge-model "
            f"{judge_model!r}. Start `vllm serve {judge_model}` or fix --port/--judge-model."
        )
    print(f"Verified judge {judge_model} is served at {base_url}", file=sys.stderr)


async def _judge_all(
    base_url: str,
    api_key: str,
    model: str,
    prompts: list[list[int]],
    max_tokens: int,
    seed: int,
    concurrency: int,
    timeout: float,
) -> list[tuple[str | None, str | None]]:
    """Send every prompt (as token ids) to the server at once; vLLM batches them.

    Returns ``(text, error)`` per prompt in input order. Concurrency is capped by a
    semaphore so we don't open thousands of sockets, and a single failed request
    (e.g. a prompt that slips past the length filter) does not abort the rest.
    """
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
    sem = asyncio.Semaphore(concurrency)

    async def one(ids: list[int]) -> tuple[str | None, str | None]:
        async with sem:
            try:
                resp = await client.completions.create(
                    model=model, prompt=ids, max_tokens=max_tokens, seed=seed,
                )
                return resp.choices[0].text, None
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"

    try:
        return await asyncio.gather(*(one(ids) for ids in prompts))
    finally:
        await client.close()


def main() -> None:
    args = parse_args()
    rollouts_path = Path(args.rollouts)
    output_path = Path(args.output) if args.output else default_output(args, rollouts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(stream_incorrect(rollouts_path, args.max_traces))
    print(f"Loaded {len(records)} incorrect rollouts from {rollouts_path}", file=sys.stderr)
    if not records:
        print("No incorrect rollouts to label.", file=sys.stderr)
        return

    base_url = f"http://{args.host}:{args.port}/v1"
    verify_server_model(base_url, args.api_key, args.judge_model)

    student_model = records[0]["student_model"]
    student_tok = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    # The judge tokenizer is loaded locally only to template + length-count the
    # prompts (the weights live on the server); we send the resulting token ids.
    judge_tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    solution_map = build_solution_map(records)

    # Segment every rollout, build judge prompts; skip those without a reference
    # solution or whose prompt is too long.
    prompts: list = []
    meta: list[dict] = []  # parallel to prompts: {rec, steps}
    n_skip_sol = n_skip_len = 0
    for rec in records:
        sol = solution_map.get(str(rec["problem_id"]), "")
        if not sol.strip():
            n_skip_sol += 1
            continue
        comp_ids = [t["token_id"] for t in rec["tokens"]]
        steps = segment_rollout(rec["tokens"], args.min_tokens, args.max_tokens)
        if not steps:
            continue
        step_texts = faithful_step_texts(student_tok, comp_ids, steps)
        steps_block = build_steps_block(step_texts)
        ids = build_judge_prompt_ids(
            judge_tok, rec["problem"], rec["gold_answer"], sol, steps_block
        )
        if args.max_model_len and len(ids) + args.max_gen_tokens > args.max_model_len:
            n_skip_len += 1
            continue
        prompts.append(ids)
        meta.append({"rec": rec, "steps": steps})

    print(f"Judging {len(prompts)} rollouts "
          f"({n_skip_sol} no-solution, {n_skip_len} too-long, skipped)", file=sys.stderr)

    results = asyncio.run(_judge_all(
        base_url, args.api_key, args.judge_model, prompts,
        max_tokens=args.max_gen_tokens, seed=args.seed,
        concurrency=args.max_concurrency, timeout=args.request_timeout,
    ))

    n_written = n_unparsed = n_error = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for m, (raw, err) in zip(meta, results, strict=True):
            rec, steps = m["rec"], m["steps"]
            label = extract_label(raw) if raw else None
            if err:
                n_error += 1
            elif label is None:
                n_unparsed += 1
            fe = label.get("first_error_step") if label else None
            try:
                fe = int(fe)
            except (TypeError, ValueError):
                fe = None
            in_range = fe is not None and 0 <= fe < len(steps)
            span = (
                [steps[fe]["tok_start"], steps[fe]["tok_end"]] if in_range else None
            )
            handle.write(json.dumps({
                "problem_id": rec["problem_id"],
                "row_index": rec.get("row_index"),
                "correct": rec.get("correct"),
                "student_model": student_model,
                "judge_model": args.judge_model,
                "gold_answer": rec.get("gold_answer"),
                "pred_answer": rec.get("pred_answer"),
                "num_completion_tokens": rec.get("num_completion_tokens"),
                "segmentation": {
                    "min_tokens": args.min_tokens, "max_tokens": args.max_tokens,
                    "n_steps": len(steps),
                },
                "steps": [
                    {"idx": s["idx"], "tok_start": s["tok_start"],
                     "tok_end": s["tok_end"], "n_tokens": s["n_tokens"],
                     "region": s["region"]}
                    for s in steps
                ],
                "first_error_step": fe if in_range else (-1 if fe == -1 else None),
                "first_error_tok_span": span,
                "label": label,
                "judge_error": err,
                "judge_raw": raw,
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} labels ({n_unparsed} unparsed, {n_error} request errors) "
          f"-> {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
