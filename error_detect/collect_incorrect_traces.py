"""Generate math traces and save incorrect responses.

1. Load math training examples from DeepMath-103K.
2. Prompt a small Qwen model to produce a reasoning trace and final answer.
3. Extract the boxed answer and compare against gold with eval.utils.is_equiv
   (same extraction/verification as eval.run_eval).
4. Save only incorrect or unparseable model responses as JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from vllm import LLM, SamplingParams

from eval.utils import extract_boxed_answer, is_equiv


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


@dataclass
class IncorrectTrace:
    dataset: str
    split: str
    problem_id: str
    row_index: int
    model: str
    question: str
    gold_answer: str
    pred_answer: str | None
    response: str
    correct: bool
    num_tokens_generated: int
    verification_status: str
    difficulty: float | None = None
    topic: str | None = None


@dataclass
class Counters:
    seen: int = 0
    generated: int = 0
    saved_incorrect: int = 0
    correct: int = 0
    extraction_failures: int = 0
    generation_failed: int = 0
    skipped_existing: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect incorrect Qwen traces on DeepMath-103K."
    )
    parser.add_argument("--dataset", default= "zwhe99/DeepMath-103K")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--output",
        default="data/incorrect_traces/deepmath_qwen_incorrect.jsonl",
        help="JSONL path containing only incorrect/unparseable model traces.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum dataset rows to attempt after start-index filtering.",
    )
    parser.add_argument(
        "--target-incorrect",
        type=int,
        default=None,
        help="Stop once this many incorrect traces have been saved.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for vLLM tensor parallelism.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory vLLM may use for model weights and KV cache.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip problem_ids already present in the output JSONL.",
    )
    return parser.parse_args()


def load_seen_problem_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem_id = record.get("problem_id")
            if isinstance(problem_id, str):
                seen.add(problem_id)
    return seen


def build_prompt(tokenizer, question, template_tok=None) -> str:

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tok = template_tok or tokenizer
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    return tok.apply_chat_template(messages, **kwargs)


def load_llm_and_tokenizer(args: argparse.Namespace) -> tuple[LLM, Any]:
    llm_kwargs: dict[str, Any] = {}

    llm = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        **llm_kwargs,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def generate_batch(
    llm: LLM,
    prompts: list[str],
    max_new_tokens: int,
    seed: int,
) -> list:
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        seed=seed,
    )
    return llm.generate(prompts, sampling_params)


def write_record(path: Path, record: IncorrectTrace) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def process_batch(
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: Any,
    output_path: Path,
    rows: list[tuple[int, dict[str, Any]]],
    counters: Counters,
) -> None:
    prompts = [build_prompt(tokenizer, row["question"]) for _, row in rows]

    try:
        outputs = generate_batch(
            llm=llm,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        counters.generation_failed += len(rows)
        print(f"generation_failed batch_size={len(rows)} error={exc}", file=sys.stderr)
        return

    for (row_index, row), output in zip(rows, outputs, strict=True):
        counters.generated += 1
        problem_id = f"{args.split}:{row_index}"

        completion = output.outputs[0]
        response = completion.text
        gold_answer = str(row["final_answer"])
        pred_answer = extract_boxed_answer(response)
        correct = is_equiv(pred_answer, gold_answer) if pred_answer else False

        if correct:
            counters.correct += 1
            continue

        if pred_answer is None:
            counters.extraction_failures += 1
        status = "no_answer" if pred_answer is None else "incorrect"

        record = IncorrectTrace(
            dataset=args.dataset,
            split=args.split,
            problem_id=problem_id,
            row_index=row_index,
            model=args.model,
            question=str(row["question"]),
            gold_answer=gold_answer,
            pred_answer=pred_answer,
            response=response,
            correct=correct,
            num_tokens_generated=len(completion.token_ids),
            verification_status=status,
            difficulty=row.get("difficulty"),
            topic=row.get("topic"),
        )
        write_record(output_path, record)
        counters.saved_incorrect += 1


def print_summary(counters: Counters) -> None:
    print(json.dumps(asdict(counters), indent=2), file=sys.stderr)


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_problem_ids = load_seen_problem_ids(output_path) if args.resume else set()

    llm, tokenizer = load_llm_and_tokenizer(args)
    dataset = load_dataset(args.dataset, split=args.split)
    stop_index = len(dataset) if args.max_samples is None else args.start_index + args.max_samples
    stop_index = min(stop_index, len(dataset))

    counters = Counters()
    batch: list[tuple[int, dict[str, Any]]] = []

    for row_index in range(args.start_index, stop_index):
        row = dict(dataset[row_index])
        counters.seen += 1
        problem_id = f"{args.split}:{row_index}"
        if problem_id in seen_problem_ids:
            counters.skipped_existing += 1
            continue

        if "question" not in row or "final_answer" not in row:
            raise KeyError("Expected dataset columns 'question' and 'final_answer'.")

        batch.append((row_index, row))
        if len(batch) < args.batch_size:
            continue

        process_batch(args, llm, tokenizer, output_path, batch, counters)
        batch = []
        print_summary(counters)

        if (
            args.target_incorrect is not None
            and counters.saved_incorrect >= args.target_incorrect
        ):
            break

    if batch and (
        args.target_incorrect is None
        or counters.saved_incorrect < args.target_incorrect
    ):
        process_batch(args, llm, tokenizer, output_path, batch, counters)

    print_summary(counters)


if __name__ == "__main__":
    main()
