r"""
Stage 1 of the trained-self-teacher arm: sample rollouts from the FROZEN student, grade them,
and record the student's own log-probabilities for the sampled tokens.

Run ONCE per (model, dataset). The rollouts are drawn from the student's UN-PRIVILEGED prompt --
the PI only enters when the teacher scores them in stage 2 -- so a single cache serves every
`--pi-mode`.

Why the student's logprobs are cached here rather than recomputed in stage 2: the student is
frozen for the entire E-step, so `log pi_theta(y_t | x, y_<t)` is a constant of the problem.
Caching it means stage 2 holds only the teacher in memory.

MULTIPLE ROLLOUTS PER PROMPT MATTER. With one rollout per question the teacher can drive the
E-step loss down by reading the QUESTION's difficulty and ignoring the trace entirely. Sampling
`--n` rollouts means the same question appears with different outcomes, so stage 2 can measure
within-question discrimination. The mixed-outcome fraction is reported below, and `--mixed-only`
keeps just those questions.

Output: an on-disk HF dataset at data/rollouts/<dataset>/<model-slug>/ with columns
question, final_answer, completion_ids, completion_text, reward, student_logps, n_tokens,
gen_model, dataset.

SCORING ALWAYS RUNS IN A SPAWNED PROCESS, and that is a correctness requirement rather than
hygiene.
Initialising vLLM leaves global torch state altered, and freeing the engine does not restore it,
so an HF forward that follows vLLM in the same process gives measurably different logprobs from
the same forward in a clean one (Qwen3-1.7B, identical inputs: 1741/3072 tokens differing, std
0.045, max 0.39). Stage 2 runs without vLLM, so that whole difference would land on rho -- the
`--pi-mode none` control, where rho is analytically zero, reads 0.043 off a same-process cache
and exactly 0.000 off a subprocess-scored one.

# Generate/reuse rollouts and score them; the clean-process boundary is automatic.
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_self_teacher.gen_rollouts \
    --model Qwen/Qwen3-4B --dataset deepmath --max-samples 1024 --n 4
"""

import argparse
import gc
import multiprocessing
import os
from collections import defaultdict

import torch
from datasets import Dataset, load_from_disk

from train.opsd.train_self_teacher.lib import per_token_logps, rollout_path
from utils import (
    DATASET_REGISTRY_TRAIN,
    format_prompt_math,
    grade,
    load_hint_cache,
    load_train_dataset,
)


def summarize_outcomes(rows: list[dict]) -> tuple[float, float]:
    """(pass rate over rollouts, fraction of questions with BOTH outcomes)."""
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question"]].append(row["reward"])
    mixed = sum(1 for rewards in by_question.values() if 0 < sum(rewards) < len(rewards))
    pass_rate = sum(r["reward"] for r in rows) / max(len(rows), 1)
    return pass_rate, mixed / max(len(by_question), 1)


def generate(args, out_dir: str) -> None:
    """Sample `--n` completions per question from the frozen student and grade each."""
    from vllm import LLM, SamplingParams

    if args.questions_from == "hints":
        # The hint cache is a heavily filtered subset -- gen_hints drops answer leaks, empty
        # hints and unclosed think traces, keeping roughly a fifth of the rows. Drawing questions
        # from the DATASET prefix instead would (a) waste ~80% of this generation pass, since the
        # hint arm can only use rollouts whose question has a hint to join, and (b) hand each
        # --pi-mode a different question set, so the arms would no longer be comparable. Drawing
        # from the cache fixes ONE question set that all four arms can use, which is the same
        # reason eval/passk_pi.py restricts its arms to a common full-PI-feasible subset.
        ds = load_hint_cache(args.model, args.dataset, max_samples=args.max_samples)
        print(f"Loaded {len(ds)} questions from the {args.model} hint cache")
    else:
        ds = load_train_dataset(args.dataset, max_samples=args.max_samples)
        print(f"Loaded {len(ds)} {args.dataset} questions")
    print(f"  rolling out with {args.model}")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # The student prompt, byte-identical to train_grpo / train_sdft / eval: whatever the student
    # is trained to continue is what it is asked to continue here.
    prompts = [
        tokenizer.apply_chat_template(
            format_prompt_math(row["question"]), tokenize=False, add_generation_prompt=True
        )
        for row in ds
    ]
    # Sampling left at vLLM's defaults (temperature 1.0, top_p 1.0, top_k 0), which is the
    # distribution the trainers roll out at -- the rollouts must be on-policy for the student.
    sampling = SamplingParams(n=args.n, max_tokens=args.max_completion_length, seed=args.seed)
    outputs = llm.generate(prompts, sampling)

    rows = []
    for row, output in zip(ds, outputs, strict=True):
        for completion in output.outputs:
            _, correct = grade(completion.text, row["final_answer"])
            rows.append({
                "question": row["question"],
                "final_answer": str(row["final_answer"]),
                "completion_ids": list(completion.token_ids),
                "completion_text": completion.text,
                "reward": float(correct),
                "n_tokens": len(completion.token_ids),
                "gen_model": args.model,
                "dataset": args.dataset,
            })

    pass_rate, mixed_rate = summarize_outcomes(rows)
    print(f"Generated {len(rows)} rollouts over {len(ds)} questions")
    print(f"  pass rate {pass_rate:.3f}  |  questions with mixed outcomes {mixed_rate:.3f}")
    if mixed_rate < 0.2:
        print("  warning: few questions carry both outcomes, so the teacher can fit the label "
              "from question difficulty alone. Raise --n, or pick a dataset slice nearer this "
              "model's ability. Watch `within_question_auc_q*` in stage 2.")
    if args.mixed_only:
        by_question = defaultdict(list)
        for row in rows:
            by_question[row["question"]].append(row["reward"])
        kept = [r for r in rows if 0 < sum(by_question[r["question"]]) < len(by_question[r["question"]])]
        print(f"  --mixed-only: kept {len(kept)}/{len(rows)} rollouts")
        rows = kept
    if not rows:
        raise RuntimeError("No rollouts survived; nothing to cache.")

    Dataset.from_list(rows).save_to_disk(out_dir)
    print(f"Saved rollouts -> {out_dir}")

    # The engine holds the GPU; drop it before the scoring pass loads the HF student.
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def score(args, out_dir: str) -> None:
    """Add `student_logps`: the frozen student's logprob for each sampled token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.trainer.utils import pad

    ds = load_from_disk(out_dir)
    if "student_logps" in ds.column_names:
        if not args.force:
            print(f"{out_dir} already carries student_logps; use --force to rescore.")
            return
        ds = ds.remove_columns("student_logps")  # add_column below refuses to overwrite

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    model.to("cuda")

    # Pre-tokenize the student prompts once; they repeat across a question's rollouts.
    prompt_cache: dict[str, list[int]] = {}

    def student_prompt_ids(question: str) -> list[int]:
        if question not in prompt_cache:
            prompt_cache[question] = tokenizer.apply_chat_template(
                [format_prompt_math(question)],
                add_generation_prompt=True, tokenize=True, return_dict=True,
            )["input_ids"][0]
        return prompt_cache[question]

    all_logps: list[list[float]] = []
    for start in range(0, len(ds), args.score_batch_size):
        rows = ds[start : start + args.score_batch_size]
        prompts = [torch.tensor(student_prompt_ids(q), dtype=torch.long) for q in rows["question"]]
        completions = [torch.tensor(c, dtype=torch.long) for c in rows["completion_ids"]]

        # Same layout as stage 2 and as GRPO: prompt LEFT-padded so its real tokens sit flush
        # against the completion, completion RIGHT-padded so trailing pads fall after every real
        # token. Anything else misaligns the first completion token's score.
        prompt_ids = pad(prompts, padding_value=tokenizer.pad_token_id, padding_side="left")
        prompt_mask = pad([torch.ones_like(p) for p in prompts], padding_value=0, padding_side="left")
        completion_ids = pad(completions, padding_value=tokenizer.pad_token_id, padding_side="right")
        completion_mask = pad(
            [torch.ones_like(c) for c in completions], padding_value=0, padding_side="right"
        )

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(model.device)
        with torch.no_grad():
            logps = per_token_logps(
                model, input_ids, attention_mask, completion_ids.to(model.device)
            ).float().cpu()

        for row_idx, completion in enumerate(completions):
            all_logps.append(logps[row_idx, : len(completion)].tolist())
        if (start // args.score_batch_size) % 50 == 0:
            print(f"  scored {min(start + args.score_batch_size, len(ds))}/{len(ds)}")

    ds = ds.add_column("student_logps", all_logps)
    # save_to_disk cannot overwrite a directory it is reading from; stage through a sibling.
    tmp_dir = out_dir.rstrip("/") + ".scoring"
    ds.save_to_disk(tmp_dir)
    del ds
    import shutil

    shutil.rmtree(out_dir)
    os.rename(tmp_dir, out_dir)
    print(f"Added student_logps -> {out_dir}")


def score_in_clean_process(args, out_dir: str) -> None:
    """Score in a fresh interpreter so vLLM's Torch mutations cannot contaminate logprobs."""
    print("Scoring in a clean subprocess (vLLM has perturbed this process's torch state)")
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=score, args=(args, out_dir))
    process.start()
    process.join()
    exitcode = process.exitcode
    process.close()
    if exitcode != 0:
        raise RuntimeError(f"Rollout scoring subprocess failed with exit code {exitcode}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="The frozen student. MUST be the model trained in stage 3.")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()))
    p.add_argument("--output-root", default="data/rollouts")
    p.add_argument("--max-samples", type=int, default=1024,
                   help="Questions to roll out. Total rollouts = this x --n.")
    p.add_argument("--questions-from", default="hints", choices=["hints", "dataset"],
                   help="Where the questions come from. 'hints' (default) uses this model's hint "
                        "cache, which is the INTERSECTION every --pi-mode can serve.")
    p.add_argument("--n", type=int, default=4,
                   help="Rollouts per question. >1 is what stops the teacher fitting the outcome "
                        "from question difficulty alone; see the module docstring.")
    p.add_argument("--mixed-only", action="store_true",
                   help="Keep only questions that produced BOTH a correct and an incorrect "
                        "rollout. Strongest form of the difficulty-shortcut guard, at the cost of "
                        "dropping the easiest and hardest questions entirely.")
    p.add_argument("--max-completion-length", type=int, default=8192,
                   help="Completion budget.")
    p.add_argument("--score-batch-size", type=int, default=1,
                   help="Rows per scoring forward. LEAVE AT 1 UNLESS YOU HAVE MEASURED THE COST: "
                        "batching pads the shorter rows, and in bfloat16 attention over pad "
                        "positions perturbs the real tokens' logits by ~0.03 nats (measured on "
                        "Qwen3-1.7B: 17 left pads -> std 0.031, max 0.12; 64 pads -> max 0.30). "
                        "That lands directly on rho, whose whole scale is ~0.1. Batching at EQUAL "
                        "width is bit-exact, so the cost is padding, not batching -- but rows here "
                        "are ragged. Stage 2's default forwards are unpadded too, so 1 keeps the "
                        "two stages exactly comparable.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate and rescore even if a compatible cache exists.")
    p.add_argument("--seed", type=int, default=42)
    # vLLM
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    return p


def main():
    args = build_parser().parse_args()

    out_dir = rollout_path(args.model, args.dataset, args.output_root)
    print(f"model: {args.model}  dataset: {args.dataset}  ->  {out_dir}")

    # Reuse guard, matching utils/gen_hints.py: same model, at least as many rollouts.
    wanted = args.max_samples * args.n
    if not args.force and os.path.isdir(out_dir):
        cached = load_from_disk(out_dir)
        if set(cached.unique("gen_model")) == {args.model} and len(cached) >= wanted:
            print(f"Reusing {len(cached)} cached rollouts at {out_dir} (>= {wanted}, model "
                  f"matches). Use --force to regenerate.")
        else:
            generate(args, out_dir)
    else:
        generate(args, out_dir)

    # A spawned interpreter preserves clean HF numerics without exposing separate modes.
    score_in_clean_process(args, out_dir)


if __name__ == "__main__":
    main()
