r"""
Stage 1 of the trained-self-teacher arm: sample rollouts from the FROZEN student, grade them,
and record the student's own log-probabilities for the sampled tokens.

Run ONCE per (model, dataset). The rollouts are drawn from the student's UN-PRIVILEGED prompt --
the PI only enters when the teacher scores them in stage 2 -- so a single cache serves every
`--pi-mode`. Generation is paid once for the whole PI ladder.

Why the student's logprobs are cached here rather than recomputed in stage 2: the student is
frozen for the entire E-step, so `log pi_theta(y_t | x, y_<t)` is a constant of the problem.
Caching it means stage 2 holds only the teacher in memory and a sweep over objectives / beta /
tau costs one forward per step instead of two. The cached values come from a TRAINING forward
(HF, bf16), not from vLLM's sampler, because that is the quantity stage 2's ratio is defined
against -- vLLM and HF disagree slightly on logprobs, which is the same train/infer mismatch
GRPO corrects for with `importance_sampling_ratio`.

MULTIPLE ROLLOUTS PER PROMPT MATTER. With one rollout per question the teacher can drive the
E-step loss down by reading the QUESTION's difficulty and ignoring the trace entirely -- a
shortcut that looks like success on every calibration metric. Sampling `--n` rollouts means the
same question appears with different outcomes, so difficulty alone cannot explain the label.
The mixed-outcome fraction is reported below, and `--mixed-only` keeps just those questions.
(`value_at_start` in stage 2's diagnostics is the corresponding guard at read time.)

Output: an on-disk HF dataset at data/rollouts/<dataset>/<model-slug>/ with columns
question, final_answer, completion_ids, completion_text, reward, student_logps, n_tokens,
gen_model, dataset.

SCORING RUNS IN A SUBPROCESS, and that is a correctness requirement rather than hygiene.
Initialising vLLM leaves global torch state altered, and freeing the engine does not restore it,
so an HF forward that follows vLLM in the same process gives measurably different logprobs from
the same forward in a clean one (Qwen3-1.7B, identical inputs: 1741/3072 tokens differing, std
0.045, max 0.39). Stage 2 runs without vLLM, so that whole difference would land on rho -- the
`--pi-mode none` control, where rho is analytically zero, reads 0.043 off a same-process cache
and exactly 0.000 off a subprocess-scored one. `--stage all` therefore re-invokes this module
with `--stage score`.

# generate + score (the subprocess split is automatic)
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_self_teacher.gen_rollouts \
    --model Qwen/Qwen3-4B --dataset deepmath --max-samples 1024 --n 4

# or drive the halves by hand, e.g. to put them on different GPUs
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_self_teacher.gen_rollouts --stage generate ...
CUDA_VISIBLE_DEVICES=1 uv run python -m train.opsd.train_self_teacher.gen_rollouts --stage score ...
"""

import argparse
import gc
import os
import subprocess
import sys
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
              "model's ability. Watch `value_at_start` in stage 2.")
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


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="The frozen student. MUST be the model trained in stage 3 -- the rollouts "
                        "are only on-policy for the weights that produced them.")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()))
    p.add_argument("--output-root", default="data/rollouts")
    p.add_argument("--max-samples", type=int, default=1024,
                   help="Questions to roll out. Total rollouts = this x --n.")
    p.add_argument("--questions-from", default="hints", choices=["hints", "dataset"],
                   help="Where the questions come from. 'hints' (default) uses this model's hint "
                        "cache, which is the INTERSECTION every --pi-mode can serve: the hint arm "
                        "can only use questions that have a hint, and the cache keeps ~1/5 of the "
                        "dataset. That both avoids wasting most of the generation pass and gives "
                        "all four PI arms an identical question set. Use 'dataset' only when no "
                        "hint arm is planned; run utils.gen_hints first otherwise.")
    p.add_argument("--n", type=int, default=4,
                   help="Rollouts per question. >1 is what stops the teacher fitting the outcome "
                        "from question difficulty alone; see the module docstring.")
    p.add_argument("--mixed-only", action="store_true",
                   help="Keep only questions that produced BOTH a correct and an incorrect "
                        "rollout. Strongest form of the difficulty-shortcut guard, at the cost of "
                        "dropping the easiest and hardest questions entirely.")
    p.add_argument("--max-completion-length", type=int, default=4096,
                   help="Completion budget. Half the RL arms' 8192 by default: the teacher's "
                        "forward in stage 2 is over [teacher_prompt || completion], and `full` PI "
                        "adds a long prompt on top.")
    p.add_argument("--stage", default="all", choices=["all", "generate", "score"],
                   help="'all' generates here and then runs the scoring half in a FRESH "
                        "SUBPROCESS -- required for correctness, not tidiness: vLLM leaves global "
                        "torch state altered, which changes the HF forward's numerics and would "
                        "bake a ~0.04-nat offset into every cached student logprob. Split the two "
                        "manually if you want them on different GPUs.")
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
                   help="Regenerate / rescore even if a compatible cache exists.")
    p.add_argument("--seed", type=int, default=42)
    # vLLM
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    args = p.parse_args()

    out_dir = rollout_path(args.model, args.dataset, args.output_root)
    print(f"model: {args.model}  dataset: {args.dataset}  ->  {out_dir}")

    if args.stage in ("all", "generate"):
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

    if args.stage in ("all", "score"):
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(
                f"No rollout cache at {out_dir} to score. Run with --stage generate first."
            )
        if args.stage == "all":
            # SCORE IN A FRESH PROCESS. Importing and initialising vLLM mutates global torch
            # state (matmul precision / kernel selection), and it does not undo that when the
            # engine is freed -- so an HF forward that follows a vLLM engine in the same process
            # is numerically DIFFERENT from the same forward in a clean one. Measured on
            # Qwen3-1.7B: identical inputs, same weights, 1741/3072 tokens disagreeing, std 0.045,
            # max 0.39. Those logprobs become the student half of rho, and stage 2 runs without
            # vLLM, so the whole difference lands on the signal: the `--pi-mode none` control,
            # where rho is analytically zero, read 0.043 dispersion off a same-process cache and
            # exactly 0.000 off a subprocess-scored one.
            forwarded = [
                sys.executable, "-m", "train.opsd.train_self_teacher.gen_rollouts", "--stage", "score",
                "--model", args.model, "--dataset", args.dataset,
                "--output-root", args.output_root,
                "--score-batch-size", str(args.score_batch_size),
            ]
            if args.force:
                forwarded.append("--force")
            print(f"Scoring in a clean subprocess (vLLM has perturbed this one's torch state)")
            subprocess.run(forwarded, check=True)
        else:
            score(args, out_dir)


if __name__ == "__main__":
    main()
