"""
PPO with a **PI-conditioned value function**: the critic reads the question *plus privileged
information* (a self-hint, the gold answer, or the worked solution); the policy reads only the
question, and rollouts are still drawn from the un-privileged policy.

Everything except the critic's prompt is inherited from train/train_ppo.py -- same GRPOTrainer
machinery, same accuracy_reward, same GAE, same clipped value loss -- so the arm is structurally
"train_ppo plus PI".

WHY PUT PI IN THE CRITIC (rather than in a teacher, as train_sdft.py does)
The plain critic has to predict P(correct | prefix) from the question alone, which means it has to
essentially solve the problem to score it -- we are asking a 4B critic to be better than the policy
it scores. Given the gold, the task collapses to VERIFICATION ("is this prefix still tracking
toward *this* answer"), which is enormously easier. Three properties make this the right place:

  1. NO EXPLORATION COLLAPSE. In SDFT, PI strength and exploration damage are entangled: the PI
     changes the teacher's GENERATION distribution, and that distribution is the imitation target
     (`full` PI -> 880-token teacher, 98.6% never closing </think> -> a student at 5.2 pass@1). A
     value head has no generation distribution, so there is nothing to imitate and nothing to
     collapse. Here PI strength affects only where credit lands.
  2. UNBIASED. f is fixed at episode start and invisible to the policy, so a_t indep. f | s_t and
     the baseline term vanishes in expectation. (s_t, f) is Markov with rewards measurable w.r.t.
     it, so this is NOT the POMDP asymmetric-critic bias case. What f buys is not information --
     the gold is a deterministic function of the question, which is already in the state -- but
     TRACTABILITY: it is a computational shortcut for a finite-capacity critic.
  3. IT LOCALIZES BLAME EVEN AT lam=1. With A_t = G - V(s_t) and a calibrated critic, a wrong
     rollout whose fatal error is at t* gets A_t ~ -V before the error and ~ 0 after it: the
     thousands of tokens written after the mistake stop being punished. A flat critic (what an
     under-informed one degenerates to) blames every token equally -- i.e. GRPO. V's first
     difference V(y<=t) - V(y<t) is also exactly the counterfactual credit this project treats as
     the ideal signal, so lam dials MC outcome credit <-> learned counterfactual credit.

Hence lam defaults to 1.0 here: no bootstrap, no bias from critic error, and the critic's target is
then exactly the 0/1 outcome at every real completion position -- a clean calibration problem.

WHAT IS ACTUALLY DIFFERENT FROM train_ppo.py
  * `_value_inputs` is overridden to hand the critic `[value_prompt || completion]` instead of
    `[prompt || completion]`. The value prompt is the SAME chat messages with the PI folded into
    the user turn (utils.compose_pi_messages), rendered with add_generation_prompt=True so the
    completion attaches at the same boundary. Alignment is preserved because the critic's values
    are sliced from the END (the trailing `logits_to_keep`), so a longer value prompt shifts
    nothing: `cat([value_prompt_ids, completion_ids])[:, -C:] == completion_ids` regardless of
    how the two prompts differ in length.
  * `--critic-lr` defaults to 1e-5 rather than inheriting the policy's 1e-6. The `.score` head is
    randomly initialised and has to learn P(correct|prefix) from scratch inside --max-steps.
  * Calibration diagnostics (see the metrics block in `_generate_and_score_completions`).

The critic's forward is the only thing that sees the PI. Nothing about generation changes, and
the PI column never reaches the policy or the reward.

READ THE DIAGNOSTICS BEFORE SPENDING AN EVAL. The go/no-go is `ppo/value_brier_q*`: the critic's
squared error against the realized outcome at 25/50/75/100% of each trace. If it does not fall
across quantiles, and does not separate from the no-PI run at the EARLY quantiles, the PI is not
reaching the critic and an AIME eval is premature. `ppo/credit_mass_last_quartile` guards the main
failure mode: a shallow late-firing verifier that only string-matches the answer at the end,
giving late blame rather than process credit (if it fires, `--pi-mode hint` -- approach without
value -- is the designed mitigation, which is also why hint is the default).

KNOWN CONFOUNDS vs the train_ppo 4B lam=1 control: PI vs none, critic lr 1e-5 vs 1e-6, and data
pool (the hint cache is deepmath's ~19.5k solution-bearing subset, not the full 103k). If the
result is ambiguous, run this same script with `--pi-mode none` for an exactly matched control.

RESUME behaves exactly as in train_ppo.py -- read that docstring for the full rules. The two that
bite most often: --max-steps is the TOTAL budget and is free to raise on resume, but LEARNING
RATES COME FROM THE CHECKPOINT (a changed --learning-rate / --critic-learning-rate is refused by
validate_resume rather than silently ignored). Note this arm's example pool is the hint cache, so
one epoch is ~2,438 steps on the 4B cache and ~495 on the 1.7B one -- extend past that and prompts
start repeating. PI-specific: `pi_mode`, `gen_model` and `max_value_prompt_length` are all in
run_meta.json, so resuming a hint run under a different --pi-mode is a hard error rather than a
silent change to what the critic conditions on halfway through a run.

# 4B: vLLM on its own GPU (colocate OOMs from ~4B), policy + critic on another.
CUDA_VISIBLE_DEVICES=7 uv run trl vllm-serve --model Qwen/Qwen3-4B --gpu-memory-utilization 0.9 &
CUDA_VISIBLE_DEVICES=6 uv run python -m train.train_ppo_pi \
    --model Qwen/Qwen3-4B --dataset deepmath --pi-mode hint --vllm-server-port 8000

# smoke (<=1.7B, colocate)
CUDA_VISIBLE_DEVICES=0 uv run python -m train.train_ppo_pi \
    --model Qwen/Qwen3-1.7B --pi-mode hint --vllm-mode colocate \
    --max-steps 2 --max-samples 32 --max-completion-length 256 --save-steps 2
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, set_seed
from trl.rewards import accuracy_reward
from trl.trainer.utils import pad

from train.train_ppo import PPOConfig, PPOTrainer, is_bitsandbytes_optim
from train.train_sdft import PI_ANSWER, PI_FULL, PI_HINT
from utils import (
    DATASET_REGISTRY_TRAIN,
    compose_pi_messages,
    format_prompt_math,
    load_hint_cache,
    load_train_dataset,
    validate_resume,
)


PI_MODES = ("hint", "answer", "full", "none")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_ppo_pi_dataset(
    dataset: str = "deepmath",
    pi_mode: str = "hint",
    model: str | None = None,
    max_samples: int | None = None,
):
    """Columns:
      `prompt`             -- conversational [system, user], identical to train_grpo /
                              train_ppo / eval, so the POLICY's prompt is unchanged by the PI.
      `solution`           -- gold wrapped in \\boxed{} for accuracy_reward (as train_grpo).
      `privileged_context` -- the critic-only string (PI_* from train_sdft, so the PI ladder is
                              worded identically to the SDFT arms). Empty for `--pi-mode none`.

    `hint` draws its rows from the hint cache (every row has a hint), so the example pool is the
    dataset's solution-bearing subset rather than the full set -- see KNOWN CONFOUNDS above.
    `full` needs a worked solution; `answer` and `none` run on the whole pool.
    """
    if pi_mode not in PI_MODES:
        raise ValueError(f"unknown pi_mode {pi_mode!r}; expected one of {PI_MODES}")

    if pi_mode == "hint":
        if model is None:
            raise ValueError("pi_mode='hint' needs --model to locate its hint cache.")
        ds = load_hint_cache(model, dataset, max_samples)
    else:
        ds = load_train_dataset(
            dataset, max_samples=max_samples, require_solution=(pi_mode == "full")
        )

    def _map(row):
        if pi_mode == "hint":
            privileged_context = PI_HINT.format(hint=row["hint"])
        elif pi_mode == "answer":
            privileged_context = PI_ANSWER.format(answer=str(row["final_answer"]))
        elif pi_mode == "full":
            privileged_context = PI_FULL.format(demo=row["solution"])
        else:  # none -- the matched control; the critic sees the plain prompt
            privileged_context = ""
        return {
            "prompt": format_prompt_math(row["question"]),
            "solution": "\\boxed{" + str(row["final_answer"]) + "}",
            "privileged_context": privileged_context,
        }

    return ds.map(_map, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# Trainer
#
# No config subclass: the PI is a dataset column, not a hyperparameter, so PPOConfig is
# unchanged. `pi_mode` is recorded in run_meta.json -- the same place train_sdft.py keeps it,
# and the file validate_resume checks.
# ---------------------------------------------------------------------------


class PPOPITrainer(PPOTrainer):
    """PPOTrainer whose critic is conditioned on privileged information.

    The single override is `_value_inputs`: the critic reads `[value_prompt || completion]` where
    the value prompt carries the PI. Everything else -- rollouts, reward, GAE, the policy loss,
    checkpointing -- is inherited untouched, so this arm differs from train_ppo.py in exactly the
    critic's view.
    """

    def __init__(self, *args, max_value_prompt_length: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_value_prompt_length = max_value_prompt_length
        self._pi_rows = None  # dataset rows for the batch being scored (set below)

    # -- the PI value prompt ------------------------------------------------

    def _generate_and_score_completions(self, inputs):
        # Stash the raw rows BEFORE super() runs: it computes old_values (and therefore calls
        # _value_inputs) inside, so the PI must already be reachable by then. `inputs` is the
        # list of dataset row dicts, row-aligned with the tensors super() builds -- GRPO uses an
        # identity collator and RepeatSampler emits each prompt's num_generations repeats
        # consecutively -- so row i's PI belongs to completion i.
        self._pi_rows = inputs
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            self._pi_rows = None  # never let a stale batch's PI be reused silently
        self._log_calibration_metrics(output)
        return output

    def _value_inputs(self, batch):
        """Override: give the critic the PI-conditioned prompt instead of the policy's.

        Built lazily on the first call (inside super()._generate_and_score_completions) and
        WRITTEN BACK into `batch` -- which is the dict that call returns, so the value prompt
        rides GRPO's shuffle/split buffering alongside every other rollout tensor and is already
        present when `_compute_loss` calls this on a micro-batch slice. That is what guarantees
        `vpred` and `old_values` are computed from the same state, as cliprange_value assumes.
        """
        if "value_prompt_ids" not in batch:
            if self._pi_rows is None:
                raise RuntimeError(
                    "No value prompt in the batch and no rows stashed to build one from. "
                    "_value_inputs must be reached either inside _generate_and_score_completions "
                    "(which stashes the rows) or on a batch that already carries "
                    "'value_prompt_ids'; a TRL change to the buffering may have dropped the key."
                )
            n_rows, n_completions = len(self._pi_rows), batch["completion_ids"].size(0)
            if n_rows != n_completions:
                raise RuntimeError(
                    f"Stashed PI rows ({n_rows}) != completions in the batch ({n_completions}). "
                    "The privileged context would be paired with the wrong rollout, silently "
                    "conditioning each value on another problem's hint."
                )
            ids, mask = self._build_value_prompts(self._pi_rows, batch["completion_ids"].device)
            batch["value_prompt_ids"], batch["value_prompt_mask"] = ids, mask

        input_ids = torch.cat([batch["value_prompt_ids"], batch["completion_ids"]], dim=1)
        attention_mask = torch.cat([batch["value_prompt_mask"], batch["completion_mask"]], dim=1)
        return input_ids, attention_mask, batch["completion_ids"].size(1)

    def _build_value_prompts(self, rows, device):
        """Tokenize the PI-conditioned prompts -> (B, P_v) ids + mask, left-padded.

        Mirrors GRPOTrainer._tokenize_prompts (same chat template, same chat_template_kwargs, same
        add_generation_prompt) so the value prompt differs from the policy prompt ONLY by the PI
        text, and left-pads like GRPO's prompts so prompt and completion stay contiguous.
        """
        # A row that reaches here without the column at all means the PI never survived into the
        # trainer (a dataset built without it, or TRL dropping unused columns). That must be loud:
        # falling back to the plain prompt would silently turn this arm INTO its own control while
        # every log kept saying "pi_mode=hint".
        missing = [i for i, row in enumerate(rows) if "privileged_context" not in row]
        if missing:
            raise RuntimeError(
                f"{len(missing)}/{len(rows)} rows reached the critic without a "
                "'privileged_context' column, so the critic would be conditioned on nothing while "
                "the run still reported a PI mode. Check that the dataset was built by "
                "build_ppo_pi_dataset and that GRPOConfig.remove_unused_columns is False."
            )
        conversations = [
            compose_pi_messages(row["prompt"], row["privileged_context"])
            if row["privileged_context"]
            else row["prompt"]  # legitimately empty only for --pi-mode none
            for row in rows
        ]
        tokenized = self.processing_class.apply_chat_template(
            conversation=conversations,
            tools=self.tools or None,  # mirror _tokenize_prompts exactly
            chat_template=self.chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **self.chat_template_kwargs,
        )
        ids_list = tokenized["input_ids"]

        # No truncation happens upstream (TRL 1.6.0's GRPOTrainer has no max_prompt_length), so
        # the cap is ours. Left-truncate, like SDFT's teacher prompt: the tail (the question and
        # the PI's closing instruction) is what the critic most needs.
        n_truncated = 0
        if self.max_value_prompt_length is not None:
            capped = []
            for ids in ids_list:
                if len(ids) > self.max_value_prompt_length:
                    n_truncated += 1
                    ids = ids[-self.max_value_prompt_length:]
                capped.append(ids)
            ids_list = capped

        n = max(len(ids_list), 1)
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["ppo/value_prompt_len_mean"].append(sum(map(len, ids_list)) / n)
        self._metrics[mode]["ppo/value_prompt_truncated_rate"].append(n_truncated / n)
        # The fraction of rollouts whose critic prompt actually carried privileged info. Must be
        # 1.0 for every --pi-mode except `none` (where it is 0.0). This is the one number that
        # distinguishes "the PI reached the critic" from "the PI silently didn't" -- prompt
        # LENGTH cannot, since a long question with no PI and a short question with one are
        # indistinguishable by length alone.
        self._metrics[mode]["ppo/value_prompt_pi_rate"].append(
            sum(1 for row in rows if row["privileged_context"]) / n
        )

        tensors = [torch.tensor(ids, dtype=torch.long) for ids in ids_list]
        masks = [torch.ones_like(t) for t in tensors]
        ids = pad(tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left").to(device)
        mask = pad(masks, padding_value=0, padding_side="left").to(device)
        return ids, mask

    # -- calibration diagnostics --------------------------------------------

    def _log_calibration_metrics(self, output):
        """Is the critic actually predicting the outcome, and where does its credit land?

        All from tensors already computed -- no extra forward passes. `old_values` are the
        critic's pre-update predictions for this batch, so these describe the critic that
        produced the advantages being trained on.
        """
        mode = "train" if self.model.training else "eval"
        values = output["old_values"]  # (B, C)
        mask = output["completion_mask"] * output["scorable"]  # (B, C)
        outcome = output["terminal_reward"]  # (B, 1) -- 0/1 (minus any missing-EOS penalty)
        lengths = mask.sum(dim=1)  # (B,)
        valid = lengths > 0
        if not valid.any():
            return

        # Brier score at 25/50/75/100% of each row's real length. The headline: with a working
        # PI critic this should FALL across quantiles (the outcome gets more predictable as the
        # trace unfolds) and beat the no-PI run hardest at the EARLY quantiles, which is where
        # the gold buys the most and where dense credit matters.
        for q in (25, 50, 75, 100):
            idx = (lengths.float() * q / 100).ceil().long().clamp(min=1) - 1  # (B,)
            v_q = values.gather(1, idx.unsqueeze(1))  # (B, 1)
            brier = ((v_q - outcome) ** 2)[valid].mean()
            self._metrics[mode][f"ppo/value_brier_q{q}"].append(brier.item())

        # Calibration at the first completion token: V here can only encode "how often does this
        # policy solve this problem", so PI should help LEAST -- a useful negative control.
        self._metrics[mode]["ppo/value_at_start"].append(values[valid, 0].mean().item())
        self._metrics[mode]["ppo/outcome_mean"].append(outcome[valid].mean().item())

        # Where the credit lands. A critic that has learned to string-match the answer only moves
        # at the very end, concentrating |A| in the last quartile: late blame, not process credit.
        advantages = (output["advantages"] * mask).abs()
        total = advantages.sum(dim=1)  # (B,)
        pos = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)  # (1, C)
        last_q = (pos.float() >= 0.75 * lengths.unsqueeze(1).float()) & mask.bool()
        share = (advantages * last_q).sum(dim=1) / total.clamp(min=1e-8)
        keep = valid & (total > 0)
        if keep.any():
            self._metrics[mode]["ppo/credit_mass_last_quartile"].append(share[keep].mean().item())


# ---------------------------------------------------------------------------
# Resume metadata
# ---------------------------------------------------------------------------


def build_run_meta(args, num_train_examples: int) -> dict:
    return {
        "method": "ppo_pi_vllm",
        "model": args.model,
        "pi_mode": args.pi_mode,
        # Hint provenance: hints are only "self" if the trained model wrote them (asserted by
        # utils.load_hint_cache), and the PI is otherwise invisible in the checkpoint.
        "gen_model": args.model if args.pi_mode == "hint" else None,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "reward": "accuracy_reward",
        "gamma": args.gamma,
        "lam": args.lam,
        "vf_coef": args.vf_coef,
        "cliprange_value": args.cliprange_value,
        "critic_max_grad_norm": args.critic_max_grad_norm,
        "max_value_prompt_length": args.max_value_prompt_length,
        "loss_type": args.loss_type,
        "vllm_mode": args.vllm_mode,
        # See train_ppo.build_run_meta: optim x vllm_mode decides whether the policy NaNs.
        "optim": args.optim,
        # resume-critical, learning rates: on resume these come from the CHECKPOINT, not the CLI
        # (see train_ppo.build_run_meta). Recorded so validate_resume errors instead of silently
        # ignoring a changed rate.
        "learning_rate": args.learning_rate,
        "critic_learning_rate": args.critic_learning_rate,
        # resume-critical: dataset order + batch chunking.
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="Policy to train; the critic is this arch + a scalar head. For "
                        "--pi-mode hint this must also be the model that WROTE the hints.")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()))
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/ppo_pi")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>_<pi-mode>")
    p.add_argument("--max-samples", type=int, default=None)
    # privileged context
    p.add_argument("--pi-mode", default="hint", choices=list(PI_MODES),
                   help="What the CRITIC sees on top of the question (the policy never sees it): "
                        "'hint' precomputed self-hints (run train.gen_hints first), 'answer' the "
                        "boxed gold, 'full' the worked solution, 'none' the matched no-PI control.")
    p.add_argument("--max-value-prompt-length", type=int, default=4096,
                   help="Left-truncate the critic's PI prompt to this many tokens. Nothing "
                        "upstream truncates prompts, and 'full' demos are long.")
    # PPO / GAE
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1.0,
                   help="GAE lambda. Defaults to 1.0 here (not train_ppo's 0.95): no bootstrap, "
                        "so critic error cannot bias the advantages, and the critic's target is "
                        "exactly the 0/1 outcome at every position.")
    p.add_argument("--vf-coef", type=float, default=0.1)
    p.add_argument("--cliprange-value", type=float, default=0.2)
    p.add_argument("--critic-max-grad-norm", type=float, default=1.0)
    p.add_argument("--critic-learning-rate", type=float, default=1e-5,
                   help="Learning rate for the critic, SEPARATE from the policy's. Defaults 10x "
                        "the policy's 1e-6: the scalar head is randomly initialised and the whole "
                        "experiment rests on it converging inside --max-steps. Pass the policy's "
                        "rate for strict parity with train_ppo.py.")
    p.add_argument("--no-whiten-advantages", dest="whiten_advantages", action="store_false")
    p.add_argument("--missing-eos-penalty", type=float, default=0.0)
    p.add_argument("--num-ppo-epochs", type=int, default=1)
    p.add_argument("--loss-type", default="dapo", choices=["dapo", "bnpo"])
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=1.0)
    # generation
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=2)
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="adafactor",
                   help="Optimizer. Defaults to adafactor, NOT train_ppo.py's paged_adamw_8bit, "
                        "because this script defaults to --vllm-mode server, where every "
                        "bitsandbytes 8-bit optimizer corrupts the policy on the first "
                        "optimizer step (re-measured 2026-07-23; see train_ppo.py's docstring). "
                        "adamw_torch is also clean but its fp32 moments cost ~64GB across policy "
                        "+ critic at 4B. In colocate mode the 8-bit optimizers are fine.")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    # vLLM
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-mode", default="server", choices=["colocate", "server"],
                   help="Defaults to 'server' (not train_ppo's 'colocate'): this arm targets 4B, "
                        "where a colocate engine OOMs against policy + critic. Launch "
                        "`trl vllm-serve --model <same model>` on its own GPU first.")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                   help="COLOCATE ONLY (default 0.25).")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=None,
                   help="COLOCATE ONLY (default 1).")
    p.add_argument("--vllm-server-host", default="0.0.0.0")
    p.add_argument("--vllm-server-port", type=int, default=8000)
    p.add_argument("--vllm-server-timeout", type=float, default=240.0)
    p.add_argument("--vllm-group-port", type=int, default=51216)
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    # resume
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume dir ('checkpoint-<step>'). Restores policy, critic (weights + "
                        "optimizer state), scheduler and RNG, and skips seen examples. Pass the "
                        "SAME --model, --dataset, --pi-mode, --max-samples, --seed and batch "
                        "config as the original run (verified against its run_meta.json).")
    p.add_argument("--force-resume", action="store_true")
    args = p.parse_args()

    # Same guards as train_ppo.py: these are the server's properties there, and TRL would ignore
    # them silently -- exactly the flags you would reach for after an OOM.
    if args.vllm_mode == "server":
        misplaced = [
            f"{flag} (use `trl vllm-serve {serve_flag}`)"
            for flag, val, serve_flag in (
                ("--vllm-gpu-memory-utilization", args.vllm_gpu_memory_utilization,
                 "--gpu-memory-utilization"),
                ("--vllm-tensor-parallel-size", args.vllm_tensor_parallel_size,
                 "--tensor-parallel-size"),
            )
            if val is not None
        ]
        if misplaced:
            p.error(
                "these only apply to --vllm-mode colocate and would be silently ignored in server "
                "mode, where they are the server's properties: " + "; ".join(misplaced)
            )
    if args.vllm_mode == "server" and is_bitsandbytes_optim(args.optim):
        p.error(
            f"--optim {args.optim} corrupts the policy under --vllm-mode server: finite "
            "gradients, NaN params straight out of optimizer.step(). Use the default "
            "--optim adafactor. See train_ppo.py's module docstring."
        )

    vllm_gpu_mem = 0.25 if args.vllm_gpu_memory_utilization is None else args.vllm_gpu_memory_utilization
    vllm_tp = 1 if args.vllm_tensor_parallel_size is None else args.vllm_tensor_parallel_size

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(
        args.output_root, model_slug, f"{args.dataset}_{args.pi_mode}"
    )
    print(f"model: {model_slug}  dataset: {args.dataset}  pi: {args.pi_mode}  ->  output: {output_dir}")
    if args.vllm_mode == "server":
        print(f"  vLLM: server at {args.vllm_server_host}:{args.vllm_server_port} "
              f"(weight-sync group port {args.vllm_group_port})")

    train_dataset = build_ppo_pi_dataset(
        args.dataset, args.pi_mode, model=args.model, max_samples=args.max_samples
    )
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample prompt: {train_dataset[0]['prompt'][-1]['content'][:120]!r}")
    print(f"  sample solution: {train_dataset[0]['solution']!r}")
    print(f"  sample privileged_context (CRITIC ONLY): "
          f"{train_dataset[0]['privileged_context'][:160]!r}")

    training_args = PPOConfig(
        output_dir=output_dir,
        # PPO / GAE
        gamma=args.gamma,
        lam=args.lam,
        vf_coef=args.vf_coef,
        cliprange_value=args.cliprange_value,
        critic_max_grad_norm=args.critic_max_grad_norm,
        critic_learning_rate=args.critic_learning_rate,
        whiten_advantages=args.whiten_advantages,
        missing_eos_penalty=args.missing_eos_penalty,
        # policy surrogate (inherited GRPO machinery)
        num_iterations=args.num_ppo_epochs,
        loss_type=args.loss_type,
        epsilon=args.epsilon,
        temperature=args.temperature,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # generation backend
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=vllm_gpu_mem,
        vllm_tensor_parallel_size=vllm_tp,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_group_port=args.vllm_group_port,
        # optimization
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        # bookkeeping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    # Seed FIRST: the critic's `.score` head is randomly initialised right here, before Trainer
    # calls set_seed() -- otherwise --seed would not reproduce a run and any A/B over critic
    # hyperparameters would be confounded by a different value function in each arm.
    set_seed(args.seed)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.bfloat16, trust_remote_code=True
    )

    meta = build_run_meta(args, len(train_dataset))
    if args.resume_from_checkpoint:
        validate_resume(args.resume_from_checkpoint, meta, args.force_resume)
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote run metadata -> {os.path.join(output_dir, 'run_meta.json')}")

    trainer = PPOPITrainer(
        model=args.model,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=train_dataset,
        value_model=value_model,
        max_value_prompt_length=args.max_value_prompt_length,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)  # policy only (eval never needs the critic)
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()
