"""
Train the PPO baseline -- the actor-critic RL contrast to GRPO, in our RLVR setting
(verifiable binary outcome reward, NO reward model, NO reference model, vLLM rollouts).

Where GRPO scores a rollout against a *group-relative* mean baseline (no critic), PPO
learns a **value function** and estimates per-token advantages with GAE. On our sparse
terminal binary reward that learned critic is the whole point: it is the dense per-token
credit-assignment signal, the RL analogue of the OPD/OPSD credit this project studies.

Design -- built ON TOP of GRPOTrainer rather than TRL's classic PPOTrainer, so we inherit
the fast machinery for free and only add the critic:

  * vLLM colocate generation + policy->vLLM weight sync, prompt tokenization, reward-func
    invocation, per-token logprobs, and the clipped-surrogate policy loss are ALL reused
    from GRPOTrainer unchanged. GRPO's `_compute_loss` already accepts per-token (B,T)
    advantages (for subclasses like MiniLLM), so our GAE advantages plug straight in.
  * NO reference model: GRPO builds one only when beta>0; we force beta=0 (matches our
    GRPO baseline). NO reward model: the outcome reward is `trl.rewards.accuracy_reward`
    over the gold `solution` column -- identical to the GRPO baseline, so PPO-vs-GRPO
    isolates exactly the critic.
  * The critic is a SEPARATE value model (policy arch + scalar `.score` head, via
    AutoModelForSequenceClassification num_labels=1), kept out of `self.model` so the
    vLLM weight-sync (which iterates `self.model.named_parameters()`) is untouched.

What we override on GRPOTrainer:
  * `_calculate_rewards`  -- stash the raw (gathered) rewards so we can rebuild the
                            terminal scalar reward for GAE (GRPO discards it after
                            group-normalizing).
  * `_generate_and_score_completions` -- call super() for all the generation/reward/
                            logprob work, then REPLACE the group-normalized advantages
                            with a value-model forward (old values) + GAE (advantages,
                            returns). Stored as extra (B,T) tensors in the batch dict;
                            GRPO's split/shuffle buffering handles them like any other.
  * `_compute_loss`       -- reuse GRPO's policy loss (super) and ADD a clipped value
                            (MSE) loss over the critic's fresh predictions vs returns.
  * `create_optimizer`    -- add the value model's parameters so the critic trains.

GAE: gamma=1.0 (no discounting over a single reasoning episode), lam=0.95. lam is the
PPO-vs-GRPO dial -- lam->1 makes advantages ~ Monte-Carlo return minus the critic baseline
(the interpretable ablation); lam<1 leans on the critic's bootstrap for denser credit.

Loss aggregation (--loss-type) only sets the DENOMINATOR that turns per-token losses into a
scalar, i.e. how tokens are weighted against each other. It matters more here than in GRPO:
GRPO broadcasts one scalar advantage over a sequence's tokens, whereas our A_t genuinely
varies per token, so a length-dependent denominator would rescale exactly the credit the
critic just assigned. Hence only token-uniform options are exposed:
  dapo (default) -- each token equal across the whole optimizer step (num_items_in_batch).
                    Also train_grpo.py's default, so PPO-vs-GRPO differs only in the critic.
  bnpo           -- each token equal within the micro-batch. This is EXACTLY the aggregation
                    TRL's classic PPOTrainer uses (its `masked_mean(pg_loss_max, ~padding_mask)`);
                    dapo is the same idea with the denominator promoted from the micro-batch to
                    the full accumulation window, so bnpo/accum ~= dapo.
Two of GRPO's loss types are deliberately NOT offered, both because they would break the
per-token credit the critic exists to produce:
  grpo    -- per-sequence mean: dividing by each sequence's own length down-weights tokens in
             long traces, rescaling the per-token GAE credit by trace length.
  dr_grpo -- a CONSTANT denominator (B * max_completion_length). Token-uniform, but it ties the
             policy loss's scale to the budget's fill fraction while vf_loss (below) tracks the
             realized token count -- so the pg:vf ratio, i.e. the effective vf_coef, would drift
             as completions get shorter during training.
vf_loss uses the bnpo-style masked_mean -- what classic PPO uses for its value loss too. Since
both surviving loss types share that normalizer (exactly for bnpo, and ~= for dapo, whose global
token count ~= accum * micro-batch count), vf_coef keeps its textbook meaning under either.

Scope (v1): single-GPU (like the 1.7B student baseline). Multi-GPU DDP wrapping of the
separate value model + cross-process grad sync is a follow-up. No resume (the value-model
state is not in the standard HF checkpoint).

# single GPU, colocate vLLM
CUDA_VISIBLE_DEVICES=0 uv run python -m train.train_ppo \
    --model Qwen/Qwen3-1.7B --dataset deepmath --max-samples 8192

aside :  GRPOTrainer also supports custom rollout logic, in case we want to use this later
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForSequenceClassification
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

from train.train_grpo import build_grpo_dataset
from utils import DATASET_REGISTRY_TRAIN, validate_resume


# ---------------------------------------------------------------------------
# Small masked reducers (defined here rather than imported from trl's experimental
# PPO internals, to avoid depending on that module's private surface).
# ---------------------------------------------------------------------------


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten `x` over the masked (real) tokens; matches trl PPO's masked_whiten."""
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    out = (x - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        out = out + mean
    return out


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level GAE over right-padded completions.

    `rewards`, `values`, `mask` are all (B, T) aligned to completion tokens; the terminal
    reward sits at each row's last real token and `values`/`rewards` are zero on padding.
    Because padding tails carry zero reward and zero value, GAE decays to 0 there and the
    bootstrap at the terminal token uses next-value = 0 (episode end). Returns
    (advantages, returns) with returns = advantages + values (the value-fn targets),
    both computed from the RAW (un-whitened) advantages.
    """
    values = values * mask
    T = rewards.size(1)
    lastgaelam = 0.0
    advantages_reversed = []
    for t in reversed(range(T)):
        nextvalues = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig(GRPOConfig):
    """GRPOConfig + the PPO critic/GAE knobs. beta is forced to 0 (no reference model)
    and scale_rewards to 'none' (advantages come from GAE, not group normalization)."""

    gamma: float = field(default=1.0, metadata={"help": "GAE discount (1.0 = no discounting)."})
    lam: float = field(default=0.95, metadata={"help": "GAE lambda."})
    vf_coef: float = field(default=0.1, metadata={"help": "Value-loss weight."})
    cliprange_value: float = field(default=0.2, metadata={"help": "Value-clipping range."})
    whiten_advantages: bool = field(default=True, metadata={"help": "Whiten GAE advantages over the mask."})
    missing_eos_penalty: float = field(
        default=0.0,
        metadata={"help": "Subtract from a completion's terminal reward if it did not end in EOS "
                          "(reasoning truncated at the budget). 0.0 disables."},
    )

    def __post_init__(self):
        # Reference-free objective; advantages are GAE, so no group-normalization.
        self.beta = 0.0
        self.scale_rewards = "none"
        super().__post_init__()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PPOTrainer(GRPOTrainer):
    """GRPOTrainer + a learned critic (separate value model) and GAE advantages."""

    def __init__(self, *args, value_model, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the critic OUT of self.model so vLLM weight-sync (which loads
        # self.model.named_parameters() into the engine) stays a pure-policy sync.
        value_model = value_model.to(self.accelerator.device)
        value_model.train()
        if self.args.gradient_checkpointing:
            value_model.gradient_checkpointing_enable()
            value_model.config.use_cache = False
            # Inputs are token ids (no grad); without this, checkpointing drops the
            # value model's gradients entirely (same fix GRPO applies to the policy).
            value_model.enable_input_require_grads()
        self.value_model = value_model
        self._rewards_per_func_buf = None  # stashed by _calculate_rewards for GAE

    # -- critic parameters into the optimizer -------------------------------

    def create_optimizer(self):
        optimizer = super().create_optimizer()  # built over the policy (self.model)
        value_params = [p for p in self.value_model.parameters() if p.requires_grad]
        optimizer.add_param_group({"params": value_params})
        return optimizer

    # -- per-token value predictions ----------------------------------------

    def _get_per_token_values(self, input_ids, attention_mask, logits_to_keep):
        """Per-token state values aligned to the completion tokens, mirroring
        `_get_per_token_logps_and_entropies`'s shift: run the value backbone, apply the
        scalar `.score` head at every position, drop the last (next-token) position, and
        keep the trailing `logits_to_keep` -> (B, C)."""
        vm = self.value_model
        backbone = getattr(vm, vm.base_model_prefix)
        hidden = backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state  # (B, L, H)
        values = vm.score(hidden).squeeze(-1)  # (B, L)
        values = values[:, :-1]  # drop last position (predicts the token after the sequence)
        values = values[:, -logits_to_keep:]  # (B, C) aligned to completion tokens
        return values

    # -- stash raw rewards for GAE ------------------------------------------

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._rewards_per_func_buf = rewards_per_func  # gathered (B_all, num_funcs)
        return rewards_per_func

    # -- replace group-norm advantages with value + GAE ---------------------

    def _generate_and_score_completions(self, inputs):
        # Clear the stash first so the checks below prove super() populated it for THIS
        # batch, rather than us silently reusing a stale one from a previous step.
        self._rewards_per_func_buf = None
        output = super()._generate_and_score_completions(inputs)
        device = self.accelerator.device

        prompt_ids = output["prompt_ids"]
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        b_local = completion_ids.size(0)
        pidx = self.accelerator.process_index

        # Rebuild the terminal scalar reward from the stashed (gathered) rewards_per_func,
        # then take this process's slice (super() slices `advantages` the same way).
        # The stash is a side-channel: GRPO computes the raw rewards internally and only
        # returns the group-normalized advantages, so we intercept _calculate_rewards.
        # These guards pin the assumption that makes that sound -- super() calls it exactly
        # once per batch -- so a TRL change breaks loudly instead of silently mis-crediting.
        rpf = self._rewards_per_func_buf  # (B_all, num_funcs)
        if rpf is None:
            raise RuntimeError(
                "GRPOTrainer._generate_and_score_completions did not call _calculate_rewards, "
                "so the raw rewards PPO's GAE needs were never stashed. TRL's internals likely "
                "changed; rebuild the terminal reward from the new reward path."
            )
        expected = b_local * self.accelerator.num_processes
        if rpf.size(0) != expected:
            raise RuntimeError(
                f"Stashed rewards batch ({rpf.size(0)}) != expected gathered batch "
                f"({expected} = {b_local} local x {self.accelerator.num_processes} processes). "
                "The reward stash is misaligned with the completions, which would assign each "
                "rollout the wrong terminal reward."
            )
        # A completion whose every reward func returned None is *unscorable* -- there is no
        # verdict on it (e.g. accuracy_reward could not parse the gold). nansum would collapse
        # such a row to 0.0, making it indistinguishable from a graded-WRONG answer, which
        # would push the policy away from a possibly-correct trace. GRPO instead excludes them
        # from its baseline and forces their advantage to 0; we mirror that below via
        # `scorable`, so an unscorable rollout contributes no gradient to either actor or critic.
        unscorable_all = torch.isnan(rpf).all(dim=1)  # (B_all,)
        weights = self.reward_weights.to(rpf.device).unsqueeze(0)
        rewards_all = (rpf * weights).nansum(dim=1)  # (B_all,); all-NaN rows collapse to 0.0
        local = slice(pidx * b_local, (pidx + 1) * b_local)
        rewards = rewards_all[local].to(device)  # (B,)
        scorable = (~unscorable_all[local]).to(device=device, dtype=torch.float32).unsqueeze(1)  # (B, 1)

        # Terminal reward at each row's last real completion token (right-padded).
        seq_len = completion_mask.sum(dim=1).long()  # (B,)
        last_idx = (seq_len - 1).clamp(min=0)  # (B,)
        rows = torch.arange(b_local, device=device)
        if self.args.missing_eos_penalty and self.args.missing_eos_penalty > 0:
            last_tok = completion_ids[rows, last_idx]
            no_eos = last_tok != self._tokenizer.eos_token_id
            rewards = rewards - no_eos.float() * self.args.missing_eos_penalty
        token_rewards = torch.zeros_like(completion_mask, dtype=torch.float32)  # (B, C)
        token_rewards[rows, last_idx] = rewards
        token_rewards = token_rewards * completion_mask

        # Old values from the critic (no grad), then GAE.
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([output["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        with torch.no_grad():
            old_values = self._get_per_token_values(input_ids, attention_mask, logits_to_keep)
        old_values = old_values.float() * completion_mask

        # (B, T) per-token
        advantages, returns = compute_gae(
            token_rewards, old_values, completion_mask, self.args.gamma, self.args.lam
        )
        # Drop unscorable rows from the whitening statistics too (they carry a fabricated
        # reward of 0), then zero them so they yield no policy gradient.
        loss_mask = completion_mask * scorable  # (B, C)
        if self.args.whiten_advantages:
            advantages = masked_whiten(advantages, loss_mask)
        advantages = advantages * loss_mask

        output["advantages"] = advantages  # (B, C) -- GRPO's loss accepts per-token advantages
        output["returns"] = returns
        output["old_values"] = old_values
        output["scorable"] = scorable  # (B, 1) -- masks the value loss in _compute_loss

        # Log critic-side stats. value/returns use loss_mask so the reported means describe
        # the rollouts that actually train the critic. unscorable_rate is expected to be ~0
        # (our golds come from build_grpo_dataset's \boxed{} wrap over answer-bearing rows);
        # a non-zero reading means rollouts are being silently dropped -- worth investigating.
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["ppo/value_mean"].append(masked_mean(old_values, loss_mask).item())
        self._metrics[mode]["ppo/returns_mean"].append(masked_mean(returns, loss_mask).item())
        self._metrics[mode]["ppo/unscorable_rate"].append(unscorable_all.float().mean().item())
        return output

    # -- add the clipped value loss to GRPO's policy loss -------------------

    def _compute_loss(self, model, inputs):
        pg_loss = super()._compute_loss(model, inputs)  # policy loss (already normalized)

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]
        # Unscorable rollouts carry a fabricated reward of 0, so their `returns` are not a
        # real target -- keep them out of the critic's regression (their advantages were
        # already zeroed, so the policy loss ignores them too).
        if "scorable" in inputs:
            mask = mask * inputs["scorable"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        vpred = self._get_per_token_values(input_ids, attention_mask, logits_to_keep).float()
        old_values = inputs["old_values"]
        returns = inputs["returns"]

        vpredclipped = old_values + torch.clamp(
            vpred - old_values, -self.args.cliprange_value, self.args.cliprange_value
        )
        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)

        # Match the policy loss's gradient-accumulation normalization so accumulated
        # grads average correctly (vf_coef absorbs any constant scale difference).
        mode = "train" if self.model.training else "eval"
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        vf_loss = vf_loss / normalizer

        clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), mask)
        self._metrics[mode]["ppo/vf_loss"].append(self.accelerator.gather(vf_loss.detach()).mean().item())
        self._metrics[mode]["ppo/vf_clipfrac"].append(self.accelerator.gather(clipfrac.detach()).mean().item())

        return pg_loss + self.args.vf_coef * vf_loss


# ---------------------------------------------------------------------------
# Resume metadata
# ---------------------------------------------------------------------------


def build_run_meta(args, num_train_examples: int) -> dict:
    return {
        "method": "ppo_vllm",
        "model": args.model,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "reward": "accuracy_reward",
        "gamma": args.gamma,
        "lam": args.lam,
        "vf_coef": args.vf_coef,
        "cliprange_value": args.cliprange_value,
        "loss_type": args.loss_type,
        # resume-critical (see validate_resume): dataset order + batch chunking.
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
    p.add_argument("--model", default="Qwen/Qwen3-1.7B",
                   help="Policy to train; the value model is this arch + a scalar head.")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()),
                   help="Training dataset (see utils.DATASET_REGISTRY_TRAIN).")
    p.add_argument("--output-root", default="outputs/ppo")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>")
    p.add_argument("--max-samples", type=int, default=None, help="Subset the training set")
    # PPO / GAE
    p.add_argument("--gamma", type=float, default=1.0,
                   help="GAE discount. 1.0 = no discounting over the reasoning episode.")
    p.add_argument("--lam", type=float, default=0.95,
                   help="GAE lambda. ->1.0 = Monte-Carlo return minus critic baseline; "
                        "<1.0 leans on the critic bootstrap.")
    p.add_argument("--vf-coef", type=float, default=0.1, help="Value-loss weight.")
    p.add_argument("--cliprange-value", type=float, default=0.2, help="Value-clipping range.")
    p.add_argument("--no-whiten-advantages", dest="whiten_advantages",
                   action="store_false", help="Disable GAE advantage whitening.")
    p.add_argument("--missing-eos-penalty", type=float, default=0.0,
                   help="Subtract from a completion's terminal reward if it did not end in EOS. "
                        "0.0 disables (matches the GRPO baseline).")
    p.add_argument("--num-ppo-epochs", type=int, default=1,
                   help="Gradient passes reusing each rollout (maps to GRPO num_iterations; "
                        ">1 enables PPO's clip-and-reuse via stored old logprobs).")
    p.add_argument("--loss-type", default="dapo", choices=["dapo", "bnpo"],
                   help="Token-loss aggregation for the clipped policy surrogate (see module "
                        "docstring). 'dapo' matches the GRPO baseline; 'bnpo' is TRL classic PPO's "
                        "exact aggregation. Both are token-uniform and share vf_loss's normalizer. "
                        "GRPO's 'grpo' and 'dr_grpo' are excluded: they would rescale or destabilize "
                        "the per-token GAE credit.")
    p.add_argument("--epsilon", type=float, default=0.2, help="PPO clip range (policy).")
    p.add_argument("--temperature", type=float, default=1.0, help="Rollout sampling temperature.")
    # generation
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=2,
                   help="Rollouts per prompt. PPO uses the critic (not a group) as the "
                        "baseline, so grouping is unused -- each rollout gets its own GAE. "
                        "GRPO's config requires >=2, so 2 is the minimum/default.")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    # vLLM
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25,
                   help="Fraction of GPU memory vLLM may reserve (colocate). Keep modest: "
                        "policy + critic + colocate engine share the GPU.")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    # resume
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume dir ('checkpoint-<step>'). Restores the POLICY/optimizer/RNG "
                        "and skips seen data; the critic state is NOT checkpointed, so it "
                        "resets to init on resume -- account for this.")
    p.add_argument("--force-resume", action="store_true")
    args = p.parse_args()

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, args.dataset)
    print(f"model: {model_slug}  dataset: {args.dataset}  ->  output: {output_dir}")

    train_dataset = build_grpo_dataset(args.dataset, max_samples=args.max_samples)
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample prompt: {train_dataset[0]['prompt'][-1]['content'][:120]!r}")
    print(f"  sample solution: {train_dataset[0]['solution']!r}")

    training_args = PPOConfig(
        output_dir=output_dir,
        # PPO / GAE
        gamma=args.gamma,
        lam=args.lam,
        vf_coef=args.vf_coef,
        cliprange_value=args.cliprange_value,
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
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
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

    # Critic: policy arch + a scalar value head, initialised from --model.
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

    trainer = PPOTrainer(
        model=args.model,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=train_dataset,
        value_model=value_model,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)  # saves the policy (eval only needs it)
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()
