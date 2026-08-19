"""
Design -- built ON TOP of GRPOTrainer rather than TRL's PPOTrainer :

  * vLLM colocate generation + policy->vLLM weight sync, prompt tokenization, reward-func
    invocation, per-token logprobs, and the clipped-surrogate policy loss are reused
    from GRPOTrainer unchanged. GRPO's `_compute_loss` already accepts per-token (B,T)
    advantages (for subclasses like MiniLLM), so our GAE advantages plug straight in.
  * NO reference model: GRPO builds one only when beta>0; we force beta=0. No reward model
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

Critic warmup (--critic-warmup-steps, default 0 = off) freezes the policy for the first N
optimizer steps so the randomly-initialised `.score` head can fit P(correct|prefix) against
rollouts from a FIXED policy before its predictions start steering that policy. Without it the
first updates hand the actor advantages from a critic that is still near its random init.
Warmup steps are counted against --max-steps, so `--critic-warmup-steps 20 --max-steps 200`
leaves the policy 180 updates.

GAE: gamma=1.0 (no discounting over a single reasoning episode)
lam->1 makes advantages ~ Monte-Carlo return minus the critic baseline,
lam<1 leans on the critic's bootstrap.

Loss aggregation (--loss-type) only sets the DENOMINATOR that turns per-token losses into a
scalar, i.e. how tokens are weighted against each other. It matters more here than in GRPO:
GRPO broadcasts one scalar advantage over a sequence's tokens, whereas our A_t varies per token,
so a length-dependent denominator would rescale exactly the credit the critic just assigned.
Hence only token-uniform options are exposed:
  dapo (default) -- each token equal across the whole optimizer step (num_items_in_batch).
                    Also train_grpo.py's default, so PPO-vs-GRPO differs only in the critic.
  bnpo           -- each token equal within the micro-batch. This is the aggregation
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
both surviving loss types share that normalizer, vf_coef keeps its textbook meaning under either.

Resume (--resume-from-checkpoint) restores the policy/optimizer/scheduler/RNG, skips seen
data, and restores the CRITIC too.

What resume does and does not let you change:
  * --max-steps IS free to change --
  * LEARNING RATES COME FROM THE CHECKPOINT, not the command line. To change a rate, start a
    new run.
  * Changing --max-steps is only safe with lr_scheduler_type=constant (the default).
  * Extending far enough re-enters the dataset.

!! vLLM SERVER MODE + ANY bitsandbytes 8-bit OPTIMIZER CORRUPTS THE POLICY.  Use adafactor.  !!
Prefer adafactor for 4B+: adamw_torch's fp32 moments cost ~8 bytes/param
`_assert_policy_finite` (below) now catches this at the step it happens.

# single GPU, colocate vLLM (<=1.7B)
CUDA_VISIBLE_DEVICES=0 uv run python -m train.ppo.train_ppo \
    --model Qwen/Qwen3-1.7B --dataset deepmath --max-samples 8192

# 4B+: vLLM on its own GPU, policy + critic on another.
CUDA_VISIBLE_DEVICES=7 uv run trl vllm-serve --model Qwen/Qwen3-4B --gpu-memory-utilization 0.9 &
CUDA_VISIBLE_DEVICES=6 uv run python -m train.ppo.train_ppo \
    --model Qwen/Qwen3-4B --dataset deepmath --vllm-mode server --vllm-server-port 8000

"""

import argparse
import json
import os
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForSequenceClassification, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward
from trl.trainer.utils import pad

from train.grpo.train_grpo import build_grpo_dataset
from utils import DATASET_REGISTRY_TRAIN, validate_resume


# ---------------------------------------------------------------------------
# Small masked reducers
# ---------------------------------------------------------------------------


# Critic weights inside each `checkpoint-<step>` dir
VALUE_MODEL_FILE = "value_model.pt"


def is_bitsandbytes_optim(optim: str) -> bool:
    """True for bitsandbytes optimizers, which are unsafe under --vllm-mode server.

    Both 8-bit variants tested (`adamw_bnb_8bit`, `paged_adamw_8bit`) corrupt the policy in the
    optimizer step; the other bnb variants are untested. Matched by NAME.
    """
    o = optim.lower()
    return "8bit" in o or "bnb" in o or "paged" in o


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

    `rewards`, `values`, `mask` are all (B, T) aligned to completion tokens.
    Returns (advantages, returns) with returns = advantages + values (the value-fn targets),
    both computed from un-whitened advantages.
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
    lam: float = field(default=1.0, metadata={"help": "GAE lambda."})
    vf_coef: float = field(default=0.1, metadata={"help": "Value-loss weight."})
    critic_learning_rate: float | None = field(
        default=None,
        metadata={"help": "Learning rate for the critic's parameter group. None inherits the "
                          "policy's `learning_rate` -- which is the RLVR-appropriate 1e-6, far "
                          "below what the randomly-initialised `.score` head needs to converge "
                          "inside a 200-step run."},
    )
    cliprange_value: float = field(default=0.2, metadata={"help": "Value-clipping range."})
    critic_max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Grad-norm clip for the critic, applied SEPARATELY from the policy's "
                          "max_grad_norm. 0.0 measures the norm without clipping."},
    )
    whiten_advantages: bool = field(default=True, metadata={"help": "Whiten GAE advantages over the mask."})
    critic_warmup_steps: int = field(
        default=0,
        metadata={"help": "Optimizer steps at the start of training during which the policy is "
                          "frozen and only the critic trains. Counted against max_steps. 0 "
                          "disables (joint training from step 0)."},
    )
    missing_eos_penalty: float = field(
        default=0.0,
        metadata={"help": "Subtract from a completion's terminal reward if it did not end in EOS "
                          "(reasoning truncated at the budget). 0.0 disables."},
    )

    def __post_init__(self):
        # Reference-free objective; advantages are GAE, so no group-normalization.
        self.beta = 0.0
        self.scale_rewards = "none"

        if self.critic_warmup_steps < 0:
            raise ValueError(
                f"critic_warmup_steps must be >= 0; got {self.critic_warmup_steps}."
            )

        requested_num_generations = self.num_generations
        if requested_num_generations < 1:
            raise ValueError(
                "PPO requires num_generations to be at least 1; "
                f"got {requested_num_generations}."
            )

        # GRPOConfig rejects num_generations=1 because ordinary GRPO needs at least two
        # completions to form a within-prompt baseline. PPO does not: below we replace
        # GRPO's advantages with critic-based GAE. Temporarily satisfy that GRPO-only
        # validation, then restore the value that the sampler and trainer should use.
        #
        # The inherited validation therefore still checks divisibility as if G=2. This is
        # intentionally acceptable for this script's even generation batches (16 by
        # default), and avoids copying TRL's large, version-sensitive __post_init__.
        self.num_generations = max(2, requested_num_generations)
        try:
            super().__post_init__()
        finally:
            # Restore even when another parent validation fails, so an exception never
            # leaves a partially constructed config reporting the wrong value.
            self.num_generations = requested_num_generations


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

    def create_optimizer(self, *args, **kwargs):
        # Appending the critic's params here is also
        # what makes the critic's optimizer STATE checkpoint for free.
        optimizer = super().create_optimizer(*args, **kwargs)  # built over the policy (self.model)
        value_params = [p for p in self.value_model.parameters() if p.requires_grad]
        # The lr is set here.
        group = {"params": value_params}
        if self.args.critic_learning_rate is not None:
            group["lr"] = self.args.critic_learning_rate
        optimizer.add_param_group(group)
        # 3 groups : decay, no decay, critic
        return optimizer

    # -- critic gradient clipping + norm logging ----------------------------

    def _clip_grad_norm(self, model):
        """Clip the policy via super(), then the critic -- with its own, separate budget.

        Trainer only ever clips `self.model`, and the critic deliberately lives outside it
        (see __init__), so without this the policy is bounded and the critic is not.

        Adam already normalizes per-parameter, so the clip is not what bounds the step size;
        it exists to stop an outlier spike from polluting the second-moment estimate and
        distorting the update direction for many steps after. `critic_max_grad_norm=0`
        measures without clipping, so the two can be compared.

        Trainer calls this only when max_grad_norm > 0; this script never sets it otherwise.
        """
        policy_grad_norm = super()._clip_grad_norm(model)

        params = [p for p in self.value_model.parameters() if p.grad is not None]
        if params:
            limit = self.args.critic_max_grad_norm
            # clip_grad_norm_ returns the norm from BEFORE any scaling, so the metric is the
            # true pre-clip norm either way
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                params, limit if limit and limit > 0 else float("inf")
            )
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["ppo/critic_grad_norm"].append(critic_grad_norm.item())
        return policy_grad_norm

    # -- critic checkpointing -----------------------------------------------
    #
    # Trainer only saves `self.model`, so the critic's WEIGHTS need handling here; its
    # optimizer state already rides along (see create_optimizer).

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)  # GRPO's override (model card) -> HF's
        if self.args.should_save:  # rank 0 only
            ckpt_dir = os.path.join(
                self._get_output_dir(trial), f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            )
            torch.save(self.value_model.state_dict(), os.path.join(ckpt_dir, VALUE_MODEL_FILE))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        path = os.path.join(resume_from_checkpoint, VALUE_MODEL_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No critic weights ({VALUE_MODEL_FILE}) in {resume_from_checkpoint}. Resuming "
                "would restart the value function from its random init while the policy carries "
                "on. Was this checkpoint written before critic checkpointing existed?"
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Load IN-PLACE. The optimizer holds references to these exact tensors, so the module
        # must not be reassigned.
        self.value_model.load_state_dict(state)
        print(f"Restored critic weights <- {path}")

    # -- end-of-step: reset the critic's grads, catch a corrupted policy ------

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        """Trainer calls this immediately after `optimizer.step()` and `model.zero_grad()`, which
        is both where the server-mode optimizer corruption is born (module docstring) and the only
        seam at which the critic's gradients can be cleared.
        Cost is ~one small kernel per parameter per optimizer step, negligible against a step
        that spends tens of seconds in generation.
        """
        self._assert_policy_finite()
        self.value_model.zero_grad(set_to_none=True)
        return super()._maybe_log_save_evaluate(*args, **kwargs)

    def _assert_policy_finite(self):
        bad = [n for n, p in self.model.named_parameters() if not torch.isfinite(p).all()]
        if not bad:
            return
        raise RuntimeError(
            f"Policy corrupted: {len(bad)}/{sum(1 for _ in self.model.parameters())} parameters "
            f"are non-finite immediately after optimizer.step() at step {self.state.global_step} "
            f"(first: {bad[:3]}). Gradients are typically FINITE here -- the corruption is in the "
            f"optimizer, not the loss. Known cause: a bitsandbytes 8-bit optimizer "
            f"(--optim {self.args.optim}) while --vllm-mode server holds an NCCL weight-sync "
            f"group. Use --optim adafactor. See the module docstring."
        )

    # -- critic warmup -------------------------------------------------------

    @property
    def in_critic_warmup(self) -> bool:
        """True while only the critic trains (see `_compute_loss`).

        `state.global_step` counts COMPLETED optimizer steps and is incremented after
        `optimizer.step()`, so it holds k throughout every micro-batch of optimizer step k:
        `< critic_warmup_steps` therefore covers exactly that many steps. It is also restored
        by `--resume-from-checkpoint`, so a resumed run does not repeat a warmup it finished.
        """
        return (
            self.args.critic_warmup_steps > 0
            and self.model.training
            and self.state.global_step < self.args.critic_warmup_steps
        )

    # -- per-token value predictions ----------------------------------------

    def _value_inputs(self, batch):
        """What the critic reads: (input_ids, attention_mask, logits_to_keep)."""
        input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        return input_ids, attention_mask, batch["completion_ids"].size(1)

    def _render_value_prompts(self, conversations, device, max_length=None):
        """Tokenize critic-only conversations exactly as GRPO tokenizes policy prompts.

        Unused by this trainer -- its critic reads the policy's already-tokenized
        `prompt_ids` -- but it is shared with train_ppo_val.py, whose critic has its own
        prompt.
        Kept here, on the base class, for two reasons:

          * It mirrors GRPOTrainer._tokenize_prompts (same chat_template, chat_template_kwargs,
            tools, add_generation_prompt=True) so the critic's prompt ends on the IDENTICAL
            assistant/<think> header the completion was sampled after.
          * LEFT padding is not a style choice. `_get_per_token_values` reads values[:, 0] from
            the position just before the first completion token.

        `max_length` left-truncates. Returns (ids, mask, n_truncated).
        """
        tokenized = self.processing_class.apply_chat_template(
            conversation=conversations,
            tools=self.tools or None,
            chat_template=self.chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **self.chat_template_kwargs,
        )
        ids_list = tokenized["input_ids"]

        n_truncated = 0
        if max_length is not None:
            capped = []
            for ids in ids_list:
                if len(ids) > max_length:
                    n_truncated += 1
                    ids = ids[-max_length:]
                capped.append(ids)
            ids_list = capped

        mode = "train" if self.model.training else "eval"
        n = max(len(ids_list), 1)
        self._metrics[mode]["ppo/value_prompt_len_mean"].append(sum(map(len, ids_list)) / n)

        tensors = [torch.tensor(ids, dtype=torch.long) for ids in ids_list]
        masks = [torch.ones_like(tensor) for tensor in tensors]
        ids = pad(
            tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left"
        ).to(device)
        mask = pad(masks, padding_value=0, padding_side="left").to(device)
        return ids, mask, n_truncated

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
        # inputs -> solution, prompt
        # Clear the stash first so the checks below prove super() populated it for THIS
        # batch, rather than us reusing a stale one from a previous step.
        self._rewards_per_func_buf = None
        # each input repeated num_generations times to match GRPOConfig
        # update : default is now 1, no repeated examples
        # output contains non-zero GRPO advantage if num_generations > 1
        output = super()._generate_and_score_completions(inputs)
        device = self.accelerator.device

        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        b_local = completion_ids.size(0)
        pidx = self.accelerator.process_index

        # Rebuild the terminal scalar reward from the stashed (gathered) rewards_per_func,
        # then take this process's slice (super() slices `advantages` the same way).
        # The stash is a side-channel: GRPO computes the raw rewards internally and only
        # returns the group-normalized advantages, so we intercept _calculate_rewards.
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
        # A completion whose every reward func returned None is unscorable.
        # GRPO excludes them from its baseline and forces their advantage to 0;
        # we mirror that below via `scorable`,
        # so an unscorable rollout contributes no gradient to either actor or critic.
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
        # Note : The current critic currently doesn't use the value model
        # as an instruction tuned language model
        with torch.no_grad():
            old_values = self._get_per_token_values(*self._value_inputs(output))
        old_values = old_values.float() * completion_mask

        # (B, T) per-token
        advantages, returns = compute_gae(
            token_rewards, old_values, completion_mask, self.args.gamma, self.args.lam
        )
        # Drop unscorable rows from the whitening statistics
        # then zero them so they yield no policy gradient.
        loss_mask = completion_mask * scorable  # (B, C)
        if self.args.whiten_advantages:
            advantages = masked_whiten(advantages, loss_mask)
        advantages = advantages * loss_mask

        output["advantages"] = advantages  # (B, C) -- GRPO's loss accepts per-token advantages
        output["returns"] = returns
        output["old_values"] = old_values
        output["scorable"] = scorable  # (B, 1) -- masks the value loss in _compute_loss
        # The per-rollout outcome the critic is regressing towards
        output["terminal_reward"] = rewards.unsqueeze(1)  # (B, 1)

        # Log critic-side stats
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["ppo/value_mean"].append(masked_mean(old_values, loss_mask).item())
        self._metrics[mode]["ppo/returns_mean"].append(masked_mean(returns, loss_mask).item())
        self._metrics[mode]["ppo/unscorable_rate"].append(unscorable_all.float().mean().item())
        return output

    # -- add the clipped value loss to GRPO's policy loss -------------------

    def _compute_loss(self, model, inputs):
        pg_loss = super()._compute_loss(model, inputs)  # policy loss (already normalized)

        # Warmup: keep the surrogate in the graph but scale it to zero, so the policy receives
        # exactly-zero gradients while the rollout, reward, GAE and logging paths stay identical
        # to a normal step. GRPO's own metrics (clip ratio, entropy) are recorded by the super()
        # call above and therefore still report what the policy WOULD have done.
        # This does not skip the policy's backward pass -- it costs the same as a real step. The
        # optimizer still steps on zero grads, which is a no-op for adafactor/Adam at the default
        # weight_decay=0; a nonzero weight decay would still shrink the frozen policy.
        warmup = self.in_critic_warmup
        if warmup:
            pg_loss = pg_loss * 0.0

        completion_mask = inputs["completion_mask"]
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]
        # Unscorable rollouts carry a reward of 0, so their `returns` are not a
        # real target -- keep them out of the critic's regression.
        # Their advantages were already zeroed, so the policy loss ignores them too.
        if "scorable" in inputs:
            mask = mask * inputs["scorable"]

        vpred = self._get_per_token_values(*self._value_inputs(inputs)).float()
        old_values = inputs["old_values"]
        returns = inputs["returns"]

        vpredclipped = old_values + torch.clamp(
            vpred - old_values, -self.args.cliprange_value, self.args.cliprange_value
        )
        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss_max = torch.max(vf_losses1, vf_losses2)

        # Normalize EXACTLY as the policy loss does for this loss_type, so the critic weights
        # tokens against each other the same way the actor does.
        #   dapo -- every token equal across the whole optimizer step. `num_items_in_batch` is the
        #           generation batch's total completion token count (a 0-dim tensor, which TRL's
        #           split/shuffle pass through unchanged), so summing the micro-batches gives one
        #           token-uniform mean and no /grad_accum belongs here.
        #   bnpo -- every token equal within the micro-batch, then /grad_accum.
        # Both used to take the bnpo form. Under dapo that made the critic's weighting
        # SEQUENCE-uniform -- token t of rollout i carried weight 1/(accum * L_i) -- while the
        # policy's stayed token-uniform. At per_device_train_batch_size=1 a micro-batch IS one
        # rollout, so with completion lengths spanning ~2.5k-8.2k, tokens in short traces got up
        # to ~3x the weight of tokens in long ones: exactly the length-rescaling of per-token
        # credit that excludes the 'grpo' loss type (module docstring), applied to the critic.
        # Both forms are a mean of squared errors, so vf_coef keeps its previous magnitude.
        mode = "train" if self.model.training else "eval"
        if self.loss_type == "dapo":
            vf_loss = 0.5 * (vf_loss_max * mask).sum() / (
                inputs["num_items_in_batch"] / self.accelerator.num_processes
            )
        else:  # bnpo -- the only other type this script exposes
            vf_loss = 0.5 * masked_mean(vf_loss_max, mask)
            vf_loss = vf_loss / (self.current_gradient_accumulation_steps if mode == "train" else 1.0)

        clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), mask)
        # Logged every step (not just during warmup) so the transition is visible as an edge in
        # TensorBoard rather than a gap, and so runs with warmup disabled still plot a flat 0.
        self._metrics[mode]["ppo/critic_warmup"].append(1.0 if warmup else 0.0)
        self._metrics[mode]["ppo/vf_loss"].append(self.accelerator.gather(vf_loss.detach()).mean().item())
        self._metrics[mode]["ppo/vf_clipfrac"].append(self.accelerator.gather(clipfrac.detach()).mean().item())

        return pg_loss + self.args.vf_coef * vf_loss


# ---------------------------------------------------------------------------
# Resume metadata
# ---------------------------------------------------------------------------


def build_run_meta(args, num_train_examples: int) -> dict:
    return {
        "method": "ppo_vllm",
        # Critic-state identity, recorded as None because THIS trainer's critic reads the
        # policy's prompt. It is here so the comparison FIRES: validate_resume only iterates
        # the keys of the meta it is handed, so without this key a checkpoint written by
        # train_ppo_val.py (whose meta stamps 'verifier_v1') would resume here silently -- a
        # critic trained to score verifier-framed prefixes, now fed policy prompts, with
        # every log still looking healthy. `method` alone would not catch it either if the
        # arms are ever renamed. The sibling arms stamp their own value.
        "value_prompt_version": None,
        "model": args.model,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "reward": "accuracy_reward",
        "gamma": args.gamma,
        "lam": args.lam,
        "vf_coef": args.vf_coef,
        "cliprange_value": args.cliprange_value,
        "critic_max_grad_norm": args.critic_max_grad_norm,
        # resume-critical: warmup is keyed on the RESTORED global_step, so a resumed run
        # silently trains jointly if it finished warmup -- recording it makes a changed
        # value an error rather than an invisible difference between the two halves.
        "critic_warmup_steps": args.critic_warmup_steps,
        "loss_type": args.loss_type,
        "vllm_mode": args.vllm_mode,
        # Recorded because the optimizer x vllm_mode combination decides whether the policy
        # silently NaNs (module docstring), and a crashed run leaves no training_args.bin to
        # read it back from.
        "optim": args.optim,
        # resume-critical, learning rates: on resume these come from the CHECKPOINT, not the CLI.
        # optimizer.load_state_dict restores each param group's `lr` and `initial_lr`, and it runs
        # after create_scheduler, so a different --learning-rate on the command line is silently
        # ignored. Recording both here turns that into a validate_resume error.
        "learning_rate": args.learning_rate,
        "critic_learning_rate": args.critic_learning_rate,
        # resume-critical: dataset order + batch chunking
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
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/ppo")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>")
    p.add_argument("--max-samples", type=int, default=None, help="Subset the training set")
    # PPO / GAE
    p.add_argument("--gamma", type=float, default=1.0,
                   help="GAE discount. 1.0 = no discounting over the reasoning episode.")
    p.add_argument("--lam", type=float, default=1.0,
                   help="GAE lambda. ->1.0 = Monte-Carlo return minus critic baseline; "
                        "<1.0 leans on the critic bootstrap.")
    p.add_argument("--vf-coef", type=float, default=0.1, help="Value-loss weight.")
    p.add_argument("--cliprange-value", type=float, default=0.2, help="Value-clipping range.")
    p.add_argument("--critic-max-grad-norm", type=float, default=1.0,
                   help="Clip the critic's gradients to this norm separately from the policy "
                        "(which Trainer clips to max_grad_norm=1.0). Pass 0 to disable clipping")
    p.add_argument("--critic-warmup-steps", type=int, default=0,
                   help="Freeze the policy for this many optimizer steps at the start of "
                        "training so the randomly-initialised value head can fit against a "
                        "FIXED policy before its advantages start steering one. Counted "
                        "against --max-steps. 0 = joint training from step 0.")
    p.add_argument("--no-whiten-advantages", dest="whiten_advantages",
                   action="store_false", help="Disable GAE advantage whitening.")
    p.add_argument("--missing-eos-penalty", type=float, default=0.0,
                   help="Subtract from a completion's terminal reward if it did not end in EOS. "
                        "0.0 disables (matches the GRPO baseline).")
    p.add_argument("--num-ppo-epochs", type=int, default=1,
                   help=">1 enables PPO's clip-and-reuse via stored old logprobs.")
    p.add_argument("--loss-type", default="dapo", choices=["dapo", "bnpo"],
                   help="Token-loss aggregation for the clipped policy surrogate (see module "
                        "docstring). 'dapo' matches the GRPO baseline; 'bnpo' is TRL classic PPO's "
                        "exact aggregation. Both are token-uniform and share vf_loss's normalizer. "
                        "GRPO's 'grpo' and 'dr_grpo' are excluded.")
    p.add_argument("--epsilon", type=float, default=0.2, help="PPO clip range (policy).")
    p.add_argument("--temperature", type=float, default=1.0, help="Rollout sampling temperature.")
    # generation
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=1,
                   help="Rollouts per prompt. PPO uses the critic as the "
                        "baseline, so grouping is unused -- each rollout gets its own GAE. ")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--critic-learning-rate", type=float, default=None,
                   help="Learning rate for the critic. Omit to inherit --learning-rate. That "
                        "default (1e-6) suits a pretrained policy but might be very slow for the "
                        "critic's scalar head")
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="paged_adamw_8bit",
                   help="Optimizer. The 8-bit default keeps the memory footprint of the GRPO "
                        "baseline and is fine in COLOCATE mode. In --vllm-mode server EVERY "
                        "bitsandbytes 8-bit optimizer corrupts the policy; "
                        "pass --optim adafactor there.")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    # vLLM
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"],
                   help="'colocate': run the engine in-process on the same GPU, OOMs from ~4B."
                        "'server': talk to a standalone `trl vllm-serve` on its own GPU.")
    # Colocate-only. Defaults are None so we can tell "user set it" from "left alone" and
    # reject it in server mode, where it is the SERVER's property (see main()).
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                   help="COLOCATE ONLY (default 0.25)")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=None,
                   help="COLOCATE ONLY (default 1). In server mode pass --tensor-parallel-size to "
                        "`trl vllm-serve` instead.")
    p.add_argument("--vllm-server-host", default="0.0.0.0", help="SERVER MODE: vLLM server host.")
    p.add_argument("--vllm-server-port", type=int, default=8000, help="SERVER MODE: vLLM server port.")
    p.add_argument("--vllm-server-timeout", type=float, default=240.0,
                   help="SERVER MODE: seconds to wait for the server to be reachable.")
    p.add_argument("--vllm-group-port", type=int, default=51216,
                   help="SERVER MODE: port for the NCCL weight-sync group.")
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    # resume
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume dir ('checkpoint-<step>'). Restores policy, critic (weights + "
                        "optimizer state), scheduler and RNG, and skips already-seen examples. "
                        "Pass the SAME --model, --dataset, --max-samples, --seed and batch config "
                        "as the original run. --max-steps is the TOTAL budget: training continues up to it.")
    p.add_argument("--force-resume", action="store_true")
    args = p.parse_args()


# ---------------------------------------------------------------------------
# Verify CLI options
# ---------------------------------------------------------------------------
#
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
    # Refuse the combination that corrupts training
    if args.vllm_mode == "server" and is_bitsandbytes_optim(args.optim):
        p.error(
            f"--optim {args.optim} corrupts the policy under --vllm-mode server: finite "
            "gradients, NaN params out of optimizer.step(), Use --optim adafactor "
            "(recommended -- adamw_torch is also clean but its fp32 moments cost ~64GB across "
            "policy + critic at 4B)."
        )

    vllm_gpu_mem = 0.25 if args.vllm_gpu_memory_utilization is None else args.vllm_gpu_memory_utilization
    vllm_tp = 1 if args.vllm_tensor_parallel_size is None else args.vllm_tensor_parallel_size


# ---------------------------------------------------------------------------
# Set output directory
# ---------------------------------------------------------------------------


    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, args.dataset)
    print(f"model: {model_slug}  dataset: {args.dataset}  ->  output: {output_dir}")
    if args.vllm_mode == "server":
        print(f"  vLLM: server at {args.vllm_server_host}:{args.vllm_server_port} "
              f"(weight-sync group port {args.vllm_group_port})")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
        critic_max_grad_norm=args.critic_max_grad_norm,
        critic_warmup_steps=args.critic_warmup_steps,
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
        # colocate-only; ignored by TRL in server mode (the server owns these)
        vllm_gpu_memory_utilization=vllm_gpu_mem,
        vllm_tensor_parallel_size=vllm_tp,
        # server-only; ignored in colocate
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_group_port=args.vllm_group_port,
        # optimization
        learning_rate=args.learning_rate,
        critic_learning_rate=args.critic_learning_rate,
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
    # The `.score` head is randomly initialised
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
