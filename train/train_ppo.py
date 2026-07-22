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

Resume (--resume-from-checkpoint) restores the policy/optimizer/scheduler/RNG, skips seen
data, and restores the CRITIC too: its optimizer state rides in the standard optimizer.pt
(the value params share the policy's optimizer), and its weights are saved alongside each
checkpoint as value_model.pt. Resuming a checkpoint that lacks that file is a hard error --
a critic restarting from random init would make every advantage meaningless.
Note each checkpoint therefore costs ~2x the model bytes; consider --save-total-limit.

What resume does and does not let you change:
  * --max-steps IS free to change -- it is the TOTAL budget and is deliberately absent from
    run_meta.json, so resuming a finished 200-step run with --max-steps 400 trains 200 more.
    Verified end to end (a run that had reached its budget, resumed to a larger one).
  * LEARNING RATES COME FROM THE CHECKPOINT, not the command line. optimizer.load_state_dict
    restores each param group's `lr` AND `initial_lr`, and Trainer calls it after create_scheduler
    -- so a different --learning-rate / --critic-learning-rate on the resume command would be
    silently discarded, and the run would quietly continue at the old rate. Both are recorded in
    run_meta.json so validate_resume raises instead of ignoring you. To change a rate, start a
    new run. (Checkpoints written before those keys existed simply skip the check.)
  * Changing --max-steps is only safe with lr_scheduler_type=constant (the default). LambdaLR's
    state_dict excludes the lambda, so it is rebuilt from the NEW budget while last_epoch is
    restored: a cosine run resumed at step 200 with --max-steps 400 jumps its LR back UP to the
    new schedule's midpoint.
  * Extending far enough re-enters the dataset. One epoch is
    len(dataset) / (per_device_train_batch_size * gradient_accumulation_steps / num_generations)
    steps -- 12,877 on full DeepMath at the defaults, but only 2,438 on a 19.5k hint cache.
    Past that the seeded permutation reshuffles and prompts start repeating.

Scope (v1): single-GPU (like the 1.7B student baseline). Multi-GPU DDP wrapping of the
separate value model + cross-process grad sync is a follow-up.

Generation backend (--vllm-mode):
  colocate (default) -- the engine runs in-process on the training GPU. Simplest, and nothing
                        idles (generation and training timeshare the GPU), but its KV cache
                        competes with the policy + critic for memory: OOMs from ~4B up.
  server             -- talk to a standalone `trl vllm-serve` on its OWN GPU, freeing that
                        memory and giving generation a full GPU of KV cache. Costs a GPU that
                        idles during forward/backward, so it buys memory, not throughput.
Weight sync is the same in both: once per optimizer step (gated on global_step), only the
POLICY (the critic is not in self.model, so vLLM never sees it). In server mode HTTP carries
only metadata (name/dtype/shape) while the tensors go GPU->GPU over a NCCL group the trainer
joins as the last rank -- so the per-step cost is ~2N bytes over NVLink/PCIe, negligible next
to a 60s step. The server must serve the SAME model as --model.

!!  SERVER MODE + `adamw_bnb_8bit` CORRUPTS THE POLICY -- use the default paged_adamw_8bit.  !!
(Diagnosed 2026-07-17. An earlier note here blamed the weight sync and declared server mode
broken outright; that was wrong, and the evidence below retires it.)
Symptom: everything *looks* healthy -- server starts, communicator forms, all 310 params POST
200 OK per step, /generate/ returns 200, step 1 trains with a finite loss -- then step 2 dies
in TRL's own grpo_trainer.py:1912 `torch.tensor(logps)` / "Could not infer dtype of NoneType",
because the server's model now emits NaN logprobs (extract_logprobs maps NaN -> None).

The corruption is born in the OPTIMIZER STEP, not the sync. Instrumenting one step: gradients
are finite (0/310) after backward, 21/310 params are non-finite immediately after
optimizer.step(), and the NEXT broadcast leaves that count at exactly 21/310 -- the sync only
ships NaN we had already made. The dead tensors form a contiguous run in named_parameters
order, their grads are tiny and healthy (absmax ~5e-3), and the count varies run to run
(6, 21, 33): memory corruption, not arithmetic.

BOTH ingredients are needed, which is why colocate looked like the safe *mode*:
    server   + adamw_bnb_8bit   -> NaN        server + adamw_torch       -> clean
    colocate + adamw_bnb_8bit   -> clean      server + paged_adamw_8bit  -> clean
                                              server + adafactor         -> clean
NOT PPO-specific: a stock GRPOTrainer (no critic, none of our overrides) reproduces it.
NOT the broadcast: at the root, vLLM sets sendbuff == recvbuff (pynccl.py) -- a legal in-place
no-op -- and a two-process NCCL probe returns the root's buffer untouched, so P2P here is fine.
Note lr=0 does NOT prove a corruption is upstream of the optimizer in general (0 * NaN = NaN
under torch AdamW); it happens to hold for bnb, which survives NaN grads at lr=0.

Why paged: the discriminator is where bitsandbytes puts the 8-bit state. Non-paged allocates it
with torch.zeros_like in torch's caching allocator (corrupts); paged uses CUDA managed memory
for params >1e5 elements -- exactly the large tensors that die. Same kernels and same update
math, so it costs nothing numerically and keeps the 8-bit memory footprint that server mode
exists to buy. Root cause is upstream (bitsandbytes/NCCL); this is a workaround.
If it resurfaces, check the params right after optimizer.step() -- not after the sync.

# single GPU, colocate vLLM (<=1.7B)
CUDA_VISIBLE_DEVICES=0 uv run python -m train.train_ppo \
    --model Qwen/Qwen3-1.7B --dataset deepmath --max-samples 8192

# 4B+: vLLM on its own GPU, policy + critic on another. Engine memory/TP are now the
# SERVER's flags -- passing --vllm-gpu-memory-utilization to the trainer is a hard error.
CUDA_VISIBLE_DEVICES=7 uv run trl vllm-serve --model Qwen/Qwen3-4B --gpu-memory-utilization 0.9 &
CUDA_VISIBLE_DEVICES=6 uv run python -m train.train_ppo \
    --model Qwen/Qwen3-4B --dataset deepmath --vllm-mode server --vllm-server-port 8000

aside :  GRPOTrainer also supports custom rollout logic, in case we want to use this later
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

from train.train_grpo import build_grpo_dataset
from utils import DATASET_REGISTRY_TRAIN, validate_resume


# ---------------------------------------------------------------------------
# Small masked reducers (defined here rather than imported from trl's experimental
# PPO internals, to avoid depending on that module's private surface).
# ---------------------------------------------------------------------------


# Critic weights inside each `checkpoint-<step>` dir. A bare state_dict (not save_pretrained):
# one file regardless of model size, no sharding to reassemble, and no shared-tensor errors.
# The critic's config is reconstructible from --model + num_labels=1 (see main()).
VALUE_MODEL_FILE = "value_model.pt"


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

    def create_optimizer(self, *args, **kwargs):
        # Appending the critic's params here (rather than giving it its own optimizer) is also
        # what makes the critic's optimizer STATE checkpoint for free: it lands in the same
        # `optimizer.pt` Trainer already writes. This runs before _load_optimizer_and_scheduler
        # and rebuilds an identical param-group layout, so that state reloads by index.
        # *args/**kwargs, not an explicit signature: transformers 4.x is `create_optimizer(self)`
        # while 5.x is `create_optimizer(self, model=None)`. Both call it with no arguments, so
        # forwarding whatever we're given keeps this working across the vllm-driven version pin.
        optimizer = super().create_optimizer(*args, **kwargs)  # built over the policy (self.model)
        value_params = [p for p in self.value_model.parameters() if p.requires_grad]
        # The lr is set HERE rather than patched onto param_groups[-1] afterwards, so the
        # scheduler (built right after this, and which records each group's `initial_lr`)
        # schedules the critic from its own base rate rather than the policy's.
        group = {"params": value_params}
        if self.args.critic_learning_rate is not None:
            group["lr"] = self.args.critic_learning_rate
        optimizer.add_param_group(group)
        return optimizer

    # -- critic gradient clipping + norm logging ----------------------------

    def _clip_grad_norm(self, model):
        """Clip the policy via super(), then the critic -- with its own, separate budget.

        Trainer only ever clips `self.model`, and the critic deliberately lives outside it
        (see __init__), so without this the policy is bounded and the critic is not.

        SEPARATE rather than one joint norm over both: a joint norm would let a critic spike
        shrink the policy's update, making the policy's effective step size a function of
        critic noise -- PPO-vs-GRPO would then differ by more than the critic, which is the
        same reason the 'grpo'/'dr_grpo' loss types are excluded (see module docstring).
        Separate clipping leaves the policy's gradients byte-identical to the GRPO baseline.

        Adam already normalizes per-parameter, so the clip is not what bounds the step size;
        it exists to stop an outlier spike from polluting the second-moment estimate and
        distorting the update direction for many steps after. `critic_max_grad_norm=0`
        measures without clipping, so the two can be compared.

        Trainer calls this only when max_grad_norm > 0; this script never sets it otherwise.
        torch's clip_grad_norm_ rather than accelerate's because the critic is not
        accelerator.prepare'd -- consistent with the single-GPU scope.
        """
        policy_grad_norm = super()._clip_grad_norm(model)

        params = [p for p in self.value_model.parameters() if p.grad is not None]
        if params:
            limit = self.args.critic_max_grad_norm
            # clip_grad_norm_ returns the norm from BEFORE any scaling, so the metric is the
            # true pre-clip norm either way; an inf limit scales nothing and only measures.
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                params, limit if limit and limit > 0 else float("inf")
            )
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["ppo/critic_grad_norm"].append(critic_grad_norm.item())
        return policy_grad_norm

    # -- critic checkpointing -----------------------------------------------
    #
    # Trainer only saves `self.model`, so the critic's WEIGHTS need handling here; its
    # optimizer state already rides along (see create_optimizer). `final/` stays policy-only
    # -- eval loads a plain causal LM and never needs the critic.

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
                "on, so every GAE advantage would be measured against a meaningless baseline -- "
                "silently, since training would still look healthy. Was this checkpoint written "
                "before critic checkpointing existed?"
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Load IN-PLACE. The optimizer holds references to these exact tensors, so the module
        # must not be reassigned: that would leave the optimizer updating orphaned parameters
        # while the forward pass used new ones, and the critic would never learn.
        self.value_model.load_state_dict(state)
        print(f"Restored critic weights <- {path}")

    # -- per-token value predictions ----------------------------------------

    def _value_inputs(self, batch):
        """What the critic reads: (input_ids, attention_mask, logits_to_keep).

        Here that is exactly what the POLICY reads -- the same prompt, the same completion,
        one tensor shared by both models. It is a method, and takes the whole `batch`, so
        that a subclass can give the critic a DIFFERENT prompt (train_ppo_pi.py conditions
        it on privileged info) without touching either call site.

        `batch` is the rollout dict during `_generate_and_score_completions` and the
        micro-batch dict during `_compute_loss`; both carry these four keys, and both must
        resolve to the same sequence for a row, or `cliprange_value` would be clipping
        `vpred` against an `old_values` computed from a different state.
        """
        input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        return input_ids, attention_mask, batch["completion_ids"].size(1)

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
        # batch, rather than us reusing a stale one from a previous step.
        self._rewards_per_func_buf = None
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
        with torch.no_grad():
            old_values = self._get_per_token_values(*self._value_inputs(output))
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
        # The per-rollout outcome the critic is regressing towards, kept so subclasses can
        # score the critic against it (train_ppo_pi.py's calibration metrics) without
        # re-deriving it from the reward stash.
        output["terminal_reward"] = rewards.unsqueeze(1)  # (B, 1)

        # Log critic-side stats. value/returns use loss_mask so the reported means describe
        # the rollouts that actually train the critic. unscorable_rate is expected to be ~0
        # (our golds come from build_grpo_dataset's \boxed{} wrap over answer-bearing rows);
        # a non-zero reading means rollouts are being dropped -- worth investigating.
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["ppo/value_mean"].append(masked_mean(old_values, loss_mask).item())
        self._metrics[mode]["ppo/returns_mean"].append(masked_mean(returns, loss_mask).item())
        self._metrics[mode]["ppo/unscorable_rate"].append(unscorable_all.float().mean().item())
        return output

    # -- add the clipped value loss to GRPO's policy loss -------------------

    def _compute_loss(self, model, inputs):
        pg_loss = super()._compute_loss(model, inputs)  # policy loss (already normalized)

        completion_mask = inputs["completion_mask"]
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]
        # Unscorable rollouts carry a fabricated reward of 0, so their `returns` are not a
        # real target -- keep them out of the critic's regression (their advantages were
        # already zeroed, so the policy loss ignores them too).
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
        "critic_max_grad_norm": args.critic_max_grad_norm,
        "loss_type": args.loss_type,
        "vllm_mode": args.vllm_mode,
        # resume-critical, learning rates: on resume these come from the CHECKPOINT, not the CLI.
        # optimizer.load_state_dict restores each param group's `lr` and `initial_lr`, and it runs
        # after create_scheduler, so a different --learning-rate on the command line is silently
        # ignored. Recording both here turns that into a validate_resume error.
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
    p.add_argument("--lam", type=float, default=0.95,
                   help="GAE lambda. ->1.0 = Monte-Carlo return minus critic baseline; "
                        "<1.0 leans on the critic bootstrap.")
    p.add_argument("--vf-coef", type=float, default=0.1, help="Value-loss weight.")
    p.add_argument("--cliprange-value", type=float, default=0.2, help="Value-clipping range.")
    p.add_argument("--critic-max-grad-norm", type=float, default=1.0,
                   help="Clip the critic's gradients to this norm, SEPARATELY from the policy "
                        "(which Trainer clips to max_grad_norm=1.0). Pass 0 to disable clipping "
                        "while still logging ppo/critic_grad_norm, so the two can be compared.")
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
    p.add_argument("--critic-learning-rate", type=float, default=None,
                   help="Learning rate for the critic. Omit to inherit --learning-rate. That "
                        "default (1e-6) suits a pretrained policy but is very slow for the "
                        "critic's RANDOMLY INITIALISED scalar head, which has to learn "
                        "P(correct|prefix) from scratch within --max-steps.")
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="paged_adamw_8bit",
                   help="Optimizer. Default is the PAGED 8-bit AdamW rather than train_grpo.py's "
                        "`adamw_bnb_8bit`: the non-paged one corrupts the policy in the optimizer "
                        "step whenever --vllm-mode server is active (see the module docstring). "
                        "It is the same 8-bit AdamW numerically -- paged only moves the state to "
                        "CUDA managed memory -- so the GRPO baseline stays comparable.")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    # vLLM
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"],
                   help="'colocate': run the engine in-process on the training GPU (simplest, "
                        "nothing idles, but its KV cache competes with policy+critic -- OOMs from "
                        "~4B). 'server': talk to a standalone `trl vllm-serve` on its own GPU. "
                        "See the module docstring for the launch recipe.")
    # Colocate-only. Defaults are None so we can tell "user set it" from "left alone" and
    # reject it in server mode, where it is the SERVER's property (see main()).
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                   help="COLOCATE ONLY (default 0.25): fraction of the training GPU vLLM may "
                        "reserve. Keep modest -- policy + critic + engine share it. In server mode "
                        "pass --gpu-memory-utilization to `trl vllm-serve` instead.")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=None,
                   help="COLOCATE ONLY (default 1). In server mode pass --tensor-parallel-size to "
                        "`trl vllm-serve` instead.")
    p.add_argument("--vllm-server-host", default="0.0.0.0", help="SERVER MODE: vLLM server host.")
    p.add_argument("--vllm-server-port", type=int, default=8000, help="SERVER MODE: vLLM server port.")
    p.add_argument("--vllm-server-timeout", type=float, default=240.0,
                   help="SERVER MODE: seconds to wait for the server to be reachable.")
    p.add_argument("--vllm-group-port", type=int, default=51216,
                   help="SERVER MODE: port for the NCCL weight-sync group the trainer joins as the "
                        "last rank. Weights cross GPU->GPU over this group; HTTP carries only "
                        "metadata.")
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
                        "as the original run (verified against its run_meta.json). --max-steps is "
                        "the TOTAL budget: training continues up to it.")
    p.add_argument("--force-resume", action="store_true")
    args = p.parse_args()

    # In server mode the engine's memory/TP are the SERVER's properties, configured when it is
    # launched; TRL ignores these config fields entirely. Accepting them here would silently do
    # nothing -- exactly the flag you'd reach for after an OOM -- so reject them instead.
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
    # Refuse the one combination that trains silently and wrong: the non-paged bnb 8-bit AdamW
    # NaNs the policy inside the optimizer step once the NCCL weight-sync group exists (module
    # docstring). It surfaces two steps later as an unrelated-looking dtype error in TRL, so a
    # hard stop here is worth more than the flexibility.
    if args.vllm_mode == "server" and args.optim == "adamw_bnb_8bit":
        p.error(
            "--optim adamw_bnb_8bit corrupts the policy under --vllm-mode server: finite "
            "gradients, NaN params straight out of optimizer.step(). Use the default "
            "--optim paged_adamw_8bit (same 8-bit AdamW, state in CUDA managed memory), or "
            "adafactor / adamw_torch. See the module docstring."
        )

    vllm_gpu_mem = 0.25 if args.vllm_gpu_memory_utilization is None else args.vllm_gpu_memory_utilization
    vllm_tp = 1 if args.vllm_tensor_parallel_size is None else args.vllm_tensor_parallel_size

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, args.dataset)
    print(f"model: {model_slug}  dataset: {args.dataset}  ->  output: {output_dir}")
    if args.vllm_mode == "server":
        print(f"  vLLM: server at {args.vllm_server_host}:{args.vllm_server_port} "
              f"(weight-sync group port {args.vllm_group_port})")

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
    # Seed FIRST: the `.score` head is the one weight in this script with no pretrained values
    # to load, so it is randomly initialised right here -- and Trainer only calls set_seed()
    # when it is constructed, several lines below. Without this the critic starts from ambient
    # OS entropy on every run: --seed would not reproduce a run, run_meta.json's `seed` would
    # be a lie about the critic, and any A/B over critic hyperparameters would be confounded
    # by a different value function in each arm.
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
