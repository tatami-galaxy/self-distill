# `train_ppo_val.py` — call trace for one training iteration

What actually runs, from `Trainer.train()` down, for each iteration of the outermost
training loop. Traced against the pinned libraries, not from memory:

| | |
|---|---|
| `transformers` | 5.10.1 (`transformers/trainer.py`, line numbers below refer to it) |
| `trl` | 1.6.0 (`trl/trainer/grpo_trainer.py`) |
| ours | `train/train_ppo.py` + `train/train_ppo_val.py` |

**Legend** — `HF` = `transformers.Trainer`, `TRL` = `GRPOTrainer`, **`★`** = ours
(`PPOTrainer` in `train_ppo.py`, or `PPOValTrainer` in `train_ppo_val.py` where noted).

This traces the **verifier-prompt arm**, because it is the one with an extra moving part.
`train_ppo.py` on its own is the same trace with Phase A's value-prompt construction removed:
its `_value_inputs` reads the policy's already-tokenized `prompt_ids` straight out of the
rollout dict, so there is nothing to compose, render, cache or stash. `train_ppo_pi.py` is
this trace with `compose_value_messages` replaced by `compose_pi_messages`. Our references
below name symbols rather than line numbers — the numbers have gone stale twice.

**Config assumed throughout** (the script's defaults):

```
per_device_train_batch_size = 1     num_generations   = 1
gradient_accumulation_steps = 16    num_iterations    = 1   (--num-ppo-epochs)
steps_per_generation        = 16    loss_type         = dapo
   (GRPOConfig defaults it to gradient_accumulation_steps)
generation_batch_size       = per_device_bs × num_procs × steps_per_generation = 16
generate_every              = steps_per_generation × num_iterations            = 16
```

So one optimizer step consumes **one generation batch of 16 rows**, generated once and reused
across all 16 micro-batches. At the default `num_generations=1` those 16 rows are 16 **distinct**
prompts — stock GRPO's config forbids `<2`, but `PPOConfig` overrides that since the critic, not
a group, is the baseline (see [`train_ppo.py`](../train/train_ppo.py) `PPOConfig.__post_init__`).
The batch *size* is `per_device_bs × num_procs × steps_per_generation` and does **not** depend on
`num_generations`, so passing 2 instead (the earlier runs' value) fills the same 16 rows with
8 prompts × 2 rollouts and changes nothing else in this trace — `generation_batch_size`, the step
count, and the control flow are all identical.

**The policy and the critic read different prompts.** The policy gets the dataset's
problem-solving prompt; the critic gets the same question under `VALUE_SYSTEM_PROMPT`, a verifier
instruction. That second prompt is tokenized on our own path (`_build_value_prompts`), not TRL's,
and is built **once per generation batch** then cached into the rollout dict — see Phase A. Both
forwards still end on the identical generation header, so the completion attaches where it was
sampled.

---

## Top level

```
Trainer.train()                                                          trainer.py:1331
│
├─ HF  if resume_from_checkpoint:                                              1416
│  │     # non-deepspeed / non-FSDP path loads HERE, before the optimizer exists
│  ├─ ★ PPOTrainer._load_from_checkpoint(ckpt)                                 1421
│  │  ├─ HF  super()._load_from_checkpoint()          → policy weights
│  │  └─ ★  torch.load(<ckpt>/value_model.pt) → value_model.load_state_dict()
│  │        (in-place: the optimizer holds refs to these exact tensors)
│  └─ HF  TrainerState.load_from_json(...)            → global_step, epoch
│
└─ HF  _inner_training_loop()                                                  1440
   │
   ├─ HF  get_train_dataloader()                          ── TRL override      1454
   │  │     batch_size = per_device_bs × steps_per_generation = 16 rows
   │  └─ TRL _get_train_sampler() → RepeatSampler(
   │           mini_repeat_count = num_generations   = 1,   (2 ⇒ each prompt ×2)
   │           repeat_count      = num_iterations × steps_per_generation = 16)
   │        ⇒ 16 distinct prompts; that SAME 16-row batch is yielded 16× in a row
   │          (repeat_count is the buffering reuse, independent of num_generations)
   │
   ├─ HF  set_initial_training_values(...)  → max_steps, steps_in_epoch        1467
   ├─ HF  _init_training_state(...)         → epochs_trained, steps_to_skip    1469
   │
   ├─ HF  _prepare_for_training(max_steps, dl, ckpt)  ──── ONCE ───────        1472
   │  ├─ ★ PPOTrainer.create_optimizer()                                       1582
   │  │  ├─ HF  super().create_optimizer()        → param groups over self.model
   │  │  └─ ★  optimizer.add_param_group({params: value_model,
   │  │                                   lr: critic_learning_rate})
   │  ├─ HF  _wrap_model(); accelerator.prepare(model, optimizer)         1586-1608
   │  │        (value_model is NOT prepared — single-GPU scope, and it must stay
   │  │         out of self.model so the vLLM weight sync is pure-policy)
   │  ├─ HF  create_scheduler(num_training_steps=max_steps)                    1613
   │  │        reads each group's lr → records initial_lr  ⟵ why the critic lr
   │  │        must be set inside create_optimizer, not patched on afterwards
   │  └─ HF  if resume: _load_optimizer_and_scheduler(ckpt)                    1652
   │           restores lr AND initial_lr per group ⟹ a changed --learning-rate
   │           / --critic-learning-rate is silently overwritten  (validate_resume)
   │
   ├─ HF  model.zero_grad(); callback_handler.on_train_begin()             1506-1508
   │
   ├─ for epoch in range(epochs_trained, num_train_epochs):                    1513
   │  └─ HF _run_epoch(...)                                                    1662
   │     │
   │     ├─ HF  if resuming mid-epoch: skip_first_batches(dl, n)               1687
   │     │
   │     └─ for update_step in range(...):        ◀══ ONE ITERATION = ONE OPTIMIZER STEP
   │        │                                                                  1706
   │        ├─ HF  get_batch_samples(epoch_iterator, num_batches=16)           1710
   │        │      → 16 dataloader items, all the SAME 16 rows (16 distinct prompts,
   │        │        or 8 prompts × 2 at num_generations=2)
   │        │      → _get_num_items_in_batch(...) = None  ⚠ no "labels"; the
   │        │        collator yields a list, not a dict            (note 1)
   │        │
   │        ├─ for i, inputs in enumerate(batch_samples):   ◀── micro-batch, i = 0..15
   │        │  │                                                               1724
   │        │  ├─ HF  accelerator.gradient_state._set_sync_gradients(i == 15)
   │        │  ├─ HF  sync_context = no_sync  (i < 15)  |  nullcontext  (i == 15)
   │        │  │
   │        │  └─ TRL training_step(model, inputs, None)                       1743
   │        │     ├─ HF  Trainer.training_step()                               1876
   │        │     │  ├─ model.train()
   │        │     │  ├─ TRL _prepare_inputs(inputs)     ─────────────  PHASE A
   │        │     │  │     real work only at i == 0; i = 1..15 read the buffer
   │        │     │  ├─ HF  compute_loss() → TRL compute_loss()
   │        │     │  │     └─ ★ PPOTrainer._compute_loss()  ────────  PHASE B
   │        │     │  ├─ HF  loss = loss / current_gradient_accumulation_steps  1936
   │        │     │  │        ⚠ extra ÷16 on top of the loss_type's own
   │        │     │  │          normalization                       (note 2)
   │        │     │  └─ HF  accelerator.backward(loss)
   │        │     └─ TRL self._step += 1  ; log step_time every 16 calls
   │        │
   │        └─ if do_sync_step  (i == 15):        ─────────────────  PHASE C    1762
   │           ├─ ★ PPOTrainer._clip_grad_norm(model)                          1765
   │           │  ├─ HF  super() → accelerator.clip_grad_norm_(model, 1.0)  POLICY
   │           │  └─ ★  clip_grad_norm_(value_model, critic_max_grad_norm)  CRITIC
   │           │        → log ppo/critic_grad_norm        (separate budgets, by design)
   │           ├─ HF  optimizer.step()      one optimizer, 2 groups (policy | critic lr)
   │           ├─ HF  lr_scheduler.step()
   │           ├─ HF  model.zero_grad()                 ⚠ POLICY ONLY          1780
   │           ├─ HF  state.global_step += 1   ⟵ what makes the next step's
   │           │                                  vllm sync_weights() fire
   │           ├─ HF  callback_handler.on_step_end()
   │           └─ ★ PPOTrainer._maybe_log_save_evaluate(...)                   1784
   │              ├─ ★ _assert_policy_finite()      catches the server-mode NaN here
   │              ├─ ★ value_model.zero_grad(set_to_none=True)
   │              │       the ONLY place the critic's grads are cleared
   │              └─ HF  super()._maybe_log_save_evaluate()                    2055
   │                 ├─ if should_log:  TRL log() → flush _metrics (all ppo/*)
   │                 └─ if should_save: ★ PPOTrainer._save_checkpoint()
   │                    ├─ TRL super() (model card) → HF Trainer._save_checkpoint
   │                    │     save_model / _save_optimizer_and_scheduler /
   │                    │     _save_rng_state          (critic optim state rides along)
   │                    └─ ★ torch.save(value_model.state_dict(), value_model.pt)
   │
   └─ HF  _finalize_training(...)                                              1531
```

Two important points:

* **`create_optimizer_and_scheduler()` (`trainer.py:1141`) is dead code** — it has no call site
  anywhere in `trainer.py` (the other mentions are docstrings and error messages at 601, 660,
  682, 4202). The two halves are called separately, 30 lines apart, with `accelerator.prepare`
  between them. Our critic param group is appended in the first half precisely so the second
  half picks up its `lr` as `initial_lr`.
* **Resume touches our state at two different depths**: policy + critic *weights* in `train()`
  before the optimizer exists (1421), critic *optimizer state* inside `_prepare_for_training`
  after the scheduler is built (1652). That split is why the weights need their own
  `value_model.pt` while the optimizer state needs nothing.

---

## Phase A — rollout + credit assignment

`TRL _prepare_inputs`. Real work happens **only at `i == 0`**; micro-batches 1–15 fall
straight through to the last line.

```
TRL _prepare_inputs(generation_batch)                            grpo_trainer.py:1149
 │   generate_every = steps_per_generation × num_iterations = 16
 ├─ if self._step % 16 == 0:                        ◀── true only at i == 0
 │  │
 │  ├─ ★ PPOValTrainer._generate_and_score_completions(inputs)   train_ppo_val.py
 │  │  ├─ ★ self._value_rows = inputs      # raw rows; GRPO's dict has only TOKENIZED
 │  │  │                                   #   policy prompts. Stashed BEFORE super(),
 │  │  │                                   #   which computes old_values inside itself.
 │  │  │                                   #   try/finally -> cleared      (note 5)
 │  │  │
 │  │  ├─ ★ PPOTrainer._generate_and_score_completions(inputs)      train_ppo.py
 │  │  ├─ ★ self._rewards_per_func_buf = None            # clear stale stash
 │  │  │
 │  │  ├─ TRL super()._generate_and_score_completions(inputs)                1828
 │  │  │  ├─ TRL _generate(prompts)                                         1702
 │  │  │  │  ├─ _tokenize_prompts(prompts)                     → prompt_ids
 │  │  │  │  ├─ _generate_single_turn(prompt_ids, ...)                      1348
 │  │  │  │  │  ├─ vllm_generation.sync_weights()   # iff global_step != _last_loaded_step
 │  │  │  │  │  └─ vllm_generation.generate(...)               → completion_ids, logprobs
 │  │  │  │  └─ log completions/{mean,min,max}_length, clipped_ratio
 │  │  │  │     returns num_items_in_batch = Σ completion tokens      (note 1)
 │  │  │  ├─ pad → prompt_ids (B,P) left-padded, completion_ids (B,C) right-padded
 │  │  │  ├─ torch.no_grad():
 │  │  │  │  ├─ _get_per_token_logps_and_entropies(model, …, batch_size=1)
 │  │  │  │  │      → old_per_token_logps    # chunked: 16 sequential 1-row forwards
 │  │  │  │  ├─ importance_sampling_ratio  (vLLM train/infer mismatch correction)
 │  │  │  │  └─ ref_per_token_logps  ──────────────── SKIPPED, beta forced to 0
 │  │  │  ├─ ★ PPOTrainer._calculate_rewards(inputs, prompts, completions, ids)
 │  │  │  │  ├─ TRL super()._calculate_rewards()          → accuracy_reward, (B_all, 1)
 │  │  │  │  └─ ★ stash → self._rewards_per_func_buf
 │  │  │  ├─ group-normalize → advantages (B,)      ⟵ computed, then thrown away (note 4)
 │  │  │  └─ return dict{prompt_ids, prompt_mask, completion_ids, completion_mask,
 │  │  │                 advantages, num_items_in_batch, old_per_token_logps, …}
 │  │  │
 │  │  ├─ ★ read stash + 2 guards (populated? gathered size == B × n_procs?)
 │  │  ├─ ★ rewards = (rpf × weights).nansum(1)[local_slice];  scorable = ~all-NaN
 │  │  ├─ ★ optional: −missing_eos_penalty for rows not ending in EOS
 │  │  ├─ ★ token_rewards: place the scalar reward at each row's LAST real token
 │  │  │
 │  │  ├─ ★ torch.no_grad():
 │  │  │     _get_per_token_values( *★_value_inputs(output) )  → old_values (B,C)
 │  │  │      ├─ ★ _value_inputs: 'value_prompt_ids' not in batch → build it once
 │  │  │      │                                                   train_ppo_val.py
 │  │  │      │   └─ ★ _build_value_prompts(self._value_rows)      train_ppo_val.py
 │  │  │      │      ├─ compose_value_messages: swap the SOLVER system turn for
 │  │  │      │      │     VALUE_SYSTEM_PROMPT, keep the question verbatim
 │  │  │      │      │     ⟵ THE ONLY THING THIS ARM CHANGES. train_ppo_pi.py
 │  │  │      │      │        substitutes compose_pi_messages right here.
 │  │  │      │      └─ ★ _render_value_prompts(conversations)      train_ppo.py
 │  │  │      │         ├─ apply_chat_template(add_generation_prompt=True, same
 │  │  │      │         │     chat_template/kwargs/tools as TRL _tokenize_prompts)
 │  │  │      │         ├─ pad(..., padding_side="left")   ⟵ must be LEFT: right-padding
 │  │  │      │         │     would put PADs between prompt and completion (note 6)
 │  │  │      │         └─ log ppo/value_prompt_len_mean
 │  │  │      │   └─ ★ cache batch['value_prompt_ids'/'value_prompt_mask']
 │  │  │      │         ⟵ written into the dict super() returned, so it rides the
 │  │  │      │            shuffle/split below and Phase B reuses it (no rebuild)
 │  │  │      ├─ ★ → [value_prompt ‖ completion]
 │  │  │      │     (train_ppo.py's own critic reads [prompt ‖ completion] here instead)
 │  │  │      └─ ★ backbone fwd → .score → [:, :-1][:, -C:]     ★_get_per_token_values
 │  │  │            ONE unchunked B=16 forward at ~8k tokens            (note 3)
 │  │  ├─ ★ compute_gae(token_rewards, old_values, mask, gamma, lam) → advantages, returns
 │  │  ├─ ★ masked_whiten(advantages, completion_mask × scorable)
 │  │  │        ⟵ OVERWRITES GRPO's group-normalized advantages
 │  │  ├─ ★ output += {returns, old_values, scorable, terminal_reward}
 │  │  │        (+ value_prompt_ids / value_prompt_mask, cached above)
 │  │  └─ ★ log ppo/{value_mean, returns_mean, unscorable_rate}
 │  │
 │  ├─ TRL shuffle_sequence_dict(generation_batch)
 │  └─ TRL split_tensor_dict(..., 16) → self._buffered_inputs   # 16 slices of 1 row
 │
 └─ return self._buffered_inputs[self._step % 16]   ◀── i = 1..15 take ONLY this line
```

`compute_gae` (`train_ppo.py`) walks `t` backwards over the completion axis with
`gamma=1.0`; padding carries zero reward and zero value, so GAE decays to 0 there and the
bootstrap at the terminal token uses next-value = 0. `returns = advantages + values` is taken
from the **raw** (pre-whitening) advantages.

---

## Phase B — loss, every micro-batch

```
HF  Trainer.compute_loss()  →  TRL GRPOTrainer.compute_loss()          grpo_trainer.py:2408
 └─ ★ PPOTrainer._compute_loss(model, inputs)                            train_ppo.py
    │
    ├─ TRL super()._compute_loss(model, inputs)  ───────────── POLICY LOSS        2489
    │  ├─ _get_per_token_logps_and_entropies(model, ...)   ← policy fwd WITH grad
    │  ├─ advantages = inputs["advantages"]    # (B,C): dim()==2, so NOT unsqueezed
    │  │      TRL unsqueezes only 1-D advantages — this is the seam that lets our
    │  │      per-token GAE plug into GRPO's loss unchanged
    │  ├─ coef_1 = exp(logps − old_logps);  coef_2 = clamp(coef_1, 1±ε)
    │  ├─ per_token_loss = −min(coef_1·A, coef_2·A)
    │  ├─ × inputs["importance_sampling_ratio"]         (vLLM correction)
    │  ├─ + β·KL  ──────────────────────────────── SKIPPED, beta = 0
    │  ├─ dapo: loss = (ptl·mask).sum() / (num_items_in_batch / num_procs)
    │  │  bnpo: loss = masked_mean(ptl, mask) / current_gradient_accumulation_steps
    │  └─ log entropy, clip_ratio/*
    │
    ├─ ★ mask = completion_mask × scorable   (× tool_mask if present)
    ├─ ★ vpred = _get_per_token_values( *★_value_inputs(inputs) )  ← critic fwd WITH grad
    │        _value_inputs finds 'value_prompt_ids' already in the micro-batch (Phase A
    │        cached it before the shuffle/split), so it reuses — never rebuilds — here
    ├─ ★ vpredclipped = old_values + clamp(vpred − old_values, ±cliprange_value)
    ├─ ★ vf_loss = 0.5 · max((vpred−returns)², (vpredclipped−returns)²)
    │        normalized to MATCH the policy's loss_type:
    │          dapo → sum / (num_items_in_batch / num_procs)      [token-uniform]
    │          bnpo → masked_mean / current_gradient_accumulation_steps
    ├─ ★ log ppo/{vf_loss, vf_clipfrac}
    └─ ★ return pg_loss + vf_coef · vf_loss
```

Two full-size forwards **with** grad per micro-batch: the policy (inside
`super()._compute_loss`) and the critic (`_get_per_token_values`). `_value_inputs` must resolve
to the same sequence here as it did in Phase A, or `cliprange_value` would be clipping `vpred`
against an `old_values` computed from a different state. Caching `value_prompt_ids` into the
rollout dict in Phase A — rather than re-rendering the verifier prompt here — is what makes that
hold by construction.

---

## Phase C — optimizer step, after `i == 15`

```
if do_sync_step (i == 15):                                              trainer.py:1762
 ├─ ★ PPOTrainer._clip_grad_norm(model)                                  train_ppo.py
 │  ├─ HF super()._clip_grad_norm → accelerator.clip_grad_norm_(model.parameters(), 1.0)
 │  │                                                                          POLICY
 │  └─ ★ torch.nn.utils.clip_grad_norm_(value_model.parameters(),
 │                                      critic_max_grad_norm)                  CRITIC
 │     └─ ★ log ppo/critic_grad_norm       (pre-clip norm; 0.0 measures without clipping)
 ├─ HF  _get_grad_norm(model, grad_norm)
 ├─ HF  optimizer.step()          # ONE optimizer, 2 param groups: policy lr | critic lr
 ├─ HF  lr_scheduler.step()
 ├─ HF  model.zero_grad()         # ⚠ POLICY ONLY — the critic is outside self.model
 ├─ HF  state.global_step += 1    # ⟵ what makes the NEXT step's sync_weights() fire
 ├─ HF  callback_handler.on_step_end()
 │
 └─ ★ PPOTrainer._maybe_log_save_evaluate(...)                           train_ppo.py
    ├─ ★ _assert_policy_finite()          # catches the server-mode optimizer NaN here,
    │                                     #   at the step it happens
    ├─ ★ value_model.zero_grad(set_to_none=True)
    │       the ONLY seam where the critic's grads clear; without it the critic
    │       enters the next step holding clip(G_{k−1}), an undocumented momentum term
    └─ HF super()._maybe_log_save_evaluate()                              trainer.py:2055
       ├─ if should_log:  TRL log() → flush self._metrics (all ppo/* land here)
       └─ if should_save: ★ PPOTrainer._save_checkpoint(model, trial)    train_ppo.py
          ├─ TRL super()._save_checkpoint (model card) → HF Trainer._save_checkpoint
          │     save_model / _save_optimizer_and_scheduler / _save_rng_state
          │     (the critic's optimizer state rides along in optimizer.pt for free)
          └─ ★ torch.save(value_model.state_dict(), <ckpt>/value_model.pt)
```

---

## Notes

1. **Two different `num_items_in_batch`.** HF's `_get_num_items_in_batch` (`trainer.py:2123`)
   returns `None` here — it requires `"labels" in batch_samples[0]`, but GRPO's identity
   collator yields a *list of dicts*, so the membership test fails. The one `_compute_loss`
   actually reads is `inputs["num_items_in_batch"]`, which TRL puts into the batch dict from
   `_generate` (total completion tokens across the generation batch). Same name, different
   object, different scope.

2. **`training_step` divides the returned loss by `current_gradient_accumulation_steps` again**
   (`trainer.py:1936`), because that `None` makes the condition true. So `dapo`'s token-mean
   gets an extra ÷16, and `bnpo` gets ÷16 twice. It is a uniform scale on
   `pg_loss + vf_coef·vf_loss`, so both the pg:vf ratio and the token-uniform weighting are
   intact — and the GRPO baseline eats the identical factor, so the PPO-vs-GRPO comparison
   holds. But the *effective* learning rate is 16× smaller than the flag reads.

3. **The critic's rollout-time forward is unchunked.** TRL calls
   `_get_per_token_logps_and_entropies(..., batch_size=per_device_train_batch_size)`, i.e. 16
   sequential 1-row forwards. `_get_per_token_values` runs a single
   forward over all 16 rows at ~8k tokens. It is under `no_grad` so there is no activation
   graph, but it is the largest single activation footprint in the step and the first thing
   that will OOM if `steps_per_generation` or the completion budget rises.

4. **GRPO's group-normalized advantages are fully computed, then discarded** — `nanstd`,
   `repeat_interleave`, the gather, all of it — before we overwrite `output["advantages"]`.
   Cheap (it is `(B,)` arithmetic), but it is why `frac_reward_zero_std` still appears in the
   logs and means nothing for PPO: the critic is the baseline, so GRPO's group statistics are
   never used. At the default `num_generations=1` every "group" is a single rollout, so its std
   is trivially zero and the metric pins to **1.0**; at `num_generations=2` it drifts (the older
   runs logged 0.75–0.83). Either way it describes a mechanism this trainer does not use.

5. **Why `_value_rows` is cleared in `finally`, not at method entry.** The neighbouring
   `_rewards_per_func_buf` clears at entry, which is safe because it is *read* in the same
   method right after `super()`. `_value_rows` is read from a different method
   (`_value_inputs`), which `_compute_loss` also calls. If a TRL buffering change ever dropped
   `value_prompt_ids` from the micro-batch, a stale stash would be the same length as the batch
   — the `n_rows == n_completions` guard compares batch sizes, which always match — so it would
   sail through and pair every verifier prompt with a *previous* batch's question. Clearing in
   `finally` makes that case raise instead.

6. **The padding side is the real alignment invariant.** `cat([value_prompt, completion])[:, -C:]
   == completion_ids` is a tautology of the concatenation and cannot fail. What can fail is the
   padding side: right-padding the value prompt leaves PADs *between* prompt and completion, so
   `values[:, 0]` — which `_get_per_token_values` reads from the position just before the first
   completion token — comes off a pad instead of the generation header, offsetting each row's
   value curve by its own pad count. Both this and the generation-boundary match against TRL's
   `_tokenize_prompts` are pinned in `tests/test_ppo_config.py`.
