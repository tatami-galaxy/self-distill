# GRPO/PPO loss scaling under gradient accumulation

This note records a non-obvious interaction between TRL's `GRPOTrainer` and
Transformers' `Trainer`. It was verified against the environment used by this repository
on 2026-08-18:

- TRL 1.6.0
- Transformers 5.10.1
- Accelerate 1.13.0

Re-check the source and the gradient-scaling regression described below when upgrading any
of these packages.

## The important rule

**Do not add another gradient-accumulation correction to losses returned by
`PPOTrainer`.**

`GRPOTrainer` explicitly disables the outer loss division in
`Trainer.training_step`. Our `PPOTrainer` subclasses `GRPOTrainer`, calls its
constructor, and preserves that behavior.

## How TRL disables the outer division

In `trl/trainer/grpo_trainer.py`, `GRPOTrainer.__init__` passes a deliberately non-None
sentinel to the base trainer:

```python
super().__init__(
    ...
    compute_loss_func="non-None value to disable scaling",
)
```

In `transformers/trainer.py`, `Trainer.training_step` divides by
`current_gradient_accumulation_steps` only when this condition is true:

```python
if (
    (not self.model_accepts_loss_kwargs or num_items_in_batch is None)
    and self.compute_loss_func is None
):
    loss = loss / self.current_gradient_accumulation_steps
```

The TRL sentinel makes the final clause false, so the division is skipped. The sentinel is
not a real loss function and is not called: `GRPOTrainer` overrides `compute_loss` and
routes directly to its own `_compute_loss`.

This is intentional upstream behavior, not a customization in `train_grpo.py`.

## Two different `num_items_in_batch` values

The call path contains two unrelated values with the same name:

1. The `num_items_in_batch` argument passed by `Trainer.training_step` is `None` for
   GRPO. The identity collator returns a list of rollout dictionaries rather than a labeled
   language-model batch, so Transformers cannot infer a label count.
2. `inputs["num_items_in_batch"]` is created by TRL during generation. It is the global
   accumulated batch's completion-token count and is available to `_compute_loss`.

The first value would normally cause Transformers to apply its outer division, but the
non-None `compute_loss_func` sentinel disables that branch. The second value is the
denominator used by DAPO.

## Where accumulation normalization happens

TRL owns the complete normalization:

- **DAPO:** each micro-batch contributes its token-loss sum divided by the global accumulated
  completion-token count. Summing gradients over the accumulation window produces one
  token-uniform mean. No explicit `/ gradient_accumulation_steps` belongs in this branch.
- **BNPO:** each micro-batch computes a token mean and divides it by
  `current_gradient_accumulation_steps`. Summing those gradients produces the mean of the
  micro-batch means.

The custom PPO policy loss comes from `GRPOTrainer._compute_loss`. Its value loss mirrors
the selected policy normalizer:

- DAPO value loss uses the same global completion-token denominator.
- BNPO value loss uses a masked micro-batch mean divided by the current accumulation count.

The returned `policy_loss + vf_coef * value_loss` is passed unchanged to
`accelerator.backward`; Transformers does not divide it again.

## What would introduce a scaling bug

Any of the following should trigger a new end-to-end check:

- setting `trainer.compute_loss_func = None`;
- bypassing `GRPOTrainer.__init__`;
- replacing the inherited `training_step`;
- upgrading TRL or Transformers such that the sentinel or guard changes;
- multiplying the PPO loss by `gradient_accumulation_steps` to “cancel” an outer division;
- removing BNPO's internal accumulation division without enabling an equivalent outer one.

The fifth case would currently make both actor and critic gradients too large by the
accumulation factor.

## Regression check

Use a one-parameter model and compare one optimizer step:

1. With accumulation 1, return a loss whose derivative is 1.
2. With accumulation 16, return sixteen already-window-normalized contributions whose
   derivatives are each `1 / 16`.
3. Keep the `GRPOTrainer` sentinel and overridden `compute_loss` behavior in the probe.

Both runs must produce a parameter update of 1. A result of `1 / 16` in the second run
means an outer accumulation division has been reintroduced.

For a runtime sanity check on a real trainer:

```python
assert trainer.compute_loss_func is not None
```

This assertion checks the safeguard's presence. The gradient regression checks that the
full trainer stack still honors it.
