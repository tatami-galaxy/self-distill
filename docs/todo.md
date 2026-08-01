### Baselines
- [ ] train on deepscaleR
- [x] base models
- [x] sft on base models 
- [ ] sdft on sft-ed models
- [ ] sdft with all PI
- [ ] instruct models
- [ ] all aime sets
- [ ] beyondaime
- [ ] hmmt
- [ ] qwen 8B
- [ ] opd from qwen 8B
- [ ] coding, knowledge tasks


### Research
- Leverage ICL ability of LLM value functions
  - [x] value function prompt
  - [ ] value pretraining
  - [ ] decoupled gae (VC-PPO)
  - [ ] sweep, anneal lambda with modified value functions
  - [ ] `value(h_t) = ⟨h_t, e_Yes − e_No⟩ = logit(Yes) − logit(No)`
  - [ ] Model value = σ(margin) with a 0/1 target -> make the output a calibrated probability
  - [ ] sweep, anneal lambda with modified value functions
  - [ ] Compare value function behaviours
  - [ ] Different LR for head and backbone
  - [ ] LoRA
- PI conditioned value function
- Train self-teacher in SDFT (analogous to ppo value-pretraining?)
  - [ ] SDFT with negative advantage clipped at high-entropy positions (or in general)
  - [ ] Train SDFT self-teacher objectives
    - [x] Pointwise: for each sampled token ratio
      `rho_t = log pi_teacher(y_t | x, PI, y_<t) - log pi_student(y_t | x, y_<t)`,
      either regress the raw ratio to `tau * (2R - 1)` or apply
      `BCEWithLogits(beta * rho_t, R)`.
    - [x] Endpoint: apply
      `BCEWithLogits(beta * mean_t(rho_t) + b, R)` once per trajectory, with a learned
      base-rate bias `b`, leaving the allocation of credit across tokens unconstrained.
    - [ ] Asymmetric initialization-anchored objective: preserve the initial PI-conditioned
      ratio `rho_t^0` except where a successful rollout contradicts it. Use
      `rho_t^target = rho_t^0 + alpha * R * max(m - rho_t^0, 0)`, so successful,
      PI-disfavored tokens are lifted toward a zero or small positive margin `m`, while failed
      trajectories and non-targeted tokens are anchored to `rho_t^0` rather than pushed more
      negative. Train with `L_lift + lambda * L_anchor` and balance successful rollouts by
      question so easy questions do not dominate.
