### Baselines
- [ ] train on deepscaleR
- [x] base models
- [x] sft on base models 
- [ ] sdft on sft-ed models
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
    - [x] Pointwise
    - [x] Endpoint
    - [x] Asymmetric
    - [ ] Hyperparameter sweep
- [ ] Analyze PI conditioned self-teacher behavior (rollouts, log ratios)
