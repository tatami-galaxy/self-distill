### Baselines
- [x] train on deepscaleR
- [ ] base models, sft-ed models, instruct models
- [ ] beyondaime
- [ ] hmmt
- [ ] qwen 8B as student, teacher
- [ ] coding, knowledge tasks
- [ ] hyperparameter sweep


### Research
- Leverage ICL ability of LLM value functions
  - [x] value function prompt
  - [ ] value pretraining
  - [ ] decoupled gae (VC-PPO)
  - [ ] sweep, anneal lambda (currently 1) with modified value functions
  - [ ] `value(h_t) = ⟨h_t, e_Yes − e_No⟩ = logit(Yes) − logit(No)`
  - [ ] Model value = σ(margin) with a 0/1 target -> make the output a calibrated probability
  - [ ] sweep, anneal lambda with modified value functions
  - [ ] Analyze value function behaviours
  - [ ] Different LR for head and backbone
  - [ ] LoRA
- PI conditioned value function
- Train self-teacher in SDFT (analogous to ppo value-pretraining?)
  - [ ] SDFT with negative advantage clipped at high-entropy positions (or in general)
  - [ ] Train self-teacher with rollout outcome signal
    - [x] Asymmetric
    - [x] Asymmetric aggregate
    - [ ] Max Ent RL?
    - [ ] Hyperparameter sweep
  - [ ] Train self-teacher for hint generation
    - [ ] Objective from self-teacher behavior (example : entropy)
    - [ ] Objective from teacher-student log ratio
    - [ ] Hint generation with information bottleneck -> pass@k with generated hint -> RL (VAE/Autoencoder)
- [x] Analyze PI conditioned self-teacher behavior
