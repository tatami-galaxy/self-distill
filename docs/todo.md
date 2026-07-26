### Baselines
- [x] eval with temp 1.0 (default TRL configs)
- [ ] train with KL
- [ ] train on deepscaleR
- [ ] base, sft and instruct models
- [ ] eval on all aime sets, beyondaime, hmmt
- [ ] opd from other models
- [ ] qwen 8B student, teacher
- [ ] coding, knowledge tasks


### Research
- Leverage ICL ability of LLM value functions
  - [x] value function prompt 
  - [ ] `value(h_t) = ⟨h_t, e_Yes − e_No⟩ = logit(Yes) − logit(No)`
  - [ ] Model value = σ(margin) with a 0/1 target -> make the output a calibrated probability
  - [ ] Different LR for head and backbone? LoRA?
- PI conditioned value function
