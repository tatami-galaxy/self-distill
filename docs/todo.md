### Training data

- [x] Deepmath
- [ ] Reasoning Gym
- [ ] L3 reasoning subset of SciKnowEval
- [ ] Countdown-CoT-20k

### Benchmarks

- [x] Aime 24, 25
- [-] Beyondaime
- [ ] Reasoning Gym
- [ ] Countdown heldout
- [ ] SciKnowEval

### Models

- [x] think models
- [ ] sft-ed models
- [-] instruct models

### SD Objective

- [x] Reverse KL
- [ ] JSD
- [ ] Forward CE

### Research

- PI conditioned value function
  - [ ] $A^{\mathrm{actor}}_{t,k}=(1-\rho_k)A^{\mathrm{SD}}_{0,t}+\rho_k A^{R,\mathrm{GAE}}_{t},\qquad\rho_k\in[0,1]$
  - [x] value warmup
  - [ ] PVF analysis, comparison with SDFT

- PI conditioned Q function

  placeholder

- Train self-teacher in SDFT 

  - [x] Train self-teacher with outcome signal
    - [x] Asymmetric
    - [x] Asymmetric aggregate

  - [ ] Train self-teacher for hint generation
    - [x] $R(h) = \alpha\,S(h)-\frac{|h|}{B}-\gamma\,T(h)$
    - [x] Tune hyperparameters
    - [ ] Other cost functions
    - [ ] Dual objective : $\min_\phi\;\mathbb E_{h\sim g_\phi}\left[C(h)+\gamma\,T(h)\right]\quad\text{subject to}\quad S(h)\ge \tau$
    - [ ] LoRA on top of self-teacher 
