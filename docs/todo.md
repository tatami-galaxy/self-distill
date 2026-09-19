### Training

- [x] Deepmath
- [ ] Reasoning Gym
- [ ] Countdown-CoT-20k
- [ ] CodeIO
- [ ] Hyperparameter sweep, more steps
- [x] Validation set

### Benchmarks

- [x] Aime 24, 25
- [ ] Beyondaime
- [ ] Reasoning Gym
- [ ] Countdown heldout
- [x] CodeIO
- [ ] Comparison with other SD methods

### Models

- [x] think models
- [ ] sft-ed models
- [x] instruct models

### SD Objective

- [x] Reverse KL
- [ ] Forward KL
- [ ] JSD
- [ ] Forward CE

### Analysis

- [x] Self-teacher behavior analysis
- [x] Trained student behavior analysis
- [ ] PI information content vs Student performance
- [ ] PVF analysis
- [ ] Distributional analysis, V-information


### Research

- **PI conditioned V**
  - [ ] $A^{\mathrm{actor}}_{t,k}=(1-\rho_k)A^{\mathrm{SD}}_{0,t}+\rho_k A^{R,\mathrm{GAE}}_{t},\qquad\rho_k\in[0,1]$
  - [x] value warmup
  - [ ] judge instruction

- **Soft Q, NLAC, reward shaping**

- **Train self-teacher for hint generation**
  - [x] $R(h) = \alpha\,S(h)-\frac{|h|}{B}-\gamma\,T(h)$
  - [x] Constrained optimization : $\min_\phi\;\mathbb E_{h\sim g_\phi}\left[C(h)+\gamma\,T(h)\right]\quad\text{subject to}\quad \mathbb E_{h\sim g_\phi}[S(h)]\ge \tau$
  - [ ] Training objective theoretical justification
  - [x] Tune hyperparameters
  - [ ] Sufficiency proxy
  - [ ] V information for minimal hint
  - [ ] Other objectives
  - [x] LoRA on top of self-teacher 

