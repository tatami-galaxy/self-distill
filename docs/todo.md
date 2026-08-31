### Training

- [x] Deepmath
- [ ] Reasoning Gym`
- [ ] Countdown-CoT-20k
- [ ] Hyperparameter sweep, more steps

### Benchmarks

- [x] Aime 24, 25
- [ ] Beyondaime
- [ ] Reasoning Gym
- [ ] Countdown heldout
- [ ] avg@12

### Models

- [x] think models
- [ ] sft-ed models
- [ ] instruct models

### SD Objective

- [x] Reverse KL
- [ ] Forward KL
- [ ] JSD
- [ ] Forward CE

### Analysis

- [x] Self-teacher behavior analysis
- [ ] Trained student behavior analysis
- [ ] PVF analysis
- [ ] Soft PQF analysis

### Research

- **PI conditioned V**
  - [ ] $A^{\mathrm{actor}}_{t,k}=(1-\rho_k)A^{\mathrm{SD}}_{0,t}+\rho_k A^{R,\mathrm{GAE}}_{t},\qquad\rho_k\in[0,1]$
  - [x] value warmup
  - [ ] PVF analysis, comparison with SDFT

- **PI conditioned Q function actor-critic**

  - [ ] $\sum_{t=1}^T\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\cdot(Q_\phi(\hat{y}_t, s_t)-\beta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}))]$

- **Train self-teacher for hint generation**
  - [x] $R(h) = \alpha\,S(h)-\frac{|h|}{B}-\gamma\,T(h)$
  - [x] Tune hyperparameters
  - [ ] Other cost functions
  - [ ] Dual objective : $\min_\phi\;\mathbb E_{h\sim g_\phi}\left[C(h)+\gamma\,T(h)\right]\quad\text{subject to}\quad S(h)\ge \tau$
  - [ ] LoRA on top of self-teacher 
