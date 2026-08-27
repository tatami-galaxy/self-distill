### Training

- [x] Deepmath
- [ ] Reasoning Gym`
- [ ] Countdown-CoT-20k
- [ ] Hyperparameter sweep, more steps

### Benchmarks

- [x] Aime 24, 25
- [-] Beyondaime
- [ ] Reasoning Gym
- [ ] Countdown heldout

### Models

- [x] think models
- [ ] sft-ed models
- [-] instruct models

### SD Objective

- [x] Reverse KL
- [-] JSD
- [-] Forward CE

### Research

- **PI conditioned V**
  - [ ] $A^{\mathrm{actor}}_{t,k}=(1-\rho_k)A^{\mathrm{SD}}_{0,t}+\rho_k A^{R,\mathrm{GAE}}_{t},\qquad\rho_k\in[0,1]$
  - [x] value warmup
  - [ ] PVF analysis, comparison with SDFT

- **PI conditioned Q function actor-critic**

  $J_{\mathrm{SDPO}}\propto\sum_t\left\{\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s_t)}\left[Q(s_t,a)\right]+H\!\left(\pi_\theta(\cdot\mid s_t)\right)\right\}, \text{where }\\ Q(s_t,a)=\operatorname{logit}_{\mathrm{teacher}}(a\mid s_t).$

- **Train self-teacher for hint generation**
  - [x] $R(h) = \alpha\,S(h)-\frac{|h|}{B}-\gamma\,T(h)$
  - [x] Tune hyperparameters
  - [ ] Other cost functions
  - [ ] Dual objective : $\min_\phi\;\mathbb E_{h\sim g_\phi}\left[C(h)+\gamma\,T(h)\right]\quad\text{subject to}\quad S(h)\ge \tau$
  - [ ] LoRA on top of self-teacher 
