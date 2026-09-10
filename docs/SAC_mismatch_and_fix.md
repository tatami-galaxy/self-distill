# Why the SDPO-initialized SAC degrades, and two consistent fixes

Companion to `SDPO_to_SAC.md`. Notation: $\ell_T(a|s)=\log\pi_T(a|x,f,y_{<t})$ (privileged
teacher), $\pi_\theta$ the student, $H_t$ the student entropy at prefix $t$,
$\mathrm{KL}_t=\mathrm{KL}(\pi_\theta(\cdot|s_t)\,\|\,\pi_T(\cdot|s_t))$, $R\in\{0,1\}$ the
terminal correctness reward.

## 1. What the runs did

All five runs under `outputs/sac/` share one signature. Numbers below are from
`trainer_state.json` (logging every 10 optimizer steps, 16 rollouts per step).

Qwen3-1.7B, hint PI, linear head, $\lambda=0$:

| step | reward_mean | sampled_q | soft_values | q_targets | clipped_ratio | mean_length |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.66 | 0.29 | 0.55 | 0.55 | 0.32 | 5939 |
| 70 | 0.44 | 4.31 | 4.58 | 4.58 | 0.58 | 7076 |
| 130 | 0.38 | 6.73 | 6.99 | 6.99 | 0.72 | 7531 |
| 190 | 0.20 | 8.69 | 8.98 | 8.98 | 0.87 | 7979 |

- $Q$, $V$ and the targets drift **upward linearly** (about 0.047 per step) and never settle.
  The gap $V - Q_{\text{sampled}}\approx 0.25$–$0.45$ is constant.
- The policy stops terminating: clipped ratio goes 0.32 → 0.87, so reward goes to 0.2.
  AIME24 pass@1 holds near base (0.44–0.50) until checkpoint 200 (0.34).
- Same shape for full PI (reward 0.67 → 0.19) and for Qwen3-4B hint (0.64 → 0.006).
- $\lambda=1$: the Monte-Carlo soft return is **774 at step 10** ($q\_loss\approx 10^6$),
  reward is 0 by step 50.
- `state_scaled_linear`: the learned multiplier $\alpha(s)$ falls 0.96 → 0.23, i.e. the critic
  shrinks the teacher term, reward is 0 by step 70.

The `soft_advantages` mean stays at $\sim 10^{-3}$ and `actor_loss` at $\sim 0$ throughout, so the
damage is not a large advantage. It is a systematically *biased* one (Section 2c).

## 2. Mechanism

### (a) $Q_0=\ell_T$ is the soft $Q$ of a different MDP than the one being fit

`SDPO_to_SAC.md` shows the per-prefix SDPO loss equals a MaxEnt objective with
$Q_\phi(s_t,\hat y_t)=\ell_T(\hat y_t|s_t)$ and $\beta=1$. That is a **contextual bandit**
at every prefix: reward $\ell_T$, no future term, i.e. $\gamma=0$. $\ell_T$ is the *exact* soft
$Q$ of that bandit.

The trainer then regresses $Q$ onto the soft Bellman target of an **entirely different MDP**:
per-token reward 0, terminal reward $R$, entropy bonus $\beta=1$, $\gamma=1$. Its fixed point is

$$
Q^*(s_t,a_t)=\mathbb{E}\Big[R+\sum_{k>t}H_k\Big],\qquad
V^*(s_t)=\mathbb{E}\Big[R+\sum_{k\ge t}H_k\Big].
$$

With $H\approx 0.3$–$0.45$ nats/token and 6000-token completions, that is $\sim 2000$ nats,
of which the outcome contributes at most 1. $Q_0=\ell_T\approx -1$ per token is nowhere near
it, and nothing in the target refers to $\ell_T$. So "critic learning has no requirement to
preserve the teacher's preferences" is exactly right: the teacher only ever lived in the
initialization, and the target erases it.

### (b) The linear drift is an entropy ratchet, not slow convergence

At $\lambda=0$ the target is $y_t=V(s_{t+1})$ with $V(s)=\mathbb{E}_{\pi_\theta}[Q]+H$.
Whatever $Q$ currently is, the target sits $H$ above its mean. The critic chases it, $V$ moves
up by the same amount, and so on. Predicted drift per "critic convergence time" $\approx H$;
observed $V-Q\approx 0.25$–$0.45$ and 0.047/step. The head only reaches $\sim 9$ in 200 steps
because it learns at lr $10^{-5}$ through an action-dependent projection; the target it is
chasing is $\sim 2000$.

At step 0 the mismatch is also visible directly: $y_0=V_0(s_{t+1})=-\mathrm{KL}_{t+1}\approx
-0.015$ (hint) while $Q_0=\ell_T(a_t|s_t)$ with mean $-H-\mathrm{KL}\approx -0.4$, so
$q\_loss_0\approx\mathbb{E}[\ell_T^2]\approx 0.8$–$1.4$, as logged. The very first critic
update is "raise $Q$ on every sampled token by $|\ell_T|$".

### (c) Why the policy stops terminating

EOS is terminal, so $Q(s,\text{EOS})\to R\in\{0,1\}$. Every other token bootstraps into
$V(s')$, which carries all future entropy and is drifting upward. The actor advantage of EOS
relative to continuing is $R-V(s')$, i.e. $\approx -9$ by step 190 and heading to $-2000$.
The MaxEnt objective with $\beta=1$ *pays for length*. This is the clipped-ratio curve.
The $\lambda=1$ and state-scaled runs simply reach the same place faster because their
targets or their parameterization expose the full offset immediately.

### (d) Scales, per token

| quantity | 1.7B hint | 1.7B full | source |
|---|---:|---:|---|
| $\ell_T$ (mean, $= -H-\mathrm{KL}$) | $\approx -0.4$ | $\approx -0.5$ | logged $q\_loss_0$ |
| raw SD advantage $\ell_T-\log\pi_\theta$ (mean $=-\mathrm{KL}_t$) | $-0.015$ | $-0.11$ | `advantage_dynamics` step 0 |
| entropy return $\sum_{k>t}H_k$ | $10^3$–$3\cdot 10^3$ | same | $H\times$ length |
| outcome $R$ | 0/1 per sequence | | |
| per-token outcome credit $\Delta P(\text{correct})$ | $\sim 10^{-2}$, mostly 0 | | Vine, `advantage_comparison` |

Two further facts from `results/advantage_comparison`: per-token OPSD credit has
Pearson $\le 0.03$ with Vine outcome credit, and Vine itself has split-half reliability
$0.00$ (tokens) / $0.09$ (steps) at $K=16$ while **prefix values are reliable at 0.95**.
So the outcome signal is learnable at the *state* level and essentially unmeasurable at the
*per-action* level from a handful of rollouts. Any design that asks a per-action residual to
absorb the outcome from single samples is fighting that.

## 3. The fix: put the teacher in the reward, not in the initialization

Replace the outcome-only MDP with the MDP whose soft $Q$ **is** $\ell_T$ at $\gamma=0$:

$$
\boxed{r_t=\ell_T(a_t|s_t)+\alpha\,R\,\mathbb{1}[t=T]},\qquad \beta=1 .
$$

Soft Bellman: $Q(s_t,a_t)=r_t+\gamma V(s_{t+1})$, $V(s)=\mathbb{E}_{\pi_\theta}[Q(s,a)-\log\pi_\theta(a|s)]$.

**No ratchet.** $\mathbb{E}_{\pi_\theta}[\ell_T]-\mathbb{E}_{\pi_\theta}[\log\pi_\theta]=
-\mathrm{KL}_t$, so the teacher reward's expectation cancels the entropy bonus exactly:

$$
V(s_t)=-\sum_{k\ge 0}\gamma^k\,\mathbb{E}[\mathrm{KL}_{t+k}]+\alpha\gamma^{T-t}P(\text{correct}|s_t)\le \alpha ,
$$

$$
Q(s_t,a_t)=\ell_T(a_t|s_t)-\gamma\sum_{k\ge 1}\gamma^{k-1}\mathbb{E}[\mathrm{KL}_{t+k}\mid s_t,a_t]
+\alpha\gamma^{T-t}P(\text{correct}\mid s_t,a_t).
$$

- $\gamma=0$: $Q\equiv\ell_T=Q_0$ exactly. Initial $q\_loss$ is $\approx 0$ except at the
  terminal token where it is $\alpha^2 R^2$. The critic has nothing to unlearn.
- $\gamma\to 1$: the residual learns two things. A **look-ahead SD term**, "after $a$, how much
  will the student keep disagreeing with the privileged teacher", bounded by cumulative KL
  ($\approx 0.015\times 6000\approx 90$ nats for hint, $\approx 1.5$ nats at $\gamma=0.99$).
  And the **outcome term** $\alpha P(\text{correct}|s,a)$.
- EOS is no longer penalized: $Q(s,\text{EOS})=\ell_T(\text{EOS}|s)+\alpha R$ versus
  $Q(s,a)=\ell_T(a|s)-\gamma(\text{future KL})+\ldots$. Length is not rewarded.

**Actor gradient.** With $A_t=Q(s_t,a_t)-\log\pi_\theta(a_t|s_t)-V(s_t)$:

$$
A_t=\underbrace{\ell_T(a_t|s_t)-\log\pi_\theta(a_t|s_t)+\mathrm{KL}_t}_{\text{centered SDPO advantage}}
+\gamma\big[\bar V(s_{t+1}\mid a_t)-\mathbb{E}_{a\sim\pi_\theta}\bar V(s_{t+1}\mid a)\big]
+\alpha\gamma^{T-t}\big[P(\text{c}|s_t,a_t)-P(\text{c}|s_t)\big].
$$

At initialization the second and third brackets are zero (zero-init head), so the gradient
is *exactly* the SDPO gradient with no warm-up, and outcome credit fades in as the critic
learns. This is the "start from the SD gradient, then edit it with outcome regression" goal,
with the edit living in additive terms that cannot erase the teacher term.

**Scale.** $\alpha$ now has units: nats of teacher log-probability per correct answer. The
per-token outcome advantage is $\alpha\,\Delta P\approx 10^{-2}\alpha$ against an SD advantage
of $0.015$ (hint) to $0.11$ (full). So $\alpha\in[1,10]$ is the natural range and $\alpha=1$
is a sensible first value. The mismatch is no longer a corruption of the critic; it is one
explicit knob.

### Two-stream form (recommended implementation)

Keep the SD and outcome parts as separate return streams so each has its own discount and
the outcome part can use a long horizon while the SD part stays local:

$$
Q(s,a)=Q^{SD}(s,a)+\alpha\,Q^{R}(s,a),\qquad
Q^{SD}(s,a)=\ell_T(a|s)+c_T(s,a),\qquad Q^{R}(s,a)=v_R(s)+c_R(s,a),
$$

with $c_T,c_R$ the existing zero-initialized linear residuals $u_a^\top D\,h_T(s)$ and
$v_R(s)=w^\top h_T(s)$ a new zero-initialized **state-only scalar** (this is the missing
state-only component named in the `train_sac.py` docstring; a state offset cancels in
$Q-V$ so it never touches the actor, it only stops $D$ from having to fake it).

Targets, both with the existing forward-view $\lambda$ recursion:

$$
G^{SD}_t=\ell_T(a_t|s_t)+\gamma_T\Big[(1-\lambda)V^{SD}_{t+1}+\lambda\big(G^{SD}_{t+1}-\log\pi_\theta(a_{t+1}|s_{t+1})\big)\Big],
\quad V^{SD}(s)=\mathbb{E}_{\pi_\theta}[Q^{SD}-\log\pi_\theta]
$$

$$
G^{R}_t=R\,\mathbb{1}[t=T]+\gamma\Big[(1-\lambda)V^{R}_{t+1}+\lambda G^{R}_{t+1}\Big],
\quad V^{R}(s)=\mathbb{E}_{\pi_\theta}[Q^{R}] .
$$

The entropy term belongs to the SD stream only (it is SDPO's entropy); the outcome stream is
a plain hard $Q$ in $[0,1]$. Critic loss $=(Q^{SD}-G^{SD})^2+(Q^R-G^R)^2$ on scorable tokens;
with $\gamma_T=0$ the first term is identically zero and $c_T$ is not needed.

Two settings of one trainer:

| | $\gamma_T$ | what $Q^{SD}$ is | critic scale |
|---|---|---|---|
| **F1** KL-to-privileged-teacher regularized outcome RL | 0 | $\ell_T$ frozen | $Q^R\in[0,1]$ |
| **F2** SDPO-consistent soft MDP | $\gamma$ or $0.99$ | $\ell_T$ + look-ahead disagreement | $+$ future KL |

F1 is the same object as KL-regularized RLHF with reference $\pi_T(\cdot|x,f)$ and the KL
kept in the loss rather than in the return: $\nabla\,\mathbb{E}_\pi[Q^R]-\nabla\,\mathrm{KL}(\pi_\theta\|\pi_T)$.
Its $R=0$ special case has $Q^{R*}\equiv 0$ and $\pi^*=\pi_T$, i.e. SDPO. It is also the
`todo.md` item $A=(1-\rho)A^{SD}+\rho A^{R}$ with $\rho\leftrightarrow\alpha$ and the outcome
critic reading the privileged teacher's hidden states, as in `train_ppo_pi.py`.

### What to expect at step 0, as a check

- `q_loss` $\approx \alpha^2\,\mathbb{E}[R^2]/\text{len}\approx 10^{-4}$ (F1) instead of 0.8.
- `sampled_q` $\approx\ell_T$, mean $\approx -0.4$, and it should **stay** there.
- `soft_values` $\approx -\mathrm{KL}\approx -0.015$; under F2 with $\gamma=1$ it drifts to
  $\approx -90$ at rollout start, which the scalar $v$ absorbs.
- `state_scale` $\approx 1$ if the multiplicative head is kept.
- clipped ratio flat at the base model's $\approx 0.3$; reward flat or rising.

Recommended first sweep on 1.7B hint (SDFT hint reference: 0.47–0.50 AIME24; current SAC
0.44 → 0.34): F1 with $\alpha\in\{0,1,5\}$, $\lambda=1$ for the outcome stream (TD from a
zero critic over 6000 tokens propagates too slowly, and Vine says prefix values are the
reliable part); then F2 with $\gamma_T\in\{0.99,1\}$. $\alpha=0$ must reproduce SDFT hint
exactly; that is the regression test.

## 4. Choices deliberately not taken

- **Whitening / normalizing $Q$ targets.** Would hide the ratchet, not remove it, and would
  still leave EOS penalized relative to continuation.
- **$\beta\ll 1$ to tame the entropy.** Scales the SD gradient down by the same $\beta$ and
  leaves the outcome-only MDP's fixed point unrelated to $\ell_T$.
- **Critic warm-up / random $Q$.** Sensible as a pure-SAC baseline, but it gives up the
  free SD gradient that motivates the whole design, and under the outcome-only MDP it still
  learns to pay for length.
- **Target network / twin $Q$.** Stabilizers for a well-posed target. The target here was
  not well posed.
