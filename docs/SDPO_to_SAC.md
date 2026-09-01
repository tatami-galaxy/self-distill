## SDPO gradient

Lets derive the SDPO gradient from the SDPO objective. As in the SDPO paper we derive (and use) the per-prefix semi-gradient and not the full gradient through the on-policy trajectory distribution  :

$$
\begin{aligned}

\mathcal{L}_{SD}&=\sum_{t=1}^TKL(\pi_{\theta}(\cdot|x,y_{<t})||sg(\pi_T(\cdot|x,f,y_{<t}) \\

\nabla_{\theta}\mathcal{L}_{SD}&=\nabla_{\theta}\sum_{t=1}^TKL(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})||sg(\pi_T(\hat{y_{t}}|x,f,y_{<t}) \\

&=\nabla_{\theta}\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\frac{\pi_{\theta}(\hat{y_{t}}|x,y_{<t})}{sg(\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))} \\

&\text{Let }A_t=\text{log}\frac{sg(\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))}{\pi_{\theta}(\hat{y_{t}}|x,y_{<t})}. \text{ Then,} \\

\nabla_{\theta}\mathcal{L}_{SD}&=-\nabla_{\theta}\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\cdot A_t \\

&=-\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_{\theta}A_t + A_t\nabla_{\theta}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}) \\

\text{Now, }\nabla_{\theta}A_t &= -\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}). \text{ Therefore the first term is,} \\
&\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}), \\ 
&\text{applying score function identity,} \\
&=\sum_{t=1}^T\sum_{\hat{y_{t}}}\nabla_{\theta}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})=\sum_{t=1}^T\nabla_{\theta}\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})=0 \\

\text{Therefore, } \\
\nabla_{\theta}\mathcal{L}_{SD}&=-\sum_{t=1}^T\sum_{\hat{y_{t}}}A_t\nabla_{\theta}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}) \\

&=-\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\cdot A_t \\

&=-\sum_{t=1}^T\mathbb{E}_{\hat{y_{t}}\sim\pi_\theta}[\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\cdot A_t] \\

\end{aligned}
$$

This means minimzing $\mathcal{L}_{SD}$ is the same as using $\sum_{t=1}^T\mathbb{E}_{\hat{y_{t}}\sim\pi_\theta}[\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})A_t]$ as the policy gradient. 


## Soft Actor Critic

Now lets write the SDPO objective in a slightly different way : 

$$
\begin{aligned}

\mathcal{L}_{SD}&=\sum_{t=1}^TKL(\pi_{\theta}(\cdot|x,y_{<t})||sg(\pi_T(\cdot|x,f,y_{<t}) \\

&=\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\frac{\pi_{\theta}(\hat{y_{t}}|x,y_{<t})}{sg(\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))} \\

&=\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})-\pi_{\theta}(\hat{y_{t}}|x,y_{<t})sg(\text{log}\pi_{T}(\hat{y_{t}}|x,f,y_{<t})) \\

&=\sum_{t=1}^T-H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})) - \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[sg(\text{log}\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))] \\\\

\text{Let } &\text{log}\pi_T(\hat{y_{t}}|x,f,y_{<t}))=Q_{\phi}(\hat{y}_t,s_t), \text{where }s_t = sg(x, f, y_{<t}).\text{ Then,} \\

\mathcal{L}_{SD}&=\sum_{t=1}^T-H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})) - \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}Q_{\phi}(\hat{y}_t,s_t), \\

&\text{we can remove the stopgrad since we optimize w.r.t }\theta.
\end{aligned}
$$

Minimizing $\mathcal{L}_{SD}$ is the same as :

$$
\underset{\theta}{\operatorname{argmax}} \sum_{t=1}^T \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}Q_{\phi}(\hat{y}_t,s_t) +\beta H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})), \text{where }\beta=1
$$

This is the (per-prefix) maximum entropy RL objective. Lets call it MaxEnt. Differentiating this w.r.t $\theta$ gives us the soft actor critic (SAC) policy gradient : 

$$
\begin{aligned}

\nabla_\theta\text{MaxEnt}&=\nabla_\theta\sum_{t=1}^T \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}Q_{\phi}(\hat{y}_t,s_t) +\beta H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})) \\

&=\sum_{t=1}^T\nabla_\theta\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})Q_{\phi}(\hat{y}_t,s_t)+\beta\nabla_\theta H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})) \\\\

&\text{Lets exclude the outer summation for now. Then the first term is : } \\

&\nabla_\theta\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})Q_{\phi}(\hat{y}_t,s_t) \\

&=\sum_{\hat{y_{t}}}\nabla_\theta\pi_{\theta}(\hat{y_{t}}|x,y_{<t})Q_{\phi}(\hat{y}_t,s_t) \\

&=\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})Q_{\phi}(\hat{y}_t,s_t) \\

&=\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})Q_{\phi}(\hat{y}_t,s_t)] \\\\

&\text{The second term is :} \\
&-\beta\nabla_\theta\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}) \\

&=-\beta\sum_{\hat{y_{t}}}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_\theta\pi_{\theta}(\hat{y_{t}}|x,y_{<t})-\beta\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}) \\

&=-\beta\sum_{\hat{y_{t}}}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_\theta\pi_{\theta}(\hat{y_{t}}|x,y_{<t}) \\

&=-\beta\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}) \\

&=-\beta\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})]. \\\\
\text{ Therefore,}  \\

\nabla_\theta\text{MaxEnt}&=\sum_{t=1}^T\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})Q_{\phi}(\hat{y}_t,s_t)]-\\
&\quad\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\beta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})] \\

&=\sum_{t=1}^T\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\cdot(Q_\phi(\hat{y}_t, s_t)-\beta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}))] \\

\text{If }\beta=1, \\
\nabla_\theta\text{MaxEnt}&=\sum_{t=1}^T\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\cdot A_t]=-\nabla_\theta\mathcal{L}_{SD}, \text{the SDPO gradient.}

\end{aligned}
$$

The soft $Q$ function can be fitted with a $1$-step or $\lambda$ returns more generally. We can also subtract the soft $V$ from the entropy regulazied soft $Q$ $(Q_\phi(\hat{y}_t, s_t)-\beta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}))$ to get an advantage estimate. This is then equal to the centered SDPO advantage at initialization (not the raw SDPO advantage). 

$$
\begin{aligned}

V^{soft}_\phi(s_t) &= \mathbb{E}_{\hat{y}_t\sim\pi_\theta}[Q_\phi(\hat{y}_t, s_t)-\beta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})] \\

1\text{-step target :} \\

y_t^{(1)}&=r(\hat{y}_t, s_t)+\gamma V^{soft}_\phi(s_{t+1}) \\

2\text{-step target :} \\

y_t^{(2)}&=r(\hat{y}_t, s_t)+\gamma[r(\hat{y}_{t+1}, s_{t+1})-\beta\text{log}\pi_{\theta}(\hat{y}_{t+1}|x,y_{<t+1})+\gamma V^{soft}_\phi(s_{t+2})] \\

n\text{-step target :} \\

y_t^{(n)}&=r(\hat{y}_t, s_t) + \sum_{k=1}^{n-1}\gamma^k(r(\hat{y}_{t+k}, s_{t+k})-\beta\text{log}\pi_{\theta}(\hat{y}_{t+k}|x,y_{<t+k}))+\gamma^nV^{soft}_\phi(s_{t+n}) \\

\lambda\text{-return :} \\

y_t^{(n+1)}-y_t^{(n)}&=\gamma^n[r(\hat{y}_{t+n}, s_{t+n})-\beta\text{log}\pi_{\theta}(\hat{y}_{t+n}|x,y_{<t+n}))+γV_\phi^{soft}​(s_{t+n+1}​)−V_\phi^{soft}​(s_{t+n}​)] \\

\text{Define } \delta_{t+k}&=r(\hat{y}_{t+k}, s_{t+k})-\beta\text{log}\pi_{\theta}(\hat{y}_{t+k}|x,y_{<t+k}))+γV_\phi^{soft}​(s_{t+k+1}​)−V_\phi^{soft}​(s_{t+k}​). \\

\text{ Therefore,} \\

y_t^{(n)}&=y_t^{(1)}+\sum_{k=1}^{n-1}\gamma^k\delta_{t+k} \\

y_t^{(\lambda)}&​=(1−\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}y_t^{(n)}. \text{ Then it can be shown that,} \\

y_t^{(\lambda)}&​=r(\hat{y}_t, s_t)+\gamma V^{soft}_\phi(s_{t+1})+\sum_{k=1}^{\infty}(\gamma\lambda)^k\delta_{t+k} \\

\text{The soft }Q \text{ can then be fitted by minimizing :} \\

\mathcal{L}(\phi)&=\mathbb{E}[(Q_\phi(\hat{y}_t, s_t)-y_t^{(\lambda)})^2]

\end{aligned}
$$

$Q_\phi$ is initialized with the teacher log-probability and subsequently trained toward the soft outcome $Q$.

## Sampled soft SARSA

Computing $V^{soft}$ exactly requires an expectation over the full vocabulary at every prefix. We can instead use the next action in the on-policy rollout as a Monte Carlo sample of this expectation. This is State–action–reward–state–action (SARSA). To distinguish the two inputs, define the observable actor state and the privileged critic state as

$$
o_t=(x,y_{<t}), \qquad s_t=(x,f,y_{<t}),
$$

and let the rollout policy be $\pi_b(a_t|o_t)$. During on-policy training, $\pi_b$ is the frozen policy $\pi_{old}$ that generated the batch. The soft value under this policy is

$$
V^{soft}_{\bar\phi}(s_{t+1})
=
\mathbb{E}_{a\sim\pi_b(\cdot|o_{t+1})}
\left[
Q_{\bar\phi}(s_{t+1},a)-\beta\log\pi_b(a|o_{t+1})
\right],
$$

where $\bar\phi$ denotes critic parameters held fixed while the targets are constructed. Since the observed next token was sampled as $a_{t+1}\sim\pi_b(\cdot|o_{t+1})$, the sampled soft-SARSA value is

$$
\widehat V^{SARSA}_{t+1}
=
Q_{\bar\phi}(s_{t+1},a_{t+1})
-\beta\log\pi_b(a_{t+1}|o_{t+1}).
$$

It is conditionally unbiased for the soft value:

$$
\mathbb{E}_{a_{t+1}\sim\pi_b}
\left[
\widehat V^{SARSA}_{t+1}\mid s_{t+1}
\right]
=
V^{soft}_{\bar\phi}(s_{t+1}).
$$

The one-step sampled soft-SARSA target is therefore

$$
Y_t^{(1)}
=
r_t
+\gamma(1-d_t)
\left[
Q_{\bar\phi}(s_{t+1},a_{t+1})
-\beta\log\pi_b(a_{t+1}|o_{t+1})
\right],
$$

where $d_t=1$ when action $a_t$ ends the episode. Thus a terminal action has target $Y_t=r_t$ and does not bootstrap from a state after termination. With outcome-only rewards, $r_t=0$ except on the final generated token, where it is the outcome reward.

Provided no terminal state occurs within the next $n$ transitions, the sampled $n$-step target is

$$
\begin{aligned}
Y_t^{(n)}
&=r_t
+\sum_{k=1}^{n-1}\gamma^k
\left[
r_{t+k}-\beta\log\pi_b(a_{t+k}|o_{t+k})
\right] \\
&\quad+\gamma^n
\left[
Q_{\bar\phi}(s_{t+n},a_{t+n})
-\beta\log\pi_b(a_{t+n}|o_{t+n})
\right].
\end{aligned}
$$

If a terminal action occurs first, the target stops there and the final bootstrap term is omitted. The corresponding finite-trajectory $\lambda$-return can be computed backwards without constructing every $n$-step return:

$$
\boxed{
G_t^\lambda
=
r_t
+\gamma(1-d_t)
\left[
-\beta\log\pi_b(a_{t+1}|o_{t+1})
+(1-\lambda)Q_{\bar\phi}(s_{t+1},a_{t+1})
+\lambda G_{t+1}^\lambda
\right]
}
$$

with $G_T^\lambda=r_T$ at the terminal token. At $\lambda=0$, this reduces to the one-step sampled soft-SARSA target. At $\lambda=1$, it becomes the sampled Monte Carlo soft return. The critic is fitted with

$$
\mathcal{L}_Q(\phi)
=
\mathbb{E}_t
\left[
\left(Q_\phi(s_t,a_t)-sg(G_t^\lambda)\right)^2
\right].
$$

For an on-policy rollout batch, the values $Q_{\bar\phi}(s_t,a_t)$ and the behavior log-probabilities can be cached before any optimization and kept fixed while forming and fitting these targets. This gives a fitted soft-SARSA update without requiring a separate target-model copy. If the actions instead come from a policy different from $\pi_b$, the estimator no longer directly targets $V^{soft,\pi_b}$ without an off-policy correction.

Sampled soft SARSA is used here to estimate the continuation value in the critic target. It is not necessary to use the same sample as a soft-$V$ baseline in the actor update. In particular, if

$$
Q_\phi(s_t,a)=c_\phi(s_t)+\ell_\phi(s_t,a),
\qquad
\ell_\phi(s_t,a)=\log\operatorname{softmax}(z_\phi(s_t))_a,
$$

then the action-independent $c_\phi(s_t)$ can be dropped as a policy-gradient baseline. The sampled actor score can be written as

$$
g_t
=
sg\left[
\ell_{\bar\phi}(s_t,a_t)
-\beta\log\pi_b(a_t|o_t)
\right].
$$

At initialization, with $c=0$, $\ell=\log\pi_T$, and $\beta=1$, this is exactly the raw SDPO log-ratio advantage. Subtracting the sampled value constructed from the same action would instead make the actor score identically zero; a centered actor advantage would require an independent baseline estimate or an explicit vocabulary expectation.

We can also have $\beta\neq1$ and still get the self-distillation initialization by initializing the soft $Q$ to be : 

$$Q_\phi(\hat{y}_t, s_t)=sg[\text{log}\pi_T(\hat{y_{t}}|x,f,y_{<t})+(\beta-1)\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})]$$
