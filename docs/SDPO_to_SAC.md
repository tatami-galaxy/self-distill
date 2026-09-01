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


## Maximum Entropy RL

Now lets write the SDPO objective in a slightly different way : 

$$
\begin{aligned}

\mathcal{L}_{SD}&=\sum_{t=1}^TKL(\pi_{\theta}(\cdot|x,y_{<t})||sg(\pi_T(\cdot|x,f,y_{<t}) \\

&=\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\frac{\pi_{\theta}(\hat{y_{t}}|x,y_{<t})}{sg(\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))} \\

&=\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})-\pi_{\theta}(\hat{y_{t}}|x,y_{<t})sg(\text{log}\pi_{T}(\hat{y_{t}}|x,f,y_{<t})) \\

&=\sum_{t=1}^T-H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})) - \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[sg(\text{log}\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))] \\\\

\end{aligned}
$$

Now let $\boxed{Q_{\phi}(\hat{y}_t,s_t)=\text{log}\pi_T(\hat{y_{t}}|x,f,y_{<t})),}$ where $s_t = sg(x, f, y_{<t})$. Then, 

$$
\mathcal{L}_{SD}=\sum_{t=1}^T-H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})) - \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}Q_{\phi}(\hat{y}_t,s_t)
$$


We can remove the stopgrad since we optimize w.r.t $\theta$. Minimizing $\mathcal{L}_{SD}$ is the same as :

$$
\boxed{\underset{\theta}{\operatorname{argmax}} \sum_{t=1}^T \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}Q_{\phi}(\hat{y}_t,s_t) +\beta H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})), \text{where }\beta=1}
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

\end{aligned}
$$

Therefore,

$$
\boxed{\nabla_\theta\text{MaxEnt}=\sum_{t=1}^T\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\cdot(Q_\phi(\hat{y}_t, s_t)-\beta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}))]}
$$

If $\beta=1$, $\nabla_\theta\text{MaxEnt}=\sum_{t=1}^T\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\cdot A_t]=-\nabla_\theta\mathcal{L}_{SD}$, the SDPO gradient.


## Learning Q

The soft $Q$ function can be fit with a $1$-step or $\lambda$ returns more generally. We can also subtract the soft $V$ from the entropy regulazied soft $Q$ $(Q_\phi(\hat{y}_t, s_t)-\beta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}))$ to get an advantage estimate. This is then equal to the centered SDPO advantage at initialization (not the raw SDPO advantage). 

$$
\boxed{V^{soft}_\phi(s_t) = \mathbb{E}_{\hat{y}_t\sim\pi_\theta}[Q_\phi(\hat{y}_t, s_t)-\beta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})]}
$$

$$
\begin{aligned}

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

\end{aligned}
$$

$$
\boxed{y_t^{(\lambda)}​=r(\hat{y}_t, s_t)+\gamma V^{soft}_\phi(s_{t+1})+\sum_{k=1}^{\infty}(\gamma\lambda)^k\delta_{t+k}}
$$

The soft $Q$ can then be fit by minimizing :

$$
\boxed{\mathcal{L}(\phi)=\mathbb{E}[(Q_\phi(\hat{y}_t, s_t)-y_t^{(\lambda)})^2]}
$$


$Q_\phi$ is initialized with the teacher log-probability and subsequently trained toward the soft outcome $Q$. We can also have $\beta\neq1$ and still get the self-distillation initialization by initializing the soft $Q$ to be : 

$$Q_\phi(\hat{y}_t, s_t)=sg[\text{log}\pi_T(\hat{y_{t}}|x,f,y_{<t})+(\beta-1)\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})]$$

Computing $V_\phi^{soft}$ exactly requires an expectation over the full vocabulary at every prefix. One option is to instead take an expectation over top-$K$ tokens but then the value estimate is no longer unbiased. Another option is to use the unbiased but high variance SARSA estimate : 

$$
\boxed{V_{SARSA}=Q_\phi(\hat{y}_{t+1}, s_{t+1})-\beta\text{log}\pi_{\theta}(\hat{y}_{t+1}|x,y_{<t+1})}
$$

