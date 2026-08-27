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

This means minimzing $\mathcal{L}_{SD}$ is the same as using $\sum_{t=1}^T\mathbb{E}_{\hat{y_{t}}\sim\pi_\theta}[\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})A_t]$ as the policy gradient. Now lets write the SDPO objective in a slightly different way : 

$$
\begin{aligned}

\mathcal{L}_{SD}&=\sum_{t=1}^TKL(\pi_{\theta}(\cdot|x,y_{<t})||sg(\pi_T(\cdot|x,f,y_{<t}) \\

&=\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\frac{\pi_{\theta}(\hat{y_{t}}|x,y_{<t})}{sg(\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))} \\

&=\sum_{t=1}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})-\pi_{\theta}(\hat{y_{t}}|x,y_{<t})sg(\text{log}\pi_{T}(\hat{y_{t}}|x,f,y_{<t})) \\

&=\sum_{t=1}^T-H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})) - \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}sg(\text{log}\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))) \\\\

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

&\text{The the second term is :} \\
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
\nabla_\theta\text{MaxEnt}&=\sum_{t=1}^T\mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[\nabla_\theta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\cdot A_t]=\nabla_\theta\mathcal{L}_{SD}, \text{the SDPO gradient.}

\end{aligned}
$$

The soft $Q$ function can be fitted with a $1$-step or $\lambda$ returns more generally. We can also subtract the soft $V$ from the soft $Q$ to get an advantage estimate. This is then equal to the centered SDPO advantage at initialization (not the raw SDPO advantage). 

$$
\begin{aligned}

V^{soft}_\phi(s_t) &= \mathbb{E}_{\hat{y}_t\sim\pi_\theta}[Q_\phi(\hat{y}_t, s_t)-\beta\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}))] \\

1\text{-step target :} \\

y_t &=r(\hat{y}_t, s_t)+\gamma V^{soft}_\phi(s_{t+1}) \\

\end{aligned}
$$