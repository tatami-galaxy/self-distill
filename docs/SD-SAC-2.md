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

&=\sum_{t=1}^T-H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})) - \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[sg(\text{log}\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))] \\

\end{aligned}
$$

Minimizing $\mathcal{L}_{SD}$ is the same as : 

$$
\underset{\theta}{\operatorname{argmax}} \sum_{t=1}^T \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[sg(\text{log}\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))] +\beta H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})), \text{where }\beta=1
$$

This is identical to the maximum entropy RL objective as noted by the SDPO paper : 

$$
\underset{\theta}{\operatorname{argmax}} \sum_{t=1}^T \mathbb{E_{\hat{y_{t}}\sim\pi_\theta}}[r(\hat{y_{t}},s_t)] +\beta H(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})), \text{where }\beta=1
$$

Therefore $\text{log}\pi_{T}(\hat{y_{t}}|x,f,y_{<t})$ can be viewed as an implicit reward. 