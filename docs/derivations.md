Lets derive the SDPO gradient from the SDPO objective :

$$
\begin{aligned}

\mathcal{L}_{SD}&=\sum_{t=i}^TKL(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})||sg(\pi_T(\hat{y_{t}}|x,f,y_{<t}) \\

\nabla_{\theta}\mathcal{L}_{SD}&=\nabla_{\theta}\sum_{t=i}^TKL(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})||sg(\pi_T(\hat{y_{t}}|x,f,y_{<t}) \\

&=\nabla_{\theta}\sum_{t=i}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\frac{\pi_{\theta}(\hat{y_{t}}|x,y_{<t})}{sg(\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))} \\

&\text{Let }A_t=\text{log}\frac{sg(\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))}{\pi_{\theta}(\hat{y_{t}}|x,y_{<t})}. \text{ Then,} \\

\nabla_{\theta}\mathcal{L}_{SD}&=-\nabla_{\theta}\sum_{t=i}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})A_t \\

&=-\sum_{t=i}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_{\theta}A_t + A_t\nabla_{\theta}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}) \\

\text{Now, }\nabla_{\theta}A_t &= -\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}). \text{ Therefore the first term is,} \\
&\sum_{t=i}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}) \\
&=\sum_{t=i}^T\sum_{\hat{y_{t}}}\nabla_{\theta}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})=\sum_{t=i}^T\nabla_{\theta}\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})=0 \\

\text{Therefore, } \\
\nabla_{\theta}\mathcal{L}_{SD}&=-\sum_{t=i}^T\sum_{\hat{y_{t}}}A_t\nabla_{\theta}\pi_{\theta}(\hat{y_{t}}|x,y_{<t}) \\

&=-\sum_{t=i}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})A_t \\

&=-\sum_{t=i}^T\mathbb{E}_{\hat{y_{t}}\sim\pi_\theta}[\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})A_t] \\

\end{aligned}
$$

This means minimzing $\mathcal{L}_{SD}$ is the same as using $\sum_{t=i}^T\mathbb{E}_{\hat{y_{t}}\sim\pi_\theta}[\nabla_{\theta}\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})A_t]$ as the policy gradient. Now lets write the SDPO objective in a slightly different way : 

$$
\begin{aligned}

\mathcal{L}_{SD}&=\sum_{t=i}^TKL(\pi_{\theta}(\hat{y_{t}}|x,y_{<t})||sg(\pi_T(\hat{y_{t}}|x,f,y_{<t}) \\

&=\sum_{t=i}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\frac{\pi_{\theta}(\hat{y_{t}}|x,y_{<t})}{sg(\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))} \\

&=\sum_{t=i}^T\sum_{\hat{y_{t}}}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})\text{log}\pi_{\theta}(\hat{y_{t}}|x,y_{<t})-\pi_{\theta}(\hat{y_{t}}|x,y_{<t})sg(\pi_{T}(\hat{y_{t}}|x,f,y_{<t}))

\end{aligned}
$$