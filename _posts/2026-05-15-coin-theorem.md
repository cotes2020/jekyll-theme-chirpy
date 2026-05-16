---
title: "Coin Theorem"
date: 2026-05-15
categories: [Math]
tags: [theorem]
math: true
---

Most of the work poured into the culmination of my Master Thesis involved different approaches and iterations. Among many pages of scratches, this theorem sat close to my heart. This is one of the first (original) high probability bounds I ever worked on which was crucial in the development of the algebra of my thesis. Unfortunately, it will not make it into the final draft, I etch it on your screen because I cherished it. The original proof I wrote was not entirely correct, and is attached further below. Claude Code gave a slick proof and hopefully a great read.


$$\newcommand{\p}{\mathcal{P}}$$
$$\newcommand{\coloneqq}{:=}$$
$$\newcommand{\I}{\mathbb{1}}$$
$$\newcommand{\E}{\mathbb{E}}$$

### Game
Suppose you have full power to change the bias of coin before you toss it. Every time you toss a tail, you get some reward every step. However, if you toss a head, the game is over. The reward you get is equal to the bias of the coin you just tossed. What is the high probabibility upper bound on the total reward earned by you?


### Math

<blockquote>
<strong>Theorem (Coin).</strong> Suppose we have a coin that changes its bias every time it is tossed. Let $X_t=1$ if the toss at time $t$ results in heads and $X_t=0$ if tails, and let $\mathcal{F}_t = \sigma(X_1, \ldots, X_t)$ be the natural filtration. Define $b_t$ as the conditional probability of heads given the past:
$$b_t \coloneqq \p(X_t=1 \mid \mathcal{F}_{t-1})$$ and assume $b_t \in (0,1)$ almost surely. Let $\tau$ be the first time we toss heads. That is, we toss this coin $\tau$ times.
$$\tau = \min\{t \mid X_t=1\}$$
Define the cost variable $Y$ as the sum of biases until we toss heads for the first time. That is:
$$Y \coloneqq \sum_{t=1}^{\tau-1} b_t$$ Then, for any $\delta \in (0, 1)$,
$$Y \leq \log(1/\delta) \quad \text{w.p.} \quad 1-\delta$$
</blockquote>

*Proof (Claude).* Define the cumulative sum of biases:

$$S_t \coloneqq \sum_{k=1}^{t}b_k,\quad \text{so,}\;S_0=0.$$

With $0 < b_t < 1$, and $b_t$ predictable, $S_t$ is strictly increasing and predictable.
Define a new sequence:

$$M_0=1,\quad M_t = \I(X_t=0)\exp(b_t)\;M_{t-1}.$$

Starting from $1$, we keep on multiplying $\exp$ of the bias when we toss tails. The moment we toss heads, we set the sequence to zero and we never change it thenonwards. Thus $M$ is an absorbing process.

Clearly, $M_t$ is non-negative.
With $\E_t[.] := \E[.|\mathcal{F}_{t-1}]$, we have:

$$
\begin{align*}
\E_t[M_t] &= \E[\I(X_t=0)\exp(b_t)\;M_{t-1}|\mathcal{F}_{t-1}]\\
&= M_{t-1}\exp(b_t)\;\E[\I(X_t=0)|\mathcal{F}_{t-1}]\\
&= M_{t-1}\exp(b_t)\;\p(X_t=0|\mathcal{F}_{t-1})\\
&= M_{t-1}\exp(b_t)\;(1-b_t)\\
&\leq M_{t-1} \qquad \text{since $\exp(-x) \geq 1-x$}
\end{align*}
$$

Hence, $(M_t)_{t \geq 0}$ is a non-negative supermartingale.

Moreover, since $S_t$ is increasing, we get:

$$Y = \sum_{t < \tau} b_t = \sup_{t < \tau}S_t$$

$$\sup_{t \geq 0}M_t = \sup_{t < \tau} \exp(S_t) = \exp(\sup_{t < \tau} S_t) = \exp(Y)$$


Using Ville's inequality:

$$\p\left(\sup_{t \geq 0} M_t \geq a\right)\leq \frac{\E[M_0]}{a}.$$

and also $M_0=1$. This essentially boils down to:

$$\p(\exp(Y) \geq a) \leq \frac{1}{a}$$

Setting $a=1/\delta$, we get the required result.
<div style="text-align:right">
$\blacksquare$
</div>
