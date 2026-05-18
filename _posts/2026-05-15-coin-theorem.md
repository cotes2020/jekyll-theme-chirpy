---
title: "A Beautiful Rock"
date: 2026-05-15
categories: [Math]
tags: [theorem]
math: true
---
>Every once in a while, in a period of time that feels long and filled with exhaustion, though very rarely, you might see something like this. Just thinking about it gets your heart racing, and you can feel your confidence coming back. After a long and strenuous climb to the top, it serves as the foothold that you desperately needed. It is not a miracle, maybe it's one out of a hundred, or even one out of a thousand,  but it's the one you went to reach and managed to grab. By grabbing and connecting these rare moments, you are able to keep climbing higher and higher.

My masters thesis was about proving the [Differential Privacy]({% post_url 2026-05-16-differential-privacy %}) of Thompson Sampling [add reference] for two armed bandits.
The progression of my master thesis demanded from the skill of proving high probability bounds. Calculating expectations of random variable was well known to me, but any progress demanded the skill of proving high probability bounds which inturn demanded a deeper knowledge of the underlying distribution.


## Preface
In a two-armed bandit setting with finite horizon $T$, we consider that the reward sequences are inputs to the algorithm and the action sequences are random outputs. Our goal was to show that for any, arbitrary, reward seqeuence $R \in [0.1]^T$ and its [neighbor]({% post_url 2026-05-16-differential-privacy %}) $R^\prime$, Thompson Sampling plays actions sequences $a_{1:t}$ such that

$$\sum_{t \leq T}\log\frac{P_t(A_t=a_t)}{P'_t(A_t=a_t)} \leq \log^2 T\quad\text{w.p.}\quad1-\delta$$

Sometime in July 2025, I believed in the ways of Taylor series. After plugging in probabilities for the closed form, I get something along the lines of

$$\sum_{t \leq T}\frac{f(t)}{\sqrt{n_2(t)+1}}$$

Here, $f$ represented the log derivative of $P_t$ w.r.t its parameters, and $n_2$ is the pull count of the some arm (wlog the second arm). Let $\tau_j$ refer to the time index when this second arm was played for the $j$th time. The problem is

$$\sum_{\tau_j}^{\tau_{j+1}} f(t).$$

Thompson sampling explores both arms initially and plays with constant probability. After sometime, the probability of the sub-optimal arm decreases with more and more until it becomes very unlikely the worse arm is ever played. In that case, if I showed $f(t)$ was bounded by a number, however small, the sum would not have a non-trivial $\Omega(T)$ bound.

>I figured the only way to bound this sum was to factor $P_t$ from $f$. Then, for example, in expectation, if I toss a fair coin, I should get heads in two tries or about $\frac{1}{b}$ tries if the coin had bias $b$. That is, a certain action is not being played because its chance is low, so would be the privacy loss and we would have reached the summit.

---
>**Lemma 1:** For $p \in (0,1)$ and $X \sim \mathrm{Geom}(p)$
>
>$$X < \frac{\log(1/\delta)}{p}\quad\text{w.p.}\quad 1 - \delta.$$


**Proof.**
We seek the smallest $k$ such that

$$\mathbb{P}(X > k) = (1-p)^k < \delta.$$

We use the standard inequality $1 - p < e^{-p}$. Moreover with $p \in [0,1]$, we also get

$$(1-p)^k < e^{-pk},$$

so it suffices to require $e^{-pk} \leq \delta$, i.e.
$$k \;\geq\; \frac{\log(1/\delta)}{p}.$$
---

However, what if the bias (although predictable) was changing with time? Most of the work poured into the main proof involved different approaches and iterations. Among many pages of scratches, this theorem sat close to my heart. This is one of the first (original) high probability bounds I ever worked on. This result turned out to provide a valid, illustrative, intermediate goal that gave me the confidence for the development of the algebra of my thesis.  Unfortunately, it will not make it into the final draft, I etch it on your screen because I cherished it. The original proof I wrote was not entirely checked, and is attached further below. Claude Code gave a slick proof and hopefully a great read.


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


**Proof**

Since we play the game only until we toss heads for the first time, define

$$w_t := \p(X_t=1|X_1=0,X_2=0,\ldots,X_{t-1}=0)$$

With this, the key observation is that $w_t$ is a known quantity. We are only evaluating this variable along a path with all tails. We define a function

$$
\begin{align*}
    S(T) &:= \sum_{t\leq T}w_t\\
    Y &= S(\tau - 1)
\end{align*}
$$

Here, note that $S$ is also a known function, not just predictable. Also, it is non-negative and strictly increasing.

And since $\tau$ is a stopping time, $\I(\tau = t-1)$ is indeed  $\mathcal{F}_t$-measurable. Define another known quantity $T^\star$, such that


$$
\begin{align*}
T^\star := \inf \{t \in \mathbb{N}: S(t) > \log(1/\delta)\}\\
\implies S(T^\star-1) \leq \log(1/\delta) < S(T^\star).
\end{align*}
$$

Now, if the set over which we take the infimum happens to be empty, then clearly $Y \leq \log(1/\delta)$ with probability 1. Therefore, for the non-trivial case, $T^\star$ is finite. So, with $S$ strictly increasing, we have that $S$ is invertible. Therefore, bounding $Y$ and $\tau$ are essentially the same problem. With $Y = S(\tau - 1)$,

$$\{\tau \leq T^\star\} = \{Y \leq \log(1/\delta)\}$$

Since $S$ is deterministic, and $w_t \in (0,1)$, we apply $1 - x \leq \exp(-x)$.

$$
\begin{align*}
    \p(\tau > T^\star) = \prod_{t\leq T^\star}(1-w_t) \leq \exp(-S(T^\star))\\
    \p\left(Y > \log(1/\delta)\right) \leq \exp(-\log(1/\delta)) = \delta.
\end{align*}
$$

<div style="text-align:right">
$\blacksquare$
</div>

---

**Proof (Claude).** Define the cumulative sum of biases:

$$S_t \coloneqq \sum_{k=1}^{t}b_k,\quad \text{so,}\;S_0=0.$$

With $0 < b_t < 1$, and $b_t$ predictable, $S_t$ is strictly increasing and predictable.
Define a new sequence:

$$M_0=1,\quad M_t = \I(X_t=0)\exp(b_t)\;M_{t-1}.$$

Starting from $1$, we keep on multiplying $\exp$ of the bias when we toss tails. The moment we toss heads, we set the sequence to zero and we never change it thereafter. Thus $M$ is an absorbing process.

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
