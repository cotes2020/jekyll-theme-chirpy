---
title: "f-DP and Gaussian Differential Privacy"
date: 2026-05-16
categories: [Math]
tags: [differential-privacy, statistics, hypothesis-testing]
math: true
---

In the [previous post]({% post_url 2026-05-16-differential-privacy %}), we introduced differential privacy and the privacy loss random variable. Here we lift the definition to a richer framework — one that describes privacy not as two numbers but as an entire curve.

## Hypothesis Testing View of Differential Privacy

Differential privacy can be naturally interpreted through the lens of hypothesis testing, where an adversary attempts to distinguish between two neighboring datasets based on the output of the mechanism. The following exposition is adapted from Dong, Roth, and Su [^gdp].

Suppose an adversary observes the output $M(\cdot) = x$ and must determine whether the underlying dataset was $S$ or its neighbor $S'$. They formulate the test:

- **Null hypothesis** ($H_0$): the dataset is $S'$,
- **Alternate hypothesis** ($H_1$): the dataset is $S$.

The adversary selects a rejection region $O$ such that the Type I error satisfies $\mathcal{P}(M(S') \in O) = \alpha$. Under differential privacy, the power of the test is bounded by

$$\mathcal{P}(M(S) \in O) := 1 - \beta \leq e^{\epsilon} \alpha + \delta.$$

For suitably small $(\epsilon, \delta)$, any adversarial test has low distinguishing power. But $(\epsilon,\delta)$ only constrains the trade-off at a *single* significance level $\alpha$. A more complete picture tracks the entire curve.

## Rejection Rules and Trade-off Functions

A *rejection rule* $\phi$ maps outputs to $[0,1]$, encoding the probability of rejecting the null hypothesis. The two types of errors are:

- **Type I error** $(\alpha_\phi)$: probability of rejecting $H_0$ when $H_0$ is true.
- **Type II error** $(\beta_\phi)$: probability of failing to reject $H_0$ when $H_1$ is true.

> **Definition (Trade-off Function [^gdp]):** Given two distributions $P$ and $Q$, the *trade-off function* $T(P,Q): [0,1] \to [0,1]$ maps a Type I error level $\alpha$ to the minimal achievable Type II error:
>
> $$T(P,Q)(\alpha) = \inf\{ \beta_\phi : \alpha_\phi \leq \alpha \},$$
>
> where the infimum is over all randomized rejection rules $\phi$.

Since we are evaluating the infimum over all possible rejection rules, the trade-off function captures the fundamental difficulty of distinguishing $P$ from $Q$. A larger trade-off function (higher curve) means the two distributions are harder to tell apart — better privacy.

## $f$-Differential Privacy

Building on trade-off functions, $f$-DP provides a functional generalization of $(\epsilon,\delta)$-DP.

> **Definition ($f$-DP [^gdp]):** A mechanism $M$ satisfies $f$-Differential Privacy if, for all neighboring datasets $S$ and $S'$:
>
> $$T(M(S), M(S')) \geq f,$$
>
> where $f$ is a trade-off function.

In this formulation, privacy is described by an entire curve, fully characterizing the adversary's advantage at every significance level. $(\epsilon,\delta)$-DP is a special case of $f$-DP, where $f$ corresponds to a specific piecewise linear trade-off function.

## Gaussian Differential Privacy

Among all $f$-DP guarantees, Gaussian Differential Privacy (GDP) stands out as particularly natural and convenient.

> **Definition ($\mu$-GDP [^gdp]):** A mechanism satisfies $\mu$-GDP if, for all neighboring datasets $S$ and $S'$:
>
> $$T(M(S), M(S')) \geq G_\mu, \qquad G_\mu(\alpha) = \Phi\!\left(\Phi^{-1}(1-\alpha) - \mu\right),$$
>
> where $\Phi$ is the standard normal CDF.

This is precisely the trade-off function one obtains when trying to distinguish $\mathcal{N}(0,1)$ from $\mathcal{N}(\mu, 1)$ using a single sample. The parameter $\mu$ directly controls the hardness of this test: larger $\mu$ means easier to distinguish, i.e., weaker privacy.

![Trade-off functions for shifted Gaussians](/assets/img/tradeoff_lines.png){: width="450"}
*Trade-off curves $G_\mu$ for $\mu = 1, 2, 3, 4$: minimum Type II error $\beta$ vs. significance level $\alpha$.*

The curves are Blackwell-ordered — a higher $\mu$ lies strictly below a lower one at every $\alpha$, meaning the two distributions are harder to confuse at every significance level simultaneously, not just on average. Concretely, at $\alpha = 5\%$: a $\mu=1$-GDP mechanism still forces ~65% Type II error on the adversary (good privacy), while a $\mu=4$ mechanism drops near zero (essentially no protection).

GDP enjoys several favorable properties:
- It is fully described by a single parameter $\mu$.
- It preserves the hypothesis testing interpretation.
- Just as the central limit theorem tells us that sums of random variables converge to a Gaussian, composition of multiple privacy mechanisms converges to GDP in the limit (Theorem 3.6 of [^gdp]).

## Conversion to $(\epsilon,\delta)$-DP

> **Corollary [^gdp]:** A mechanism is $\mu$-GDP if and only if it is $(\epsilon, \delta(\epsilon))$-differentially private for all $\epsilon > 0$, where
>
> $$\delta(\epsilon) = \Phi\!\left(-\frac{\epsilon}{\mu} + \frac{\mu}{2}\right) - e^{\epsilon}\,\Phi\!\left(-\frac{\epsilon}{\mu} - \frac{\mu}{2}\right).$$

## Composition of GDP Mechanisms

When a mechanism releases a sequence of statistics over $t$ rounds, where each round-$i$ release satisfies $\mu_i$-GDP, the overall mechanism satisfies a cumulative $\mu$-GDP guarantee with

$$\mu = \sqrt{\sum_{i=1}^{t} \mu_i^2}.$$

This clean composition formula — a Pythagorean sum — is one of the key practical advantages of the GDP framework. It mirrors the composition of the privacy loss random variable: independent Gaussian losses add in quadrature.

---

[^gdp]: Dong, J., Roth, A., and Su, W. J. (2022). Gaussian Differential Privacy. *Journal of the Royal Statistical Society Series B*, 84(1), 3–37. [https://doi.org/10.1111/rssb.12454](https://doi.org/10.1111/rssb.12454)
