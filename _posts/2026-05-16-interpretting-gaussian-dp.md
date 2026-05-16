---
title: "Interpreting Gaussian Differential Privacy"
date: 2026-05-16
categories: [Math]
tags: [differential-privacy, statistics, hypothesis-testing]
math: true
---

## Differential Privacy

A plethora of algorithms extract invaluable insights from vast amounts of data. But releasing data — even in anonymized form — has repeatedly proven dangerous.

**The Netflix Prize (2006–2009).** Netflix publicly released a dataset of 100 million movie ratings from 500,000 subscribers, replacing usernames with random IDs. Narayanan and Shmatikoff [^narayanan2008] showed that knowing as few as 8 movies a person rated (and their approximate dates) was enough to uniquely identify that person in the dataset with high confidence, simply by cross-referencing with public IMDb reviews. Anonymization alone was not enough.

**The AOL Search Leak (2006).** AOL released 20 million search queries from 650,000 users, again replacing usernames with numeric IDs. Within days, *New York Times* reporters identified user #4417749 as Thelma Arnold, a 62-year-old widow from Georgia — purely from the content of her searches [^aol2006]. Her queries included her town, her friends' names, and her medical conditions. No technical attack was needed; a human reading the queries was sufficient.

These incidents illustrate a fundamental issue: data that appears anonymous in isolation can become identifying when combined with other information. Differential privacy is the mathematical response to this challenge. It is a property of an *algorithm* (not a dataset) that guarantees the output is useful in aggregate while placing a hard limit on how much any single individual's data can influence what an adversary learns. We follow definitions from Near and Abuah's *Programming Differential Privacy* [^near_abuah_2021].

>**Definition (Neighboring Datasets):** Two sequences $S = [x_i \ldots x_N]$ and $S^\prime = [x^\prime_i \ldots x^\prime_N]$ are called $t$-neighboring sequences if they differ by only one row, at $i = t$ only. That is, $x_i = x^\prime_i$ for $i \neq t$ and $x_i \neq x^\prime_i$ for $i = t$.

The algorithm should add just enough noise to its output such that it is very hard for an adversary to figure out if the underlying dataset was $S$ or its neighbor $S^\prime$.

>**Definition (Differential Privacy):** A mechanism $M$ satisfies $\epsilon$-differential privacy if for all neighboring datasets $S$ and $S^\prime$, and all possible sets of outputs $O$, we have
>
>$$\mathcal{P}(M(S) \in O) \leq e^{\epsilon} \mathcal{P}(M(S^\prime) \in O).$$
>
>Instead, $M$ is $(\epsilon, \delta)$-differentially private if it satisfies
>
>$$\mathcal{P}(M(S) \in O) \leq e^{\epsilon} \mathcal{P}(M(S^\prime) \in O) + \delta.$$

## Hypothesis Testing View of Differential Privacy

Differential privacy can be naturally interpreted through the lens of hypothesis testing, where an adversary attempts to distinguish between two neighboring datasets based on the output of the mechanism. The following exposition is adapted from Dong, Roth, and Su [^gdp].

Suppose an adversary observes the output $M(\cdot) = x$ and must determine whether the underlying dataset was $S$ or its neighbor $S'$. They formulate the test:

- **Null hypothesis** ($H_0$): the dataset is $S'$,
- **Alternate hypothesis** ($H_1$): the dataset is $S$.

The adversary selects a rejection region $O$ such that the Type I error satisfies $\mathcal{P}(M(S') \in O) = \alpha$. Under differential privacy, the power of the test is bounded by

$$\mathcal{P}(M(S) \in O) := 1 - \beta \leq e^{\epsilon} \alpha + \delta.$$

For suitably small $(\epsilon, \delta)$, this bound implies that any adversarial test would have low distinguishing power, providing strong privacy guarantees.

### Rejection Rules and Trade-off Functions

A *rejection rule* $\phi$ maps outputs to $[0,1]$, encoding the probability of rejecting the null hypothesis. The two types of errors are:

- **Type I error** $(\alpha_\phi)$: probability of rejecting $H_0$ when $H_0$ is true.
- **Type II error** $(\beta_\phi)$: probability of failing to reject $H_0$ when $H_1$ is true.



> **Definition (Trade-off Function[^gdp]):** Given two distributions $P$ and $Q$, the *trade-off function* $T(P,Q): [0,1] \to [0,1]$ maps a Type I error level $\alpha$ to the minimal achievable Type II error:
>
> $$T(P,Q)(\alpha) = \inf\{ \beta_\phi : \alpha_\phi \leq \alpha \},$$
>
> where the infimum is over all randomized rejection rules $\phi$.

Since we are evaluating the infimum over all possible rejection rules, the trade-off function tells us how different two distributions are. A larger trade-off function implies that distinguishing the two distributions is harder — and thus better privacy is achieved.

### $f$-Differential Privacy

Building on trade-off functions, $f$-Differential Privacy provides a functional generalization of $(\epsilon,\delta)$-DP.

> **Definition ($f$-DP [^gdp]):** A mechanism $M$ satisfies $f$-Differential Privacy if, for all neighboring datasets $S$ and $S'$:
>
>$$T(M(S), M(S')) \geq f,$$
>
>where $f$ is a trade-off function.

In this formulation, privacy is described not by two numbers but by an entire curve, fully characterizing the adversary's advantage across all significance levels. Notably, $(\epsilon,\delta)$-DP is a special case of $f$-DP, where $f$ corresponds to a specific piecewise linear trade-off function.

### Gaussian Differential Privacy

Among all $f$-DP guarantees, Gaussian Differential Privacy (GDP) stands out as particularly natural and convenient.

>**Definition ($\mu$-GDP [^gdp]):** A mechanism satisfies $\mu$-GDP if, for all neighboring datasets $S$ and $S'$:
>
>$$T(M(S), M(S')) \geq G_\mu,$$
>
>$$G_\mu(\alpha) = \Phi\!\left(\Phi^{-1}(1-\alpha) - \mu\right),$$
>
>and $\Phi$ is the standard normal CDF.

This is precisely the trade-off function one obtains when trying to distinguish $\mathcal{N}(0,1)$ from $\mathcal{N}(\mu, 1)$ using a single sample. The parameter $\mu$ directly controls the hardness of this test: larger $\mu$ means easier to distinguish, i.e., weaker privacy.

![Trade-off functions for shifted Gaussians](/assets/img/tradeoff_lines.png){: width="450"}
*Trade-off curves $G_\mu$ for $\mu = 1, 2, 3, 4$: minimum Type II error $\beta$ vs. significance level $\alpha$.*

The curves are Blackwell-ordered — a higher $\mu$ lies strictly below a lower one at every $\alpha$, meaning the two distributions are harder to confuse at every significance level simultaneously, not just on average. Concretely, at $\alpha = 5\%$: a $\mu=1$-GDP mechanism still forces ~65% Type II error on the adversary (good privacy), while a $\mu=4$ mechanism drops near zero (essentially no protection).

GDP enjoys several favorable properties:
- It is fully described by a single parameter $\mu$.
- It preserves the hypothesis testing interpretation.
- Just as the central limit theorem tells us that sums of random variables converge to a Gaussian, composition of multiple privacy mechanisms converges to GDP in the limit (Theorem 3.6 of [^gdp]).

>**Corollary (Conversion from GDP to $(\epsilon,\delta)$-DP [^gdp]):** A mechanism is $\mu$-GDP if and only if it is $(\epsilon, \delta(\epsilon))$-differentially private for all $\epsilon > 0$, where
>
>$$\delta(\epsilon) = \Phi\!\left(-\frac{\epsilon}{\mu} + \frac{\mu}{2}\right) - e^{\epsilon}\,\Phi\!\left(-\frac{\epsilon}{\mu} - \frac{\mu}{2}\right).$$

### Composition of GDP Mechanisms

When a mechanism releases a sequence of statistics over $t$ rounds, where each round-$i$ release satisfies $\mu_i$-GDP, the overall mechanism satisfies a cumulative $\mu$-GDP guarantee with

$$\mu = \sqrt{\sum_{i=1}^{t} \mu_i^2}.$$

This clean composition formula is one of the key practical advantages of the GDP framework.

---

[^near_abuah_2021]: Near, J. P. and Abuah, C. (2021). *Programming Differential Privacy*. [https://programming-dp.com/](https://programming-dp.com/)

[^gdp]: Dong, J., Roth, A., and Su, W. J. (2022). Gaussian Differential Privacy. *Journal of the Royal Statistical Society Series B*, 84(1), 3–37. [https://doi.org/10.1111/rssb.12454](https://doi.org/10.1111/rssb.12454)

[^narayanan2008]: Narayanan, A. and Shmatikoff, V. (2008). Robust De-anonymization of Large Sparse Datasets. *IEEE Symposium on Security and Privacy*. [https://ieeexplore.ieee.org/document/4531148](https://ieeexplore.ieee.org/document/4531148)

[^aol2006]: Barbaro, M. and Zeller, T. (2006). A Face Is Exposed for AOL Searcher No. 4417749. *The New York Times*. [https://www.nytimes.com/2006/08/09/technology/09aol.html](https://www.nytimes.com/2006/08/09/technology/09aol.html)
