---
title: "Differential Privacy"
date: 2026-05-16
categories: [Math]
tags: [differential-privacy, statistics]
math: true
---

## Motivation

A plethora of algorithms extract invaluable insights from vast amounts of data. But releasing even anonymized data has repeatedly proven dangerous.

**The Netflix Prize (2006–2009).** Netflix publicly released a dataset of 100 million movie ratings from 500,000 subscribers, replacing usernames with random IDs. Narayanan and Shmatikoff [^narayanan2008] showed that knowing just 8 movies a person rated (and their approximate dates) was enough to uniquely identify that person in the dataset with high confidence, simply by cross-referencing with public IMDb reviews. Anonymization alone was not enough.

**The AOL Search Leak (2006).** AOL released 20 million search queries from 650,000 users, again replacing usernames with numeric IDs. Within days, *New York Times* reporters identified user #4417749 as Thelma Arnold, a 62-year-old widow from Georgia, purely from the content of her searches [^aol2006]. Her queries included her town, her friends' names, and her medical conditions. A human reading the queries was sufficient to blow her privacy.

These incidents illustrate a fundamental issue: data that appears anonymous in isolation can become identifying when combined with other information. Differential privacy is one way to tackle this challenge. It is a property of an *algorithm* (not a dataset) that guarantees the output is useful in aggregate while placing a hard limit on how much any single individual's data can influence what an adversary learns. We follow definitions from Near and Abuah [^near_abuah_2021] and Dwork and Roth [^dwork2014].

---

## Differential Privacy

> **Definition (Neighboring Datasets):** Two datasets $D$ and $D'$ over a universe $\mathcal{X}$ are *neighboring*, written $D \sim D'$, if they differ in exactly one record: one can be obtained from the other by adding, removing, or replacing a single element.

The algorithm should produce outputs that are nearly indistinguishable whether it ran on $D$ or $D'$. Formally:

> **Definition ($(\varepsilon,\delta)$-Differential Privacy [^dwork2014]):** A randomized mechanism $M$ is $(\varepsilon,\delta)$-DP if for all neighboring datasets $D \sim D'$ and all measurable output sets $\mathcal{S}$,
>
> $$\mathbb{P}(M(D) \in \mathcal{S}) \leq e^{\varepsilon}\,\mathbb{P}(M(D') \in \mathcal{S}) + \delta.$$

When $\delta = 0$ this is called *pure* $\varepsilon$-DP. The $\delta > 0$ relaxation permits a small probability of failure — events on which no privacy guarantee is made — but $\delta$ should be negligibly small for the guarantee to be meaningful.

---
## Post-Processing Immunity

A fundamental and practically useful property of differential privacy is that it is immune to post-processing: any computation applied to a private output cannot make it *less* private.

> **Theorem (Post-Processing [^dwork2014]):** Let $M$ be an $(\varepsilon,\delta)$-DP mechanism and let $f$ be any (possibly randomized) function. Then $f \circ M$ — the mechanism that first runs $M$ and then applies $f$ to its output — is also $(\varepsilon,\delta)$-DP.

The proof is immediate from the definition: $f$ cannot look at $D$ or $D'$ directly; it only sees the output of $M$. Since $M$'s output distributions over $D$ and $D'$ already satisfy the DP inequality, composing with any $f$ preserves it.

**Why this matters.** Post-processing immunity means that once you hold a private output, you are free to analyze, transform, or publish it in any way without weakening the privacy guarantee. It also justifies *modular* algorithm design: prove that a base mechanism is private, and every downstream computation inherits that guarantee for free — no matter how complex.

---

## Privacy Loss Function

For any fixed output $o$, the *privacy loss* is the log-ratio of the probabilities (or densities) of producing $o$ under $D$ versus $D'$.

**Discrete case.** When $M$ has a discrete output space:

$$\mathcal{L}^{(o)}_{M,D,D'} := \log\frac{\mathbb{P}(M(D) = o)}{\mathbb{P}(M(D') = o)}.$$

**Continuous case.** When $M$ has a continuous output space with densities $p_D$ and $p_{D'}$:

$$\mathcal{L}^{(o)}_{M,D,D'} := \log\frac{p_D(o)}{p_{D'}(o)}.$$

In both cases this is the log-likelihood ratio: a large positive value means $o$ is much more likely under $D$ than $D'$, so observing $o$ would strongly "reveal" that $D$ was the true input. The $(\varepsilon, \delta)$-DP condition says that such large values should be rare.

### A Worked Example

Consider a medical dataset $D$ with $n = 1000$ records, each containing age, sex, and smoking status, along with whether the person developed lung cancer within five years. Suppose a researcher wants to release the answer to the query: *"Does smoking increase cancer risk?"* — modeled as outputting **Yes** or **No**.

From the data, suppose the estimated cancer rate among smokers is 18% and among non-smokers is 6%. A naive mechanism $M$ answers **Yes** (smoking increases risk) whenever the estimated difference exceeds a threshold, else **No**. With 1000 records this gap is large and stable, so $M(D) = \text{Yes}$ with probability 0.95 and **No** with probability 0.05.

Now consider the neighboring dataset $D'$ obtained by replacing one record: a smoker who developed cancer becomes a non-smoker who did not. The rates shift slightly — say the gap narrows by $\approx 0.2$ percentage points — and the mechanism now answers **Yes** with probability 0.90 and **No** with probability 0.10.

The privacy loss for each output is:

$$\mathcal{L}^{(\text{Yes})} = \log\frac{0.95}{0.90} \approx 0.054, \qquad \mathcal{L}^{(\text{No})} = \log\frac{0.05}{0.10} \approx -0.693.$$

A positive value ($\approx 0.054$) means observing **Yes** gives the adversary mild evidence that the underlying dataset was $D$ (the one with the smoker). A negative value ($\approx -0.693$) means observing **No** is actually *more likely* under $D'$, so it points toward $D'$. In this case neither value is large, so the privacy cost of a single query is small — but the loss accumulates with every additional query.

More generally, when some output $o$ yields a very large privacy loss, an adversary who observes $o$ can decisively reject the null hypothesis that the input was $D'$ — and in doing so, infer private information about the individual whose record differs between $D$ and $D'$. This connection between privacy loss and hypothesis testing is the starting point of a richer framework explored in the [next post]({% post_url 2026-05-16-interpretting-gaussian-dp %}).

---

## Privacy Loss as a Random Variable

The output $o$ is itself random (drawn from $M(D)$), so $\mathcal{L}^{(o)}_{M,D,D'}$ is a **random variable** — call it $\mathcal{L}$. Its distribution is induced by the mechanism's randomness when run on $D$.

This reframing is powerful: instead of checking the DP inequality for every possible output set $\mathcal{S}$, we can work with the tail of a single scalar random variable. Specifically, one can show that the DP condition is equivalent to controlling this tail [^dwork2014]:

> **Sufficient Condition:** $M$ is $(\varepsilon, \delta)$-DP if for all neighboring $D \sim D'$,
>
> $$\mathbb{P}_{o \sim M(D)}\!\left(\mathcal{L}^{(o)}_{M,D,D'} > \varepsilon\right) \leq \delta.$$

Intuitively: the mechanism is private if the log-likelihood ratio exceeds $\varepsilon$ only with probability $\delta$. The bad event — where an output $o$ strongly distinguishes $D$ from $D'$ — must be rare. This condition reduces a universal quantifier over all output sets $\mathcal{S}$ to a single tail probability, making it much easier to work with analytically.

**A useful consequence.** For the Gaussian mechanism — adding $\mathcal{N}(0, \sigma^2)$ noise to a query with $\ell_2$-sensitivity $\Delta$ — the privacy loss is Gaussian [^dwork2016cdp]:

$$\mathcal{L} \sim \mathcal{N}\!\left(\frac{\Delta^2}{2\sigma^2},\, \frac{\Delta^2}{\sigma^2}\right).$$

The mean of $\mathcal{L}$ is $\Delta^2 / (2\sigma^2)$, and its tail can be bounded precisely using the Gaussian CDF, yielding an explicit $(\varepsilon, \delta)$ guarantee.

---

## Composition

A private mechanism is rarely used just once. A data analyst may run several analyses on the same dataset, and each query incurs its own privacy cost. *Composition* refers to how the overall privacy guarantee degrades when a mechanism is applied multiple times.

### Sequential Composition

Suppose two mechanisms $M_1$ and $M_2$ are applied sequentially to the same dataset $D$. The composed mechanism releases $(M_1(D),\, M_2(D, M_1(D)))$, where $M_2$ may depend on both $D$ and the output of $M_1$.

> **Theorem (Sequential Composition [^dwork2014]):** If $M_1$ is $(\varepsilon_1, \delta_1)$-DP and $M_2$ is $(\varepsilon_2, \delta_2)$-DP (for any fixed value of its first argument), then their composition is $(\varepsilon_1 + \varepsilon_2,\, \delta_1 + \delta_2)$-DP.

The structural insight is that the joint privacy loss factors as a sum of individual losses. For any output $(o_1, o_2)$ and neighboring $D \sim D'$, the log-ratio of joint densities decomposes as:

$$\log\frac{p_D(o_1, o_2)}{p_{D'}(o_1, o_2)} = \underbrace{\log\frac{p_D(o_1)}{p_{D'}(o_1)}}_{\mathcal{L}_1} + \underbrace{\log\frac{p_D(o_2 \mid o_1)}{p_{D'}(o_2 \mid o_1)}}_{\mathcal{L}_2}.$$

This decomposition motivates the result, but the full proof for the $(\varepsilon,\delta)$ case is more delicate — the privacy loss serves only as a sufficient condition, and bounding the composed guarantee requires careful manipulation of the DP inequalities for $M_1$ and $M_2$ directly; see Theorem 3.16 of Dwork and Roth [^dwork2014].

### Parallel Composition

> **Theorem (Parallel Composition [^dwork2014]):** If $M_1$ and $M_2$ are applied to *disjoint subsets* of the dataset, the composed mechanism satisfies $(\max(\varepsilon_1,\varepsilon_2),\, \max(\delta_1,\delta_2))$-DP.

This is because an individual's data appears in at most one subset, so only one mechanism's guarantee is ever relevant for any given individual.

### Advanced Composition

Sequential composition gives $(k\varepsilon, k\delta)$-DP for $k$ applications of an $(\varepsilon,\delta)$-mechanism. This linear growth in $\varepsilon$ is often too pessimistic. The advanced composition theorem exploits the random-variable structure of privacy losses to obtain a square-root improvement:

> **Theorem (Advanced Composition [^dwork2010]):** For $k$ mechanisms each satisfying $(\varepsilon, \delta)$-DP, their composition satisfies $(\varepsilon', k\delta + \delta')$-DP for any $\delta' > 0$, where
>
> $$\varepsilon' = \sqrt{2k\ln(1/\delta')}\,\varepsilon + k\varepsilon(e^\varepsilon - 1).$$
>
> For small $\varepsilon$, this is approximately $O\left(\varepsilon\sqrt{k\ln(1/\delta')}\right)$.

This square-root improvement is significant in practice and motivates the GDP composition rule $\mu = \sqrt{\sum \mu_i^2}$, covered in the [next post]({% post_url 2026-05-16-interpretting-gaussian-dp %}).


---

[^near_abuah_2021]: Near, J. P. and Abuah, C. (2021). *Programming Differential Privacy*. [https://programming-dp.com/](https://programming-dp.com/)

[^dwork2014]: Dwork, C. and Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3–4), 211–407. [https://doi.org/10.1561/0400000042](https://doi.org/10.1561/0400000042)

[^dwork2016cdp]: Dwork, C. and Rothblum, G. N. (2016). Concentrated Differential Privacy. *arXiv:1603.01887*. [https://arxiv.org/abs/1603.01887](https://arxiv.org/abs/1603.01887)

[^dwork2010]: Dwork, C., Rothblum, G. N., and Vadhan, S. (2010). Boosting and Differential Privacy. *IEEE FOCS 2010*, pp. 51–60. [https://doi.org/10.1109/FOCS.2010.12](https://doi.org/10.1109/FOCS.2010.12)

[^narayanan2008]: Narayanan, A. and Shmatikoff, V. (2008). Robust De-anonymization of Large Sparse Datasets. *IEEE Symposium on Security and Privacy*. [https://ieeexplore.ieee.org/document/4531148](https://ieeexplore.ieee.org/document/4531148)

[^aol2006]: Barbaro, M. and Zeller, T. (2006). A Face Is Exposed for AOL Searcher No. 4417749. *The New York Times*. [https://www.nytimes.com/2006/08/09/technology/09aol.html](https://www.nytimes.com/2006/08/09/technology/09aol.html)
