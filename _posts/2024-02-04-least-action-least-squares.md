---
title: Least Action vs. Least Squares
author: jake
date: 2024-02-04 12:00:00 +0800
categories: [Physics]
tags: [statistics, physics, least squares, least action]
math: true
mermaid: true
---
In this post I want to describe two methods that both use the word "Least" to describe them. Both of these concepts are foundational in their respective field (physics and statistics), and I find it interesting to compare/contrast their formulation.

## Path Minimization
![Path Minimization](assets/img/path_minimization.png)
We start with a path. A path consists of $(x, y(x))$ points in space. When we vary $x + \delta x$, we can move to the corresponding $y$ coordinate along the step defined by $y'(x) = \frac{dy}{dx}$. We can encode this entire relation inside a *functional*:

$$
F(y(x), y'(x), x)
$$

Which has a path integral (the "length" of the path from points $a$ to $b$) of:

$$
\begin{equation*}
I[y] = \int_a^b F(y(x), y'(x), x)dx
\end{equation*}
$$

Now, if we wanted to say, minimize this path, we can consider an alternative path starting and ending at the same points ($x=a$ and $x=b$) but "perturbed" by some small amount, $\epsilon$:

$$
\tilde y(x) = y(x) + \epsilon \eta(x), \eta(a) = \eta(b) = 0
$$

Where $\tilde y(x)$ only differs from $y(x)$ on the interior of $\[a, b\]$. $\tilde y(x)$ also has a path length,

$$
\begin{equation*}
I[\tilde y] = \int_a^b F(\tilde y(x), \tilde y'(x), x)dx
\end{equation*}
$$

Since $y$ and $\tilde y$ were arbitrary functions, we know that for any $\epsilon > 0$, $I[\tilde y(x)] > I[y(x)]$ because $\epsilon$ is making $\tilde y(x) > y(x)$. So to "find" the smallest version of $I[y]$, we can tighten the $\epsilon$ bound around it by minimizing:

$$
\begin{gather*}
\frac{dI[\tilde y]}{d\epsilon}\bigg |_{\epsilon=0} = 0 \\
\implies \int_a^b \frac{d}{d\epsilon}F(\tilde y(x), \tilde y'(x), x)\bigg |_{\epsilon=0} dx = 0
\end{gather*}
$$

<mark>
Lets take a pause and review before the algebra. We have already defined $y(x)$ and its path from $a$ to $b$ as $I[y]$. For whichever $y(x)$, we defined a larger version of it $\tilde y(x)$ which has a nonnegative component $\epsilon \eta(x)$. We are now minimizing $I[\tilde y]$ as a bound on $I[y]$, which has the affect of searching, over a space of functions, for the smallest path. Okay, onto the algebra...
<mark>

First,
<details><summary>$\int_a^b \frac{d}{d\epsilon}F(\tilde y(x), \tilde y'(x), x)\bigg |_{\epsilon=0} dx = 0 \implies \int_a^b (\frac{\partial F}{\partial \tilde y}\eta + \frac{\partial F}{\partial \tilde y'}\eta')\bigg |_{\epsilon=0} dx = 0$</summary>
$$
\begin{flalign*}
\frac{dF}{d\epsilon} &= \frac{\partial F}{\partial \tilde y} \frac{\partial \tilde y}{\partial \epsilon} + \frac{\partial F}{\partial \tilde y'} \frac{\partial\tilde y'}{\partial \epsilon} \\
&= \frac{\partial F}{\partial \tilde y} \eta + \frac{\partial F}{\partial \tilde y'} \eta'
\end{flalign*}
$$
</details>

then,
<details><summary>$\int_a^b (\frac{\partial F}{\partial \tilde y}\eta + \frac{\partial F}{\partial \tilde y'}\eta')\bigg |_{\epsilon=0} dx = 0 \implies \int_a^b (\frac{\partial F}{\partial \tilde y} - \frac{\partial}{\partial x} \frac{\partial F}{\partial \tilde y'})\bigg |_{\epsilon=0} \eta dx = 0$</summary>
Use a substitution for the second term, $\frac{\partial F}{\partial \tilde y'}\eta'$
$$
\begin{flalign*}
\int_a^b \frac{\partial F}{\partial \tilde y'}\eta'dx = \frac{\partial F}{\partial \tilde y}\eta \bigg |_a^b - \int_a^b \frac{\partial}{\partial x}\frac{\partial F}{\partial \tilde y'}\eta dx
\end{flalign*}
$$
And recognize that $\frac{\partial F}{\partial \tilde y}\eta \bigg |_a^b$ is 0 by construction ($\eta(a) = \eta(b) = 0$)
</details>

And finally we are left with,

$$
\begin{equation*}
\frac{dI[\tilde y]}{d\epsilon} = \int_a^b (\frac{\partial F}{\partial \tilde y} - \frac{\partial}{\partial x} \frac{\partial F}{\partial \tilde y'})\bigg |_{\epsilon=0} \eta dx = 0
\end{equation*}
$$

which goes to zero for any functional $F$ at $\epsilon = 0$ provided that,

$$
\begin{equation*}
\frac{\partial F}{\partial y} - \frac{\partial}{\partial x} \frac{\partial F}{\partial y'} = 0
\end{equation*}
$$

By the [fundamental lemma of calculus of variations](https://en.wikipedia.org/wiki/Fundamental_lemma_of_the_calculus_of_variations). This last expression is known as the [Euler Lagrange equation](https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation). By design, when following the trajectory of this second-order differential equation, you also minimize a trajectory along a path.

### Least Action
Turning Path Minimization into the [Least Action Principle](https://en.wikipedia.org/wiki/Stationary-action_principle) is straightforward at this point. Replace $x \rightarrow t$, $y \rightarrow q$ and $y' \rightarrow \dot q$:

$$
\begin{equation*}
A[q] = \int_{t_1}^{t_2} L(q(t), \dot q(t), t) dt
\end{equation*}
$$

$q$ is known as a *generalized coordinate* representing for example cartesian or polar coordinates and $\dot q(t)$ is short-hand for the time derivative of q wrt t, $\frac{dq}{dt}$. Here is the really cool part, when we specify $L$, we can then apply the Euler Lagrange equation to automatically find the Least Action path. The most famous example is if we let $L$ be the difference between the kinetic and potential energy and $q$ be cartesian coordinates, $L = \frac{1}{2}m \dot q^2 - V(q)$:

$$
\begin{flalign*}
\frac{\partial L}{\partial q} - \frac{\partial}{\partial t}\frac{\partial L}{\partial \dot q} &= 0 \\
\frac{\partial L}{\partial q} &= \frac{\partial}{\partial t}\frac{\partial L}{\partial \dot q} \\
-\frac{\partial V}{\partial q} &= \frac{\partial}{\partial t} m\dot q \\
F &= m \ddot q
\end{flalign*}
$$

So, turns out if we use $L = T - V$, and pass that through Euler Lagrange Equations, we recover Newton's Second Law. Equivalently, Newton's Second Law is the exact trajectory that minimizes the action over the functional $L = T - V$. 

Here is another example, but this time taking $q$ to be polar coordinates:

$$
\begin{equation*}
L = \frac{1}{2}m(\dot \rho^2 + \rho^2\dot\phi^2) - V(\rho)
\end{equation*}
$$

Which yields the equation (in the $\phi$ axis):

$$
\begin{equation*}
\frac{\partial}{\partial t}\frac{\partial L}{\partial \dot \phi} = m\rho^2\dot\phi = 0
\end{equation*}
$$

Interestingly, $\frac{\partial L}{\partial \dot \phi}$ is equal to the (conjugate) momentum in the $\phi$ direction, $p_\phi$, implying that angular momentum is conserved with respect to time. You can see both of these examples [here](https://en.wikipedia.org/wiki/Lagrangian_mechanics#Conservative_force) along with more exotic examples [here](https://en.wikipedia.org/wiki/Lagrangian_(field_theory)#Examples).

<mark>
Here is the takeaway, these physical systems only differ in the form of $L$. Once we can express the system in terms of $L$, Least Action & Euler Lagrange are a common framework to determine the equations of motions, conserved quantities, and conjugate momentum.
<mark>

## KL Divergence Minimization
![KL Divergence Minimization](assets/img/kl_minimization.png)
Now, lets switch tracks. Instead of a path between two points, lets abstract the discussion to two probability distributions. Our goal is now to find the smallest distance between two probability distributions. Instead of taking a path integral like before, we will use tools from Information Theory to define a notion of "distance":
- *Entropy* - best possible long term average bits per message (optimal) that can be achieved under a symbol distribution $P(X)$ by using an encoding scheme (possibly unknown) specifically designed for $P(X)$.

$$
H(P) = -\sum_{x \in \mathcal X} P(x) \log P(x) = \mathbb E_{x \sim P}[-\log P(x)]
$$

- *Cross Entropy* - long term average bits per message (suboptimal) that results under a symbol distribution $P(X)$, by reusing an encoding scheme (possibly unknown) designed to be optimal for a scenario with symbol distribution $Q(X)$.

$$
H(P, Q) = -\sum_{x \in \mathcal X} P(x) \log Q(x) = \mathbb E_{x \sim P}[-\log Q(x)] 
$$

- *KL Divergence* - penalty we pay, as measured in average number of bits, for using the *optimal* scheme for $Q(X)$, under the scenario where symbols are *actually* distributed as $P(X)$.

$$
D_{KL}(P || Q) = H(P, Q) - H(P) = \mathbb E_{x \sim P}[-\log Q(x)]  - \mathbb E_{x \sim P}[-\log P(x)]
$$

<mark>
So, it seems KL divergence is a good candidate to use a notion of "distance" between two distributions (where "distance" is nominal only, since KL divergence does <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Relation_to_metrics">not satisfy the triangle inequality</a>). 
<mark>

I remember from [CS229 PS3](https://cs229.stanford.edu/summer2020/ps3.pdf), there was a problem asking us to prove that minimizing the KL Divergence between a parameterized distribution $$P_\theta$$ (i.e. Normal, Bernoulli, Binomial etc.) and the empirical distribution of the population $$\hat P$$, this is equivalent with maximizing likelihood estimate for $$\theta$$. We are looking for single expression that, when satisfied, is equivalent with minimizing the KL divergence between two distributions:

<details><summary>CS 229 Problem Set 3 2.C</summary>
<IMG src="assets/img/PS3.png"  alt="CS 229 Problem Set 3 2.C"/>
</details>

First, lets define the empirical distribution:

$$
\hat P(x) = \frac{1}{n} \sum_{i=1}^n \mathbb I\{x^{(i)} = x\}
$$

To show the equivalence, we work from the definition of KL divergence towards something that maximizes the log-likelihood:

$$
\begin{flalign*}
\arg \min_\theta D_{KL}(\hat P || P_{\theta}) &= \arg \min_\theta \sum_{x \in \mathcal X} \hat P(x)\log{\hat P(x)} - \sum_{x \in \mathcal X} \hat P(x)\log{P(x)}\\
&= \arg \max_\theta \sum_{x \in \mathcal X} \hat P(x)\log{P(x)}\\
&= \arg \max_\theta \sum_{x \in \mathcal X} \frac{1}{n} \sum_{i=1}^n \mathbb I\{x^{(i)} = x\}\log{P(x)} \\
&= \arg \max_\theta \frac{1}{n} \sum_{i=1}^n \sum_{x \in \mathcal X} \mathbb I\{x^{(i)} = x\}\log{P(x)}
\end{flalign*}
$$

The key here is that the inner sum is only nonzero for the unique value $$x \in \mathcal X$$ such that $$x = x^{(i)}$$. Hence,

$$
= \arg \max_\theta \frac{1}{n} \sum_{i=1}^n\log{P(x^{(i)})}
$$

Which is the Maximum Likelihood Estimate. Similar to the Euler Lagrange Equation above, when we are able to satisfy this equation, we also minimize the KL divergence between two distributions at the same time.

### Least Squares
Write about least square and least absolute deviance

## Lagrangians and Loss Functions
talk about specific instantiations of Lagrangians and Loss functions (Least Squares, Least Absolute Deviations). Class hierarchy perspective.

```python
class LeastAction:
    L: Expression

class KineticPotentialEnergy(LeastAction):
    L = (1 / 2) * m * q_dot ** 2 - V_q

class RotationalEnergy(LeastAction):
    L = (1 / 2) * m * (rho_dot ** 2 + rho ** 2 * phi_dot ** 2) - V_rho
```


```python
class MaximumLikelihoodEstimate:
    Loss: Expression

class LeastSquares(MaximumLikelihoodEstimate):
    Loss = (y_i - y_i_hat) ** 2

class LeastAbsoluteDeviation(MaximumLikelihoodEstimate):
    Loss = abs(y_i - y_i_hat)
```