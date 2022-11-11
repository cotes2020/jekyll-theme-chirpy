---
title: "Linearity of Classical Gauge Fields"
description: "Revisiting the linearity of classical fields under the lens of gauge invariance"
categories: [classical field theory]
tags: [linearity, gauge invariance, symmetry]
---

In [Combining Valid Solutions Into New Ones in Classical Field Theory]({% post_url 2022-04-23-combining-solutions-classical-field-theory %}), we showed that any linear combination of valid solutions to classical field equations, is in turn a solution.

We then took the idea of solution fields behaving like coordinates further, in [Gauge Invariance in Classical Field Theory]({% post_url 2022-05-10-gauge-invariance-classical-field-theory %}). Here, we found that it is possible for certain differentiable transformations of solution fields to be new solution fields. To admit such families of solutions (where each choice is called a 'gauge') into the Lagrangian, we replace covariant derivatives with the gauge-invariant gauge covariant derivatives. This is functionally equivalent to replacing partial derivatives with covariant derivatives to allow general covariance, in differential geometry.

For this post, it is best to read the ones mentioned above as we will rely on the concepts and notation in them. Armed with their ideas, we will now go against the chronology of the said posts i.e. we will begin with gauge fields and derive their linearity.

Such a construction is equally logical as the reverse, as gauge invariance is a fundamental feature of classical fields [^1] (just as general covariance is, of physics on manifolds). In fact, a classical field is _defined_ to possess such structure among others. [^2] Therefore, just as deriving linearity from the fundamental Euler-Lagrange equations is rigorous, so is deriving it from other basic concepts such as gauge invariance (if possible, that is). Let us see how this can be done, and gain a new perspective on linearity in the process.

[^1]: While constructing gauge invariance, we did borrow the coordinate-like behaviour of solution fields seen in linearity-related contexts. However, we did not use the idea of linearity itself to explore gauge invariance, as it is not a prerequisite for such structure.

[^2]: The others being, namely: locality, Lorentz invariance and the principle of stationary action.

## Derivation

### Gauge transformation

Let us start with a solution set $$\left\{ \phi_a \right\}$$ for the equations of motion for a Lagrangian $$\mathcal{L}$$. We now apply a gauge transformation as,

$$\phi_a \to \widetilde{\phi}_a$$

The new equations of motion become,

$$\frac{\partial \mathcal{L}}{\partial \widetilde{\phi}^a} - D_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} = 0$$

where,

$$
\begin{align}
\phi^a & = \delta^{a b} \phi_b \\
\widetilde{\pi}^\mu_{\phantom{\mu} a} & = \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi^a \right)} \\
D_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} & = \nabla_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} - \widetilde{\pi}^\mu_{\phantom{\mu} b} G^b_{\phantom{b} \mu a} \\ \\
G^b_{\phantom{b} \mu a} & = J^b_{\phantom{b} c} \partial_\mu J^c_{\phantom{c} a} \\
J^b_{\phantom{b} a} & = \frac{\partial \widetilde{\phi}_a}{\partial \phi_b}
\end{align}
$$

### Invariance of equations of motion

Now, we ask, when are the fields in the new gauge still valid solutions for the equations of motion in the _original_ gauge? The most general such scenario is when gauge covariant derivatives reduce to covariant derivatives under the said gauge transformation, thereby preserving the form of the equations of motion. Here, we find,

$$G^b_{\phantom{b} \mu a} = J^b_{\phantom{b} c} \partial_\mu J^c_{\phantom{c} a} = 0$$

This is further true in general only if $$\partial_\mu J^c_{\phantom{c} a} = 0$$. Thus,

$$\partial_\mu \frac{\partial \widetilde{\phi}_a}{\partial \phi_c} = 0$$

Lastly, for the above to be generally true, the fields in the new gauge must be of the form,

$$\widetilde{\phi}_a = C^b_{\phantom{b} a} \phi_b \\$$

where $$\partial_\mu C^b_{\phantom{b} a} = 0$$. This is precisely the notion of linearity: linear combinations of solution fields obey the same equations of motion as the original.

## Conclusion

We have thus found a new way to interpret the above statement on linearity. Namely, linear combinations constitute the most general gauge transformations under which a given Lagrangian is automatically gauge-invariant (as gauge covariant derivatives reduce to covariant derivatives).

Thanks for reading!
