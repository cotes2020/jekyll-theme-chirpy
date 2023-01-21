---
title: "Scalar Field Lagrangian From Symmetry Considerations: Part 2 (Gauge Invariance)"
description: "This time, gauge invariance leaves us little choice"
categories: [classical field theory]
tags: [Klein-Gordon theory, gauge invariance, symmetry]
---

## Related concepts from older posts

In [Scalar Field Lagrangian From Symmetry Considerations]({% post_url 2021-12-01-scalar-field-lagrangian-symmetry-considerations %}), we derived the Lagrangian for the Klein-Gordon theory of a scalar field $$\phi$$ evolving in a potential $$V$$,

$$\mathcal{L} = \frac{1}{2} \partial_\mu \phi \partial^\mu \phi - V \left( \phi \right)$$

We did so, by deriving the field-theoretic energy-momentum tensor using Noether's theorem applied to the symmetry of the action under small translations in spacetime and borrowing its symmetry from continuum mechanics:

$$
\begin{align}
T^{\mu \nu} & = \pi^\mu \partial^\nu \phi - \eta^{\mu \nu} \mathcal{L} \\
T^{\mu \nu} & = T^{\nu \mu}
\end{align}
$$

where $$\displaystyle{ \pi^\mu = \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)} }$$ is the conjugate momentum tensor. The above equations suffice to lead to the aforementioned Lagrangian.

Additionally, in [Harmonic Oscillators in Scalar Field Theory]({% post_url 2021-11-06-harmonic-oscillators-scalar-field-theory %}), we constructed the perturbation theory for Klein-Gordon fields oscillating about the local minima of the potential $$V \left( \phi \right)$$. We found that only differences in the field matter, and since these are small perturbations, their squares and higher powers vanish. As the equations of motion are obtained by differentiating the Lagrangian, it follows that we must have $$\phi^2$$ appear as a potential term, and it does so in the form,

$$\mathcal{L} = \frac{1}{2} \partial_\mu \phi \partial^\mu \phi - \frac{1}{2} m^2 \phi^2$$

where $$m$$ is a 'mass' of the field, of sorts (this becomes clear only after quantizing the field, which is a subject of quantum field theory).

Last but not the least, in the following posts, we discussed the idea of gauge invariance and indexed fields behaving much like coordinates in classical mechanics:

[Combining Valid Solutions Into New Ones in Classical Field Theory]({% post_url 2022-04-23-combining-solutions-classical-field-theory %})

[Gauge Invariance in Classical Field Theory]({% post_url 2022-05-10-gauge-invariance-classical-field-theory %})

In this post, we will incorporate gauge invariance into the Klein-Gordon theory and see how it provides deep insights about the field theory (such as the structure of its perturbation theory) for free!

## Lagrangian

Consider an indexed family of independent scalar fields, $$\left\{ \phi_a \right\}$$. Put briefly, gauge invariance is the idea that under differentiable transformations of the indexed fields, i.e. gauge transformations $$\phi_a \to \widetilde{\phi}_a$$, the structure of the equations of motion for each field must remain preserved.

This condition is enforced on the theory simply by replacing the usual geometric covariant derivatives $$\nabla_\mu \phi_a$$, with gauge covariant derivatives $$D_\mu \phi_a = \nabla_\mu \phi_a - \phi_b G^b_{\phantom{b} \mu a}$$. Here, $$G^b_{\phantom{b} \mu a}$$ are the gauge connection coefficients $$G^b_{\phantom{b} \mu a} = J^b_{\phantom{b} c} \partial_\mu J^c_{\phantom{c} a}$$ and $$J^b_{\phantom{b} a}$$ is the gauge Jacobian $$J^b_{\phantom{b} a} = \frac{\partial \widetilde{\phi}_a}{\partial \phi_b}$$.

Now, recall the Klein-Gordon Lagrangian,

$$\mathcal{L} = \frac{1}{2} \partial_\mu \phi \partial^\mu \phi - V \left( \phi \right)$$

The prescription for gauge invariance is to replace $$\phi$$ with $$\phi_a$$ and $$\partial_\mu \phi$$ with $$D_\mu \phi_a$$. Since the quantity on the left hand side has no free indices, all indices on the right hand side must be dummy indices. This severe restriction, with the fact that our new Lagrangian must reduce to the original one in suitable gauges (where $$\partial_\mu J^b_{\phantom{b} a} = 0$$, as we had seen [here]({% post_url 2022-05-14-linearity-classical-gauge-fields %})), motivates us to propose the new Lagrangian,

$$\mathcal{L} = \frac{1}{2} D_\mu \phi_a D^\mu \phi^a - V \left( \phi_a \right)$$

However, the above is still not quite right; we have a gauge scalar potential $$V$$ which depends on $$\phi_a$$ (which, as far as gauge transformations are concerned, does not look like a scalar i.e. it does not resemble a gauge scalar). The simplest gauge scalar we can construct from $$\phi_a$$, which depends on no other field, is $$\phi_a \phi^a$$. Therefore, we write,

$$\mathcal{L} = \frac{1}{2} D_\mu \phi_a D^\mu \phi^a - V \left( \phi_a \phi^a \right)$$

## Some insights

Let us now apply the above ideas to gain new insight on classical field theory.

### Perturbation theory

Similar to our previous exploration of the perturbation theory for the classical Klein-Gordon model, we ignore contractions of $$\phi_a$$ and its powers in the equations of motion. Since the equations of motion are obtained by differentiating the Lagrangian in various terms, it can contain only the contraction of $$\phi_a$$ up to its first power, i.e. $$\phi_a \phi^a$$. Such a Lagrangian can be written in the form,

$$\mathcal{L} = \frac{1}{2} D_\mu \phi_a D^\mu \phi^a - \frac{1}{2} m^2 \phi_a \phi^a$$

### Single index theory

To realize the power of the above argument, consider a single indexed field $$\phi$$ (we didn't have to index it explicitly as there is only one!). Now, our argument becomes the following. We cannot have $$\phi$$ appear in the Lagrangian, since it is not a gauge scalar (much like $$\phi_a$$ is not, for a theory with more gauge indices). As a matter of fact, no odd power of $$\phi$$ is a gauge scalar, since an odd power corresponds to a free gauge index,

$$\phi^{2n+1} = \phi^{2n} \cdot \phi \leftrightarrow \left( \phi_b \phi^b \right)^n \phi_a$$

Therefore, only even powers of $$\phi$$ can appear in the Lagrangian, which is a gauge scalar. This internal working of the field $$\phi$$ would not come to limelight without the idea of gauge invariance and multiples degrees of freedom - thus providing an excellent theoretical justification for gauge field theories.

Armed with this knowledge and the Klein-Gordon Lagrangian from perturbation theory, we assert that a fully gauge-invariant Lagrangian for a single field $$\phi$$ has the series representation,

$$\mathcal{L} = \frac{1}{2} D_\mu \phi D^\mu \phi - \frac{1}{2} m^2 \phi^2 - \sum_{n=2}^{\infty} \frac{1}{n!} g_n \phi^{2 n}$$

where $$\left\{ g_n \vert n = 1, 2, \dots \right\}$$ are a set of coupling constants, with the canonical condition $$\displaystyle{ g_1 = \frac{1}{2} m^2 }$$. Clearly, we obtain the Klein-Gordon perturbation theory from the above Lagrangian if we consider small oscillations of $$\phi$$ about stable equilibria (so that $$\phi^3$$ and higher odd powers vanish from the equations of motion, effectively removing $$- \frac{1}{4!} g_4 \phi^4$$ and higher even terms from the Lagrangian).

### Complex scalar field theory

At last, let us venture towards the scalar field theory of complex-valued fields. In this model, every field $$\phi$$ has a corresponding dual $$\bar{\phi}$$ which maps it to a real, $$\bar{\phi} \phi \in \mathbb{R}$$. Here, juxtaposition of complex numbers represents complex multiplication, dictated by a bilinear map corresponding to $$\text{SO} \left( 2 \right)$$. For readers who are interested, [Algebra Done Tensorially: Part 2 (Algebras Over Fields)]({% post_url 2021-10-23-algebras-over-fields %}) explores complex algebra from a tensor algebraic perspective.

In the notation we have used so far, $$\phi^a$$ represents the components of $$\phi$$ while $$\phi_a$$ represents those of $$\bar{\phi}$$. In particular,

$$
\begin{align}
\phi & = \phi^a e_a \\
\bar{\phi} & = \phi_a \theta^a \\
\theta^a \left( e_b \right) & = e_b \left( \theta^a \right) = \delta^a_{\phantom{a} b}
\end{align}
$$

The intuition for the last equation has been elaborated in [Demystifying the Definition Of a Covector Basis]({% post_url 2021-10-23-covector-basis-definition %}). Note that in this general picture, it is not necessarily true that $$\phi_a = \delta_{a b} \phi^b$$. In general, $$\phi_a = g_{a b} \phi^b$$ where $$g_{a b} \theta^a \otimes \theta^b$$ is the metric tensor $$g$$.

Thus, from the above discussion, the full Lagrangian for complex scalar fields (with gauge scalar potential terms) is,

$$\mathcal{L} = \frac{1}{2} D_\mu \bar{\phi} D^\mu \phi - \frac{1}{2} m^2 \bar{\phi} \phi - \sum_{n=2}^\infty \frac{1}{n!} g_n \left( \bar{\phi} \phi \right)^n$$

## Conclusion

The above Lagrangian is where much of relativistic quantum mechanics and quantum field theory begin. Through a series of posts, we have managed to construct it using purely classical methods. In doing so, we discovered gauge field theories, which are yet another deep chapter of modern quantum mechanics and quantum field theory. Hopefully, these subjects will form a major part of the upcoming posts on this blog.

Thank you for reading till here. Feel free to leave a comment or reaction below :)
