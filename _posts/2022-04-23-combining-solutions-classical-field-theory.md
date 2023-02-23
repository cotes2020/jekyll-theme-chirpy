---
title: "Combining Valid Solutions Into New Ones in Classical Field Theory"
description: "When and how can given solutions of the classical field-theoretic Euler-Lagrange equations be combined into new solutions?"
categories: [classical field theory]
tags: [linearity]
redirect_from: combining-valid-solutions-classical-field-theory
---

A popular theme in some field theories is linearity — wherein valid solutions of the field equations in question can be added and scaled to generate new valid solutions. For example, Maxwell's equations are linear in vacuum and homogeneous media. A different kind of field equation that is linear, is the time-independent Schrödinger equation for the wavefunction (a complex-valued scalar field). However, we will not go into quantum mechanics or quantum field theory for now.

In classical field theory, we build a unified framework which reproduces the myriad of different field theories in different physical contexts. It is only assumed that the classical fields obey some basic principles, such as locality, Lorentz invariance and the principle of stationary action. [^1]

[^1]: We can add further structure to field theories by introducing what are called 'gauge symmetries'. This is explored in the post [Gauge Invariance in Classical Field Theory]({% post_url 2022-05-10-gauge-invariance-classical-field-theory %}).

This makes us wonder, that in the general construct of classical field theory, when are solutions of field equations linear? And can they be combined in ways other than adding and scaling, to generate new solutions? Let us find for ourselves.

For simplicity, we will stick to real-valued scalar fields in inertial coordinates. However, the results can be extended to tensor fields in arbitrary coordinate systems.

## Solutions as coordinates

Suppose we have a Lagrangian density $$\mathcal{L}$$ and a set of scalar fields $$\left\{ \phi_{\left( i \right)} \right\}$$ which independently obey the Euler-Lagrange equations for the given Lagrangian:

$$\frac{\partial \mathcal{L}}{\partial \phi_{\left( i \right)}} = \nabla_\mu \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)}$$

The parentheses around the index $$i$$ remind us that it is not a tensorial index, but a label for each scalar field.

In the above equations, we may interpret the role of $$\left\{ \phi_{\left( i \right)} \right\}$$ as a set of independent coordinates respecting the said equations of motion. Therefore, we have,

$$\frac{\partial \phi_{\left( i \right)}}{\partial \phi_{\left( j \right)}} = \delta_{\left( i \right) \left( j \right)}$$

Now, we consider a new field $$\phi$$ that is a function of the solution set $$\left\{ \phi_{\left( i \right)} \right\}$$,

$$\phi = \phi \left( \left\{ \phi_{\left( i \right)} \right\} \right)$$

Therefore, we can frame our problem as a twofold question:

1) When is the following true?

$$\frac{\partial \mathcal{L}}{\partial \phi} = \nabla_\mu \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)}$$

2) If the above is true, when is $$\phi$$ permitted to be of the following form?

$$\phi = \sum_i C_{\left( i \right)} \phi_{\left( i \right)}$$

Where $$\left\{ C_{\left( i \right)} \right\}$$ are constants i.e. $$\partial_\mu C_{\left( i \right)} = 0$$.

## The 'solution'

To answer the two questions above, we will, respectively,

1. Expand the Euler-Lagrange equations for the new coordinate $$\phi$$ and investigate when it is true.

2. Find the situation in which $$\phi$$ reduces to a linear combination of the solutions $$\left\{ \phi_{\left( i \right)} \right\}$$.

The multivariable chain rule from calculus will be used throughout.

### Expanding the Euler-Lagrange equations

Let us begin by expanding the left hand side of the equations of motion $$\frac{\partial \mathcal{L}}{\partial \phi} = \nabla_\mu \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)}$$, in terms of the solutions $$\left\{ \phi_{\left( i \right)} \right\}$$,

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \phi} & = \sum_i \frac{\partial \mathcal{L}}{\partial \phi_{\left( i \right)}} \frac{\partial \phi_{\left( i \right)}}{\partial \phi} \\
 & = \sum_i \nabla_\mu \left( \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \right) \frac{\partial \phi_{\left( i \right)}}{\partial \phi}
\end{align}
$$

Now, expand $$\frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \right) \phi}$$ and later find the divergence of the right hand side of the equations of motion,

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)} & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \frac{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)}{\partial \left( \partial_\mu \phi \right)} \\
 & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left[ \frac{\partial \left( \partial_\mu \phi \right)}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \right]^{-1} \\
 & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left[ \frac{\partial}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \sum_j \frac{\partial \phi}{\partial \phi_{\left( j \right)}} \partial_\mu \phi_{\left( j \right)} \right]^{-1} \\
 & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left[ \sum_j \frac{\partial}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left( \frac{\partial \phi}{\partial \phi_{\left( j \right)}} \partial_\mu \phi_{\left( j \right)} \right) \right]^{-1} \\
 & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left[ \sum_j \delta_{\left( i \right) \left( j \right)} \frac{\partial}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left( \frac{\partial \phi}{\partial \phi_{\left( j \right)}} \partial_\mu \phi_{\left( j \right)} \right) \right]^{-1} \\
 & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left[ \frac{\partial}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left( \frac{\partial \phi}{\partial \phi_{\left( i \right)}} \partial_\mu \phi_{\left( i \right)} \right) \right]^{-1} \\
 & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left[ \frac{\partial \phi}{\partial \phi_{\left( i \right)}} \frac{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \right]^{-1} \\
 & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left[ \frac{\partial \phi}{\partial \phi_{\left( i \right)}} \delta^\nu_{\phantom{\nu} \mu} \right]^{-1} \\
 & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\nu \phi_{\left( i \right)} \right)} \left( \frac{\partial \phi}{\partial \phi_{\left( i \right)}} \right)^{-1} \delta^\mu_{\phantom{\mu} \nu} \\
 & = \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \frac{\partial \phi_{\left( i \right)}}{\partial \phi}
\end{align}
$$

That was lengthy, but in the end, we have a relatively compact result! In the last step, we took the inverse of the partial derivative $$\frac{\partial \phi}{\partial \phi_{\left( i \right)}}$$ as simply its reciprocal. This is because for the independent set of functions $$\left\{ \phi_{\left( i \right)} \right\}$$, we have $$\frac{\partial \phi}{\partial \phi_{\left( i \right)}} = \frac{d \phi}{d \phi_{\left( i \right)}}$$, which is a difference quotient.

Finally, we find the divergence of the expression obtained in accordance with the Euler-Lagrange equations,

$$
\begin{align}
\nabla_\mu \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)} & = \nabla_\mu \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \frac{\partial \phi_{\left( i \right)}}{\partial \phi} \\
 & = \sum_i \nabla_\mu \left( \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \right) \frac{\partial \phi_{\left( i \right)}}{\partial \phi} + \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \partial_\mu \frac{\partial \phi_{\left( i \right)}}{\partial \phi}
\end{align}
$$

Plugging in the expression for the first term in the previous expansion $$\frac{\partial \mathcal{L}}{\partial \phi}$$,

$$\nabla_\mu \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)} = \frac{\partial \mathcal{L}}{\partial \phi} + \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \partial_\mu \frac{\partial \phi_{\left( i \right)}}{\partial \phi}$$

Thus, we have obtained the equations of motion of a general coordinate $$\phi \left( \left\{ \phi_{\left( i \right)}\right\} \right)$$ constructed from a set of solution fields. However, for $$\phi$$ to be a valid solution, it must obey the Euler-Lagrange equations for the given Lagrangian, which requires us to set:

$$\sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \partial_\mu \frac{\partial \phi_{\left( i \right)}}{\partial \phi} = 0$$

Only if the above is true, can we say that $$\phi$$ is a valid solution.

## Linearity

In the above constraint, note the appearance of canonical 4-momentum coordinates $$\frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)}$$. In general, these are not zero. One possibility which ensures the constraint always holds good is:

$$\partial_\mu \frac{\partial \phi_{\left( i \right)}}{\partial \phi} = 0$$

From the reciprocal law for differentiation, we have,

$$\left( \frac{\partial \phi}{\partial \phi_{\left( i \right)}} \right)^{-2} \partial_\mu \frac{\partial \phi}{\partial \phi_{\left( i \right)}} = 0$$

Again, generaly, $$\frac{\partial \phi}{\partial \phi_{\left( i \right)}} \neq 0$$ so we have,

$$
\begin{align}
\partial_\mu \frac{\partial \phi}{\partial \phi_{\left( i \right)}} & = 0 \\
\frac{\partial \phi}{\partial \phi_{\left( i \right)}} & = C_{\left( i \right)} \: \vert \: \partial_\mu C_{\left( i \right)} = 0
\end{align}
$$

It is easily seen the most general situation where the above is true is,

$$\phi = \sum_i C_{\left( i \right)} \phi_{\left( i \right)}$$

Therefore, if we want our constraint to be true for all solutions, they must be combined only in the above form i.e. as linear combinations.

## Conclusion

### Summary

Let us summarize the results by answering the original questions:

1) When does $$\phi \left( \left\{ \phi_{\left( i \right)} \right\} \right)$$ obey the Euler-Lagrange equations? When the following constraint is obeyed:

$$\sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \partial_\mu \frac{\partial \phi_{\left( i \right)}}{\partial \phi} = 0$$

2) When is $$\phi$$ a linear combination of the solutions $$\left\{ \phi_{\left( i \right)} \right\}$$? Well, for some unspecified $$\left\{ \phi_{\left( i \right)} \right\}$$, we could be in a stroke of luck and automatically have the above constraint to be true. However, if we want it to be true for arbitrary solutions, we must only look at linear combinations of these solutions to generate new solutions.

In other words, there are _always_ $$\phi$$'s which are linear combinations of $$\left\{ \phi_{\left( i \right)} \right\}$$ (and an infinite number of them), although there could be other $$\phi$$'s which are non-trivial, non-linear combinations of the solutions.

### New notation for solutions

In the notation for the solutions $$\left\{ \phi_{\left( i \right)} \right\}$$, we used parentheses around the index as doing otherwise would make them look dubiously like one-forms. Then, we learnt that the most general way to mix these solutions into new ones is to combine them linearly,

$$\phi = \sum_i C_{\left( i \right)} \phi_{\left( i \right)} \: \vert \: \partial_\mu C_{\left( i \right)} = 0$$

Therefore, in the abstract sense, each solution $$\phi_{\left( i \right)}$$ is, in fact, behaving like a basis vector, with the coefficients $$C_{\left( i \right)}$$ forming the components! However, the index $$i$$ here is abstract and not related to the coordinates $$x^\mu$$ as in objects like $$\partial_\mu = \frac{\partial}{\partial x^\mu}$$, so it is still better to retain the parentheses. Nevertheless, we can apply the Einstein summation convention here as it need not be restricted to 'honest' indices:

$$\phi = C^{\left( i \right)} \phi_{\left( i \right)}$$

where $$C^{\left( i \right)} = \delta^{\left( i \right) \left( j \right)} C_{\left( j \right)}$$.

The application of the Einstein summation convention here is further justified by the fact that in the context of our derivations, upper and lower $$\left( i \right)$$ indices have repeatedly appeared along with the summation operation $$\sum \limits_{i}$$. For example,

$$\nabla_\mu \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)} = \frac{\partial \mathcal{L}}{\partial \phi} + \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \partial_\mu \frac{\partial \phi_{\left( i \right)}}{\partial \phi}$$

Hence, the summation symbol becomes redundant and we can write:

$$\nabla_\mu \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)} = \frac{\partial \mathcal{L}}{\partial \phi} + \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \partial_\mu \frac{\partial \phi_{\left( i \right)}}{\partial \phi}$$

### Finishing note

All in all, we see a linear structure in classical field theory, namely in solutions to field equations of the form of the Euler-Lagrange equations. This is the beginning of the portal into the application of linear algebraic notions in classical field theory. The bridge between linear algebra (and the more general tensor algebra) and field theory is an exciting place, so hopefully, we'll tread it in the future!

I also hope you enjoyed this reading post (thanks for doing! :) Cheers.
