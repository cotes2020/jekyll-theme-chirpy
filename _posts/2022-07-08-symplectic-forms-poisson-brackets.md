---
title: "Symplectic Forms from Poisson Brackets"
description: "Hinting at the notion of symplectic forms, by studying Poisson brackets"
categories: [classical mechanics]
tags: [symplectic forms, Poisson brackets, phase space]
---

Much of analytical mechanics dedicates itself to studying the trajectories of dynamical systems. Newtonian mechanics describes trajectories in physical space. On the other hand, Lagrangian mechanics deals with trajectories in configuration space, which is the space of all generalized position and velocity coordinates; while Hamiltonian mechanics studies trajectories in phase space, which comprises all generalized position and momentum coordinates.

A deep mathematical analysis of the Hamiltonian setup reveals a startling geometric structure, namely the geometry of symplectic manifolds. Briefly, it is found that phase space has the characteristics of a differentiable manifold equipped with a closed, nondegenerate differential 2-form. This is a special case of symplectic manifolds, which are similarly equipped with what are known as symplectic forms. The study of symplectic manifolds and their geometry forms the discipline of symplectic geometry.

To expose the working of symplectic geometry, one typically begins with Hamiltonian mechanics and fleshes it out with geometric notions such as manifolds, flow, differential forms, sections and so on. This is an extremely enriching exercise, and one that deserves a more detailed exposition than can be incorporated into a single post.

The purpose of this post, is to subtly indicate the existence of symplectic forms, using an approach that resembles that of tensor calculus. The machinery of tensors can be imagined to 'sit on top' of differential geometry (which in turn relies on topology). We will not build the machinery of tensors yet, but will handwave with its power to merely steal a glimpse at the world of symplectic forms and manifolds.

This approach will admittedly have its limitations. It will ignore some questions for the moment and take for granted the idea of phase space behaving like a differentiable manifold in order to allow the usage of tensor-like language in the first place. However, by keeping at hold the deep questions for the future, the hope is to simply _begin_ sketching out symplectic forms in Hamiltonian mechanics, which would be somewhat buried at first, had we covered our ideas with the intricate tapestry of differential geometry.

## Poisson brackets

### Informal recap

Consider two observables in phase space, $$A \left( q^i, p^j, t \right)$$ and $$B \left( q^i, p^j, t \right)$$. Here, $$q^i$$ are the components of the generalized position $$\pmb{q}$$, of the concerned system. On the other hand, $$p^i$$ are the components of conjugate momentum,

$$p^i = \frac{\partial L}{\partial \dot{q}_i}$$

where $$q_i = \delta_{ij} q^j$$. Note that we have and will continue to use the Einstein summation convention, where dummy indices are implied to be summed over.

The Poisson bracket is a bilinear, anticommutative operation [^1] which acts on two dynamical variables in the manner,

[^1]: And additionally, one that obeys the Leibniz law and Jacobi identity.

$$\left\{ A, B \right\} = \frac{\partial A}{\partial q^i} \frac{\partial B}{\partial p_i} - \frac{\partial A}{\partial p^i} \frac{\partial B}{\partial q_i}$$

It is seen that Hamilton's equations of motion now read,

$$
\begin{align}
\dot{q}^i & = \left\{ q^i, H \right\} \\
\dot{p}^i & = \left\{ p^i, H \right\}
\end{align}
$$

In fact, for any variable $$A$$,

$$
\begin{align}
\dot{A} & = \frac{dA}{dt} \\
 & = \frac{\partial A}{\partial q^i} \dot{q}^i + \frac{\partial A}{\partial p^i} \dot{p}^i + \frac{\partial A}{\partial t} \\
 & = \frac{\partial A}{\partial q^i} \frac{\partial H}{\partial p_i} - \frac{\partial A}{\partial p^i} \frac{\partial H}{\partial q_i} + \frac{\partial A}{\partial t} \\
 & = \left\{ A, H \right\} + \frac{\partial A}{\partial t}
\end{align}
$$

### As bilinear maps on partial derivative operators

Isolating the Poisson bracket of its arguments, we see that it is a map acting on $$\displaystyle{ \frac{\partial}{\partial q^i} }$$ and $$\displaystyle{ \frac{\partial}{\partial p^j} }$$. Let us, therefore, break down the Poisson bracket of two variables as,

$$\left\{ A, B \right\} = \omega \left( \frac{\partial}{\partial q^i}, \frac{\partial}{\partial p^j} \right) \left( A, B \right)$$

The reason the above is bilinear with respect to the derivative operators, is,

$$
\begin{align}
\omega \left( \sum_\alpha \frac{\partial}{\partial q_\alpha^i}, \sum_\beta \frac{\partial}{\partial p_\beta^j} \right) \left( A, B \right) & = \left( \sum_\alpha \frac{\partial A}{\partial q_\alpha^i} \right) \left( \sum_\beta \frac{\partial B}{\partial p_{\beta i}} \right) - \left( \sum_\beta \frac{\partial A}{\partial p_\beta^i} \right) \left( \sum_\alpha \frac{\partial B}{\partial q_{\alpha i}} \right) \\
 & = \sum_\alpha \sum_\beta \left( \frac{\partial A}{\partial q_\alpha^i} \frac{\partial B}{\partial p_{\beta i}} - \frac{\partial A}{\partial p_\beta^i} \frac{\partial B}{\partial q_{\alpha i}} \right) \\
 & = \sum_\alpha \sum_\beta \omega \left( \frac{\partial}{\partial q_\alpha^i}, \frac{\partial}{\partial p_\beta^j} \right) \left( A, B \right)
\end{align}
$$

Here, $$\alpha$$ and $$\beta$$ are abstract indices used to represent different derivative operators.

## Symplectic forms

The map $$\omega$$ we have defined earlier, will eventually turn out to be nothing but a symplectic form! Let us proceed by deriving its properties and nature, using the ideas laid out to far.

### Some properties

We have already seen, from the definition of $$\omega$$, that it is bilinear. It follows from the same definition, that it is anticommutative:

$$\omega \left( \frac{\partial}{\partial q^i}, \frac{\partial}{\partial q^j} \right) = - \omega \left( \frac{\partial}{\partial q^j}, \frac{\partial}{\partial q^i} \right)$$

Symplectic forms are also alternating, since,

$$
\begin{align}
\omega \left( \frac{\partial}{\partial q^i}, \frac{\partial}{\partial q^j} \right) \left( A, B \right) & = \frac{\partial A}{\partial q^i} \frac{\partial B}{\partial q_i} - \frac{\partial A}{\partial q^i} \frac{\partial B}{\partial q_i} \\
 & = 0
\end{align}
$$

Furthermore, it is easily seen that $$\omega$$ is non-degenerate, i.e. if it is $$0$$ for all values of one argument, then the other argument must be $$0$$.

Let us now write $$\omega$$ explicitly as a differential 2-form, to further investigate its properties.

### Differential 2-form representation

Recall the manner in which we have defined $$\omega$$,

$$\omega \left( \frac{\partial}{\partial q^i}, \frac{\partial}{\partial p^j} \right) \left( A, B \right) = \frac{\partial A}{\partial q^i} \frac{\partial B}{\partial p_i} - \frac{\partial A}{\partial p^i} \frac{\partial B}{\partial q_i}$$

Let us factor out the exact arguments $$\displaystyle{ \frac{\partial}{\partial q^i} }$$ and $$\displaystyle{ \frac{\partial}{\partial p^j} }$$ in the right hand side of the above expression, as follows.

$$\omega \left( \frac{\partial}{\partial q^i}, \frac{\partial}{\partial p^j} \right) \left( A, B \right) = \frac{\partial A}{\partial q^i} \delta^{i j} \frac{\partial B}{\partial p^j} - \frac{\partial A}{\partial p^i} \delta^{i j }\frac{\partial B}{\partial q^j}$$

In the component representation, the above may be written as,

$$\omega \left( \frac{\partial}{\partial q^i}, \frac{\partial}{\partial p^j} \right) \left( A, B \right) = \left[ \begin{pmatrix} \frac{\partial}{\partial q^k} & \frac{\partial}{\partial p^l} \end{pmatrix} \begin{pmatrix} 0 & - \delta_{k j} \\ \delta_{li} & 0 \end{pmatrix} \begin{pmatrix} \frac{\partial}{\partial q^i} \\ \frac{\partial}{\partial p^j} \end{pmatrix} \right] \left( A, B \right)$$

Here, each index represents a component in the space represented by $$\left( q_i, p_j \right)$$. It follows that, in an arbitrary basis,

$$\omega_{i j} = \text{d} p_i \otimes \text{d} q_j - \text{d} q_i \otimes \text{d} p_j$$

I.e.,

$$\omega_{i j} = \text{d} p_i \wedge \text{d} q_j$$

### Some more properties

Now that we know the precise form of $$\omega$$, let us show that it is closed, by computing its exterior derivative:

$$
\begin{align}
\text{d} \omega & = \text{d} \left( \text{d} p \wedge \text{d} q \right) \\
 & = \text{d}^2 p \wedge \text{d} q - \text{d} q \wedge \text{d}^2 p \\
 & = 0 \wedge \text{d} q - \text{d} q \wedge 0 \\
 & = 0
\end{align}
$$

Here, $$\text{d} q$$ is the one-form whose components are $$q_i$$, and likewise for $$\text{d} p$$. We have used the statement $$d^2 = 0$$, which applies to any $$k$$-form. We thus found that the exterior derivative of $$\omega$$ is $$0$$, i.e. it is closed. From a tensorial perspective, the exterior derivative of a $$k$$-form $$A$$ is defined as the antisymmetrized $$\left( k+1 \right)$$-form,

$$\left( \text{d} A \right)_{i_1 \dots i_{k+1}} = \left( k+1 \right) \partial_{[ i_1} A_{i_2 \dots i_{k+1} ]}$$

To understand how the above works on a symplectic manifold, we will have to explicitly use tensors, by treating position and momentum as equivalent kinds of coordinates. I will likely cover the same in a future post.

## Conclusion

To summarize, we have found how symplectic forms blend into Hamiltonian mechanics, via the structure of Poisson brackets. Subsequently, we explored the important properties of symplectic forms and also their form as maps, in both coordinate-dependent and coordinate-independent representation.

For further reading, visit the following:

- [Poisson bracket](https://en.wikipedia.org/wiki/Poisson_bracket)

- [Symplectic geometry](https://en.wikipedia.org/wiki/Symplectic_geometry)

- [Symplectic manifold](https://en.wikipedia.org/wiki/Symplectic_manifold)

- [Symplectic form](https://mathworld.wolfram.com/SymplecticForm.html)

- [Exterior derivative](https://en.wikipedia.org/wiki/Exterior_derivative)

- [Special Relativity and Flat Spacetime](https://preposterousuniverse.com/wp-content/uploads/grnotes-one.pdf)
