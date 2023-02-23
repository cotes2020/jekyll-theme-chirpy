---
title: "Scalar Field Lagrangian From Symmetry Considerations"
description: "Or, how field theory doesn't leave us much choice"
categories: [classical field theory]
tags: [Klein-Gordon theory, energy-momentum tensor, Noether theorem, symmetries]
---

Welcome to another post on, well, you guessed it: scalar field theory. Today, we will be deriving a Lagrangian density for scalar fields that appears almost everywhere in physics (namely in the Klein-Gordon and related theories).

Common arguments for assuming the form of the Lagrangian in question either take motivation from too specific systems such as chains (which are promoted to scalar fields using a procedure that does not work for higher-rank field theories), or extremely formal analogies with the point-particle Lagrangian. I will, instead, use an approach I find easy to remember.

Since the method has to do with the energy-momentum tensor for scalar fields and what it physically means, most of this post elaborates on the same. However, if you are aware of the logic and properties of the energy-momentum tensor, feel free to skip to [the end](#scalar_field_lagrangian).

## Canonical 4-momentum field

Recall the special relativistic Euler-Lagrange equation for a scalar field $$\phi$$ with Lagrangian $$\mathcal{L}$$,

$$\frac{\partial \mathcal{L}}{\partial \phi} = \partial_\mu \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)}$$

It is perfectly permissible to work with the above equation in inertial coordinates, i.e. coordinates where the Christoffel symbols vanish. As a result, we can, in such coordinates, treat $$\partial_\mu$$ like a tensor, and hence, the adjacent term too,

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \left( \partial_{\mu^\prime} \phi \right)} & = \frac{\partial \left( \partial_{\mu} \phi \right)}{\partial \left( \partial_{\mu^\prime} \phi \right)} \frac{\partial \mathcal{L}}{\partial \left( \partial_{\mu} \phi \right)} \\
 & = \frac{\partial}{\partial \left( \partial_{\mu^\prime} \phi \right)} \left( \Lambda^{\mu^\prime}_{\phantom{\mu^\prime} \mu} \: \partial_{\mu^\prime} \phi \right) \frac{\partial \mathcal{L}}{\partial \left( \partial_{\mu} \phi \right)} \\
 & = \Lambda^{\mu^\prime}_{\phantom{\mu^\prime} \mu} \: \frac{\partial \mathcal{L}}{\partial \left( \partial_{\mu} \phi \right)}
\end{align}
$$

Indeed, $$\frac{\partial \mathcal{L}}{\partial \left( \partial_{\mu} \phi \right)}$$ transforms like a vector, as second-order terms vanish in inertial coordinates. However, it's not hard to promote the quantity to a tensor in arbitrary coordinates: we simply replace partial derivatives $$\partial_\mu$$ with covariant derivatives $$\nabla_\mu$$, so that all second-order terms involving the Christoffel symbols cancel out. However, let us stick to inertial coordinates as it's much simpler to manipulate expressions in them and later switch to coordinate-independent quantities.

The vectorial quantity we just found will appear repeatedly in this post, so let us simply call it the _canonical 4-momentum field_ $$\pi^\mu$$ in lieu of conjugate momentum in the Lagrangian mechanics of point particles [^1] .

[^1]: Technically, the quantity described is a momentum _density_ field. However, we will simply refer to it as momentum, much like the Lagrangian density is commonly called the Lagrangian.

## Continuity equation

The Euler-Lagrange equations tell the following story: the dependence of the dynamics of a scalar field on the field, $$\frac{\partial \mathcal{L}}{\partial \phi}$$, is linearly related to the divergence of the canonical 4-momentum field $$\pi^\mu$$,

$$\frac{\partial \mathcal{L}}{\partial \phi} = \partial_\mu \pi^\mu$$

If the dynamics of a system are invariant under changes in $$\phi$$, $$\frac{\partial \mathcal{L}}{\partial \phi} = 0$$ and hence $$\partial_\mu \pi^\mu = 0$$. If we expand the last equation using the mostly-plus convention for the metric and negate both sides,

$$\partial_0 \pi^0 + \partial_i \pi^i = 0$$

This is the continuity equation. $$\pi^0$$ can be thought of as the local energy density, and $$\pi^i$$ the corresponding flux. By the divergence theorem, $$\partial_0 \displaystyle{ \int d^3 x \: \pi^0 = 0 }$$ as $$\displaystyle{ \int d^3 x \left( \pmb{\nabla} \cdot \pmb{\pi} \right) }$$ is a constant surface integral. Hence, the total energy $$\displaystyle{ \int d^3 x \: \pi^0 = 0 }$$ is conserved.

However, when $$\frac{\partial \mathcal{L}}{\partial \phi} \neq 0$$, the above is no more true. What, then, is conserved in a general scalar field theory?

## Noether current

It might feel strange at first that when $$\frac{\partial \mathcal{L}}{\partial \phi} \neq 0$$, the energy $$\displaystyle{ \int d^3 x \: \pi^0 }$$ is not conserved. But from the point of view of Noether's theorem, that makes sense. Say the dynamics (i.e. Lagrangian) of a field are not symmetric with respect to the field. Since the field may explicitly depend on the time coordinate, energy is not conserved.

But what happens if we consider symmetries of spacetime itself? Can we then impose conservation laws on arbitrary fields with arbitrary dynamics? As it turns out, yes.

Minkowski spacetime has many symmetries, but let us start with symmetry under translations in spacetime: for small translations of a field in spacetime $$\delta x^\rho$$, the variation of the Lagrangian is zero, $$\delta \mathcal{L} = 0$$.

Since the translation is small, we can expand $$\phi$$ to first-order,

$$\delta \phi = \delta x^\mu \: \partial_\mu \phi$$

The corresponding variation of the Lagrangian is,

$$
\begin{align}
\delta \mathcal{L} & = \frac{\partial \mathcal{L}}{\partial \phi} \delta \phi + \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)} \delta \left( \partial_\mu \phi \right) \\
 & = \partial_\mu \pi^\mu \delta x^\nu \: \partial_\nu \phi + \pi^\mu \partial_\mu \left( \delta \phi \right) \\
 & = \partial_\mu \pi^\mu \delta x^\nu \: \partial_\nu \phi + \pi^\mu \partial_\mu \left( \delta x^\nu \: \partial_\nu \phi \right) \\
 & = \partial_\mu \pi^\mu \delta x^\nu \: \partial_\nu \phi + \pi^\mu \delta x^\nu \partial_\mu \partial_\nu \phi \\
 & = \left[ \partial_\mu \left( \pi^\mu \partial_\nu \phi \right) + \partial_\nu \mathcal{L} \right] \delta x^\nu \\
 & = \partial_\nu \mathcal{L} \: \delta x^\nu = 0
\end{align}
$$

In the last step, we wrote $$\delta \mathcal{L} = \partial_\nu \mathcal{L} \: \delta x^\nu$$ since only the coordinates really change under a translation. Now, as $$\delta x^\nu$$ is arbitrary,

$$
\begin{align}
\partial_\mu \left( \pi^\mu \partial_\nu \phi \right) - \partial_\nu \mathcal{L} & = 0 \\
\partial_\mu \left( \pi^\mu \partial_\nu \phi \right) - \delta^\mu_{\phantom{\mu} \nu} \partial_\mu \mathcal{L} & = 0 \\
\partial_\mu \left( \pi^\mu \partial_\nu \phi \right) - \partial_\mu \left( \delta^\mu_{\phantom{\mu} \nu} \mathcal{L} \right) & = 0 \\
\partial_\mu \left[ \pi^\mu \partial_\nu \phi - \delta^\mu_{\phantom{\mu} \nu} \mathcal{L} \right] & = 0 \\
\end{align}
$$

Voila, we have found a conserved Noether current for scalar fields!

## Energy-momentum tensor

### Definition

In the above derivation, $$\left[ \pi^\mu \partial_\nu \phi - \delta^\mu_{\phantom{\mu} \nu} \mathcal{L} \right]$$ forms the components of the canonical energy-momentum tensor with a lowered index,

$$T^\mu_{\phantom{\mu} \nu} = \pi^\mu \partial_\nu \phi - \delta^\mu_{\phantom{\mu} \nu} \mathcal{L}$$

Generally, the energy-momentum tensor (motivated by the dynamics of discrete sets of particles) is employed as a $$\left( 2, 0 \right)$$ or $$\left( 0, 2 \right)$$ tensor. We can readily obtain the same by raising/lowering indices in the tensor above:

$$T^{\mu \nu} =  \pi^\mu \partial^\nu \phi - \eta^{\mu \nu} \: \mathcal{L}$$

where $$\partial^\nu \phi = \eta^{\rho \nu} \partial_\rho \phi$$. This tensor too has vanishing divergence,

$$
\begin{align}
\partial_\mu T^{\mu \nu} & = \partial_\mu \left( \eta^{\nu \rho} \: T^\mu_{\phantom{\mu} \rho} \right) \\
 & = \eta^{\nu \rho} \partial_\mu T^\mu_{\phantom{\mu} \rho} \\
 & = 0
\end{align}
$$

which is a set of $$4$$ equations, one for each $$\nu$$ [^2] .

[^2]: We could bring the metric out from the partial derivative as in inertial coordinates, the metric is constant. In non-inertial coordinates, we'd instead work with covariant derivatives, and assuming metric compatibility of the connection, the metric can, again, be treated as a constant. Therefore, the expression for the energy-momentum tensor in curvilinear coordinates and even general relativity (where it is dubbed the 'canonical' energy-momentum tensor to distinguish it from the energy-momentum tensor appearing in the Einstein field equations) is trivially obtainable: we simple replace the Minkowski metric with a metric tensor.

### Physical interpretation

To understand the physical meaning of the energy-momentum tensor, let us express its vanishing divergence as a continuity equation. Recall that:

$$\partial_\mu T^{\mu \nu} = 0$$

We can expand the above into timelike and spacelike parts,

$$\partial_0 T^{0 \nu} + \partial_i T^{i \nu} = 0$$

We may interpret this as the continuity equation for $$\pi^\nu$$. Then, $$T^{i \nu}$$ is the flux of $$\pi^\nu$$ through a surface of constant $$x^i$$. Or in spacetime, $$T^{\mu \nu}$$ is the flux of $$\pi^\nu$$ through a surface of constant $$x^\mu$$.

So, this is what the energy-momentum tensor really encodes: _the flux of canonical 4-momentum in spacetime_ [^3] .

[^3]: In general relativity, the canonical energy-momentum tensor does not necessarily encode momentum flux like the energy-momentum tensor appearing in the Einstein field equations. However, the distinction is not important in special relativity.

### Symmetry

In the above picture of the energy-momentum tensor, we find an interesting property: the energy-momentum tensor is totally symmetric, i.e. $$T^{\mu \nu} = T^{\nu \mu}$$. Let's see why.

Recall that $$T^{\mu \nu}$$ is the flux of $$\pi^\nu$$ perpendicular to $$\pmb{e}_\mu$$. Now consider what it means for $$\pi^\nu$$ to have a flux perpendicular to $$\pmb{e}_\mu$$ : energy density $$\pi^0$$ is being transported along $$\pmb{e}_\mu$$ with conjugate-momentum $$\pi^i$$ in the $$\pmb{e}_\nu$$ direction. But due to conjugate-momentum in the direction $$\pmb{e}_\nu$$, there must flux perpendicular to the same, so $$T^{\nu \mu} = T^{\mu \nu}$$.

<a name="scalar_field_lagrangian"></a>

## Scalar field Lagrangian

Recalling the energy-momentum tensor for a scalar field and its symmetry, we can assert,

$$\pi^\mu \partial^\nu \phi - \eta^{\mu \nu} \mathcal{L} = \pi^\nu \partial^\mu \phi - \eta^{\nu \mu} \mathcal{L}$$

But the metric is symmetric, hence,

$$\pi^\mu \partial^\nu \phi = \pi^\nu \partial^\mu \phi$$

Since this is true for any $$\pi^\mu$$ and $$\partial^\nu \phi$$,

$$\pi^\mu = \partial^\mu \phi$$

It is now straightforward to guess the general form of $$\mathcal{L}$$ by expanding both sides above,

$$\frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)} = \eta^{\mu \nu} \partial_\nu \phi$$

which, after some playing around, is found to be possible only if $$\mathcal{L}$$ is of the form,

$$\mathcal{L} = \frac{1}{2} \partial_\mu \phi \: \eta^{\mu \nu} \: \partial_\nu \phi - V \left( \phi \right)$$

## Conclusion

By positing the Lagrangian for a scalar field in the above manner, we justified the expression for $$\mathcal{L}$$ based on the symmetry of $$T^{\mu \nu}$$. This tensor was itself derived from the symmetry of the dynamics of a scalar field under translations in spacetime. These are very fundamental ideas that do not depend on the final form of $$\mathcal{L}$$.
