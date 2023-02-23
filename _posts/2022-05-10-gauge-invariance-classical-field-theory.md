---
title: "Gauge Invariance in Classical Field Theory"
description: "Exploring a subtle symmetry key to many field theories"
categories: [classical field theory]
tags: [gauge invariance, symmetries]
---

In [Combining Valid Solutions Into New Ones in Classical Field Theory]({% post_url 2022-04-23-combining-solutions-classical-field-theory %}), we have seen how an arbitrary field $$\phi$$ constructed from a class of solutions $$\left\{ \phi_{\left( i \right)} \right\}$$ for some equations of motion, must obey:

$$\frac{\partial \mathcal{L}}{\partial \phi} - \nabla_\mu \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi \right)} = - \sum_i \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi_{\left( i \right)} \right)} \partial_\mu \frac{\partial \phi_{\left( i \right)}}{\partial \phi}$$

We further interpreted the new field $$\phi$$ as being physical only if the above equation resembled the Euler-Lagrange equations. This in general, turned out to be true when $$\phi$$ is a linear combination of the solution fields $$\left\{ \phi_{\left( i \right)} \right\}$$.

However, in this post, we shall see how there is a very different way to see the new equations of motion $$\phi$$ satisfies: that they can in fact be correct, warranting a modification of the original Lagrangian! Though this may seem strange at first, it is, in fact, extremely crucial to what are called 'gauge field theories' in particle physics. We will not go into quantum field theories (QFTs) yet, but we will construct the procedure in the framework of classical field theory, which is simpler and already has a lot of the structure found in QFTs.

With that said, let us begin.

## Fields as coordinates: revisited

This post begins with the same concept as in [Combining Valid Solutions Into New Ones in Classical Field Theory]({% post_url 2022-04-23-combining-solutions-classical-field-theory %}). Namely, we consider an indexed solution set $$\left\{ \phi_{\left( i \right)} \right\}$$ of scalar fields satisfying the equations of motion for a given Lagrangian density $$\mathcal{L}$$. Moreover, we interpret these fields as abstract 'coordinates', which can be used to form new coordinates.

We now take this analogue of coordinates a step further. Combining the said coordinate fields into new ones shall be parallel to coordinate transformations. Let us write such transformations as:

$$\phi_{\left( i \right)} \to \widetilde{\phi}_{\left( i \right)} \left( \left\{ \phi_{\left( j \right)} \right\} \right)$$

We assume such transformations are bijective and differentiable. Thus, the number of coordinates remains the same in such transformations [^1] and they are freely convertible to one another. This further hints that these sets of fields are different aspects of the same thing, much like coordinates are arbitrary facets of invariant objects such as tensors.

## Notation

1) We [noted]({% post_url 2022-04-23-combining-solutions-classical-field-theory %}#new-notation-for-solutions) in the previous post that despite the solution fields' index $$\left( i \right)$$ not being physical in the sense of tensor indices, it behaves very similar to one. This will become further evident in this post, and to save us from the hassle of writing the parantheses around the index repeatedly, we will simply omit them.

2) However, to distinguish the above indices from the indices $$i, j, k$$ etc. frequently used to denote spacelike components, we will use $$a, b, c$$ and so on (nevertheless, we will not need to refer to spacelike indices separately in this post).

3) Transformation of coordinate fields and their functions will be accented with tildes. For example, when $$\phi_a \to \widetilde{\phi}_a$$, conjugate momenta $$\pi^\mu_{\phantom{\mu} a}$$ are mapped as $$\pi^\mu_{\phantom{\mu} a} \to \widetilde{\pi}^\mu_{\phantom{\mu} a}$$.

We do not accent the index itself in the manner of priming regular indices. To demonstrate the reason, consider only one solution $$\phi$$, which corresponds to a scalar in the context of coordinate fields. Now, transforming it to some new coordinate $$\widetilde{\phi}$$ does not leave it invariant, unlike for real scalars. Therefore, despite possessing similarity with tensor indices, the coordinate fields' indices must be handled carefully.

4) Analogous to Jacobian tensors in geometry, we define the following scalar quantity: [^2]

$$J^b_{\phantom{b} a} = \frac{\partial \widetilde{\phi}_a}{\partial \phi_b}$$

5) We employ the Einstein summation convention. For example,

$$\widetilde{\pi}^\mu_{\phantom{\mu} a} = J^b_{\phantom{b} a} \pi^\mu_{\phantom{\mu} b}$$

6) We define the dual of a coordinate field as,

$$\phi^a = \delta^{a b} \phi_b$$

Therefore, as an example, we write,

$$\pi^\mu_{\phantom{\mu} a} = \frac{\partial \mathcal{L}}{\partial \left( \partial_\mu \phi^a \right)}$$

In general, we can raise and lower coordinate field indices respectively using $$\delta^{a b}$$ and $$\delta_{a b}$$, analogous to the contravariant and covariant metric tensors.

[^1]: Strictly speaking, the said transformations must be intrinsic i.e. they must not refer to any notion of ambient space. Otherwise, we are permitted to transform a set of coordinates into a larger number of coordinates, bijectively.

[^2]: Unlike in tensor calculus, we do _not_ have $$\widetilde{\phi}_a = J^b_{\phantom{b} a} \phi_b$$. Instead, from the chain rule in calculus, we can only say the same for differentials of the fields, i.e. $$\displaystyle{d \widetilde{\phi}_a = \frac{\partial \widetilde{\phi}_a}{\partial \phi_b} d \phi_b = J^b_{\phantom{b} a} d \phi_b}$$. However, for the conjugate momenta $$\pi^\mu_{\phantom{\mu} a}$$, we can assert $$\widetilde{\pi}^\mu_{\phantom{\mu} a} = J^b_{\phantom{b} a} \pi^\mu_{\phantom{\mu} b}$$. Such is the case with any quantity involving the geometric covariant derivatives of the fields. This can be easily demonstrated in inertial coordinates, where geometric covariant derivatives reduce to partial derivatives, which are in turn quotients of appropriate differentials.

## Gauge invariance

A question which readers may have asked by this point, is that if solution fields for a given set of Euler-Lagrange equations are like coordinates, what underlying, invariant structure do they represent?

This brings us to gauge invariance and field theories obeying them, called gauge theories. To understand them, we first recall what a field $$\phi$$ means, in the first place. It is a varying physical parameter with each value corresponding to a a degree of freedom (therefore, there are an uncountably infinite number of them as the parameter here is a real number). We encode the dynamics of the system into the Lagrangian $$\mathcal{L}$$, and solutions of the equations of motion obtained give all the possible configurations of the system.

Now, it is possible that in a system's mathematical description, there are extra degrees of freedom which are not physical. In this scenario, some of the degrees of freedom in the equations of motion become redundant, and we can express a given configuration using multiple fields. The particular choice of representing a configuration is called a 'gauge' and transformations between different gauges are called 'gauge transformations'.

The idea of gauge invariance is that the dynamics of a system, and hence its action, must remain invariant under gauge transformations, as they are arbitrary artefacts of 'unphysical' degrees of freedom. This is analogous to general covariance, the assertion that physics must remain invariant under coordinate transformations as they are artifical constructs.

To connect gauge invariance to the previous ideas, let us state how we want to incorporate it. We select a set of distinct and physical solutions $$\left\{ \phi_a \right\}$$ for some equations of motion and switch to a different gauge so that each coordinate field is mapped to a new one. Since field theory must be gauge invariant, the fields in the new gauge must be valid solutions, and thereby satisfy the Euler-Lagrange equations.

But the problem is, they do not:

$$
\begin{align}
\nabla_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} & = \nabla_\mu \left( J^b_{\phantom{b} a} \pi^\mu_{\phantom{\mu} b} \right) \\
 & = J^b_{\phantom{b} a} \nabla_\mu \pi^\mu_{\phantom{b} b} + \pi^\mu_{\phantom{b} b} \partial_\mu J^b_{\phantom{b} a} \\
 & = J^b_{\phantom{b} a} \frac{\partial \mathcal{L}}{\partial \phi^b} + \pi^\mu_{\phantom{b} b} \partial_\mu J^b_{\phantom{b} a} \\
 & = \frac{\partial \mathcal{L}}{\partial \widetilde{\phi}^a} + \pi^\mu_{\phantom{b} b} \partial_\mu J^b_{\phantom{b} a} \\
\therefore  \frac{\partial \mathcal{L}}{\partial \widetilde{\phi}^a} - \nabla_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} & = - \pi^\mu_{\phantom{b} b} \partial_\mu J^b_{\phantom{b} a} \neq 0
\end{align}
$$

## Gauge covariant derivatives

Faced with the above problem, we are forced to modify the Lagrangian so that the new equations of motion (structurally Euler-Lagrange equations) are the above, thereby introducing gauge invariance.

Let us write the general equations of motion in a way which resembles the Euler-Lagrange equations as much as possible:

$$\frac{\partial \mathcal{L}}{\partial \widetilde{\phi}^a} - D_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} = 0$$

where,

$$
\begin{align}
D_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} & = \nabla_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} - \pi^\mu_{\phantom{\mu} b} \partial_\mu J^b_{\phantom{b} a} \\
 & = \nabla_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} - J^c_{\phantom{c} b} \widetilde{\pi}^\mu_{\phantom{\mu} c} \partial_\mu J^b_{\phantom{b} a}
\end{align}
$$

Further contracting the coefficients of conjugate momenta in the above as,

$$
J^c_{\phantom{c} b} \partial_\mu J^b_{\phantom{b} a} = G^c_{\phantom{b} \mu a}
$$

we have,

$$D_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} = \nabla_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} - \widetilde{\pi}^\mu_{\phantom{\mu} c} G^c_{\phantom{b} \mu a}$$

The above quantity is the gauge covariant derivative of conjugate momentum. Here, we are adding correction terms to the usual covariant derivative, much like the latter are correction terms added to partial derivatives:

$$\nabla_\mu \omega_\nu = \partial_\mu \omega_\nu - \omega_\rho \Gamma^\rho_{\phantom{\rho} \mu \nu}$$

This also demonstrates the correspondance between the connection coefficients $$\Gamma^\rho_{\phantom{\rho} \mu \nu} = dx^\rho \left( \nabla_\mu \partial_\nu \right)$$ in geometry and the gauge connection coefficients $$G^c_{\phantom{b} \mu a} = J^c_{\phantom{c} b} \partial_\mu J^b_{\phantom{b} a}$$ in the previous equation.

In retrospect, choosing the indices of the fields to be downstairs was a good choice, for otherwise, the sign of the correction terms in the covariant derivative would be inverted with respect to the geometric case.

## Lagrangian

We have proposed that the general equations of motion for physical fields are,

$$\frac{\partial \mathcal{L}}{\partial \widetilde{\phi}^a} - D_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a} = 0$$

Now, let us find a suitable gauge-invariant Lagrangian $$\widehat{\mathcal{L}}$$ which yields equations of motion of the above form (thereby not requiring us to switch the gauge).

Firstly, we can say from the form of the equations of motion that the divergence of the new conjugate momenta, say $$\widehat{\pi}^\mu_{\phantom{\mu} a}$$, are

$$\nabla_\mu \widehat{\pi}^\mu_{\phantom{\mu} a} = D_\mu \widetilde{\pi}^\mu_{\phantom{\mu} a}$$

which implies,

$$\frac{\partial \widehat{\mathcal{L}}}{\partial \phi^a} = \frac{\partial \mathcal{L}}{\partial \widetilde{\phi}^a}$$

As the dependency of both Lagrangians on the field is the same, we have in general,

$$\widehat{\mathcal{L}} = \mathcal{L} \left( \phi, \dots \right)$$

Since the new Lagrangian is gauge-invariant, the only derivatives of the field it can contain are gauge covariant derivatives. Therefore, we replace any covariant derivatives in the original Lagrangian with gauge covariant derivatives:

$$\widehat{\mathcal{L}} = \mathcal{L} \left( \phi, D_\mu \phi \right)$$

We define the gauge covariant derivative of a general scalar field in the following manner (chain rule), so that the properties of derivatives are retained and gauge covariant derivatives can meaningfully reduce to covariant derivatives in any gauge where $$G^b_{\phantom{b} \mu a} = 0$$,

$$
\begin{align}
D_\mu \phi & = \frac{\partial \phi}{\partial \phi_a} D_\mu \phi_a \\
 & = \frac{\partial \phi}{\partial \phi_a} \left( \partial_\mu \phi_a - \phi_b G^b_{\phantom{b} \mu a} \right) \\
 & = \partial_\mu \phi - \phi_b G^b_{\phantom{b} \mu a} \frac{\partial \phi}{\partial \phi_a}
\end{align}
$$

## Conclusion

We have, at last, found a Lagrangian which preserves the gauge invariance of physical solution fields. Now, we are left to explore the deep facts it reveals about nature. One of the most striking among them is the conservation of charge, which emerges in scalar electrodynamics from the gauge invariance of complex-valued fields under $$U \left( 1 \right)$$. I shall try to cover it in a future series on quantum mechanics.

However, the manner in which the said conservation law emerges (continuity equation) can be seen even in classical field theory. I did not cover it here as it would make the post too long. So, it will be the subject of a new post in the future. Stay tuned! :)
