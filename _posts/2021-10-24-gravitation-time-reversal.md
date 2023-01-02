---
title: "All the World's Not a Stage"
description: "How gravitation behaves under time reversal"
categories: [relativity]
tags: [gravitation, time symmetry, symmetries]
redirect_from: all-world-not-stage
---

## A story without a film

Consider a gravitational system comprising the earth and an apple. The apple is released from a certain height and it plummets to the ground. How would the evolution of this system proceed, if, instead, time ran backwards?

Our intuition tells us that if time runs backwards, the apple should, as if by definition, rise up from the ground and shoot to the sky in eternally decelerated motion. In other words, if we shot a film of the usual falling apple, reversing time would be equivalent to playing the film backwards.

But our universe, strangely enough, does not function like such a film (where bodies merely play their part; ok, nevermind the reference). How can we be sure? Let us dive into the mathematics of our universe.

## Newtonian gravity

### Time symmetry

Before getting to Einstein's general theory of relativity, let us try to answer the question in the domain of Newtonian physics. Here, gravity is famously described by Newton's 'universal law of gravitation' (which is not really universal, but will do just fine for us at the moment),

$$F = - \frac{GMm}{r^2}$$

The minus sign in the equation is pedantic (it has to do with gravity being an attractive force), but it will make no difference in answering the question.

To see what happens if we reverse time, we must perform the substitution $$t \mapsto -t$$ where $$t$$ is the parameter representing time. To do this, we must incorporate time explicitly into the law of gravitation, which can be done by going back to the definition of force,

$$
\begin{align}
m \frac{d^2 r}{dt^2} & = - \frac{GMm}{r^2} \\
\frac{d^2 r}{dt^2} & = - \frac{GM}{r^2}
\end{align}
$$

Now, watch what happens when we reverse time:

$$
\begin{align}
\frac{d^2 r}{d \left( -t \right)^2} & = \frac{d}{d \left( -t \right)} \frac{d}{d \left( -t \right)} r \\
 & = \left( - \frac{d}{dt} \right) \left( - \frac{d}{dt} \right) r \\
 & = \frac{d}{dt} \frac{d}{dt} r \\
 & = \frac{d^2 r}{dt^2}
\end{align}
$$

Pardon the loose notation, but you get the idea. The left hand side of the law of gravitation, containing the acceleration term, remains invariant under time reversal! And the right hand side does not use time explicitly, so it too remains invariant.

Thus, gravity theoretically doesn't change a bit if we reverse time. Except, _motion_ does. The motion of a body is determined by the force field it is placed in, here described by Newton's law of gravitation. But motion is _also_ determined by initial conditions, which in this case are the initial position and velocity of the apple relative to the earth.

### Initial conditions

If the ball is placed at an initial distance of $$r=r_0$$ from the centre of mass of the earth, reversing time does not change the way we measure $$r$$ or $$r_0$$. Therefore, the initial positions of bodies in a gravitational system are invariant under time reversal.

But when we express $$r$$ as a function of time and compute velocity, time reversal flips the sign of velocity as it is a first-order time derivative of position:

$$\frac{dr}{d \left( -t \right)} = - \frac{dr}{dt}$$

Hence, the initial velocities of bodies in a gravitational system are reversed under time reversal.

### Result

To summarize the above results, the parameters required to predict the motion of an apple near the earth change in the following manner under time reversal:

1. The force field of gravity remains invariant.

2. The initial positions remain invariant.

3. The initial velocities are reversed.

This means that if we reverse time, an apple will fall all right to the ground. However, if the apple were thrown downward with an initial velocity, reversing time amounts to throwing the ball _upward_, so it first rises up and ultimately falls as gravity still acts in the same direction (downward). And vice-versa, if the ball were thrown upward in the forward direction of time instead.

## General relativity

### A slightly different time symmetry

Like Newtonian gravity, general relativity (GR) is time symmetric. Formally, this means that the motion obtained by reversing time is still a valid motion in the original system, even though this new motion may not be identical to the original one, as we have seen.

Time symmetry works differently for GR in general, as compared to Newtonian gravity, which is a special case of GR. In GR, gravitational force fields are replaced by the metric tensor, and they _do_ change under time reversal.

### The metric tensor

The metric tensor $$\pmb{g}$$ provides a complete description of a spacetime (at least in Riemannian/pseudo-Riemannian geometry). Its components in a holonomic basis are defined as,

$$g_{\mu \nu} = \left\langle \partial_\mu, \partial_\nu \right\rangle$$

where $$\left\{ \partial_\mu \right\}$$ is the coordinate basis for the tangent space at each event in spacetime. $$\left\langle \right\rangle$$ is the inner product, a generalization of the familiar dot product.

Under time reversal $$t \mapsto - t$$, $$\partial_0 \mapsto - \partial_0$$. As a result,

$$
\begin{align}
g_{00} & \mapsto \left\langle - \partial_0, - \partial_0 \right\rangle = \left\langle \partial_0, \partial_0 \right\rangle = g_{00} \\
g_{ij} & \mapsto \left\langle \partial_i, \partial_j \right\rangle = g_{ij} \\
g_{i0}=g_{0i} & \mapsto \left\langle \partial_i, - \partial_0 \right\rangle = - \left\langle \partial_i, \partial_0 \right\rangle = - g_{i0}
\end{align}
$$

Therefore, under time reversal, only the spacetimelike components of the metric tensor, $$g_{i0}$$ are reversed. Nevertheless, the new metric is still a valid solution of the original field equations (which do not change, by virtue of being second-order partial differential equations).

Since the metric changes under time reversal, so does the motion of bodies in the gravitational system, even if we ignore the changes in initial conditions.

### Initial conditions

In GR, the initial conditions required to predict the evolution of a dynamic metric are the initial values $$g_{\mu \nu}$$ and its first derivatives, $$\partial_\rho g_{\mu \nu}$$. From the previous subsection, we know that the spacetimelike part of the metric is reversed under time reversal. This is a key difference from the Newtonian case, where initial positions had remained invariant.

Naturally, the first derivatives of the metric change too, along with a sign inversion for the time derivative.

### Newtonian limit

Newtonian gravity, in the framework of GR, is a weak-field, spherically symmetric vacuum solution. A characteristic of spherical symmetry is that it is described by orthogonal coordinate systems. This sets the off-diagonal, or spacetimelike components $$g_{i0}$$ of the metric tensor to $$0$$.

Under time reversal, the spacetimelike components of the metric tensor switch signs as we have seen. But since they are zero now, they remain invariant. This explains why a Newtonian gravitational field remains exactly the same even if time is reversed.

## Modified and quantum gravity

What happens to spacetime under time reversal if we don't restrict ourselves to the Einstein field equations? The metric tensor plays a crucial role in every potentially correct alternative formulation. Thus, one expects that conceptually, time reversal works in the same way in these theories.

However, things get interesting if we consider formulations of GR in which not only the metric, but other objects characterize the geometry of spacetime too. For example, there are [scalar-tensor theories](https://en.wikipedia.org/wiki/Alternatives_to_general_relativity#Scalar-tensor_theories), where there are one or more free scalar parameters (such as a spatially and temporally variable universal gravitational constant in [Brans-Dicke theory](https://en.wikipedia.org/wiki/Brans%E2%80%93Dicke_theory), though the theory gives results inconsistent with observations).

Then, there are [vector-tensor theories](https://en.wikipedia.org/wiki/Alternatives_to_general_relativity#Vector-tensor_theories) and even a [scalar-tensor-vector gravity](https://en.wikipedia.org/wiki/Scalar%E2%80%93tensor%E2%80%93vector_gravity).

With such considerations in mind, there is no general answer to the question of how spacetime changes under time reversal. In most cases, it is arbitrary, depending on the scalar or vector field concerned.

But as of now (2021), Einstein's GR is the reasonably correct classical model of gravitation. Speaking of classical models, things may take a turn when a quantum theory of gravitation is discovered. It is still not certain whether the metric will still play a crucial role in a consistent theory of quantum gravity, or if its time-symmetric nature will be retained in whatever replaces the metric. 'Time' will tell.
