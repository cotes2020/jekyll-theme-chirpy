---
title: "Conservation of Mass in Classical Mechanics"
description: "The origin of conservation of mass in classical mechanics"
categories: [classical mechanics]
tags: [mass, Noether theorem, symmetries, affine space]
redirect_from: conservation-of-mass-classical-mechanics
---

In this post, we will derive the conservation of total mass of a closed system from simpler physical facts, within the framework of classical mechanics. We will first recall the key concepts which will be used to build the argument, namely:

1. The conservation of total linear momentum

2. The principle of relativity

Then, we will construct [the main argument]({% post_url 2022-03-30-conservation-mass-classical-mechanics %}#conservation-of-total-mass) on the basis of the above.

## Conservation of total linear momentum

### Noether's theorem

Consider a system of particles with generalized coordinates $$\left\{ \pmb{q}_{i} \right\}$$ and conjugate momenta $$\left\{ \pmb{p}_{i} \right\}$$. Let the system be infinitesimally displaced as,

$$\pmb{q}_{i} \to \pmb{q}_{i} + \epsilon \: \pmb{Q}$$

where $$\epsilon$$ is a very small constant and $$\pmb{Q}$$ is an arbitrary 3-vector. The variation in the generalized coordinates is,

$$\delta \pmb{q}_{i} = \epsilon \: \pmb{Q}$$

and the variation in the generalized velocities is similarly,

$$\delta \dot{\pmb{q}}_{i} = \epsilon \: \dot{\pmb{Q}}$$

The corresponding variation in the Lagrangian of the system can be computed using the above; the Euler-Lagrange equations; and the chain rule for derivatives,

$$
\begin{align}
\delta L & = \sum_i \frac{\partial L}{\partial \pmb{q}_{i}} \cdot \delta \pmb{q}_{i} + \sum_i \frac{\partial L}{\partial \dot{\pmb{q}}_{i}} \cdot \delta \dot{\pmb{q}}_{i} \\
 & = \sum_i \dot{\pmb{p}}_{i} \cdot \delta \pmb{q}_{i} + \sum_i \pmb{p}_{i} \cdot \delta \dot{\pmb{q}}_{i} \\
 & = \frac{d}{dt} \left( \sum_i \pmb{p}_{i} \cdot \delta \pmb{q}_{i} \right) \\
 & = \epsilon \: \frac{d}{dt} \left( \sum_i \pmb{p}_{i} \cdot \pmb{Q} \right)
\end{align}
$$

If the system is symmetric under the said transformation, its action remains invariant, which implies the variation of its Lagrangian must be zero. It thus follows,

$$\frac{d}{dt} \left( \sum_i \pmb{p}_{i} \cdot \pmb{Q} \right) = 0$$

We have thus found a time-invariant quantity associated with the said symmetry. This is Noether's theorem.

### Symmetry under linear translation

Suppose we translate a system linearly along a vector $$\pmb{X}$$. The infinitesimal transformation dictated by $$\pmb{X}$$ is then of the form,

$$\pmb{x}_{i} \to \pmb{x}_{i} + \epsilon \: \pmb{X}$$

Here, we have switched from q's to x's as the Cartesian coordinate system naturally incorporates linear translation via direct addition of coordinates. Taking $$\dot{\pmb{X}} = 0$$, Noether's theorem gives us,

$$\frac{d}{dt} \left( \sum_i \pmb{p}_{i} \right) \cdot \pmb{X} = 0$$

If the above is true for all $$\pmb{X} \in \mathbb{A}^3$$ where $$\mathbb{A}^3$$ is the affine 3-dimensional space, we must have,

$$\frac{d}{dt} \sum_i \pmb{p}_{i} = \pmb{0}$$

Thus, the total momentum, specifically _linear_ momentum, of a system is conserved when it is symmetric under infinitesimal linear translations.

### Form of linear momentum

For a system of particles symmetric under infinitesimal linear translations, its Lagrangian looks like so in Cartesian coordinates:

$$L \left( \left\{ \pmb{x}_{i} \right\}, \left\{ \dot{\pmb{x}}_{i} \right\} \right) = \sum_i \frac{1}{2} m_i \dot{\pmb{x}}_{i} \cdot \dot{\pmb{x}}_{i} - V \left( \left\{ \pmb{x}_{i} - \pmb{x}_j \right\} \right)$$

where $$\left\{ m_i \right\}$$ are the masses of the particles. The Lagrangian does not explicitly depend on time as we are considering closed systems only. Secondly, the potential energy is a function of only the relative positions of the particles with respect to each other, i.e. $$V : \mathbb{A}^4 \to \mathbb{R}$$ (thus, translating the system does not affect its potential energy).

Now, the conjugate momentum (here linear momentum) of each particle is,

$$
\begin{align}
\pmb{p}_{i} & = \frac{\partial L}{\partial \dot{\pmb{x}_{i}}} \\
 & = \frac{\partial}{\partial \dot{\pmb{x}_{i}}} \sum_j \frac{1}{2} m_j \dot{\pmb{x}}_j \cdot \dot{\pmb{x}}_j \\
 & = \sum_j \frac{1}{2} m_j \cdot 2 \: \delta_{ij} \: \dot{\pmb{x}}_j \\
 & = m_i \dot{\pmb{x}_i}
\end{align}
$$

Thus, the conservation of total linear momentum takes the form,

$$\frac{d}{dt} \sum_i \pmb{p}_i = \frac{d}{dt} \sum_i m_i \dot{\pmb{x}}_i = \pmb{0}$$

## Principle of relativity

### Definition

An important feature of classical mechanics is that changing the frame of reference leaves the form of physical laws invariant. When we switch from one inertial frame to another, a Galilean transformation is applied to all quantities pertaining to the former, to find their description in the latter.

Galilean transformations are specifically the transformations that leave intervals in space, and intervals in time invariant. In other words, they preserve the structure of $$\mathbb{A}^4$$.

The invariance of physical laws under such transformations is called the principle of relativity.

### Galilean group

There are 3 independent kinds of Galilean transformations.

Firstly, we have shift of the origin, so that,

$$\pmb{x}_i \to \pmb{x}_i + \pmb{s}, \: \dot{\pmb{s}} = \pmb{0}$$

Secondly, we have boosts:

$$\pmb{x}_i \to \pmb{x}_i + t \: \pmb{v}, \: \dot{\pmb{v}} = \pmb{0}$$

Lastly, we have rotations,

$$\pmb{x}_i \to \pmb{R} \: \pmb{x}_i, \: \dot{\pmb{R}} = \pmb{O}$$

where $$\pmb{R}$$ is a linear transformation and $$\pmb{O}$$ is the null operator. For lengths to be preserved, $$\pmb{R}$$ must be linear transformations satisfying the below:

$$
\begin{align}
\left( \pmb{R} \: \pmb{x}_i \right) \cdot \left( \pmb{R} \: \pmb{x}_i \right) & = \pmb{x}_i \cdot \pmb{x}_i \\
\pmb{x}_i^T \: \pmb{R}^T \pmb{R} \: \pmb{x}_i & = \pmb{x}_i^T \pmb{x}_i \\
\pmb{R}^T \pmb{R} & = \pmb{I} \\
\end{align}
$$

where $$T$$ denotes 'transpose' and $$\pmb{I}$$ is the identity operator.

Any Galilean transformation is a direct product of the above transformations. Together, they form the Galilean group $$G$$ over $$\mathbb{A}^4$$.

## Conservation of total mass

Consider an inertial frame in which the total linear momentum of a system is conserved:

$$\frac{d}{dt} \sum_i m_i \dot{\pmb{x}}_i = \pmb{0}$$

Now, we will apply a Galilean transformation on the original frame. By the principle of relativity, the conservation of total linear momentum stays true in the new frame. Note that the shift of origin does not manifest in the linear momenta as the latter are first derivatives of position. Using these facts and the conservation of total linear momentum in the initial frame,

$$
\begin{align}
\frac{d}{dt} \sum_i m_i \: \pmb{R} \left( \dot{\pmb{x}}_i + \pmb{v} \right) & = \pmb{0} \\
\frac{d}{dt} \sum_i m_i \pmb{R} \: \dot{\pmb{x}}_i + \frac{d}{dt} \sum_i m_i \: \pmb{R} \: \pmb{v} & = \pmb{0}
\end{align}
$$

As $$\pmb{R}$$ is a linear transformation and $$\dot{R} = \pmb{O}$$ and $$\dot{\pmb{v}} = \pmb{0}$$,

$$
\begin{align}
\pmb{R} \: \frac{d}{dt} \sum_i m_i \dot{\pmb{x}}_i + \frac{d}{dt} \left( \sum_i m_i \right) \pmb{R} \: \pmb{v} & = \pmb{0} \\
\pmb{R} \times \pmb{0} + \frac{d}{dt} \left( \sum_i m_i \right) \pmb{R} \: \pmb{v} & = \pmb{0} \\
\frac{d}{dt} \left( \sum_i m_i \right) \pmb{R} \: \pmb{v} & = \pmb{0} \\
\end{align}
$$

Since the boost $$\pmb{v}$$ and rotation $$\pmb{R}$$ are arbitrary,

$$\frac{d}{dt} \left( \sum_i m_i \right) = 0$$

Presto, the total mass of a closed system is conserved!

## Conclusion

From the above exercise, we see that the conservation of mass is not an isolated experimental fact in classical mechanics. In fact, it follows from two very fundamental ideas, the conservation of total linear momentum and principle of relativity. The former in turn emerges from the symmetry of systems under infinitesimal linear translations. The principle of relativity, on the other hand, is a consequence of the equations of motion being second-order, thereby making the dynamics of a system invariant no matter the reference frame.
