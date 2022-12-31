---
title: "A Brief Geometric Analysis of the Klein-Gordon Theory: Part 2 (Tensor Algebra)"
description: "An algebraic approach to studying the geometry of harmonic oscillators"
categories: [classical field theory]
tags: [harmonic oscillators, phase space, tensors]
---

In [A Brief Geometric Analysis of Harmonic Oscillators]({% post_url 2022-01-21-geometric-analysis-harmonic-oscillators %}), we examined how a convenient choice of phase space for the motion of a harmonic oscillator reveals its periodicity, without explicitly solving the equation of motion (Hooke's law). Here, we will apply this intuition to a scalar field behaving like a harmonic oscillator but at a field-theoretic level (the corresponding equation of motion is the Klein-Gordon equation).

Let us begin by reviewing the point-particle situation using more advanced mathematical tools than in the older post mentioned. This will allow us to easily upgrade our consideration to that of a scalar field and study it under a light both old and new.

## Revisiting pointlike harmonic oscillators

### Prior knowledge is power

Consider the position $$x \left( t \right)$$ of a point-particle obeying Hooke's law. This obeys the equation of motion,

$$\ddot{x} + \frac{k}{m} x = 0$$

where $$k$$ is the spring constant of the oscillator and $$m$$ is its mass. For reasons which will soon become clear, we can make the above equation look more symmetric by defining a parameter $$\omega = \sqrt{\frac{k}{m}}$$ so that we have,

$$\omega^{-1} \ddot{x} + \omega x = 0$$

Now, we invent a new variable (and consequently, and equation of motion) $$y = \omega^{-1} \dot{x}$$. We imagine this to be a new variable which together with position, completely characterizes the instantaneous state of the oscillator. Thus, the state space is a two-dimensional phase space which consists of all $$x$$ and $$y$$ coordinates, and is geometrically the linear span of the basis vectors along $$x$$ and $$y$$,

$$S = \text{span} \left( \frac{\partial}{\partial x}, \omega \frac{\partial}{\partial \dot{x}} \right)$$

Furthermore, this phase space is a Euclidean space i.e. equipped with a Euclidean notion of distance,

$$d \left( u^i, u^j \right) = \sqrt{ \left( u^i - u^j \right) \delta_{i j} \left( u^j - u^j \right) }$$

By the polarization identity for real inner products and the property that Euclidean spaces are closed under vector subtraction, we obtain the inner product for any two vectors in our phase space,

$$g \left( u^i, v^j \right) = u^i \delta_{i j} v^j$$

Armed with this knowledge, let us plug in the new variable $$y$$ into Hooke's law to find,

$$x = - \omega^{-1} \dot{y}$$

Let's take a look at the two equations of motion we conjured together,

$$
\begin{aligned}
y & = \omega^{-1} \dot{x} \\
x & = - \omega^{-1} \dot{y}
\end{aligned}
$$

The above are a pair of coupled ordinary differential equations! Geometrically, they're telling us that the velocity vector $$\dot{\pmb{u}} = \dot{x} \partial_x + \dot{y} \partial_y$$ is given in terms of the position vector $$\pmb{u}$$ (in phase space) as,

$$\dot{\pmb{u}} = \omega \left( \partial_x \otimes \text{d} y - \partial_y \otimes \text{d} x \right) \left( x \partial_x + y \partial_y \right)$$

Let us name the linear map acting on the position vector in the above equation as $$\pmb{\omega}$$ [^1]. Then, we have, succinctly, $$\dot{\pmb{u}} = \pmb{\omega} \left( \pmb{u} \right)$$. In the component form, $$\dot{u}^i = \omega^i_{\phantom{i} j} u^j$$.

[^1]: Interestingly, raising the first index of $$\omega^i_{\phantom{i} j}$$ gives the components of a symplectic form $$\omega_{i j}$$. I do not know if the choice of labelling is a coincidence but it is a striking one!

To make the above equations look a little less opaque, consider their matrix representation (the reason we will not use it is that it is only a _representation_ of the underlying tensorial objects),

$$
\begin{aligned}
\begin{pmatrix} \dot{x} \\ \dot{y} \end{pmatrix} & = \omega \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} \\
& = \omega \begin{pmatrix} y \\ - x \end{pmatrix}
\end{aligned}
$$

Now, the inner product of the position and velocity vectors turns out to be,

$$
\begin{aligned}
g \left( \dot{u}^i, u^j \right) & = \dot{u}^i \delta_{i j} u^j \\
& = \omega^i_{\phantom{i} k} u^k \delta_{i j} u^j \\
& = u^k \omega_{j k} u^j
\end{aligned}
$$

Notice that $$\pmb{\omega}$$ is antisymmetric i.e. $$\omega_{i j} = - \omega_{j i}$$. Thus,

$$
\begin{align}
g \left( \dot{u}^i, u^j \right) & = u^j \omega_{i j} u^i \\
& = - u^j \omega_{j i} u^i
\end{align}
$$

But by the commutativity of scalar multiplication applied to the terms in a tensorial expression, we have $$u^j \omega_{i j} u^i = u^i \omega_{i j} u^j$$. Furthermore, by the property that a tensorial expression remains invariant on interchanging dummy indices, we get from the two above results,

$$u^i \omega_{i j} u^j = - u^i \omega_{i j} u^j$$

which is only true if $$u^i \omega_{i j} u^j = 0$$. Thus, $$g \left( \dot{u}^i, u^j \right) = 0$$. In other words, the velocity vector is orthogonal to the position vector in the phase space we chose [^2].

[^2]: Had we chosen an arbitrary phase space represented by $$\omega^i_{\phantom{i} j}$$, we would need to choose a metric $$g_{i j}$$ such that $$g_{i j} \omega^i_{\phantom{i} k} = - g_{i k} \omega^i_{\phantom{i} j}$$ so that the above derivation would employ an antisymmetric tensor just as above. However, we would need to be careful that the geometry of the phase space is still Euclidean, which would require a rigorous usage of Riemannian geometry.

### Interpreting the above result

We intuitively know that circular motion is the only motion where the velocity vector is orthogonal to the position vector all along. Thus, the velocity vector field of the harmonic oscillator resembles concentric circles (as seen [here]({% post_url 2022-01-21-geometric-analysis-harmonic-oscillators %}#velocity-vector-field)).

Mathematics begins where intuition ends, so let us test the above notion precisely. We will do so by switching to polar coordinates $$\left( r, \theta \right)$$ in the phase space defined by,

$$
\begin{align}
r^2 & = g \left( \pmb{u}, \pmb{u} \right) \\
\pmb{u} & = r \left( \cos \left( \theta \right) \partial_x + \sin \left( \theta \right) \partial_y \right)
\end{align}
$$

Thus, we have,

$$
\begin{align}
\frac{d}{dt} r^2 & = \frac{d}{dt} g \left( \pmb{u}, \pmb{u} \right) \\
2 r \dot{r} & = 2 g \left( \dot{\pmb{u}}, u \right) \\
2 r \dot{r} & = 0 \\
\implies \dot{r} & = 0 & [r \neq 0]
\end{align}
$$

This tells us that for a given trajectory in phase space, $$r$$ is a constant. Consequently, every $$r$$ characterizes a unique trajectory. This symmetry essentially eliminates one equation of motion from the two we constructed, the other one being,

$$
\begin{align}
\dot{\pmb{u}} & = \frac{d}{dt} \left[ r \left( \cos \left( \theta \right) \partial_x + \sin \left( \theta \right) \partial_y \right) \right] \\
\pmb{\omega} \pmb{u} & = r \frac{d}{dt} \left( \cos \left( \theta \right) \partial_x + \sin \left( \theta \right) \partial_y \right) \\
 & = r \begin{pmatrix} \dot{\theta} \left( - \sin \left( \theta \right) \partial_x + \cos \left( \theta \right) \partial_y \right) \\ + \cos \left( \theta \right) \dot{\partial}_x + \sin \left( \theta \right) \dot{\partial}_y \end{pmatrix}
\end{align}
$$

At this stage, we have time derivatives of basis vectors in phase space. Let us show that they are the zero vector. Notice that had we tracked just the components in the matrix representation, we would be tempted to differentiate only the components, making the implicit assumption that the basis vectors are constant. This is yet another demonstration of the importance of tensor calculus, which keeps track of complete invariant objects, thereby never creating (or destroying) information.

$$
\begin{align}
\dot{\partial}_i & = \frac{d}{dt} \partial_i \\
 & = \dot{x}^j \frac{d}{dx^j} \partial_i \\
 & = \omega^j_{\phantom{j} k} x^k \frac{d}{dx^j} \partial_i \\
 & = \omega^j_{\phantom{j} k} x^k \Gamma^l_{\phantom{l} j i} \partial_l
\end{align}
$$

Since we are dealing with a Euclidean space, we can choose coordinates where the connection coefficients vanish everywhere. These are nothing but Riemann normal coordinates, where the metric resembles a Kronecker delta throughout in flat space. Since the phase space in consideration is Euclidean and equipped with such a metric, we can simply cancel out the connection coefficients. Using this result in the main derivation,

$$
\begin{align}
\pmb{\omega} \pmb{u} & = r \dot{\theta} \left( - \sin \left( \theta \right) \partial_x + \cos \left( \theta \right) \partial_y \right) \\
 & = \dot{\theta} \omega^{-1} \pmb{\omega} \left[ r \left( \cos \left( \theta \right) \partial_x + \sin \left( \theta \right) \partial_y \right) \right] \\
\pmb{\omega} \pmb{u} & = \dot{\theta} \omega^{-1} \pmb{\omega} \pmb{u} \\
\implies \dot{\theta} & = \omega
\end{align}
$$

On integrating the above equation of motion, we get the solution $$\theta \left( t-t_0 \right) = \theta \left( t_0 \right) + \omega \left( t - t_0 \right)$$. Since the physical significance of $$\theta$$ is in its manifestation in the expression for $$x$$ through trigonometric functions, we can identify all $$\theta$$'s separated by integral periods, $$\displaystyle{ \theta \sim \theta + \frac{2 \pi n}{\omega} }$$. In other words, $$x \left( t \right)$$ returns to the same state after integral multiples of the time period $$\displaystyle{ = \frac{2 \pi}{\omega} }$$.

To summarize, we showed that a pointlike harmonic oscillator (a system obeying Hooke's law) has a periodic evolution, simply by studying the symmetries of the equation of motion involved. We didn't need to solve the equation explicitly to predict the same, which is a win.

However, some people may find all the algebraic manipulation we performed unnecessarily complicated. So, let us take a step back and try to think why such a route can be important. While answering the question, we will gradually climb the ladder from classical mechanics to classical field theory and discover a particular field theory which set on to revolutionize modern physics.

## Why algebra?

Before we ask a 'why', we must ask a 'what'. What is algebra? Well, the term is used in many senses, but it usually refers to the symbolic treatment and manipulation of mathematical objects. It is more general than numeric manipulation in that algebraic methods can be reused across apparently different problems in mathematics.

In the section earlier, we used quite a bit of algebra, including tensor algebra (the manipulation of tensors). We deduced facts about harmonic oscillators — such as the orthogonality of their position and velocity vectors in phase space — using the structure and properties of the mathematical objects concerned ($$\pmb{u}$$, $$\dot{\pmb{u}}$$, $$\pmb{\omega}$$, $$\pmb{g}$$, etc.). This enabled us to conclude that a harmonic oscillator evolves periodically, without completely solving its equation of motion (rather, we broke it down into two, eliminated one by symmetry and solved the remaining first-order ordinary differential equation).

Such an approach may be particularly handy when it is difficult to solve the equation(s) of motion of a more complicated harmonic oscillator or system of harmonic oscillators. There are many examples of these: Klein-Gordon fields, perturbations of the metric tensor in linearized gravity, quantum harmonic oscillators, quantum fields, etc. Typically, these systems exhibit symmetry in the form of symmetry groups with Lie algebras. These greatly simplify many analyses which would otherwise require mathematical machinery such as Fourier transforms.

With this in mind, we will apply the approach developed above, to study Klein-Gordon fields. They are the simplest field-theoretic analogue of classical harmonic oscillators. In fact, these fields _are_ continuous systems of infinite coupled harmonic oscillators, but we will not need to treat them such in the following derivation as we will deduce their periodic property solely by algebraically manipulating their equation of motion.

## Klein-Gordon Fields

### Equations of motion

The equation of motion for a Klein-Gordon field is derived from the Klein-Gordon Langrangian density,

$$\mathcal{L} = \frac{1}{2} \partial_\mu \phi \partial^\mu \phi - \frac{1}{2} m^2 \phi^2$$

The motivation for this Lagrangian can be found in [this]({% post_url 2021-12-01-scalar-field-lagrangian-symmetry-considerations %}) post. Plugging the Lagrangian into the field-theoretic Euler-Lagrange equations, we get the equation of motion,

$$\partial_\mu \partial^\mu \phi + m^2 \phi = 0$$

This is the Klein-Gordon equation. Our phase space analysis of this equation will be parallel to the one for pointlike harmonic oscillators. So, let us begin by making the Klein-Gordon equation look more symmetric:

$$m^{-1} \partial_\mu \partial^\mu \phi + m \phi = 0$$

Since $$m$$ is constant and assumed to be non-zero, we can freely move it and its inverse across derivative operators. Now, let us define a new field which will be identified as a new variable in the phase space of the Klein-Gordon field,

$$\psi^\mu = m^{-1} \partial^\mu \phi$$

This is a first-order equation of motion. Plugging it into the Klein-Gordon equation, we get another such equation of motion. For brevity, let us work in inertial coordinates.

$$\phi = - m^{-1} \partial_\mu \psi^\mu$$

Let us behold the two equations of motion together,

$$
\begin{align}
\psi^\mu & = m^{-1} \partial^\mu \phi \\
\phi & = - m^{-1} \partial_\mu \psi^\mu
\end{align}
$$

Phase space treats its independent variables on an equal footing. Therefore, we imagine that in the above equations, $$u^A = \left( \psi^\mu, \phi \right)$$ are components of a higher-dimensional vector field,

$$
\begin{align}
\pmb{u} & = u^A \partial_A \\
 & =  \psi^\mu \frac{\partial}{\partial \psi^\mu} + \phi \frac{\partial}{\partial \phi} \in \mathcal{S}
\end{align}
$$

where $$\mathcal{S}$$ is an appropriate phase space. However, the problem with the equations of motion is that $$\psi^\mu$$ and $$\phi$$ are not really on the same footing — the former is a 4-vector field while the latter is a scalar field. To make the equations of motion even more symmetric, we might want a 4-vector field constructed from $$\phi$$ to appear in the second equation of motion ($$\phi = - m^{-1} \partial_\mu \psi^\mu$$). One way to do this is to find an equation of motion involving $$\partial_\nu \psi^\mu$$ on the right hand side, instead of the divergence $$\partial_\mu \psi^\mu$$. Consider the following:

$$
\begin{aligned}
\partial_\mu \psi^\mu & = - m \phi \\
\delta^\nu_{\phantom{\nu} \mu} \partial_\nu \psi^\mu & = - \frac{1}{4} \delta^\nu_{\phantom{\nu} \mu} \delta^\mu_{\phantom{\mu} \nu} m \phi \\
\partial_\nu \psi^\mu & = - \frac{1}{4} \delta^\mu_{\phantom{\mu} \nu} m \phi
\end{aligned}
$$

Rigorously proving the above is necessary but would make this post too lengthy. The key take is that tensor algebra severely restricts the nature of $$\partial_\nu \psi^\mu$$ given that of $$\partial_\mu \psi^\mu$$. We must have $$\text{tr} \left( \partial_\nu \psi^\mu \right) = \partial_\mu \psi^\mu$$ and any tensor on both sides of this equation can be factored out by multilinearity. Ultimately, we have two equations of motion which are symmetric up to some scaling factors:

$$
\begin{align}
\partial_\rho \psi^\mu & = - \frac{1}{4} \delta^\mu_{\phantom{\mu} \rho} m \phi \\
\partial_\rho \phi & = \eta_{\nu \rho} m \psi^\nu
\end{align}
$$

Now, we combine the two above equations into a single geometric one,

$$\partial_\rho \left( \psi^\mu \frac{\partial}{\partial \psi^\mu} + \phi \frac{\partial}{\partial \phi} \right) = m \left( \frac{\partial}{\partial \psi^\nu} \otimes \text{d} \phi - \frac{\partial}{\partial \phi} \otimes \text{d} \psi^\nu \right) \left( \psi^\mu \frac{\partial}{\partial \psi^\mu} + \phi \frac{\partial}{\partial \phi} \right)$$

In the matrix form,

$$
\begin{align}
\partial_\rho \begin{pmatrix} \psi^\mu \\ \phi \end{pmatrix} & = m \begin{pmatrix}  \end{pmatrix} \begin{pmatrix} \psi^\mu \\ \phi \end{pmatrix} \\
 & = m \begin{pmatrix}  \end{pmatrix}
\end{align}
$$
