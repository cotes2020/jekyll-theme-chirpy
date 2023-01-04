---
title: "A Brief Geometric Analysis of Harmonic Oscillators: Part 2 (Tensor Algebra)"
description: "An algebraic approach to studying the geometry of harmonic oscillators"
categories: [classical mechanics]
tags: [harmonic oscillators, phase space, tensors]
---

In [A Brief Geometric Analysis of Harmonic Oscillators]({% post_url 2022-01-21-geometric-analysis-harmonic-oscillators %}), we examined how a convenient choice of phase space for the motion of a harmonic oscillator reveals its periodicity, without explicitly solving the equation of motion (Hooke's law). Here, we will apply the same intuition but in the language of tensor algebra, introducing greater rigour as well as economy of notation.

## Prior knowledge is power

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

## Interpreting the above result

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

On integrating the above equation of motion, we get the solution $$\theta \left( t-t_0 \right) = \theta \left( t_0 \right) + \omega \left( t - t_0 \right)$$. Since the physical significance of $$\theta$$ is in its manifestation in the expression for $$x$$ through trigonometric functions, we can identify all $$\theta$$'s separated by integral periods, $$\displaystyle{ \theta \sim \theta + \frac{2 \pi n}{\omega} }$$. In other words, $$x \left( t \right)$$ returns to the same state after integral multiples of the time period $$\displaystyle{ T = \frac{2 \pi}{\omega} }$$.

To summarize, we showed that a pointlike harmonic oscillator (a system obeying Hooke's law) has a periodic evolution, simply by studying the symmetries of the equation of motion involved. We didn't need to solve the equation explicitly to predict the same, which is a win.

However, one may find all the algebraic manipulation we performed unnecessarily complicated. So, let us take a step back and try to think why such a route can be important.

## Why algebra?

Before we ask 'why', we must ask 'what'. What is algebra? Well, the term is used in many senses, but it usually refers to a structure where new objects can be created from given ones, using some kind of composition.

In the section earlier, we used quite a bit of algebra, including tensor algebra (the manipulation of tensors). We deduced facts about harmonic oscillators — such as the orthogonality of their position and velocity vectors in phase space — by algebraically amalgamating mathematical objects such as $$\pmb{u}$$, $$\dot{\pmb{u}}$$, $$\pmb{\omega}$$, $$\pmb{g}$$, etc. This enabled us to conclude that a harmonic oscillator evolves periodically, without completely solving its equation of motion (rather, we broke it down into two, eliminated one by symmetry and solved the remaining first-order ordinary differential equation).

This algebraic approach gets especially important in quantum mechanics and field theory, where much of physics can be gleaned from algebra. There is no clear reason why — it simply seems to be the way the universe works. But perhaps, there is more going on here. Perhaps, algebra is so fundamental to physics because like physics, it is about conjuring richer objects from simpler objects. Moreover, algebra being so abstract seems to correspond to physics working well with abstractions, i.e. the key details of systems under concerned situations.

However, to better understand this deep nature of physics, we must work out way through the mathematics. I hope to write future posts exploring the relationship between the theme of this post, and classical and quantum field theory. For instance, from the methods discussed above, one may begin to suspect that the fact that a Klein-Gordon field is a system of infinite harmonic oscillators may be algebraically demonstrated from the formal analogy between the Klein-Gordon equation and Hooke's law.

That said, it's time to wrap up our first post of the year! Thanks for reading and thanks for wanting to know more today than you did yesterday. As said in Matthew 7:7,

> Ask, and it shall be given you; seek, and ye shall find; knock, and it shall be opened unto you.