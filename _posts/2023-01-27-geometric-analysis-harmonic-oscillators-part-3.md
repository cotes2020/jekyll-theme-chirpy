---
title: "A Brief Geometric Analysis of Harmonic Oscillators: Part 3 (Matrix Exponentials)"
description: "An algebraic addendum to the geometric analysis of harmonic oscillators we developed previously"
categories: [classical mechanics]
tags: [harmonic oscillators, phase space, translation operator, Euler formula, tensors]
---

In [A Brief Geometric Analysis of Harmonic Oscillators: Part 2 (Tensor Algebra)]({% post_url 2023-01-01-geometric-analysis-harmonic-oscillators-part-2 %}), we generalized the methods used in [A Brief Geometric Analysis of Harmonic Oscillators]({% post_url 2022-01-21-geometric-analysis-harmonic-oscillators %}) to analyze the behaviour of harmonic oscillators in phase space.

We began with Hooke's law, $$\omega^{-1} \ddot{x} + \omega x = 0$$ and explored the Hamiltonian flow described by it in the phase space $$S$$ represented by $$\left( x, y \right)$$ where $$y = \omega^{-1} \dot{x}$$. We then showed using geometric notions that this flow resembles concentric circles in phase space, with the phase $$\theta \left( t \right)$$ being proportional to $$t$$ by a factor of $$\omega$$, i.e. $$\theta \left( t \right) = \omega t$$.

In this post, we will explicitly find the solution for the position vector $$\left( x, y \right)$$ in phase space and derive the phase as a consequence of the same.

## Hamilton's equations

As found in the previously mentioned posts, in phase space, the equation of motion for a harmonic oscillator is ingrained in Hamilton's equations of motion,

$$\dot{\pmb{u}} = \pmb{\omega} \left( \pmb{u} \right)$$

where $$\pmb{\omega} = \omega \left( \partial_x \otimes \text{d} y - \partial_y \otimes \text{d} x \right)$$ and $$\pmb{u} = x \partial_x + y \partial_y = x \partial_x + \dot{x} \partial_{\dot{x}}$$.

In the matrix representation,

$$\begin{pmatrix} \dot{x} \\ \dot{y} \end{pmatrix} = \omega \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

Now, we will try to solve for $$\pmb{u} \left( t \right)$$.

## Time translation operator

Suppose we want to construct a time translation operator $$\mathcal{U}$$ such that,

$$\mathcal{U}^t \pmb{u} \left( 0 \right) = \pmb{u} \left( t \right)$$

The exponentiation of the time translation operator has to do with the way its compositions (products) add up in terms of their arguments,

$$\prod_{k} \mathcal{U}^{t_k} = \mathcal{U}^{\sum_k t_k}$$

With this in mind, we have, using a series expansion of the time translation operator,

$$
\begin{align}
\mathcal{U}^t &= \sum_{n=0}^\infty \frac{1}{n!} t^n \frac{d^n}{dt^n} \mathcal{U}^t \Bigg|_{t=0} \\
 & = \exp \left( t \frac{d}{dt} \right) \mathcal{U}^0 \\
 & = \exp \left( t \frac{d}{dt} \right) \mathcal{I} \\
 & = \exp \left( t \frac{d}{dt} \right)
\end{align}
$$

where $$\mathcal{I}$$ is the identity map $$\text{id}_S$$.

Now, we perform a little trick. Since we imagine $$\mathcal{U}$$ to act on $$\pmb{u} \left( t \right)$$, the linear operator $$\frac{d}{dt}$$ is the linear map $$\widehat{L} : \pmb{u} \to \dot{\pmb{u}}$$. From Hamilton's equations of motion, we know that this map is nothing but $$\pmb{\omega}$$. Let us write $$\pmb{\omega} = \omega \pmb{J}$$ where $$\pmb{J} = \partial_x \otimes \text{d} y - \partial_y \otimes \text{d} x$$. Then, we have from the above equation,

$$\mathcal{U}^t = \exp \left( \omega t \pmb{J} \right)$$

Therefore, from the definition of the time translation operator, we have,

$$\pmb{u} \left( t \right) = \exp \left( \omega t \pmb{J} \right) \: \pmb{u} \left( 0 \right)$$

where,

$$\exp \left( \omega t \pmb{J} \right) = \sum_{n=0}^\infty \frac{1}{n!} \omega^n t^n \pmb{J}^n = \mathcal{U}^t$$

In principle, this is indeed an explicit solution for $$\pmb{u} \left( t \right)$$ in terms of initial conditions $$\pmb{u} \left( 0 \right)$$. However, to extract the notion of periodicity from the exponential, we will need to derive Euler's formula for matrix exponentials. [^1]

[^1]: Pedantically, matrices are only representations of underlying tensors. What we will really be doing is deriving Euler's formula for antisymmetric tensors.

## Euler's formula for the time translation operator

To make sense of the solution we derived above for the position vector in phase space, we will do the following:

$$\mathcal{U}^t = \exp \left( \omega t \pmb{J} \right)$$

Some useful facts to simplify the above expression are,

$$
\begin{align}
\pmb{J}^0 & = \pmb{\delta} \\
\pmb{J}^1 & = \pmb{J} \\
\pmb{J}^2 & = \left( \partial_x \otimes \text{d} y - \partial_y \otimes \text{d} x \right) \left( \partial_x \otimes \text{d} y - \partial_y \otimes \text{d} x \right) \\
 & = - \partial_x \otimes \text{d} x - \partial_y \otimes \text{d} y \\
 & = - \delta^i_{\phantom{i} j} \: \partial_i \otimes \text{d} x^j \\
 & = - \pmb{\delta} \\
\pmb{J}^3 & = \pmb{J}^2 \pmb{J} \\
 & = - \pmb{\delta} \pmb{J} \\
 & = - \pmb{J} \\
\pmb{J}^4 & = \pmb{J}^3 \pmb{J} \\
 & = - \pmb{J} \pmb{J} \\
 & = - \pmb{J}^2 \\
 & = \pmb{\delta} \\
 & \vdots \\
\pmb{J}^n & = \pmb{J}^{n+4k} \: \forall \: n, k \in \mathbb{Z}
\end{align}
$$

The above can be summarized in the following commutative diagram,

<iframe class="quiver-embed" src="https://q.uiver.app/?q=WzAsNCxbMCwwLCJcXHBtYntcXGRlbHRhfSJdLFsxLDAsIlxccG1ie0p9Il0sWzEsMSwiLSBcXHBtYntcXGRlbHRhfSJdLFswLDEsIi0gXFxwbWJ7Sn0iXSxbMCwxLCJcXHBtYntKfSJdLFsxLDIsIlxccG1ie0p9Il0sWzIsMywiXFxwbWJ7Sn0iXSxbMywwLCJcXHBtYntKfSJdXQ==&embed" width="304" height="304" style="border-radius: 8px; border: none;"></iframe>

Armed with this knowledge,let us collect even and odd terms in the power series for $$\mathcal{U}^t$$,

$$
\begin{align}
\mathcal{U}^t & = \sum_{n=0}^\infty \frac{1}{n!} \omega^n t^n \pmb{J}^n \\
 & = \sum_{n=0}^\infty \frac{1}{\left( 2n \right)!} \omega^{2n} t^{2n} \pmb{J}^{2n} + \sum_{n=0}^\infty \frac{1}{\left( 2n+1 \right)!} \omega^{2n+1} t^{2n+1} \pmb{J}^{2n+1} \\
 & = \sum_{n=0}^\infty \frac{\left( -1 \right)^n}{\left(  2n \right)!} \omega^{2n} t^{2n} \pmb{\delta}^{2n} + \sum_{n=0}^\infty \frac{\left( -1 \right)^n}{\left( 2n+1 \right)!} \omega^{2n+1} t^{2n+1} \pmb{J} \\
 & = \cos \left( \omega t \right) \pmb{\delta} + \sin \left( \omega t \right) \pmb{J}
\end{align}
$$

Thus,

$$
\begin{align}
\pmb{u} \left( t \right) & = \mathcal{U}^t \pmb{u} \left( 0 \right) \\
 & = \left[ \cos \left( \omega t \right) \pmb{\delta} + \sin \left( \omega t \right) \pmb{J} \right] \pmb{u} \left( 0 \right) \\
 & = \cos \left( \omega t \right) \pmb{u} \left( 0 \right) + \sin \left( \omega t \right) \pmb{J} \pmb{u} \left( 0 \right) \\
 & = \cos \left( \omega t \right) \left( x \left( 0 \right) \partial_x + y \left( 0 \right) \partial_y \right) + \sin \left( \omega t \right) \left( \partial_x \otimes \text{d} y - \partial_y \otimes \text{d} x \right) \left( x \left( 0 \right) \partial_x + y \left( 0 \right) \partial_y \right) \\
 & = \cos \left( \omega t \right) \left( x \left( 0 \right) \partial_x + y \left( 0 \right) \partial_y \right) + \sin \left( \omega t \right) \left( y \left( 0 \right) \partial_x - x \left( 0 \right) \partial_y \right) \\
 & = \left( \cos \left( \omega t \right) x \left( 0 \right) + \sin \left( \omega t \right) y \left( 0 \right) \right) \partial_x + \left( \cos \left( \omega t \right) y \left( 0 \right) - \sin \left( \omega t \right) x \left( 0 \right) \right) \partial_y
\end{align}
$$

Once again, in the matrix representation,

$$
\begin{align}
\mathcal{U}^t & = \cos \left( \omega t \right) \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} + \sin \left( \omega t \right) \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} \\
 & = \begin{pmatrix} \cos \left( \omega t \right) & \sin \left( \omega t \right) \\ - \sin \left( \omega t \right) & \cos \left( \omega t \right) \end{pmatrix} \\
 \pmb{u} \left( t \right) & = \mathcal{U}^t \pmb{u} \left( 0 \right) \\
 & = \begin{pmatrix} \cos \left( \omega t \right) & \sin \left( \omega t \right) \\ - \sin \left( \omega t \right) & \cos \left( \omega t \right) \end{pmatrix} \begin{pmatrix} x \left( 0 \right) \\ y \left( 0 \right) \end{pmatrix} \\
 & = \begin{pmatrix} \cos \left( \omega t \right) x \left( 0 \right) + \sin \left( \omega t \right) y \left( 0 \right) \\ - \sin \left( \omega t \right) x \left( 0 \right) + \cos \left( \omega t \right) y \left( 0 \right) \end{pmatrix}
\end{align}
$$

From this, we can concretely conclude that $$\pmb{u} \left( t \right)$$ is a periodic function with a period of $$T = \frac{2 \pi}{\omega}$$. Furthermore, the argument $$\omega t$$ assumes the role of an angle parameter $$\theta \left( t \right)$$ in characterizing $$\pmb{u} \left( t \right)$$.

## Conclusion

Through a series of posts, we have managed to demonstrate the periodic evolution of harmonic oscillators, following as a geometric consequence of the structure of their equation of motion. The basis for this argument was the Hamiltonian flow of a harmonic oscillator in phase space, which was then explored using a range of mathematical toolkits, namely: vector calculus, tensor algebra and finally, matrix exponentials.

This analysis hopefully lays the groundwork for building theories of more complicated systems resembling harmonic oscillators, such as Klein-Gordon fields and quantum harmonic oscillators — topics we will want to explore on this blog soon.

As always, thanks so much for reading and feel free to leave suggestions or any other comments below :)