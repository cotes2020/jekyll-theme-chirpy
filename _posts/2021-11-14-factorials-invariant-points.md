---
title: "Factorials as Invariant Points"
description: "Revisiting the extended factorial"
categories: [analysis]
tags: [gamma function, Laplace transforms]
---

In '[Deriving the Gamma Function from Scratch]({% post_url 2021-10-14-gamma-function-from-scratch %})', we investigated the analytic origin of the extended factorial. Namely, it comes from the complex solution for the functional equation of the discrete factorial function.

Today, we will take a more general route and interpret factorials as invariant points under a certain transformation, in order to solve for those points and obtain an explicit expression.

## Extended factorial: recap

### Functional equation

The functional equation of the extended factorial is, as we know,

$$z! = z \left( z-1 \right)!$$

with the boundary condition $$z! = 1$$.

### Complex solution

As seen in the previously mentioned post, the complex solution of the above functional equation is given as,

$$z! = \mathcal{L} \left\{ t^z \right\} \left( 1 \right) = \int_0^\infty t^z e^{-t} dt$$

where $$\mathcal{L}$$ is the Laplace transform operator in the time domain $$t$$ defined as,

$$\mathcal{L} \left\{ u \left( z, t \right) \right\} \left( s \right) = \int_0^\infty u \left( z, t \right) e^{-st} dt$$

## Invariant points

### Transformations

Given a complex function $$f \left( z \right) : \mathbb{C} \mapsto \mathbb{C}$$, a _transformation_ is an operation $$\mathcal{T}$$ which can be completely expressed in terms of linear operators in the said function space, $$\widehat{\mathcal{L}} \in \mathbb{C}^\mathbb{C}$$. _Linear transformations_ are special transformations $$\widehat{\mathcal{T}}$$ which can be expressed as the sums and scalar products of some linear operators. It turns out from these two definitions that a transformation can always be expressed as a [not necessarily linear] function of linear transformations.

More precisely, a transformation is a multivector in an uncountably-infinite dimensional Hilbert space in which linear transformations form a basis. And a linear transformation is a vector in that space. But I get ahead of myself.

### Symmetries

A _symmetry_ of a transformation $$\mathcal{T}$$ is a quantity $$\sigma \left( z \right)$$ such that $$\mathcal{T} \left\{ \sigma \right\} \left( z \right) = \sigma \left( z \right)$$. The set of points $$\left( z, \sigma \left( z \right) \right)$$, or simply the entire function $$\sigma \left( z \right)$$, are said to be the _invariant points_ under the transformation $$\mathcal{T}$$. They can be used to construct the _symmetry operator_ of $$\mathcal{T}$$,

$$
\begin{align}
\mathcal{S} & = \sigma \left( \cdot \right) \\
\mathcal{T} \left\{ \mathcal{S} \right\} & = \mathcal{S}
\end{align}
$$

### Laplace transforms

The Laplace transform is a functional $$\mathcal{L} : \mathcal{T} \left\{ \mathcal{D} \right\} \mapsto \mathcal{V}$$ where: $$\mathcal{D}$$ is the differential operator; $$\mathcal{T} \left\{ \mathcal{D} \right\}$$ is any differential operator; and $$\mathcal{V}$$ is an operator which maps a function space to itself.

In other words, the Laplace transform is a map from differential operators to operators formed from ordinary functions of the form $$v \left( z \right)$$.

$$\mathcal{L} \left\{ \mathcal{D}^n \left\{ \cdot \right\} \right\} \left( s \right) = s^n \mathcal{L} \left\{ \cdot \right\} \left( s \right) - \sum_{k=0}^{n-1} s^{k} \mathcal{D}^{n-k} \left\{ \cdot \right\} \left( 0^- \right)$$

There are infinite allowed choices for a converging Laplace transform, but the one we used, as is common practice, turns out to be one of the most useful and simple ones. It also leaves the space of complex functions closed, allowing us to construct an inverse Laplace transform operator.

### Finding invariant points

Given a transformation $$\mathcal{T}$$, how can we find an $$\mathcal{S}$$ or equivalently, some $$\sigma \left( z \right)$$ ?

The trick is to convert $$\mathcal{T}$$ to a differential operator and use the Laplace transform. If $$\mathcal{T}$$ is already a differential operator, the job is easier. If not, some work is required, which is a subject of functional analysis.

Suppose a given differential operator $$\mathcal{T}$$ has some $$\mathcal{S}$$. We identify that $$\mathcal{S}$$ is formed from an ordinary function $$\sigma \left( z \right)$$. Invoking the idea of Laplace transforms, we can define an invertible map $$\mathcal{L} : \mathcal{T} \mapsto \mathcal{S}$$ (here, $$\mathcal{T}$$ and $$\mathcal{S}$$ represent the spaces of all possible $$\mathcal{T}$$'s and $$\mathcal{S}$$'s) so that,

$$\mathcal{S} = \mathcal{L} \left\{ \mathcal{T} \right\}$$

Thus, the symmetry operator of a differentiable transformation is a Laplace transform.

## Back to factorials

### As invariant points

Recall the functional equation for factorials,

$$z! = z \left( z-1 \right)!$$

Now consider a related transformation defined as:

$$\mathcal{T} \left\{ f \right\} \left( z \right) = z f \left( z-1 \right)$$

invariant points of the above transformation then obey:

$$\mathcal{T} \left\{ \sigma \right\} \left( z \right) = z \sigma \left( z-1 \right) = \sigma \left( z \right)$$

which is precisely the functional equation of factorials.

### Differential operator

Now, we must turn $$\mathcal{T}$$ to a differential operator so that we can apply $$\mathcal{L}$$ to it. To do so, we switch domain from $$z$$ to $$t$$ (the 'time domain') so that the below is satisfied for some function $$u \left( z, t \right)$$ :

$$\mathcal{T} \left\{ f \right\} \left( z \right) = \mathcal{D}_t \left\{ u \left( z, t \right) \right\}$$

For invariant points $$\sigma \left( z \right)$$,

$$\mathcal{T} \left\{ \sigma \right\} \left( z \right) = \mathcal{D}_t \left\{ u_\sigma \left( z, t \right) \right\} = \sigma \left( z \right)$$

### Laplace transform

The linear map $$\mathcal{L} : \mathcal{T} \mapsto \mathcal{S}$$ is a Laplace transform for some unknown value $$s$$,

$$
\begin{align}
\sigma \left( z \right) & = \mathcal{L} \left\{ \mathcal{T} \left\{ \sigma \right\} \left( z \right)  \right\} \left( s \right) \\
 & = \mathcal{L} \left\{ \mathcal{D}_t \left\{ u_\sigma \left( z, t \right) \right\} \right\} \left( s \right) \\
 & = s \: \mathcal{L} \left\{ u_\sigma \left( z, t \right) \right\} \left( s \right) - u_\sigma \left( z, 0^- \right) \\
\therefore \mathcal{L} \left\{ \mathcal{T} \left\{ \sigma \right\} \left( z \right)  \right\} \left( s \right) & = \frac{1}{s} \left( \sigma \left( z \right) + u_\sigma \left( z, 0^- \right) \right) : = \sigma \left( z \right) \\
\implies s & = 1 , \\
u_\sigma \left( z, 0^- \right) & = 0, \\
\sigma \left( z \right) & = \mathcal{L} \left\{ u_\sigma \left( z, t \right) \right\} \left( 1 \right)
\end{align}
$$

### Solution

From the above, we can find $$\sigma \left( z \right)$$ if we can find $$u_\sigma \left( z, t \right)$$. From the statements below,


$$
\begin{align}
\sigma \left( z \right) = \mathcal{T} \left\{ \sigma \right\} \left( z \right) & = z \sigma \left( z-1 \right) \\
\mathcal{T} \left\{ \sigma \right\} \left( z \right) & = \mathcal{D}_t \left\{ u_\sigma \left( z, t \right) \right\}
\end{align}
$$

$$\sigma \left( z \right) = \mathcal{L} \left\{ u_\sigma \left( z, t \right) \right\} \left( 1 \right)$$

we can ascertain,

$$
\begin{align}
\mathcal{D}_t \left\{ u_\sigma \left( z, t \right) \right\} & = z \: \mathcal{L} \left\{ u_\sigma \left( z-1, t \right) \right\} \left( 1 \right) \\
\mathcal{D}_t \left\{ \mathcal{L} \left\{ u_\sigma \left( z, t \right) \right\} \left( 1 \right) \right\} & = \mathcal{L} \left\{ z \: u_\sigma \left( z-1, t \right) \right\} \left( 1 \right) \\
\mathcal{L} \left\{ \mathcal{D}_t \left\{ u_\sigma \left( z, t \right) \right\} \right\} \left( 1 \right) & = \mathcal{L} \left\{ z \: u_\sigma \left( z-1, t \right) \right\} \left( 1 \right) \\
\therefore \mathcal{D}_t \left\{ u_\sigma \left( z, t \right) \right\} & = z \: u_\sigma \left( z-1, t \right)
\end{align}
$$

As in the previous post on the extended factorial, we realize that $$u_\sigma \left( z, t \right) = t^z$$ solves the above equation as $$\frac{\partial}{\partial t} t^z = z \: t^{z-1}$$. Furthermore, it is indeed true that $$u_\sigma \left( z, 0^- \right) = 0$$.

Thus, we have,


$$z! = \sigma \left( z \right) = \mathcal{L} \left\{ t^z \right\} \left( 1 \right) = \int_0^\infty t^z e^{-t} dt$$

## Conclusion

With respect to the previous post on the extended factorial ([Deriving the Gamma Function from Scratch]({% post_url 2021-10-14-gamma-function-from-scratch %})), our most recent derivation is essentially identical to the previous one. However, there's more than symbol manipulation going on here:

1. We're thinking of factorials as special points which remain invariant under the transformation introduced. This is a very general way of stating functional equations. In fact, by stating the problem in this manner, we don't always _have_ to know the exact functional equation, though we didn't have to encounter such a scenario here.

2. By virtue of being a symmetry operation, we know that the factorial must be a Laplace transform. This wasn't justified in the previous derivation.

3. The new procedure automatically let us infer that $$s=1$$ and $$u \left( z, 0^- \right) = 0$$, whereas before, we assumed those were true and matched the final result by brute force.

That said, two things still remain mysterious:

1. Why does the peculiar integral we use for the Laplace transform work as a linear map $$\mathcal{T} \mapsto \mathcal{S}$$ ?

2. Why is there a time domain $$t$$ which allows us to write $$\mathcal{T} = \mathcal{D}_t \left\{ u \left( \cdot, t \right) \right\}$$ such that $$\mathcal{T}$$ itself is independent of $$t$$ i.e. $$\mathcal{D}_t \mathcal{T} = \mathcal{D}_t^2 \left\{ u \left( \cdot, t \right) \right\} = 0$$ ?

The above questions have very interesting answers that require a juicy dive into complex analysis, measure theory and unexpectedly, algebraic geometry (hint: the last equation, $$\mathcal{D}_t^2 = 0$$ smells very much of exterior derivatives). I will try to cover those topics in future posts ;)

Thanks for reading.
