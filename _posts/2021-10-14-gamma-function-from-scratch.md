---
title: "Deriving the Gamma Function from Scratch"
description: "Finding the famous integral representation of the gamma function using some analysis"
categories: [analysis]
tags: [gamma function, Laplace transforms]
---

## What is the gamma function about?

About 300 years ago, the influential mathematician Leonhard Euler solved the problem of extending the factorial function to non-integers. He originally found an [infinite product representation](https://en.wikipedia.org/wiki/Gamma_function#18th_century:_Euler_and_Stirling), which he soon expressed in the integral form,

$$\displaystyle{ z! = \int_0^1 \left( - \ln u \right)^z du }$$

By the substitution $$t = - \ln u$$, one obtains the more common representation of the extended factorial:

$$\displaystyle{ z! = \int_0^\infty t^z e^{-t} dt }$$

Note that we have used the symbol $$z$$ to denote complex arguments. The above function is defined for all complex numbers, except for negative reals.

In complex analysis, one uses the Gamma function more often (which is actually an arbitrary tradition), which in terms of the extended factorial is defined as,

$$\displaystyle{ \Gamma \left( z \right) = \left( z-1 \right)! = \int_0^\infty t^{z-1} e^{-t} dt }$$

This celebrated function appears in a lot of places, from complex analysis to quantum mechanics. Hence, it is important to understand how to derive it intuitively. We will derive the extended factorial first, and plug in $$z-1$$ to get the gamma function. But before that, let us look at some of the properties of the extended factorial, which make it a suitable candidate for continuing the factorial to non-integer arguments.

## Properties

The first important property of the extended factorial is,

$$z! = z \left( z-1 \right) ! \tag{1}$$

This can be seen by using the explicit integral representation of the extended factorial and using integration by parts,

$$\begin{align}
z! & = \int_0^\infty t^z e^{-t} dt \\
 & = t^z \int_0^\infty e^{-t} dt - \int_0^\infty \frac{d}{dt} t^z \left( \int e^{-t} dt \right) dt \\
 & = \left[ - t^z e^{-t} \right]_0^\infty - \int_0^\infty zt^{z-1} \left( -e^{-t} \right) dt \\
 & = z \int_0^\infty t^{z-1} e^{-t} dt \\
 & = z \left( z-1 \right) !
\end{align}$$

<a name="boundary_condition"></a>

The second important property is that $$0! = 1$$. This, again, can be derived from the integral representation,

$$
\begin{align}
0! & = \int_0^\infty t^0 e^{-t} dt \\
 & = \int_0^\infty e^{-t} dt \\
 & = \left[ -e^{-t} \right]_0^\infty \\
 & = 1
\end{align}
$$

These two properties together allow us to retrieve the regular factorial function defined over the whole numbers (denoted by $$n$$),

$$\begin{align}
n! & = n \left( n-1 \right) ! \\
 & = n \left( n-1 \right) \left( n-2 \right) ! \\
 & \vdots \\
 & = n \left( n-1 \right) \left( n-2 \right) \dots 2 \times 1 \times 0! \\
 & = n \left( n-1 \right) \left( n-2 \right) \dots 2 \times 1
\end{align}$$

## Derivation

### Strategy

After reading till here, you may think that "_I get it, the extended factorial has nice properties, but where does it come from?_", and if you do, you're on the right track.

The extended factorial first came from Euler's mind, so one could investigate how he conceived of it. However, that route isn't straightforward. It requires a deep understanding of analysis. After all, Euler got his neat equations thanks to spending a _lot_ of time with such problems.

Instead, we will begin with the [properties of the factorial](#properties), which are obvious even from whole number arguments.

Consider equation $$\left( 1 \right)$$, which is basically a functional equation for the factorial function. If we can solve for the factorial function so that the functional equation holds even for non-integer arguments, and use the boundary condition $$0! = 1$$, we have found the extended factorial.

### Interlude: the Laplace transform

The Laplace transform is an essential tool in the mathematician's and physicist's toolbox. It is used to convert differential equations to algebraic equations. This allows us to solve the algebraic equation, 'inverse Laplace transform' it and find exact solutions for the original differential equation.

The Laplace transform $$F \left( s \right)$$ of a function $$f \left( t \right)$$ is defined as,

$$\displaystyle{ F \left( s \right) = \mathcal{L} \left\{ f \left( t \right) \right\} \left( s \right) = \int_0^\infty f \left( t \right) e^{-st} dt }$$

Thus, the Laplace transform operator $$\mathcal{L}$$ takes in a function $$f \left( t \right)$$ in the time, or $$t$$ domain and gives a new function $$F \left( s \right)$$ in the $$s$$ domain.

Laplace transforms have some interesting [properties](https://en.wikipedia.org/wiki/Laplace_transform#Properties_and_theorems), which will deem to be useful for this derivation. Though I will derive the properties used in this post, it could be helpful to run through them yourself in the previous link.

Alert readers may have realized by this point that the extended factorial is itself a Laplace transform,

$$z! = \mathcal{L} \left\{ t^z \right\} \left( 1 \right)$$

Here, $$t^z$$ is not just a function of $$t$$, but also of the variable $$z$$. Similarly, $$z!$$ is a function of $$z$$, but not an explicit function of $$s$$. Hence, for our purposes, we will generalize the operation of the Laplace transform as,

$$\displaystyle{ U_s \left( z, t \right) = \mathcal{L} \left\{ u \left( z, t \right) \right\} \left( s \right) = \int_0^\infty u \left( z, t \right) e^{-st} dt }$$

We have used the symbol $$u$$ for the new argument function of the Laplace transform operator, to remind us that it is a function of both $$z$$ and $$t$$.

### Getting started

A useful property of the Laplace transform is the following,

$$\displaystyle{ \mathcal{L} \left\{ u \left( z, t \right) \right\} \left( s \right) = \frac{1}{s} \left( \mathcal{L} \left\{ \frac{\partial u \left( z, t \right)}{\partial t} \right\} \left( s \right) + \lim_{t \to 0^-} u \left( z, t \right) \right) \tag{2} }$$

Where $$\frac{\partial}{\partial t}$$ denotes differentiation with respect to the variable $$t$$, while keeping other variables constant. As usual, the above can be derived by plugging in the explicit expression for the left hand side and transforming it to the right hand side,

$$\begin{align}
\mathcal{L} \left\{ u \left( z, t \right) \right\} \left( s \right) & = \int_0^\infty u \left( z, t \right) e^{-st} dt \\
 & = u \left( z, t \right) \int_0^\infty e^{-st} dt - \int_0^\infty \frac{\partial u \left( z, t \right)}{\partial t} \left( \int e^{-st} dt \right) dt \\
 & = \left[- \frac{1}{s} u \left( z, t \right) e^{-st} \right]_0^\infty - \int_0^\infty \frac{\partial u \left( z, t \right)}{\partial t} \left( - \frac{1}{s}  e^{-st} \right) dt \\
 & = \frac{1}{s} u \left( z, 0^- \right) + \frac{1}{s} \int_0^\infty \frac{\partial u \left( z, t \right)}{\partial t} e^{-st} dt \\
 & = \frac{1}{s} \left( \mathcal{L} \left\{ \frac{\partial u \left( z, t \right)}{\partial t} \right\} \left( s \right) + u \left( z, 0^- \right) \right)
\end{align}$$

### Some brute force

Let us compare the equations $$\left( 1 \right)$$ and $$\left( 2 \right)$$,

$$z! = z \left( z-1 \right) ! + 0 \tag{1}$$

$$\displaystyle{ \mathcal{L} \left\{ u \left( z, t \right) \right\} \left( s \right) = \frac{1}{s} \mathcal{L} \left\{ \frac{\partial u \left( z, t \right)}{\partial t} \right\} \left( s \right) + \frac{1}{s} u \left( z, 0^- \right) \tag{2} }$$

Apparently, the two above equations look very different from each other. But is it possible to establish an exact correspondance between them? As it turns out, yes. Let us make the assumption that $$z!$$ has an inverse Laplace transform. Then, it can be expressed as a Laplace transform of some function, in the following manner,

$$z! = \frac{1}{s} \mathcal{L} \left\{ u \left( z, t \right) \right\} \left( s \right)$$

Those are the first terms in the two equations we are comparing. Further, let,

$$z \left( z-1 \right) ! = \frac{1}{s} \mathcal{L} \left\{ \frac{\partial u \left( z, t \right)}{\partial t} \right\} \left( s \right)$$

$$\frac{1}{s} u \left( z, 0^- \right) = 0$$

From the first two relations,

$$z \left( z-1 \right)! = z \mathcal{L} \left\{ u \left( z-1, t \right) \right\} \left( s \right) = \frac{1}{s} \mathcal{L} \left\{ \frac{\partial u \left( z, t \right)}{\partial t} \right\} \left( s \right)$$

Or explicitly,

$$\displaystyle{ \frac{1}{s} \int_0^\infty \frac{\partial u \left( z, t \right)}{\partial t} e^{-st} dt = z \int_0^\infty u \left( z-1, t \right) e^{-st} dt }$$

We need to solve for both $$u \left( z, t \right)$$ and $$s$$. Firstly, we can pull in various free variables in both sides of the equation, into their repsective integrals, since they are independent of the variable over which we are integrating,

$$\displaystyle{ \int_0^\infty \frac{1}{s} \frac{\partial u \left( z, t \right)}{\partial t} e^{-st} dt = \int_0^\infty z u \left( z-1, t \right) e^{-st} dt }$$

From the above, it is obvious that,

$$\frac{1}{s} \frac{\partial u \left( z, t \right)}{\partial t} = z u \left( z-1, t \right)$$

There is no general way to solve the above functional equation. However, if you have kept some high school differentiation formulas up your sleeve, you will remember that,

$$\frac{\partial}{\partial t} t^z = z t^{z-1}$$

We can put this in exactly the form we want, by defining $$u \left( z, t \right) = t^z$$,

$$\frac{1}{1} \frac{\partial u \left( z, t \right)}{\partial t} = z u \left( z-1, t \right)$$

Thus, $$s=1$$. Plugging all of this into the original statement $$z! = \frac{1}{s} \mathcal{L} \left\{ u \left( z, t \right) \right\} \left( s \right)$$,

$$z! = \mathcal{L} \left\{ t^z \right\} \left( 1 \right) = \int_0^\infty t^z e^{-t} dt$$

Furthermore, $$\Gamma \left( z \right)$$ can be obtained by plugging in $$z-1$$. Voila! Or, _not_ so voila, as we have derived the extended integral from simple assumptions!

### May the force be with you

But we are not yet done. Remember the second important property of the extended factorial? It must obey the boundary condition $$0! =1$$ in order to reduce to the regular factorial for integer arguments. [As shown earlier](#boundary_condition),

$$0! = \mathcal{L} \left\{ t^0 \right\} \left( 1 \right) = \mathcal{L} \left\{ 1 \right\} \left( 1 \right) = 1$$

Lastly, since we had assumed that $$u \left( z, 0^- \right) = 0$$, we must verify it now,

$$\displaystyle{ u \left( z, 0^- \right) = \lim_{t \to 0^-} t^z = 0 }$$

At long last, we can confidently say that we have found the extended factorial function defined over complex numbers.

## Conclusion

This brings us to the end of our little endeavour. Clearly, it was a not an easy one, but in Euler's own words,

> Logic is the foundation of the certainty of all the knowledge we acquire.
