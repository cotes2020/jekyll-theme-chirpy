---
title: "The Discrete Antiderivative Operator"
description: "Finding an expression for the discrete antiderivative of a function"
categories: [discrete mathematics]
tags: [discrete calculus]
redirect_from: discrete-antiderivative
---

## Discrete derivative operator

Let a function $$f: \mathbb{R} \mapsto \mathbb{R}$$. On discretizing the domain of $$f$$ into quanta $$h$$ centred at $$a_0$$, $$f: \mathbb{A} \mapsto \mathbb{A}$$ where $$\mathbb{A} = \left\{ kh+a_0 : k \in \mathbb{Z}, a_0 \in \mathbb{R} \right\}$$, the derivative operator is replaced by the _discrete derivative operator_ $$\mathcal{D}: \mathbb{A}^{\mathbb{A}} \mapsto \mathbb{A}^{\mathbb{A}}$$ defined as,

$$\mathcal{D} \left\{ f \right\} \left( a \right) = \frac{f \left( a \right) - f \left( a-h \right)}{h} \tag{1}$$

Notice that, $$\displaystyle{\lim_{h \to 0} \mathcal{D} \left\{ f \right\} \left( a \right) = f^\prime \left( a \right)}$$, if such a limit exists at $$x=a$$ where $$f^\prime \left( x \right)$$ denotes the first derivative of $$f \left( x \right)$$.

As usual, we are using the [backward difference](https://en.wikipedia.org/wiki/Finite_difference#Basic_types) to construct the difference quotient above. That's because in this blog, we'll primarily use discrete calculus in applications of mathematical physics, where the backward difference can be used to encode causality in a direct sense.

## Discrete antiderivative operator

### Operational definition

The _discrete antiderivative operator_ $$\mathcal{D}^{-1}$$ is the inverse of the discrete derivative operator $$\mathcal{D}$$,

$$\mathcal{D} \left\{ \mathcal{D}^{-1} \left\{ f \right\} \right\} \left( a \right) = \mathcal{D}^{-1} \left\{ \mathcal{D} \left\{ f \right\} \right\} \left( a \right) = f \left( a \right)$$

I.e. $$\mathcal{D} \circ \mathcal{D}^{-1} = \mathcal{D}^{-1} \circ \mathcal{D} = \mathcal{D}^0$$

From $$\left( 1 \right)$$, we have,

$$
\begin{align}
f \left( a \right) & = \mathcal{D} \left\{ \mathcal{D}^{-1} \left\{ f \right\} \right\} \left( a \right) \\
 & = \frac{1}{h} \left( \mathcal{D}^{-1} \left\{ f \right\} \left( a \right) - \mathcal{D}^{-1} \left\{ f \right\} \left( a-h \right) \right)
\end{align}
$$

$$\mathcal{D}^{-1} \left\{ f \right\} \left( a \right) - \mathcal{D}^{-1} \left\{ f \right\} \left( a-h \right) = h f \left( a \right) \tag{2}$$

### As Riemann sums

Finding $$\mathcal{D}^{-1} \left\{ f \right\} \left( a \right)$$ from the above equation alone is difficult. Instead, we will discretize the indefinite integral of the continuous function $$f \left( x \right)$$. The intuition behind this is that the integral of a function _is_ its continuous antiderivative.

To do this, we will write the indefinite integral $$F \left( x \right)$$ of $$f \left( x \right)$$ as a Riemann integral and remove the limit,

$$F \left( x \right) = \int_{F^{-1} \left( 0 \right)}^x d \xi f \left( \xi \right) = \lim_{h \to 0} \sum_{k=1}^{n} h f \left( F^{-1} \left( 0 \right) + kh \right)$$

where $$n = \frac{x - F^{-1} \left( 0 \right)}{h}$$

Removing the limit from the Riemann integral, we get a corresponding expression for the discrete antiderivative of $$f \left( a \right)$$ by changing variables in the manner,

$$f \left( x \right) \leftrightarrow f \left( a \right), F \left( x \right) \leftrightarrow \mathcal{D}^{-1} \left\{ f \right\} \left( a \right), F^{-1} \left( x \right) \leftrightarrow \left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( a \right), n \leftrightarrow \frac{a-\left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( 0 \right)}{h}$$

$$\mathcal{D}^{-1} \left\{ f \right\} \left( a \right) = \sum_{k=1}^{n} h f \left( \left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( 0 \right) + kh \right)$$

In other words,

$$\mathcal{D}^{-1} \left\{ f \right\} \left( a \right) = h \left[ f \left( \left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( 0 \right) + h \right) + f \left( \left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( 0 \right) + 2h \right) + \dots + f \left( a \right) \right]$$

We now have to express the term $$\left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( 0 \right)$$ explicitly.

### Finding the unknown term

For it to be true that:

$$\mathcal{D} \left\{ \mathcal{D}^{-1} \left\{ f \right\} \right\} \left( a \right) = \mathcal{D}^{-1} \left\{ \mathcal{D} \left\{ f \right\} \right\} \left( a \right) = f \left( a \right)$$

we will have to do the following,

$$
\begin{align}
f \left( a \right) & = \mathcal{D} \left\{ \mathcal{D}^{-1} \left\{ f \right\} \right\} \left( a \right) \\
 & = \frac{1}{h} \left( \mathcal{D}^{-1} \left\{ f \right\} \left( a \right) - \mathcal{D}^{-1} \left\{ f \right\} \left( a-h \right) \right) \\
 & = \frac{1}{h} \left( h f \left( a \right) - h f \left( \left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( 0 \right) \right) \right) \\
 & = f \left( a \right) - f \left( \left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( 0 \right) \right)
\end{align}
$$

This means that $$f \left( \left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( 0 \right) \right) = 0$$ in order for the antiderivative operator to satisfy its definition. Or,

$$\left( \mathcal{D}^{-1} \left\{ f \right\} \right)^{-1} \left( 0 \right) = f^{-1} \left( 0 \right)$$

### Explicit definition

From the previous result, we can explicitly write the discrete antiderivative of a function,

$$\mathcal{D}^{-1} \{ f \} \left( a \right) = \sum_{k=1}^{n} h f \left( f^{-1} \left( 0 \right) + kh \right)$$

Where,

$$n = \frac{a- f^{-1} \left( 0 \right)}{h}$$
