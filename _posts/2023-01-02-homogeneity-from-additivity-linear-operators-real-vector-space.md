---
title: "Homogeneity From Additivity for Linear Operators on a Real Vector Space"
description: "How Dedekind cuts provide homogeneity for free from additivity"
categories: [analysis]
tags: [linear operators, Dedekind cuts]
---

## Statement

Consider the 2 well-known properties of a linear operator $$T : \mathbb{R}^m \to \mathbb{R}^n : m, n \in \mathbb{N}$$,

$$
\begin{align}
T \left( \sum_a \pmb{u}_a \right) & = \sum_a T \left( \pmb{u}_a \right) & \forall \: \pmb{u}_a \in \mathbb{R}^m && \left( 1 \right) \\
T \left( c \pmb{u} \right) & = c T \left( \pmb{u} \right) & \forall \: c \in \mathbb{R}, \pmb{u} \in \mathbb{R}^m && \left(2 \right)
\end{align}
$$

**Proposition**: Property $$\left( 2 \right)$$ (homogeneity) can be derived from property $$\left( 1 \right)$$ (additivity) using the underlying structure of $$\mathbb{R}$$.

Thus, we propose that linearity can be unambiguously defined using axiom $$\left( 1 \right)$$ only, at least up to $$\mathbb{R}^m$$.

**Overview of proof**: Represent an arbitrary $$c \in \mathbb{R}$$ as an infinite series of rationals, using Dedekind cuts. Plug the result into $$\left( 1 \right)$$ and manipulate to obtain $$\left( 2 \right)$$.

Let us now elaborate on the above.

## Reals as infinite series of rationals

Suppose we are given any $$c \in \mathbb{R}$$. The construction of $$\mathbb{R}$$ from $$\mathbb{Q}$$ using Dedekind cuts guarantees the following:

$$\forall \: c \in \mathbb{R} : \exists \: \left\{ a_n :  n \in \mathbb{N} \right\} : c = \lim_{n \to \infty} a_n, a_n \in \mathbb{Q} \: \forall \: n \in \mathbb{N}$$

Now, we can define another sequence of rationals $$\displaystyle{ \left\{ b_k : k \in \mathbb{N} \right\} : \sum_{k=1}^n b_k = a_n }$$. It follows,

$$
\begin{align}
b_n & = \sum_{k=1}^n b_k - \sum_{k=1}^{n-1} b_k \\
 & = a_n - a_{n-1} \\
c & = \lim_{n \to \infty} a_n \\
 & = \lim_{n \to \infty} \sum_{k=1}^n b_k \\
 & = \sum_{k=1}^\infty b_k
\end{align}
$$

Thus, every real number can be represented as an infinite series of rationals,

$$\implies \forall \: c \in \mathbb{R} : \exists \: \left\{ b_k : k \in \mathbb{N} \right\} : c = \sum_{k=1}^\infty b_k, b_k \in \mathbb{Q} \: \forall \: k \in \mathbb{N}$$

Furthermore, by definition, every rational is some integer divided by some non-zero integer,

$$\forall \: b \in \mathbb{Q} : \exists \: p \in \mathbb{Z}, q \in \mathbb{Z} \backslash \left\{ 0 \right\} : b = \frac{p}{q}$$

Let us proceed to derive homogeneity from additivity for the special case $$c \in \mathbb{Q}$$ (however, we will still have $$\pmb{u} \in \mathbb{R}^m$$). Then, we will apply the result above to extend our observations to $$c \in \mathbb{R}$$.

Suppose $$c \in \mathbb{Q} , \pmb{u} \in \mathbb{R}^m$$. Consider a linear operator $$T : \mathbb{R}^m \to \mathbb{R}^n$$ which obeys additivity i.e. property $$\left( 1 \right)$$. Let us express $$c$$ as $$c = \frac{p}{q} : p \in \mathbb{Z}, q \in \mathbb{Z} \backslash \left\{ 0 \right\}$$. The signature of $$c$$ can be encoded in either the numerator $$p$$, which, pedantically speaking, is preferred over $$q$$ as the domain of the former is the entirety of $$\mathbb{Z}$$. 

## For $$c \in \mathbb{Z}$$

We will first derive homogeneity from additivity for the simple case $$q=1, p \in \mathbb{Z} \implies c \in \mathbb{Z}$$. To do so, we will investigate the individual scenarios $$p \in \mathbb{Z}^+, p=0, p \in \mathbb{Z}^-$$. 


### For $$c \in \mathbb{Z}^+ = \mathbb{N}$$

We have,

$$
\begin{align}
T \left( c \pmb{u} \right) & = T \left( p \pmb{u} \right) : p \in \mathbb{N} \\
 & = T \left( \sum_{a = 1}^p u \right)
\end{align}
$$

By additivity,

$$
\begin{align}
T \left( c \pmb{u} \right) & = \sum_{a=1}^p T \left( \pmb{u} \right) \\
 & = p T \left( \pmb{u} \right) \\
 & = c T \left( \pmb{u} \right) & \square
\end{align}
$$

### For $$c=0$$

For $$c=0$$, we have,

$$
\begin{align}
T \left( 0 \cdot \pmb{u} \right) & = T \left( \pmb{0}_m \right) \\
 & = T \left( \pmb{0}_m + \pmb{0}_m \right) \\
 & = T \left( \pmb{0}_m \right) + T \left( \pmb{0}_m \right) \\
T \left( \pmb{0}_m \right) & = T \left( \pmb{0}_m \right) + T \left( \pmb{0}_m \right) \\
 \implies T \left( \pmb{0}_m \right) & = \pmb{0}_n
\end{align}
$$

where $$\pmb{0}_k$$ is the null vector in $$\mathbb{R}^k$$.

### For $$c \in \mathbb{Z}^-$$

Let $$p = -n : n \in \mathbb{Z}^+$$,

$$
\begin{align}
\pmb{0}_n & = T \left( \pmb{0}_m \right) \\
 & = T \left( n \pmb{u} - n \pmb{u} \right)
\end{align}
$$

By additivity,

$$
\begin{align}
\pmb{0}_n & = T \left( n \pmb{u} \right) + T \left( -n \pmb{u} \right) \\
 & = n T \left( \pmb{u} \right) + T \left( -n \pmb{u} \right) & \left[ \because n \in \mathbb{Z}^+ \right] \\
\implies T \left( p \pmb{u} \right) & = T \left( - n \pmb{u} \right) \\
 & = \pmb{0}_n - n T \left( \pmb{u} \right) \\
 & = - n T \left( \pmb{u} \right) \\
 & = p T \left( \pmb{u} \right) & \square
\end{align}
$$

Combining the different scenarios above, we have indeed found,

$$
\begin{align}
\forall \: c \in \mathbb{Z} : \left( 1 \right) \implies \left( 2 \right) && \left( A \right)
\end{align}
$$

## For $$c \in \left\{ \frac{1}{q} : q \in \mathbb{Z} \backslash \left\{ 0 \right\} \right\}$$

We will now see how the above result is also true for $$c = \frac{p}{q} : p = 1, q \in \mathbb{Z} \backslash \left\{ 0 \right\}$$. Ultimately, we will combine this result with the corresponding one for $$c \in \mathbb{Z}$$ to generalize it for all $$c \in \mathbb{Q}$$ which in turn will let us generalize it to $$c \in \mathbb{R}$$ via Dedekind cuts as previously stated.

We begin with the statement,

$$
\begin{align}
T \left( \sum_{a=1}^q \frac{1}{q} \pmb{u} \right) & = T \left( \pmb{u} \right) \\
\sum_{a=1}^q T \left( \frac{1}{q} \pmb{u} \right) & = T \left( \pmb{u} \right) \\
q T \left( \frac{1}{q} \pmb{u} \right) & = T \left( u \right) \\
T \left( \frac{1}{q} \pmb{u} \right) & = \frac{1}{q} T \left( u \right) & \square
\end{align}
$$

Thus,

$$
\begin{align}
\forall \: c = \frac{1}{q} : q \in \mathbb{Z} \backslash \left\{ 0 \right\} : \left( 1 \right) \implies \left( 2 \right) && \left( B \right)
\end{align}
$$

## For $$c \in \mathbb{Q}$$

Let $$c = \frac{p}{q} : p \in \mathbb{Z}, q \in \mathbb{Z} \backslash \left\{ 0 \right\}$$. Using the previous results $$\left( A \right)$$ and $$\left( B \right)$$,

$$
\begin{align}
T \left( c \pmb{u} \right) & = T \left( \frac{p}{q} \pmb{u} \right) \\
 & = T \left( p \cdot \frac{1}{q} \pmb{u} \right) \\
 & = p T \left( \frac{1}{q} \pmb{u} \right) & \left[ \left( A \right) \right] \\
 & = \frac{p}{q} T \left( \pmb{u} \right) & \left[ \left( B \right) \right] \\
 & = c T \left( \pmb{u} \right) & \square
\end{align}
$$

$$
\begin{align}
\forall \: c \in \mathbb{Q} : \left( 1 \right) \implies \left( 2 \right) && \left( C \right)
\end{align}
$$

## For $$c \in \mathbb{R}$$

Consider an arbitrary $$c \in \mathbb{R}$$. From [the section on reals as infinite series of rationals](#reals-as-infinite-series-of-rationals), there exists a sequence of rationals $$\left\{ b_k : k \in \mathbb{N} \right\} : b_k \in \mathbb{Q} \: \forall \: k \in \mathbb{N}$$ such that it adds up to $$c$$,

$$\displaystyle{ c = \sum_{k=1}^\infty b_k }$$

Hence,

$$T \left( c \pmb{u} \right) = T \left( \sum_{k=1}^\infty b_k \pmb{u} \right)$$

By additivity,

$$
\begin{align}
T \left( c \pmb{u} \right) & = \sum_{k=1}^\infty T \left( b_k \pmb{u} \right) \\
 & = \sum_{k=1}^\infty b_k T \left( \pmb{u} \right) & \left[ \left( C \right) \right] \\
 & = c T \left( \pmb{u} \right) & \blacksquare
\end{align}
$$

$$\forall \: c \in \mathbb{R} : \left( 1 \right) \implies \left( 2 \right)$$

## Summary

To summarize the above approach, homogeneity of linear operators on vector spaces built on the base field $$\mathbb{R}$$, comes for free from their additivity. We realized this by using the construction of $$\mathbb{R}$$ from $$\mathbb{Q}$$ which is in turn constructed from $$\mathbb{Z}$$. 

The advantage of expressing reals in terms of integers is that integers can fundamentally be used for counting, which is implicitly applied in property $$\left( 1 \right)$$, additivity. Thus, this bridge from $$\left( 1 \right)$$ to $$\left( 2 \right)$$ allows us to logically proceed in the same direction.