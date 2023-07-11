---
title: "Algebra Done Tensorially: Part 1 (Bilinear Products)"
description: "A post studying bilinear products in tensor algebra"
categories: [representation theory]
tags: [bilinear products, tensors, algebras]
---

Welcome to this five-part series of posts:

| Parts | Topics |
| :-- | :-- |
| [Part 1 (Bilinear Products)]({% post_url 2021-10-18-bilinear-products %})|tensors, bilinear products |
| [Part 2 (Algebras Over Fields)]({% post_url 2021-10-23-algebras-over-fields %}) |linear maps, algebra, degrees of freedom |
| [Part 3 (Complex Numbers and Quaternions)]({% post_url 2022-02-24-complex-numbers-quaternions %}) | complex numbers, quaternions, gamma matrices |
| [Part 4 (Clifford Algebras)]() | in progress |
| [Part 5 (Lie Algebras)]() | in progress |

In this first post, I will be exploring bilinear products using tensor analysis. Since tensors will be used extensively in this series, I thought it best to quickly recap their working. Note that this is _not_ appropriate as an introduction to tensors. If you haven't already studied tensors, I'd recommend watching [this excellent playlist](https://youtube.com/playlist?list=PLJHszsWbB6hrkmmq57lX8BV-o-YIOFsiG) by YouTuber [Eigenchris](https://www.youtube.com/user/eigenchris).

If you are already aware of how tensors work, skip to [bilinear products](#bilinear_products).

## Notation

1. All tensorial objects are written in boldface. This includes basis vectors and covectors, as they remain invariant under a change of coordinates (what really changes is the labels we associate with our choice of basis).

2. Dummy indices are summed over, according to the Einstein summation convention.

3. Bilinear products are written in calligraphic font to distinguish them from ordinary tensors.

4. Tuples of indices are represented by their capital letters, even if the indices belong to different tensors, in which case the tensors are being multiplied in some form, such as through the tensor product, or by multiplying the components. Jump to [tuple index notation](#tuple_notation) to understand this better.

## Tensors

### Definition

Given a vector space $$V$$ and its dual space $$V^\text{*}$$, a rank $$\left( p, q \right)$$ _tensor_ is a multilinear map $$\pmb{T} \in \underset{i=1}{\overset{p}{\bigotimes}} V \underset{j=1}{\overset{q}{\bigotimes}} V^\text{*}$$.

This means that given a covector basis $$\left\{ \pmb{\theta}^i \right\}$$ and vector basis $$\left\{ \pmb{e}_j \right\}$$, if $$\chi_{i_1} \pmb{\theta}^{i_1}, \psi_{i_2} \pmb{\theta}^{i_2}, \dots, \omega_{i_p} \pmb{\theta}^{i_p}  \in V^\text{*}$$ and $$u^{j_1} \pmb{e}_{j_1}, v^{j_2} \pmb{e}_{j_2} , \dots, w^{j_q} \pmb{e}_{j_q} \in V$$,

$$\pmb{T} \left( \chi_{i_1} \pmb{\theta}^{i_1}, \psi_{i_2} \pmb{\theta}^{i_2}, \dots, \omega_{i_p} \pmb{\theta}^{i_p}, u^{j_1} \pmb{e}_{j_1}, v^{j_2} \pmb{e}_{j_2}, \dots, w^{j_q} \pmb{e}_{j_q} \right) = \chi_{i_1} \psi_{i_2} \dots \omega_{i_p} \: \pmb{T} \left( \pmb{\theta}^{i_1}, \pmb{\theta}^{i_2}, \dots, \pmb{\theta}^{i_p}, \pmb{e}_{j_1}, \pmb{e}_{j_2}, \dots, \pmb{e}_{j_q} \right) u^{j_1} v^{j_2} \dots w^{j_q}$$

### Components

With respect to the previous section, for a tensor $$\pmb{T}$$, its _components_ $$T^{i_1 \dots i_p}_{\phantom{i_1 \dots i_p} j_1 \dots j_q}$$ are defined in the manner,

$$\pmb{T} \left( \pmb{\theta}^{i_1}, \cdots, \pmb{\theta}^{i_p}, \pmb{e}_{j_1}, \dots, \pmb{e}_{j_q} \right) = T^{i_1 \dots i_p}_{\phantom{i_1 \dots i_p} j_1 \dots j_q} \: \underset{a=1}{\overset{p}{\bigotimes}} \pmb{e}_{i_a} \underset{b=1}{\overset{q}{\bigotimes}} \pmb{\theta}^{j_b}$$

The covector basis is defined to act on the vector basis and vice-versa as:

$$\pmb{\theta}^i \left( \pmb{e}_j \right) = \pmb{e}_j \left( \pmb{\theta}^i \right) = \delta^i_{\phantom{i}_j}$$

(The logic behind the above definition is explained in [another post]({% post_url 2021-10-23-covector-basis-definition %}).)

Assuming that covectors act on vectors only and vice-versa under the tensor product, we can extend the action to multiple copies of basis vectors and covectors so that,

$$\underset{a=1}{\overset{p}{\bigotimes}} \pmb{\theta}^{i_a} \underset{b=1}{\overset{q}{\bigotimes}} \pmb{e}_{j_b} : \underset{a=1}{\overset{p}{\bigotimes}} V \underset{b=1}{\overset{q}{\bigotimes}} V^* \mapsto \mathcal{I} \left[ V^{p+q} \right] = \mathcal{I} \left[ \left( V^* \right)^{p+q} \right]$$

where $$\mathcal{I}$$ is the identity map. Now writing the above explicitly, we get,

$$\underset{a=1}{\overset{p}{\bigotimes}} \pmb{\theta}^{i_a} \underset{b=1}{\overset{q}{\bigotimes}} \pmb{e}_{j_b} \left[ \underset{a=1}{\overset{p}{\bigotimes}} \pmb{e}_{k_a} \underset{b=1}{\overset{q}{\bigotimes}} \pmb{\theta}^{l_b} \right] = \prod_{a=1}^p \delta^{i_a}_{\phantom{i_a} k_a} \prod_{b=1}^q \delta^{l_b}_{\phantom{l_b} j_b}$$

The components of a tensor can now be written as,

$$T^{i_1 \dots i_p}_{\phantom{i_1 \dots i_p} j_1 \dots j_q} = \underset{a=1}{\overset{p}{\bigotimes}} \pmb{\theta}^{i_a} \underset{b=1}{\overset{q}{\bigotimes}} \pmb{e}_{j_b} \left[ \pmb{T} \left( \pmb{\theta}^{i_1}, \cdots, \pmb{\theta}^{i_p}, \pmb{e}_{j_1}, \dots, \pmb{e}_{j_q} \right) \right]$$

<a name="tuple_notation"></a>

To simplify our lives, let us represent tuples of indices with their capital letter, e.g. $$I \equiv i_1 \dots i_p$$. Also, let us write any multiplication of tensor components, or tensors, with a capital letter, along with tuples for the indices. Lastly, let the tensor product symbol be omitted altogether. Then, the previous set of equations becomes more readable,

$$
\begin{align}
\pmb{T} \left( \pmb{\Theta}^I, \pmb{E}_J \right) & = T^K_{\phantom{K} L} \: \pmb{E}_K \pmb{\Theta}^L \\
T^K_{\phantom{K} L} & = \pmb{\Theta}^K \pmb{E}_L \left[ \pmb{T} \left( \pmb{\Theta}^I, \pmb{E}_J \right) \right]
\end{align}
$$

Under a change of coordinates represented by a Jacobian $$\Lambda^{j^\prime}_{\phantom{j^\prime} j}$$, the coordinates of a tensor transform as,

$$
\begin{align}
T^{i_1^\prime \dots i_p^\prime}_{\phantom{i_1^\prime \dots i_p^\prime} j_1^\prime \dots j_q^\prime} & = T \left( \pmb{\theta}^{i_1^\prime}, \dots, \pmb{\theta}^{i_p^\prime}, \pmb{e}_{j_1^\prime}, \dots, \pmb{e}_{j_q^\prime} \right) \\
 & = T \left( \Lambda^{i_1^\prime}_{\phantom{i_1^\prime} i_1} \pmb{\theta}^{i_1}, \dots, \Lambda^{i_p^\prime}_{\phantom{i_p^\prime} i_p} \pmb{\theta}^{i_p}, \Lambda^{j_1}_{\phantom{j_1} j_1^\prime} \pmb{e}_{j_1}, \dots, \Lambda^{j_q}_{\phantom{j_q} j_q^\prime} \pmb{e}_{j_q} \right)
\end{align}
$$

By multilinearity,

$$
\begin{align}
T^{i_1^\prime \dots i_p^\prime}_{\phantom{i_1^\prime \dots i_p^\prime} j_1^\prime \dots j_q^\prime} & = \Lambda^{i_1^\prime}_{\phantom{i_1^\prime} i_1} \dots \Lambda^{i_p^\prime}_{\phantom{i_p^\prime} i_p} \: T \left( \pmb{\theta}^{i_1}, \dots, \pmb{\theta}^{i_p}, \pmb{e}_{j_1}, \dots, \pmb{e}_{j_q} \right) \Lambda^{j_1}_{\phantom{j_1} j_1^\prime} \dots \Lambda^{j_q}_{\phantom{j_q} j_q^\prime} \\
T^{i_1^\prime \dots i_p^\prime}_{\phantom{i_1^\prime \dots i_p^\prime} j_1^\prime \dots j_q^\prime} & = \Lambda^{i_1^\prime}_{\phantom{i_1^\prime} i_1} \dots \Lambda^{i_p^\prime}_{\phantom{i_p^\prime} i_p} \: T^{i_1 \dots i_p}_{\phantom{i_1 \dots i_p} j_1 \dots j_q} \Lambda^{j_1}_{\phantom{j_1} j_1^\prime} \dots \Lambda^{j_q}_{\phantom{j_q} j_q^\prime} \\
T^{i_1^\prime \dots i_p^\prime}_{\phantom{i_1^\prime \dots i_p^\prime} j_1^\prime \dots j_q^\prime} & = \left( \prod_{a=1}^p \Lambda^{i_a^\prime}_{\phantom{i_a^\prime} i_a} \prod_{b=1}^q \Lambda^{j_b}_{\phantom{j_b} j_b^\prime} \right) T^{i_1 \dots i_p}_{\phantom{i_1 \dots i_p} j_1 \dots j_q}
\end{align}
$$

Or using the tuple notation for indices,

$$T^{I^\prime}_{\phantom{I^\prime} J^\prime} = \Lambda^{I^\prime}_{\phantom{I^\prime} I} \: T^I_{\phantom{I} J} \: \Lambda^J_{\phantom{J} J^\prime}$$

This is known as the tensor transformation law. Note that the notation for the Jacobian is a slight deviation from the regular tuple notation, in that $$\Lambda^{I^\prime}_{\phantom{I^\prime} I}$$ is _not_ some tensor $$\Lambda^{i_1 \dots i_p}_{\phantom{i_1 \dots i_p} j_1 \dots j_q}$$, but a product of tensor components, $$\Lambda^{i_1^\prime}_{\phantom{i_1^\prime} i_1} \dots \Lambda^{i_p^\prime}_{\phantom{i_p^\prime} i_p}$$. This will be the only exception in the tuple notation.

### Invariance

Under a change of coordinates $$\Lambda^{j^\prime}_{\phantom{j^\prime} j}$$, the components of $$\pmb{T}$$ transform but $$\pmb{T}$$ itself remains invariant:

$$
\begin{align}
\pmb{T}^\prime & = T^{I^\prime}_{\phantom{I^\prime} J^\prime} \: \pmb{E}_{I^\prime} \pmb{\Theta}^\prime \\
 & = \Lambda^{I^\prime}_{\phantom{I^\prime} I} \: T^I_{\phantom{I} J} \: \Lambda^J_{\phantom{J} J^\prime} \: \Lambda^K_{\phantom{K} I^\prime} \Lambda^{J^\prime}_{\phantom{J^\prime} L} \: \pmb{E}_K \pmb{\Theta}^L \\
 & = \Lambda^{I^\prime}_{\phantom{I^\prime} I} \Lambda^K_{\phantom{K} I^\prime} \Lambda^J_{\phantom{J} J^\prime} \Lambda^{J^\prime}_{\phantom{J^\prime} L} \: T^I_{\phantom{I} J} \: \pmb{E}_K \pmb{\Theta}^L \\
 & = \Delta^K_{\phantom{K} I} \Delta^J_{\phantom{J} L} T^I_{\phantom{I} J} \: \pmb{E}_K \pmb{\Theta}^L \\
 & = T^I_{\phantom{I} J} \: \pmb{E}_I \pmb{\Theta}^J \\
 & = \pmb{T}
\end{align}
$$

where $$\Delta^K_{\phantom{K} I} = \delta^{k_1}_{\phantom{k_1} i_1} \dots \delta^{k_p}_{\phantom{k_p} i_p}$$. Once again, we see a deviation from the regular tuple notation, of similar kind as in the case of the Jacobian $$\pmb{\Lambda}$$. This is not another exception to the notation; the Kronecker delta $$\pmb{\delta}$$ is a special kind of Jacobian, which maps a vector space to itself.

<a name="bilinear_products"></a>

## Bilinear Products

### Definition

Given a vector space $$V$$, its dual space $$V^\text{*}$$ and rank $$\left( p, q \right)$$ tensors $$\pmb{\Phi}$$ and $$\pmb{T}$$, a _bilinear product_ $$\mathcal{B}$$ is a bilinear map,

$$\mathcal{B} \left( \pmb{\Phi}, \pmb{T} \right) \in \underset{i=1}{\overset{p}{\bigotimes}} V \underset{j=1}{\overset{q}{\bigotimes}} V^\text{*}$$

Thus, $$\mathcal{B}$$ linearly maps a pair of tensors belonging to the same tensor space, to another tensor in the same tensor space.

As the name suggests, bilinear products are bilinear, i.e. linear in both their arguments,

$$
\begin{align}
\mathcal{B} \left( \sum_i \pmb{\Phi_i}, \pmb{T} \right) & = \sum_i \mathcal{B} \left( \pmb{\Phi}_i, \pmb{T} \right) \\
\mathcal{B} \left( \pmb{\Phi_i}, \sum_j \pmb{T_j} \right) & = \sum_j \mathcal{B} \left( \pmb{\Phi}, \pmb{T}_j \right) \\
\implies \mathcal{B} \left( \sum_i \pmb{\Phi_i}, \sum_j \pmb{T_j} \right) & = \sum_i \sum_j \mathcal{B} \left( \pmb{\Phi}_i, \pmb{T}_j \right)
\end{align}
$$

<a name="components_bilinear_product"></a>

### Components

Given the components of two tensors, by bilinearity, their bilinear product can be expressed as,

$$
\begin{align}
\pmb{\mathcal{B}} \left( \pmb{\Phi}, \pmb{T} \right) & = \pmb{\mathcal{B}} \left( \Phi^I_{\phantom{I} J} \: \pmb{E}_I \pmb{\Theta}^J, T^K_{\phantom{K} L} \: \pmb{E}_K \pmb{\Theta}^L \right) \\
 & = \Phi^I_{\phantom{I} J} \: \pmb{\mathcal{B}} \left( \pmb{E}_I \pmb{\Theta}^J, \pmb{E}_K \pmb{\Theta}^L \right) \: T^K_{\phantom{K} L}
\end{align}
$$

In the context of the above, the components $$B^{J \phantom{I} L \phantom{K} M}_{\phantom{J} I \phantom{L} K \phantom{M} N}$$ of $$\mathcal{B}$$ may be defined as,

$$
\begin{align}
\pmb{\mathcal{B}} \left( \pmb{E}_I \pmb{\Theta}^J, \pmb{E}_K \pmb{\Theta}^L \right) & = B^{J \phantom{I} L \phantom{K} M}_{\phantom{J} I \phantom{L} K \phantom{M} N} \: \pmb{E}_M \pmb{\Theta}^N \\
\implies B^{J \phantom{I} L \phantom{K} M}_{\phantom{J} I \phantom{L} K \phantom{M} N} & = \pmb{\Theta}^M \pmb{E}_N \left[ \pmb{\mathcal{B}} \left( \pmb{E}_I \pmb{\Theta}^J, \pmb{E}_K \pmb{\Theta}^L \right) \right]
\end{align}
$$

It is noteworthy that in the above, odd tuples i.e. $$I, K, M$$ are indices running from $$1$$ to $$p$$, coupled with $$V$$ ; while even tuples $$J, L, N$$ run from $$1$$ to $$q$$ coupled with $$V^\text{*}$$.

We know that $$\pmb{\mathcal{B}} \left( \pmb{\Phi}, \pmb{T} \right)$$ is a tensor in the same space as its arguments, but that does not immediately justify that the components of $$\pmb{\mathcal{B}}$$ as we defined them transform like tensor components. Instead, we will have to verify that manually,

$$
\begin{align}
B^{J^\prime \phantom{I^\prime} L^\prime \phantom{K^\prime} M^\prime}_{\phantom{J^\prime} I^\prime \phantom{L^\prime} K^\prime \phantom{M^\prime} N^\prime} & = \pmb{\Theta}^{M^\prime} \pmb{E}_{N^\prime} \left[ \pmb{\mathcal{B}} \left( \pmb{E}_{I^\prime} \pmb{\Theta}^{J^\prime}, \pmb{E}_{K^\prime} \pmb{\Theta}^{L^\prime} \right) \right] \\
 & = \pmb{\Theta}^{M^\prime} \pmb{E}_{N^\prime} \left[ \pmb{\mathcal{B}} \left( \Lambda^I_{\phantom{I} I^\prime} \Lambda^{J^\prime}_{\phantom{J^\prime} J} \: \pmb{E}_{I} \pmb{\Theta}^{J}, \Lambda^K_{\phantom{K} K^\prime} \Lambda^{L^\prime}_{\phantom{L^\prime} L} \: \pmb{E}_{K} \pmb{\Theta}^{L} \right) \right] \\
 & = \pmb{\Theta}^{M^\prime} \pmb{E}_{N^\prime} \left[ \Lambda^I_{\phantom{I} I^\prime} \Lambda^{J^\prime}_{\phantom{J^\prime} J} \: \pmb{\mathcal{B}} \left( \pmb{E}_{I} \pmb{\Theta}^{J}, \pmb{E}_{K} \pmb{\Theta}^{L} \right) \: \Lambda^K_{\phantom{K} K^\prime} \Lambda^{L^\prime}_{\phantom{L^\prime} L} \right] \\
 & = \Lambda^I_{\phantom{I} I^\prime} \Lambda^{J^\prime}_{\phantom{J^\prime} J} \Lambda^K_{\phantom{K} K^\prime} \Lambda^{L^\prime}_{\phantom{L^\prime} L} \pmb{\Theta}^{M^\prime} \pmb{E}_{N^\prime} \left[ \pmb{\mathcal{B}} \left( \pmb{E}_{I} \pmb{\Theta}^{J}, \pmb{E}_{K} \pmb{\Theta}^{L} \right) \right] \\
 & = \Lambda^I_{\phantom{I} I^\prime} \Lambda^{J^\prime}_{\phantom{J^\prime} J} \Lambda^K_{\phantom{K} K^\prime} \Lambda^{L^\prime}_{\phantom{L^\prime} L} \left( \Lambda^{M^\prime}_{\phantom{M^\prime} M} \Lambda^{N}_{\phantom{N} N^\prime} \pmb{\Theta}^M \pmb{E}_N \right) \left[ \pmb{\mathcal{B}} \left( \pmb{E}_{I} \pmb{\Theta}^{J}, \pmb{E}_{K} \pmb{\Theta}^{L} \right) \right] \\
 & = \Lambda^I_{\phantom{I} I^\prime} \Lambda^{J^\prime}_{\phantom{J^\prime} J} \Lambda^K_{\phantom{K} K^\prime} \Lambda^{L^\prime}_{\phantom{L^\prime} L} \Lambda^{M^\prime}_{\phantom{M^\prime} M} \Lambda^{N}_{\phantom{N} N^\prime} \pmb{\Theta}^M \pmb{E}_N \left[ \pmb{\mathcal{B}} \left( \pmb{E}_{I} \pmb{\Theta}^{J}, \pmb{E}_{K} \pmb{\Theta}^{L} \right) \right] \\
 & = \Lambda^I_{\phantom{I} I^\prime} \Lambda^{J^\prime}_{\phantom{J^\prime} J} \Lambda^K_{\phantom{K} K^\prime} \Lambda^{L^\prime}_{\phantom{L^\prime} L} \Lambda^M_{\phantom{M} M^\prime} \Lambda^{N^\prime}_{\phantom{N^\prime} N} \: B^{J \phantom{I} L \phantom{K} M}_{\phantom{J} I \phantom{L} K \phantom{M} N}
\end{align}
$$

Thus, the components of $$\mathcal{B}$$ indeed transform like that of a tensor! In fact, given a rank $$\left( p, q \right)$$ tensor, the said components represent a rank $$\left( 2q+p, 2p+q \right)$$ tensor.

### A Note

We have seen above how $$B^{J \phantom{I} L \phantom{K} M}_{\phantom{J} I \phantom{L} K \phantom{M} N}$$ transforms like a tensor. It must be noted though, that the bilinear product formed from these components alone is not a tensor. I.e., the following quantity does not transform like a tensor,

$$\pmb{\mathcal{B}} \left( \pmb{E}_I \pmb{\Theta}^J, \pmb{E}_K \pmb{\Theta}^L \right) = B^{J \phantom{I} L \phantom{K} M}_{\phantom{J} I \phantom{L} K \phantom{M} N} \: \pmb{E}_M \pmb{\Theta}^N$$

Looking at the indices, the above map itself (with its components and basis) transforms like the components (not the entire map) of a rank $$\left( 2q, 2p \right)$$ tensor. We can use this fact to construct a tensor,

$$
\begin{align}
\pmb{B} & = \pmb{\mathcal{B}} \left( \pmb{E}_I \pmb{\Theta}^J, \pmb{E}_K \pmb{\Theta}^L \right) \pmb{E}_{J L} \pmb{\Theta}^{IK} \\
 & = B^{J \phantom{I} L \phantom{K} M}_{\phantom{J} I \phantom{L} K \phantom{M} N} \: \pmb{E}_M \pmb{\Theta}^N \pmb{E}_{J L} \pmb{\Theta}^{IK}
\end{align}
$$

where $$\pmb{E}_{JL} = \pmb{E}_J \otimes \pmb{E}_L$$ and $$\pmb{\Theta}^{IK} = \pmb{\Theta}^I \otimes \pmb{\Theta}^K$$. Note that the above tensor is labelled as $$\pmb{B}$$, which is different from the calligraphic label $$\pmb{\mathcal{B}}$$ for the tensor generated from two other tensors by a bilinear product.

### As linear maps

When only one argument is passed to a bilinear product, it acts as a linear map on the vector space parameterized by the other argument,

$$
\begin{align}
\mathcal{B} \left( \pmb{\Phi}, \cdot \right) \left( \sum_j \pmb{T_j} \right) & = \sum_j \pmb{\mathcal{B}} \left( \pmb{\Phi}, \cdot \right) \left( \pmb{T_j} \right) \\
\pmb{\mathcal{B}} \left( \cdot, \pmb{T} \right) \left( \sum_i \pmb{\Phi_i} \right) & = \sum_i \pmb{\mathcal{B}} \left( \cdot, \pmb{T} \right) \left( \pmb{\Phi_i} \right)
\end{align}
$$

In the component form,

$$
\begin{align}
\pmb{\mathcal{B}} \left( \pmb{\Phi}, \cdot \right) & = \pmb{\mathcal{B}} \left( \Phi^{I}_{\phantom{I} J} \: \pmb{E}_I \pmb{\Theta}^J, \cdot \right) \\
 & = \Phi^I_{\phantom{I} J} \: B^{J \phantom{I} L \phantom{K} M}_{\phantom{J} I \phantom{L} K \phantom{M} N} \: \pmb{E}_M \pmb{\Theta}^N \\
 & = \Lambda^M_{\phantom{M} K} \Lambda^{L}_{\phantom{L} N} \: \pmb{E}_M \pmb{\Theta}^N
\end{align}
$$

By linear independence of the components of the tensors involved,

$$\Phi^I_{\phantom{I} J} \: B^{J \phantom{I} L \phantom{K} M}_{\phantom{J} I \phantom{L} K \phantom{M} N} = \Lambda^M_{\phantom{M} K} \Lambda^{L}_{\phantom{L} N}$$

In other words, given a basis, a tensor (or more generally, tensor field) and a bilinear product, one can covariantly transform the components of an arbitrary tensor in the same space as the one given. This will be the starting point of the next post in this series, [Algebra Done Tensorially: Part 2 (Algebras Over Fields)]({% post_url 2021-10-23-algebras-over-fields %}).
