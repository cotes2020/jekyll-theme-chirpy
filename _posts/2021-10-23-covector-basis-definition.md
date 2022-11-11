---
title: "Demystifying the Definition Of a Covector Basis"
description: "Ever wondered why covectors are defined to act on vectors the way they are?"
categories: [tensor calculus]
tags: [dual vectors, tensors]
toc: false
---

A covector basis $$\left\{ \pmb{\theta}^i \right\}$$ is defined to act on the corresponding vector basis $$\left\{ \pmb{e}_j \right\}$$ in the manner,

$$\pmb{\theta}^i \left( \pmb{e}_j \right) = \delta^i_{\phantom{i} j}$$

Where $$\delta^i_{\phantom{i} j}$$ represents the Kronecker delta. But where does the above definition even come from? Well, turns out it's not so mysterious after all.

Say a covector $$\pmb{\phi}$$ acts on a vector $$\pmb{x}$$. In the component form,

$$\pmb{\phi} \left( \pmb{x} \right) = \left( \phi_i \: \pmb{\theta}^i \right) \left( x^j \: \pmb{e}_j \right)$$

By multilinearity,

$$\pmb{\phi} \left( \pmb{x} \right) = \phi_i \: \pmb{\theta}^i \left( \pmb{e}_j \right) \: x^j$$

We want the result of $$\pmb{\phi} \left( \pmb{x} \right)$$ to be invariant. The simplest way to do so is to define it to be a scalar of the form $$\phi_j \: x^j$$. Therefore,

$$
\begin{align}
\phi_i \: \pmb{\theta}^i \left( \pmb{e}_j \right) \: x^j & = \phi_j \: x^j \\
\phi_i \: \pmb{\theta}^i \left( \pmb{e}_j \right) \: x^j & = \phi_i \: \delta^i_{\phantom{i} j} \: x^j
\end{align}
$$

By linear independence of the components involved, we can cancel out like terms on both sides of the above equation, and are left with the required result,

$$\pmb{\theta}^i \left( \pmb{e}_j \right) = \delta^i_{\phantom{i} j}$$
