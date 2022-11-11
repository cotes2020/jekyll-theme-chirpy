---
title: "Components of Covariant Derivative of a Tensor Field"
description: "Deriving the components of the covariant derivative of an arbitrary tensor field"
categories: [tensor calculus]
tags: [covariant derivatives, tensors]
redirect_from: covariant-derivative-tensor-field-components
---

Let us find the covariant derivative of a rank $$(p, q)$$ tensor field $$\pmb{\phi}$$ by applying the Leibniz law repeatedly:

$$\nabla_\rho \left( \pmb{\phi} \otimes \pmb{\psi} \right) = \left( \nabla_\rho \pmb{\phi} \right) \otimes \pmb{\psi} + \pmb{\phi} \otimes \left( \nabla_\rho \pmb{\psi} \right)$$

Note that we will be implicitly using a Levi-Civita connection on a pseudo-Riemannian manifold.

## Derivation

$$
\begin{align}
\nabla_\rho \pmb{\phi} & = \nabla_\rho \left( \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \right) \\
 & = \partial_\rho \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & + \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \left( \nabla_\rho \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \right) \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & + \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \left( \nabla_\rho \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \right)
\end{align}
$$

In the above, when $$\nabla_\rho$$ acts explicitly on the components of $$\pmb{\phi}$$, the components behave like scalars, thereby allowing us to replace the covariant derivative with the partial derivative $$\partial_\rho$$. Let us now apply the Leibniz law,

$$
\begin{align}
\nabla_\rho \pmb{\phi} & = \partial_\rho \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 &+ \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \left[ \sum_{i=1}^p \underset{k=1}{\overset{i-1}{\bigotimes}} \partial_{\mu_k} \otimes \left( \nabla_\rho \partial_{\mu_i} \right) \underset{k=i+1}{\overset{p}{\bigotimes}} \partial_{\mu_k} \right] \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & + \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \left[ \sum_{j=1}^q \underset{k=1}{\overset{j-1}{\bigotimes}} \text{d} x^{\nu_k} \otimes \left( \nabla_\rho \text{d}x^j \right) \underset{k=j+1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_k} \right]
\end{align}
$$

Now, we use the following definitions of the connection coefficients,

$$
\begin{align}
\nabla_\rho \partial_\mu & = \Gamma^\sigma_{\phantom{\sigma} \rho \mu} \partial_\sigma \\
\nabla_\rho \text{d}x^\nu & = \Gamma^\nu_{\phantom{\nu} \rho \sigma} \text{d}x^\sigma
\end{align}
$$

as well as the linearity of the tensor product,

$$
\begin{align}
\bigotimes_k x^\sigma \partial_\sigma & = x^\sigma \bigotimes_k \partial_\sigma  \\ \bigotimes_k \theta_\sigma \text{d}x^\sigma & = \theta_\sigma \bigotimes_k \text{d}x^\sigma
\end{align}
$$

to get:

$$
\begin{align}
\nabla_\rho \pmb{\phi} & = \partial_\rho \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & + \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \left[ \sum_{i=1}^p \Gamma^{\sigma}_{\phantom{\sigma} \rho \mu_i} \underset{k=1}{\overset{i-1}{\bigotimes}} \partial_{\mu_k} \otimes \partial_\sigma \underset{k=i+1}{\overset{p}{\bigotimes}} \partial_{\mu_k} \right] \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & - \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \left[ \sum_{j=1}^q \Gamma^{\nu_j}_{\phantom{\nu_j} \rho \sigma} \underset{k=1}{\overset{j-1}{\bigotimes}} \text{d} x^{\nu_k} \otimes \text{d}x^\sigma \underset{k=j+1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_k} \right] \\ \\
 & = \partial_\rho \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & + \sum_{i=1}^p \Gamma^{\sigma}_{\phantom{\sigma} \rho \mu_i} \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{k=1}{\overset{i-1}{\bigotimes}} \partial_{\mu_k} \otimes \partial_\sigma \underset{k=i+1}{\overset{p}{\bigotimes}} \partial_{\mu_k} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & - \sum_{j=1}^q \Gamma^{\nu_j}_{\phantom{\nu_j} \rho \sigma} \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{k=1}{\overset{j-1}{\bigotimes}} \text{d} x^{\nu_k} \otimes \text{d}x^\sigma \underset{k=j+1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_k}
\end{align}
$$

Let us exchange the indices $$\sigma$$ and $$\mu_i$$, and $$\sigma$$ and $$\nu_j$$ wherever they are dummy indices:

$$
\begin{align}
\nabla_\rho \pmb{\phi} & = \partial_\rho \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & + \sum_{i=1}^p \Gamma^{\mu_i}_{\phantom{\mu_i} \rho \sigma} \phi^{\mu_1 \dots \mu_{i-1} \sigma \mu_{i+1} \dots \mu_p}_{\phantom{\mu_1 \dots \mu_{i-1} \sigma \mu_{i+1} \dots \mu_p} \nu_1 \dots \nu_q} \underset{k=1}{\overset{i-1}{\bigotimes}} \partial_{\mu_k} \otimes \partial_{\mu_i} \underset{k=i+1}{\overset{p}{\bigotimes}} \partial_{\mu_k} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & - \sum_{j=1}^q \Gamma^{\sigma}_{\phantom{\sigma} \rho \nu_j} \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_{j-1} \sigma \nu_{j+1} \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{k=1}{\overset{j-1}{\bigotimes}} \text{d} x^{\nu_k} \otimes \text{d}x^{\nu_j} \underset{k=j+1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_k} \\ \\
 & = \partial_\rho \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & + \sum_{i=1}^p \Gamma^{\mu_i}_{\phantom{\mu_i} \rho \sigma} \phi^{\mu_1 \dots \mu_{i-1} \sigma \mu_{i+1} \dots \mu_p}_{\phantom{\mu_1 \dots \mu_{i-1} \sigma \mu_{i+1} \dots \mu_p} \nu_1 \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j} \\
 & - \sum_{j=1}^q \Gamma^{\sigma}_{\phantom{\sigma} \rho \nu_j} \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_{j-1} \sigma \nu_{j+1} \dots \nu_q} \underset{i=1}{\overset{p}{\bigotimes}} \partial_{\mu_i} \underset{j=1}{\overset{q}{\bigotimes}} \text{d} x^{\nu_j}
\end{align}
$$

Thus, by factoring out the components, we get,

$$
\begin{align} \nabla_\rho \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} & = \partial_\rho \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_q} \\
  & + \sum_{i=1}^p \Gamma^{\mu_i}_{\phantom{\mu_i} \rho \sigma} \phi^{\mu_1 \dots \mu_{i-1} \sigma \mu_{i+1} \dots \mu_p}_{\phantom{\mu_1 \dots \mu_{i-1} \sigma \mu_{i+1} \dots \mu_p} \nu_1 \dots \nu_q} \\
  & - \sum_{j=1}^q \Gamma^{\sigma}_{\phantom{\sigma} \rho \nu_j} \phi^{\mu_1 \dots \mu_p}_{\phantom{\mu_1 \dots \mu_p} \nu_1 \dots \nu_{j-1} \sigma \nu_{j+1} \dots \nu_q}
\end{align}
$$

## Tuple index notation

To make the above notation less messy, let us use the tuple index notation:

1. Tuples of indices are replaced by their capital letter. For example, $$\mu_1 \dots \mu_p$$ becomes $$M$$ and $$\nu_1 \dots \nu_q$$ becomes $$N$$.

2. A subset of a tuple, running up to some index $$i-1 \leq p$$, i.e. $$\mu_1 \dots \mu_{i-1}$$, is written as $$M_i^-$$. Similarly, a subset running from an index $$i+1$$, $$\mu_{i+1} \dots \mu_p$$ is written as $$M_i^+$$.

3. From the above, it follows that we can write $$M$$ as $$M_i^- \mu_i M_i^+$$.

Using the above notation, we can write the covariant derivative of a tensor, in the component form, as,

$$\nabla_\rho \phi^M_{\phantom{M} N} = \partial_\rho \phi^M_{\phantom{M} N} + \sum_{i=1}^p \Gamma^{\mu_i}_{\phantom{\mu_i} \rho \sigma} \phi^{M_i^- \sigma M_i^+}_{\phantom{M_i^- \sigma M_i^+} N} - \sum_{j=1}^q \Gamma^\sigma_{\phantom{\sigma} \rho \nu_j} \phi^M_{\phantom{M} N_j^- \sigma N_j^+}$$
