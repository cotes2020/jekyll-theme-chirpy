---
title: "Harmonic Oscillators in Scalar Field Theory"
description: "How harmonic oscillators emerge in scalar field theory"
categories: [classical field theory]
tags: [harmonic oscillators, energy-momentum tensor, linearity]
---

Once again, let us see how classical scalar field theory naturally gives rise to common natural phenomena.  Today, we will derive the evolution of harmonic oscillators.

## Scalar form

### Scalar fields

Given a scalar field (in flat spacetime) $$\phi \left( x^\rho \right)$$ with the following Lagrangian density,

$$\mathcal{L} = \frac{1}{2} \partial_\alpha \phi \: \eta^{\alpha \beta} \: \partial_\beta \phi - V \left( \phi \right)$$

We obtain the equation of motion,

$$\square \phi + \frac{\partial V}{\partial \phi} = 0$$

### Klein-Gordon equation

Taylor-expanding the term $$\frac{\partial V}{\partial \phi} = V^\prime \left( \phi \right)$$,

$$V^\prime \left( \phi \right) = \sum_{k=0}^\infty \frac{1}{k!} \: V^{\left( k+1 \right)} \left( 0 \right) \: \phi^k$$

The equation of motion we have seen previously is a wave equation, i.e. its solutions are oscillating waves. If we are given a solution $$\phi$$ with a small amplitude so that the square and higher powers of $$\phi$$ become negligible, $$V^\prime \left( \phi \right)$$ assumes the form,

$$
\begin{align}
V^\prime \left( \phi \right) & \approx \sum_{k=0}^1 \frac{1}{k!} \: V^{\left( k+1 \right)} \left( 0 \right) \: \phi^k \\
 & = V^{\prime} \left( 0 \right) + V^{\prime \prime} \left( 0 \right) \: \phi
\end{align}
$$

If the oscillating field $$\phi$$ is stable, it must vibrate about a local minima of the potential. Without loss of generality, let this minima exist at $$\phi = 0$$ (if it does not, we can add to $$\phi$$ a constant scalar field so that the minima exists at $$\phi_{\text{new}} = 0$$, without altering the equations of motion). Hence, setting $$V^\prime \left( 0 \right) = 0$$,

$$V^\prime \left( \phi \right) = V^{\prime \prime} \left( 0 \right) \: \phi$$

Since we are considering the limiting case where the amplitude $$\phi_{\text{max}} \to 0$$, the equality sign above is appropriate. Plugging the result into the equation of motion,

$$
\begin{aligned}
\square \phi + V^{\prime \prime} \left( 0 \right) \: \phi & = 0 & \text{i.e.,} \\
\square \phi + m^2 \phi & = 0
\end{aligned}
$$

The above equation is known as the Klein-Gordon equation, named after its discoverers. $$V^{\prime \prime} \left( 0 \right)$$ is analogous to a spring constant $$m^2$$, which is the force $$- \frac{\partial V}{\partial \phi}$$ per unit displacement $$\phi$$. The reason for writing the spring constant as $$m^2$$ and not as $$m$$ will become apparent later.

### Linearity

The Klein-Gordon equation is linear, i.e. valid solutions labelled as $$\phi_{\left( i \right)}$$ may be added to yield new valid solutions, as long as they have the same $$m^2$$ (i.e. the same $$m$$ upto magnitude),

$$
\begin{aligned}
\square \sum_i \phi_{\left( i \right)} & = \sum_i \square \phi_{\left( i \right)} \\
 & = \sum_i m^2 \phi_{\left( i \right)} \\
 & = m^2 \sum_i \phi_{\left( i \right)}
\end{aligned}
$$

## Tensor form

### Stress-energy tensor

Whilst the scalar form of the Klein-Gordon equation is simple, a certain sense of direction is ingrained in its mechanism, which is not directly encoded in the scalar form. Consider a generalized spring: when it is displaced from the stable configuration by a certain amount, a proportional restoring force acts in the opposite direction of the displacement. Even though a minus sign in the expression $$\square \phi = - k \phi$$ indicates this, the formula is not manifestly vectorial, or higher-order (which is required to contain explicit information on direction).

To fix this, we will derive the evolution of harmonic oscillators in a different manner. A complete description of a field's momentum (which has a sense of direction with respect to the field's dynamics) is provided by its _stress-energy tensor_ $$T_{\mu \nu}$$,

$$T_{\mu \nu} = \eta_{\mu \rho} \frac{\partial \mathcal{L}}{\partial \left( \partial_\rho \phi \right)} \partial_\nu \phi - \mathcal{L} \: \eta_{\mu \nu}$$

The above tensor satisfies the quintessential properties of a stress-energy tensor: symmetry, conservation and the dimensions of energy density. Now, from the Lagrangian for our scalar field,

$$
\begin{align}
T_{\mu \nu} & = \eta_{\mu \rho} \: \partial^\rho \phi \: \partial_\nu \phi - \left( \frac{1}{2} \partial_\alpha \phi \: \partial^\alpha \phi - V \left( \phi \right) \right) \eta_{\mu \nu} \\
T_{\mu \nu} & = \partial_\mu \phi \: \partial_\nu \phi - \left( \frac{1}{2} \partial_\alpha \phi \: \partial^\alpha \phi - V \left( \phi \right) \right) \eta_{\mu \nu}
\end{align}
$$

The trace $$T$$ of the above stress-energy tensor is,

$$
\begin{aligned}
T & = \eta^{\mu \nu} \: T_{\mu \nu} \\
 & = \eta^{\mu \nu} \left[ \partial_\mu \phi \: \partial_\nu \phi - \left( \frac{1}{2} \partial_\alpha \phi \: \partial^\alpha \phi - V \left( \phi \right) \right) \eta_{\mu \nu} \right] \\
 & = \eta^{\mu \nu} \partial_\mu \phi \: \partial_\nu \phi - \eta_{\mu \nu} \eta^{\mu \nu} \left( \frac{1}{2} \partial_\alpha \phi \: \partial^\alpha \phi - V \left( \phi \right) \right) \\
 & = \partial_\alpha \phi \: \partial^\alpha \phi - 4 \left( \frac{1}{2} \partial_\alpha \phi \: \partial^\alpha \phi - V \left( \phi \right) \right) \\
 & = \partial_\alpha \phi \: \partial^\alpha \phi - 2 \: \partial_\alpha \phi \: \partial^\alpha \phi + 4 V \left( \phi \right) \\
\end{aligned}
$$

Or,

$$T = 4 V \left( \phi \right) - \partial_\alpha \phi \: \partial^\alpha \phi$$

### Cauchy strain tensor

For infinitesimal displacements of a continuous field $$\phi$$, its deformation is encoded in the _Cauchy strain tensor_ $$\epsilon_{\rho \sigma}$$,

$$\epsilon_{\rho \sigma} = \partial_{( \rho} \: u_{\sigma )} = \frac{1}{2} \left( \partial_\rho u_\sigma + \partial_\sigma u_\rho \right)$$

where $$u_\mu$$ is the displacement gradient defined as,

$$
\begin{aligned}
u_\mu & = \text{d} \phi \left( \frac{d}{dx^\mu} \right) \\
 & = \frac{d \phi}{dx^\mu} \\
\end{aligned}
$$

Assuming $$\phi$$ is not velocity-dependent,

$$
\begin{aligned}
u_\mu & = \frac{d \phi}{dx^\mu} \\
 & = \frac{dx^\nu}{dx^\mu} \partial_\nu \phi \\
 & = \delta^\nu_{\phantom{\nu} \mu} \: \partial_\nu \phi \\
 & = \partial_\mu \phi
\end{aligned}
$$

Therefore, the Cauchy strain tensor takes the form,

$$\epsilon_{\rho \sigma} = \frac{1}{2} \left( \partial_\rho \partial_\sigma \phi + \partial_\sigma \partial_\rho \phi \right)$$

The trace of the above tensor is a coordinate-independent scalar $$\epsilon$$,

$$
\begin{aligned}
\epsilon & = \eta^{\rho \sigma} \epsilon_{\rho \sigma} \\
 & = \frac{1}{2} \eta^{\rho \sigma} \left( \partial_\rho \partial_\sigma \phi + \partial_\sigma \partial_\rho \phi \right) \\
 & = \frac{1}{2} \left( \eta^{\rho \sigma} \partial_\rho \partial_\sigma \phi + \eta^{\rho \sigma} \partial_\sigma \partial_\rho \phi \right) \\
 & = \eta^{\rho \sigma} \partial_\rho \partial_\sigma \phi \\
\end{aligned}
$$

Or,

$$\epsilon = \square \phi$$

### Solutions as eigenfunctions of partial derivatives

Once, again, we will consider solutions of $$\phi$$ with small amplitude, so that,

$$
\begin{aligned}
\frac{\partial V}{\partial \phi} & \approx V^{\prime \prime} \left( 0 \right) \phi = m^2 \phi \\
\square \phi & \approx - m^2 \phi
\end{aligned}
$$

Therefore, we can simplify the trace of the stress-energy tensor as,

$$
\begin{aligned}
T & = 4 V \left( \phi \right) - \partial_\alpha \phi \: \partial^\alpha \phi \\
 & = 4 \int \frac{\partial V}{\partial \phi} d \phi - \partial_\alpha \phi \: \partial^\alpha \phi \\
 & = 4 \left( \frac{1}{2} m^2 \phi^2 \right) - \partial_\alpha \phi \: \partial^\alpha \phi \\
 & \approx - \partial_\alpha \phi \: \partial^\alpha \phi
\end{aligned}
$$

The arbitrary constant in $$\displaystyle{\int \frac{\partial V}{\partial \phi} d \phi}$$ may be neglected as the equation of motion remains invariant under adding a constant to $$V \left( \phi \right)$$. Therefore, the term vanishes as $$\phi_{\text{max}}$$, being negligible, ensures that $$\phi^2$$ vanishes too.

Now, we must solve for $$\partial_\alpha \phi \: \partial^\alpha \phi$$. There is no nice substitution which gives some known term from the equation of motion. However, we may make use of the linearity of the fields obtained from Hooke's law. Suppose in the statement, $$\square \phi = - m^2 \phi$$, we say $$m^2 = z^\mu z_\mu$$ where the first and higher derivatives of $$k^\mu$$ vanish. $$z^\mu$$ is therefore some constant vector whose magnitude in Minkowski space is $$m$$. This is why we wrote the spring constant as $$m^2$$; had we written it as $$m$$, we would have to substitute $$\sqrt{k^\mu k_\mu}$$ for it in the equation of motion, spoiling linearity as seen in the derivation below. But the Klein-Gordon equation was derived from the insignificance of second-order and higher terms to begin with, so such a procedure would lead to ambiguity.

Substituting $$m^2 = z^\mu z_\mu$$ in the equation of motion,

$$\eta^{\alpha \beta} \: \partial_\alpha \partial_\beta \phi = - \eta^{\alpha \beta} z_\alpha z_\beta \phi$$

By linear independence of different components of $$z_\mu$$,

$$\partial_\alpha \partial_\beta \phi = - z_\alpha z_\beta \phi$$

This severely restricts the $$k^\mu$$ we choose. Furthermore, $$\phi$$ can now be solved in terms of $$k_\mu$$ and the only sensible way of writing its first derivative, judging by the second derivative and its indices, is,

$$\partial_\alpha \phi = i z_\alpha \phi$$

where $$i^2 = -1$$. This may seem unjustified, so let us use tensors explicitly so that the restriction follows naturally. We begin by stating the first substitution a bit more rigorously,

$$
\begin{aligned}
\left( \partial_\alpha \otimes \partial_\beta \right) \phi = - \phi \left( z_\alpha \otimes z_\beta \right)
\end{aligned}
$$

The above is true throughout spacetime only if $$\partial_\alpha \phi = z_\alpha \: i \left( \phi, \cdot \right)$$ where $$i$$ is a $$\left( 0, 0 \right)$$ tensor obeying the identity,

$$i \left( \phi, \phi \right) = - \phi$$

Therefore, $$\partial_\alpha \left( \phi \right) = i z_\alpha \cdot \phi$$. I.e., all $$\phi$$ are eigenfunctions of all $$\partial_\alpha$$.

### Hooke's law, upgraded

From the above results, we have,

$$
\begin{aligned}
T & = - \partial_\alpha \phi \: \partial^\alpha \phi \\
 & = - \left( i z_\alpha \: \phi \right) \left( i z^\alpha \: \phi \right) \\
 & = - i^2 \: z_\alpha z^\alpha \phi \\
 & = m^2 \phi
\end{aligned}
$$

We can also simplify $$\text{tr} \left( \epsilon_{\rho \sigma} \right) = \epsilon$$ in the limit of small oscillations,

$$
\begin{aligned}
\epsilon & = \square \phi \\
 & = - m^2 \phi
\end{aligned}
$$

Clearly, $$T = - \epsilon$$. This can be true only if $$T_{\mu \nu}$$ and $$\epsilon_{\rho \sigma}$$ have a multilinear relationship via a $$\left( 2, 2 \right)$$ tensor:

$$T_{\mu \nu} = - C^{\rho \sigma}_{\phantom{\rho \sigma} \mu \nu} \: \epsilon_{\rho \sigma}$$

The tensor $$C^{\rho \sigma}_{\phantom{\rho \sigma} \mu \nu}$$ is known as the elasticity tensor of $$\phi$$ which, like $$k$$, has vanishing derivatives.

## Conclusion

We have derived the evolution equation of harmonic oscillators, in tensorial form, at long last! But was it necessary at all?

If our goal was to derive the existence of harmonic oscillators using simple assumptions and geometric tools, the answer is yes. The assumptions we made implicitly were:

1. The field theory should be scalar, so that internal degrees of freedom do not become important in considerations of energy.

2. The field theory should be Lorentz invariant with a second-order equation of motion, so that the relativity of position and velocity are accounted for.

3. All the fancy assumptions involving $$\phi_{\text{max}} \approx 0$$.

Given the above conditions, harmonic osillators are bound to arise very naturally.
