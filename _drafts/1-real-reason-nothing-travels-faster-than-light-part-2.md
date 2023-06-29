---
title: "The Real Reason Nothing Travels Faster Than Light: Part 2 (Dynamical Interpretation)"
description: "Interpreting the dynamics of the famous statement"
categories: [relativity]
tags: [4-momentum]
---

In [The Real Reason Nothing Travels Faster Than Light]({% post_url 2022-04-16-real-reason-nothing-travels-faster-than-light %}), we analytically proved that faster-than-light motion is logically impossible within the framework of special relativity. In doing so, we disproved the very kinematics of the situation $v>c$. This, as we discussed, is logically stronger than disproving the dynamics of the situation, i.e. the question of how such a velocity can be achieved through, say, acceleration. In this post, we will specifically look at the dynamical aspect, and how it agrees with the stronger kinematical argument.

When it comes to this topic, there are multiple layers of common misconception. The first is the supposition that disproving the dynamics also disproves the kinematics, i.e. if one can show that faster-than-light velocity cannot be achieved via any physical process, it must be theoretically impossible given the working of spacetime itself. While such an implication would be nice, it is not always true. After all, no object can be accelerated to the speed of light, and yet, light travels at, well, the speed of light! This makes sense in special relativity because things which are travelling at the speed of light cannot be accelerated in the first place, by virtue of having an invariant speed. In other words, the argument 'impossible dynamics implies impossible kinematics' implicitly assumes $v < c$, which in turn assumes that we are dealing with objects carrying positive mass. Thus, the argument fails for $v \geq c$ and we are forced to base kinematical arguments solely on the kinematics as opposed to the dynamics.

The second layer of popular misconception is the notion that theoretically, one would strictly need infinite energy to accelerate a body of positive mass from $v<c$ to $v = c$. It turns out that this is not true! In general, a body would need to convert all its mass to relativistic kinetic energy in order to approach the speed of light — but that means it would cease to exist as matter by the time it reaches the speed of light! [^1]

[^1]: However, it is true that if the body were to be accelerated to the speed of light _without_ a change in its mass, it would require infinite energy. However, this case leaves out the possibility of the body propelling itself, which does not necessarily involve infinite energy.

In the [first part](#relativistic-kinetic-energy) of this post, we will construct and geometrically interpret 4-momentum and relativistic kinetic energy for particles. In the [second part](#propulsion), we will apply these dynamical quantities to understand relativistic propulsion.

## Relativistic kinetic energy

### Constructing 4-momentum

We begin by defining a 4-vector called 4-energy-momentum or 4-momentum in short. Analogous to 3-momentum, it is defined as,

$$P^\mu = m U^\mu = m \frac{d X^\mu}{d \tau}$$

where $X^\mu$ represents the components of 4-position, which in 'laboratory' coordinates is of the form,

$$X^\mu \pmb{e}_\mu = ct \: \pmb{e}_0 + x^i \pmb{e}_i$$

In a holonomic basis, we have $\pmb{e}_0 = \frac{1}{c} \partial_t$ and $\pmb{e}_i = \partial_i$.

Another way to think of 4-momentum is that it is the conjugate momentum obtained from an appropriate action encoding dynamics in spacetime. For particles, this action would be of the form,

$$S = \int_{\tau_1}^{\tau_2} mc^2 d \tau$$

The intuition for the above action lies in the following facts:

1. The Lagrangian must be a scalar related to the body's dynamics. It must also be additive when combining non-interacting particles into larger systems. These severe restrictions leave only one kind of choice for the Lagrangian — the mass of the particle, up to linearity.

2. The Lagrangian must be generalizable to general relativity. In general relativity, test particles travel along geodesics, which extremize the spacetime interval. This must therefore also apply to special relativity.

These ideas suggest that the action along some path $C$ with the endpoints having proper times $\tau_1$ and $\tau_2$ resembles the following:

$$
\begin{align}
S & = \int_C mc \: ds \\
& = \int_{\tau_1}^{\tau_2} mc \left( \frac{dX_\mu}{d \tau} \frac{dX^\mu}{d \tau} \right)^\frac{1}{2} d \tau \\
& = \int_{\tau_1}^{\tau_2} mc^2 d \tau
\end{align}
$$

Thus, the appropriate conjugate momentum is of the form,

$$
\begin{align}
P^\mu & = \eta^{\mu \nu} \frac{\partial L}{\partial U^\nu} \\
& = \eta^{\mu \nu} \frac{\partial}{\partial U^\nu} \left( mc \left( \frac{dX_\mu}{d \tau} \frac{dX^\mu}{d \tau} \right)^\frac{1}{2} \right) \\
& = mc \eta^{\mu \nu} \frac{\partial}{\partial U^\nu} \left( U_\rho U^\rho \right)^\frac{1}{2} \\
& = mc \eta^{\mu \nu} \cdot \frac{1}{2} \left( U_\rho U^\rho \right)^{- \frac{1}{2}} \cdot 2 U_\rho \delta^\rho_{\phantom{\rho} \nu} \\
& = m c \eta^{\mu \nu} \cdot \frac{1}{c} U_\nu \\
& = m U^\mu
\end{align}
$$

### Interpreting 4-momentum

4-momentum may be interpreted as a current vector field associated with mass, in spacetime. Such a current vector field, associated with some body of mass $m$, may be defined as the vector field $J^\mu \left( x^\nu \right)$ characterized by:

1. A magnitude in spacetime, $J$, which equals the amount of mass passing in unit proper time through the hypersurface of simulteinity $\Sigma$ (in a given frame) passing through $x^\nu$.

2. Having the same direction as the motion of the mass at each point, i.e. having the same direction as the velocity vector field $U^\mu$.

By construction, the mass current vector is its magnitude times the unit vector along 4-velocity, $\frac{1}{c} U^\mu$.

In a small proper time $d \tau$, the amount of mass $dm$ which passes through $\Sigma$ is the mass of the traversing particle, $m$, times the fraction of it that passes through $\Sigma$. This fraction is $\frac{dX}{L}$ where $dX$ is the distance travelled in $d \tau$ by the particle perpendicular to $\Sigma$ and $L$ is the proper length of the particle perpendicular to $\Sigma$. Furthermore, $dX = c d \tau$. We therefore have,

$$
\begin{align}
dm & = \frac{mc}{L} d \tau \\
J^\mu & = \frac{dm}{d \tau} \cdot \frac{1}{c} U^\mu \\
& = \frac{mc}{L} \cdot \frac{1}{c} U^\mu \\
& = \frac{1}{L} m U^\mu \\
& = \frac{1}{L} P^\mu \\
\implies P^\mu & = L J^\mu
\end{align}
$$

Thus, the 4-momentum of a particle travelling in some direction (with a non-zero proper length along it) is simply its current vector for mass scaled by the proper length of the particle in its direction of motion. For particles with zero proper length along the direction of motion, we need to be more careful.

If a particle is completely localized, it has no length in any direction, i.e. $L = 0$ in every frame. Its mass is also localized, resulting in its associated mass density field resembling a Dirac delta distribution:

$$\rho \left( x^\mu; \tau \right) = m \delta \left( x^\mu - X^\mu \left( \tau \right) \right)$$

so that the total mass is as expected,

$$
\begin{align}
\int_{\mathbb{R}^4} dx^4 \rho \left( x^\mu; \tau \right) & = \int_{\mathbb{R}^4} dx^4 m \delta \left( x^\mu - X^\mu \left( \tau \right) \right) \\
& = m
\end{align}
$$

For a completely lozalized particle, its length along any direction, in the context of motion, may be taken as $d X$. The intuition for this is that no matter how small a proper time $d \tau$ we pick, the length $c d \tau$ of the particle which passes through $\Sigma$, is always the whole of $L$. This allows us to take $L = c d \tau = dX$. The particle's 4-momentum is then,

$$
\begin{align}
P^\mu & = L J^\mu \\
& = dX \cdot \frac{dm}{d \tau} \cdot \frac{1}{c} U^\mu \\
& = dX \cdot \frac{1}{d \tau} \left( \frac{mc}{dX} d \tau \right) \cdot \frac{1}{U} U^\mu \\
& = m U^\mu
\end{align}
$$

Thus, even a completely lozalized particle has 4-momentum, simply equal to its current vector for mass.

### Interpreting relativistic kinetic energy

An important fact about 4-momentum is that since it is 4-velocity scaled by mass and 4-velocity is normalized to $c^2$, we have,

$$P_\mu P^\mu = \left( mc \right)^2$$

This is known as the energy-momentum relation. To justify it, we first write the above equation in laboratory coordinates (essentially those in which $\eta_{0 i}$ disappear, $\eta_{0 0}$ can be normalized to $1$ and $\eta_{i j}$ is some induced metric $- g_{i j}$ for space):

$$P_0 P^0 - g_{i j} P^i P^j = \left( mc \right)^2$$

Since we have $P^i = m U^i = \gamma m v^i$,

$$\left( P^0 \right)^2 = \left( mc \right)^2 + \left( \gamma p \right)^2$$

where $p = \left( g_{i j} P^i P^j \right)^\frac{1}{2} = - \left( P_i P^i \right)^\frac{1}{2}$. As will be seen soon, it is useful to bring both sides of the above equation to the units of energy, which can be done by multiplying by $c^2$,

$$\left( c P^0 \right)^2 = \left( mc^2 \right)^2 + \left( \gamma p c \right)^2$$

One way to understand the above equation is to replace $P^i$s with $J^i$s, since we saw that 4-momentum and the mass current vector of a point particle are one and the same despite being apparently different concepts:

$$\left( c J^0 \right)^2 = \left( m c^2 \right)^2 + \left( jc \right)^2$$

where $j = - \left( P_i P^i \right)^\frac{1}{2} = \left( g_{i j} P^i P^j \right)^\frac{1}{2} = \gamma \left( g_{i j} p^i p^j \right)^\frac{1}{2} = \gamma p$ for particles not moving at the speed of light. [^2]

[^2]: Here, the convention is that lowercase letters for 4-vectors denote non-relativistic vectors, whereas uppercase ones denote 4-vectors. Conversion from one to the another, is, therefore, done appropriately via the Lorentz factor.

Einstein, by conceptually investigating the original form of the energy-momentum relation, interpreted the left hand side of the equation as the relativistic kinetic energy of a particle, by virtue of its motion in spacetime. Related to his interpretation is a power series approach to which we shall come soon. Before treading there, let us see what the energy-momentum relation tells us in the 'mass current form':

$$E^2 = \left( mc^2 \right)^2 + \left( j c \right)^2$$

Clearly, in the rest frame of a particle, $j=0$ and we have,

$$
\begin{align}
E & = mc^2 \\
& = mc \cdot c \\
& = c \cdot m U^0 \\
& = c J^0
\end{align}
$$

In the above relationships, we have three objects of interest: the so-called relativistic kinetic energy $E$, mass $m$ and the timelike component of mass current. 

Firstly, we notice that $m = \frac{1}{c} J^0$, indicating that measuring the mass of a body corresponds to measuring its flow of matter through time, encoded in the timelike part of the mass current vector. [^3] Secondly, $E = cJ^0$, showing that $E$ is imparted by the motion of mass along time. Last but not the least, the previous notion is expressed in terms of the mass flowing in time, by the celebrated equation $E = mc^2$. 

[^3]: An interesting result of this idea is that an object moving at the speed of light cannot have mass. This is because of the following. Such a body would not appear to move along proper time. Therefore, there _is_ no event where some non-zero $dm$ passes through some hypersurface $\Sigma$ in a small time interval $dt$, in inertial frames with $v<0$ ([a Lorentz-invariant statement]({% post_url 2022-04-16-real-reason-nothing-travels-faster-than-light %}#timelike-world-line)). This suggests that the timelike part of mass current is null.

In a frame moving with a velocity $v << c$ relative to the particle, we have $\frac{v}{c} \approx 0$, yielding:

$$
\begin{align}
E^2 & = \left( mc^2 \right)^2 + \left( \gamma m v c \right)^2 \\
& = \left( mc^2 \right)^2 \left[ 1 + \frac{\gamma^2 v^2}{c^2} \right] \\
& = \left[ 1 + \frac{v^2}{c^2} \left( 1 - \frac{v^2}{c^2} \right) \right] \left( mc^2 \right)^2 \\
& = \left( 1 + \frac{v^2}{c^2} - \frac{v^4}{c^4} \right) \left( mc^2 \right)^2 \\
& \approx \left( 1 + \frac{v^2}{c^2} + \frac{1}{4} \frac{v^4}{c^4} \right) \left( mc^2 \right)^2 \\
& = \left[ \left( 1 + \frac{1}{2} \frac{v^2}{c^2} \right) mc^2 \right]^2 \\
\implies E & \approx \left( 1 + \frac{1}{2} \frac{v^2}{c^2} \right) mc^2 \\
& = mc^2 + \frac{1}{2} mv^2
\end{align}
$$

Thus, in the non-relativistic limit, the relativistic kinetic energy still contains the non-relativistic kinetic energy.

For all of the above reasons, it is appropriate to interpret $E$ as relativistic kinetic energy by virtue of motion in spacetime.

## Propulsion

### Relativistic Tsiolkovsky rocket equation

An ideal relativistic model can be modelled in the following manner:

1. It gains 3-speed by losing mass continuously with a 3-speed $u$ in its frame.

2. The above gain in velocity happens due to the conservation of total 4-momentum of the rocket and expelled mass.

