---
title: "The Real Reason Nothing Travels Faster Than Light"
description: "Proving the famous statement analytically"
categories: [relativity]
tags: [spacetime interval]
---

## The problem

One of the celebrated results of Einstein's special theory of relativity is that no physical object can travel faster than light, no matter the frame of reference. A popular explanation for this is that as an object approaches the speed of light, its relativistic energy $$E = \gamma mc^2$$ [^1] tends to infinity. Thus, it would require infinite energy to propel a body (with mass, at least) to the speed of light or beyond.

[^1]: Here, $$\gamma = \frac{1}{\sqrt{1 - \frac{v^2}{c^2}}}$$ is the Lorentz factor, $$m$$ is the mass of the body (a Lorentz invariant) and $$c$$ is the speed of light in vacuum.

However, as a thought experiment, what if we consider a body with mass that is somehow _already_ travelling at the speed of light or even faster? What if we do not concern ourselves, at least for the moment, with the origin of this hypothetical motion (i.e. the dynamics), but only look at the situation (the kinematics)? If we can disprove the kinematics, it is stronger than disproving the dynamics as we are eliminating the very possibility of a body travelling faster than light, irrelevant of any apparent methods to achieve the same.

A popular approach to disprove the kinematics of faster-than-light motion is to calculate observables such as the body's relativistic energy $$E = \gamma mc^2$$ and momentum $$p = \gamma m v$$, and demonstrate that their values make no physical sense (they turn out to be imaginary as $$\gamma$$ is imaginary for $$v>c$$).

Yet, this philosophy is not satisfying in that there are instances where the breaking down of mathematics does not necessarily indicate the breaking down of physics. Here is an analogy: consider a straight line in a two-dimensional Cartesian coordinate system with the equation of motion $$y = mx + C$$. When the line is vertical with respect to the x-axis, we have $$m = \pm \infty$$, which is absurd. But that does not mean a line vertical with respect to a chosen axis is a geometrical impossibility. In fact, on being careful, one observes that the derivation of $$y=mx+C$$ invariably assumes in some form that the line in question is _not_ vertical. It is not surprising that this equation may, therefore, break down in a scenario excluded from the original assumptions.

Similarly, the flaw in the following logic,

$$\gamma \left( v \right) \notin \left[ 1, \infty \right) \forall \: v>c \implies v \ngtr c$$

is that perhaps the way measurable quantities like $$E$$ and $$p$$ depend on $$\gamma$$ is an artefact of knowingly or unknowingly assuming $$v<c$$. Unless we are very cautious, there is freedom for 'reality' being something like (for particles with mass): [^2]

[^2]: For particles without mass, $$E=pc$$ from the energy-momentum relation $$E^2 = m^2 c^4 + p^2 c^2$$. As their energy is completely kinetic, these particles only travel along null world lines i.e. with $$v=c$$.

$$E \left( v \right) = \begin{cases} \gamma mc^2 & v < c \\ E_+ \left( v \right) & v>c \end{cases}$$

$$\lim_{v \to c^-} E \left( v \right) = \lim_{v \to c^+} E_+ \left( v \right) = \infty$$

$$E \in \mathbb{R}^+ \cup \left\{ 0 \right\}$$

Note that we have assumed $$E \left( v \right)$$ is non-negative throughout, for it to represent a physically sensible total energy. In this overall picture, it is not surprising that $$E \left( v \right)$$ for $$v>c$$ may assume strange values if we forcefully plug in such $$v$$'s for the $$v < c$$ case.

Another way to think of the whole problem is that the manner in which we assign physical meaning to observables based on their values, is tautological — it tells us nothing about the universe in itself. On the other hand, physical arguments have an inherent logical structure that prevails 'beneath' the observables. To analytically demonstrate the incoherence of an observable with a certain value, we must investigate the corresponding physical scenario and find its logical inconsistency by reasoning. Otherwise, there always exists the pathological possibility that we have overlooked certain assumptions which are leading to the mathematical breakdown (and _not_ the physics).

Therefore, in this post, we will develop such an analytical proof for the statement 'no body can travel faster than light', within the framework of special relativity. But before building the [main argument]({% post_url 2022-04-16-real-reason-nothing-travels-faster-than-light %}#the-paradox), let us review some key concepts.

## Spacetime interval

Besides the principle of relativity, the other important postulate of special relativity is that the speed of light in vacuum, $$c$$, is the same in all inertial frames.

Let us work in local coordinates $$x^\mu = \left( ct, x^i \right)$$. Suppose we track a pulse of light with the coordinates $$x^i$$. Then, we have,

$$
\begin{align}
c^2 & = \frac{dx_i}{dt} \frac{dx^i}{dt} \\
c^2 dt^2 & = dx_i dx^i \\
c^2 dt^2 - dx_i dx^i & = 0
\end{align}
$$

For $$c$$ to be a Lorentz invariant, the above equation must be true in all inertial frames, i.e.,

$$c^2 dt^2 - dx_i dx^i = c^2 d{t^\prime}^2 - dx_{i^\prime} dx^{i^\prime} = 0$$

The above invariant is precisely the infinitesimal spacetime interval squared,

$$ds^2 = dx_\nu dx^\nu = dx^\mu \eta_{\mu \nu} dx^\nu$$

if we define the metric tensor $$\eta_{\mu \nu}$$ in local coordinates as, [^3]

$$\begin{align} \eta_{00} & = 1 \\ \eta_{ij} & = - g_{ij} \\ \eta_{i 0} & = \eta_{0 i} = 0 \end{align}$$

[^3]: This is the 'mostly minus' or particle physicist's sign convention for the metric, $$\left( +, -, -, - \right)$$. The opposite choice is the 'mostly plus' or relativist's sign convention.

where $$g_{ij}$$ is the spatial metric tensor for the chosen coordinate system. The last equation says that in local coordinates, space and time are orthogonal. This is true in all inertial frames (in local coordinates). Hence, the timelike and spacetimelike components of the metric are invariant, although the spacelike components $$-g_{ij}$$ transform like a rank-2 tensor.

On the other hand, in arbitrary coordinates, $$\eta_{\mu \nu}$$ transforms as a whole like the components of a rank-2 tensor. Therefore, $$ds^2 = dx^\mu \eta_{\mu \nu} dx^\nu$$ is always invariant, even when it is not zero. This also allows us to meaningfully venture beyond the case $$ds^2 = 0$$ and consider $$ds^2 \neq 0$$.

Note that given two events A and B, the spacetime interval between them is defined in the manner,

$$\Delta s^2 = \int_A^B ds^2 = \Delta_A^B x_\mu \Delta_A^B x^\mu$$

Let us now investigate when $$ds^2$$ is zero or non-zero, and what these physically represent.

## World lines

We now generalize our problem from tracking a pulse of light to tracking any physical body, with the coordinates $$x^i$$. Then, the world line (trajectory in spacetime) of a body is defined, in local coordinates, as the function $$x^i \left( x^0 \right) = x^i \left( ct \right)$$. [^4]

[^4]: In general, a world line is of the form $$x^i \left( \lambda \right)$$ where $$\lambda$$ is any scalar parameter (not necessarily time).

### Null world lines

A null world line is one along which $$ds^2$$ is null, or zero. As we have already seen, $$ds^2=0$$ for bodies travelling at the speed of light $$c$$.

### Timelike world line

Timelike world lines satisfy $$ds^2 > 0$$. They are called 'timelike' as $$ds^2 = c^2 dt^2 - dx_i dx^i >0$$ when the timelike part $$c^2 dt^2$$ is greater than the spacelike part $$dx_i dx^i$$.

As $$ds^2$$ is an invariant, if $$ds^2>0$$ in one inertial frame, so must it be in all other frames. In other words, a timelike world line is timelike no matter the frame of reference.

### Spacelike world line

Similar to timelike world lines, we may define spacelike world lines for which $$ds^2<0$$ (the spacelike part of $$ds^2$$ is greater than the timelike part $$c^2 dt^2$$).

Again, if $$ds^2<0$$ in one frame, this must be true in all frames.

## The paradox

Consider a body moving along a spacelike world line in some frame. We have,

$$ds^2 = c^2 dt^2 - dx_i dx^i < 0$$

Now, we rewrite the spacelike part as,

$$
\begin{align}
dx_i dx^i & = \frac{dx_i}{dt} \frac{dx^i}{dt} dt^2 \\
 & = v^2 dt^2
\end{align}
$$

Therefore, the infinitesimal spacetime interval squared is,

$$
\begin{align}
ds^2 & = c^2 dt^2 - v^2 dt^2 \\
ds^2 & = \left( c^2 - v^2 \right) dt^2
\end{align}
$$

In the body's own frame, $$v=0$$ (as the body is at rest relative to itself). Hence,

$$ds^2 = c^2 dt^2 \geq 0$$

However, $$ds^2 < 0$$ in the original frame, and it must be true in all frames, including the body's own! This results in a paradox which can only be resolved by concluding that the assumption that there can exist a frame where $$ds^2 < 0$$, is in fact _false_.

We do not see this contradiction in the cases $$ds^2 < 0$$ and $$ds^2 = 0$$. In the latter case, however, we need to be careful as it does not make sense to consider observables in the body's own frame, when it is travelling at the speed of light in some reference frame (as frames moving along null trajectories are not truly reference frames).

Thus, we can safely say that indeed, no body can travel faster than light in any inertial frame, as it not only breaks mathematics, but also logic!

## Conclusion (instantaneously inertial frames)

There is some unfinished business in our argument. The argument primarily stands on the invariance of $$ds^2$$, derived from the invariance of the speed of light in _inertial_ frames. The body we are tracking may very well be moving non-uniformly, which begs the question of why the sign of $$ds^2$$ for its world line must be the same in an inertial reference frame, and its own not-necessarily-inertial frame.

This is where the idea of instantaneously inertial frames comes in. If we 'split' the non-uniform motion of a body into infinitesimal steps, the motion along each step behaves just like uniform motion.

Or, we can construct the argument in the reverse fashion. Suppose we restrict ourselves to uniform motion. Then, the correctness of our argument is straightforward as the body's own frame is inertial. Now, nothing stops us from considering inertial motion lasting for tiny time intervals, with acceleration in between. For all these intervals, the law 'the body cannot travel faster than light' can be verified individually. As the 'gaps' of acceleration tend to zero, the motion of the body tends to continuous non-uniform motion, and the law holds at each instant.

Of course, rigorously demonstrating the above formally involves calculus as we cannot simply assume that the discontinuous motion in question, does, in fact, reduce to continuous motion in the appropriate limit. The entire exercise, if done rigorously, is unexpectedly rich as it involves measure theory, analysis and topology! I will perhaps explore it in the future in a series of posts.

Until then, I leave it to the reader to question how much rigorour is necessary to prove, in a foolproof manner, that no physical body can travel faster than light. Asking such questions can be, surprisingly or not, of immense practical value. On a related note (but in the context of pure mathematics), Terence Tao said, in his elegant lectures on analysis,

> Moreover,
as you learn analysis you will develop an “analytical way of thinking”,
which will help you whenever you come into contact with any new rules
of mathematics, or when dealing with situations which are not quite
covered by the standard rules ... <br>
... You will develop a sense of why a rule in mathematics
(e.g., the chain rule) works, how to adapt it to new situations, and what
its limitations (if any) are; this will allow you to apply the mathematics
you have already learnt more confidently and correctly.
