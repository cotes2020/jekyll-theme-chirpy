---
title: Measuring Space & Invariants (Part 3)
author: jake
date: 2024-06-14 12:00:00 +0800
categories: [Physics]
tags: [theoretical minimum, statistics, physics, math, distance, linear algebra]
math: true
mermaid: true
image:
  path: /assets/img/custom/measuring-space.png
  alt: Measuring space, stars, and the light within
---

This series of posts cover different types of distances and their invariant properties. This includes:
- Part 1: [Euclidean distance]({% link _posts/2024-06-14-distances-part-1.md %})
- Part 2: [Mahalanobis distance]({% link _posts/2024-06-14-distances-part-2.md %})
- **Part 3**: [Spacetime interval]({% link _posts/2024-06-14-distances-part-3.md %})

## Spacetime Coordinates
[Spacetime diagrams](https://en.wikipedia.org/wiki/Spacetime_diagram) are required knowledge for the next transformations. Using space as the $x$ axis, time as the $y$ axis, and the origin of the coordinate system placed directly on our noses; we can plot the movement of someone walking away from us with speed $v$. This will occur on the line $x = vt$ or $x - vt = 0$:
{% include html/distances/moving_right.html%}

> The line $x = ct$ representing the motion of light with speed $c$ is also included since it soon becomes relevant to the discussion. For all plots, we define the $x$ axis in light-seconds so that $x = ct$ yields [$c = 1$](https://en.wikipedia.org/wiki/Geometrized_unit_system) and appears as a 45º slope from the $x$ axis. As a consequence all velocities $v$ are expressed as a fraction of the speed of light, i.e. $v = 0.5$ is 1/2 the speed of light, unless I specifically added a $c^2$ term for important definitions.
{: .prompt-warning }

As time progresses, we would see their position become farther away from our nose. Due to the symmetry of this situation, we can repeat the same experiment for the other person (the one walking away). Except this time, we will call their axes $x'$ and $t'$ and put the origin of the $(x', t')$ coordinate system on their nose. Of course, from their perspective we will be moving in the opposite direction (negative) away from them:
{% include html/distances/moving_left.html%}

Since both $t$ and $t'$ refer to an absolute notion of time, we can say $t=t'$ for the moment.
### Galilean Transformation
Now we want to consider a transformation from the original coordinate frame $(x, t)$ to the moving coordinate frame $(x', t')$. This is known as the [Galilean transformation](https://en.wikipedia.org/wiki/Galilean_transformation). In vector notation this is given by:

$$
\begin{equation}
    \begin{bmatrix}
        x' \\
        t'
    \end{bmatrix} = 
    \begin{bmatrix}
        1 & -v \\
        0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        x \\
        t
    \end{bmatrix}
\end{equation}
$$

Which is another way to say $x' = x - vt$ and $t = t'$. Unlike the rotations from before, this is a [shear transformation](https://en.wikipedia.org/wiki/Shear_mapping):
{% include html/distances/newtonian_transformation.html %}

If we measure the speed (i.e. the magnitude of the velocity) at any point during the transformation, the total speed is invariant. We can see this by computing a convex combination ($\alpha \in [0, 1]$) of each person's velocity through the transformation:

$$
\begin{flalign*}
S &= \sqrt{\alpha v^2 + (1 - \alpha) (-v)^2} \\
&= \sqrt{v^2} \\
&= v
\end{flalign*}
$$

This makes intuitive sense, the person moving away from the stationary person can only move as fast as the stationary person from the moving person.

### Motivations of Special Relativity
The Galilean transformation and its corresponding shear mapping works great up until we consider the speed of light. This is the same observation Einstein made in 1905 when he tried reconciling the Galilean transformation against two key principles:
- [**Principle of Relativity**](https://en.wikipedia.org/wiki/Principle_of_relativity) - "Requirement that the equations describing the laws of physics have the same form in all admissible frames of reference".
- [**Speed of light**](https://en.wikipedia.org/wiki/Speed_of_light) - "Universal physical constant that is exactly equal to 299,792,458 metres per second" (But again, we will use $c=1$). 

[Importantly, everyone sees light propagate at exactly c (regardless of their frame of reference and whether they are moving or not)](https://en.wikipedia.org/wiki/Michelson%E2%80%93Morley_experiment). Thus, we hit a major contradiction with the Galilean transformation:

Let's consider the same setup from before ($(x, t)$ is the stationary frame and $(x', t')$ is the moving frame), but have the moving person shoot two beams of light in front and behind them at the beginning of their motion. Lets add back in the speed of light and consider the transformation from the $(x', t')$ frame to the $(x, t)$ frame:
{% include html/distances/newtonian_transformation_problem.html%}

Initially, the $(x', t')$ frame of reference sees two light beams traveling along $x' = ct'$ and $x' = -ct'$, there is no contradiction here. But, when we transform to the stationary observer's $(x, t)$ frame of reference, suddenly we see light traveling faster than $c$ (precisely, the Galilean transformation predicts light to travel at $v + c = 1.25$). But according to the principle of relativity and speed of light, both observers should always see light traveling at exactly $c$; the stationary person can never see light at a higher speed!

### Lorentz Transformation
Now consider another transformation from $(x, t)$ to $(x', t')$ called the [Lorentz transformation](https://en.wikipedia.org/wiki/Lorentz_transformation):

$$
\begin{equation}
    \begin{bmatrix}
        x' \\
        t'
    \end{bmatrix} = 
    \gamma(v)
    \begin{bmatrix}
        1 & -v \\
        -v & 1
    \end{bmatrix}
    \begin{bmatrix}
        x \\
        t
    \end{bmatrix}
\end{equation}
$$

Where $\gamma(v) = \frac{1}{\sqrt{1 - v^2}}$ and is called the [Lorentz factor](https://en.wikipedia.org/wiki/Lorentz_factor). The two key differences from the Galilean transformation from before:
1. The Lorentz factor is a nonlinear function of $v$. As $v\rightarrow 0$, then $\gamma(v) \rightarrow 1$
2. $t' \neq t$ but actually combines both $x$ and $t$ (i.e. space and time get "mixed up" into spacetime)

The Lorentz transformation has a very special property that is actually part of its [derivation](https://en.wikipedia.org/wiki/Lorentz_transformation#Derivation_of_the_group_of_Lorentz_transformations):

$$
\begin{equation}
    c^2(t_2 - t_1)^2 - (x_2 - x_1)^2 = c^2(t'_2 - t'_1)^2 - (x'_2 - x'_1)^2
\end{equation}
$$

That the speed of light is equal in both $(x, t)$ and $(x', t')$ frames of reference. We can plot the same scenario from before and see how the speed of light now behaves:
{% include html/distances/lorentz_transformation.html%}

Again, $(x', t')$ starts out observing light traveling to the right with $x' = ct'$ and to the left as  $x' = -ct'$, no contradiction as before. But, after we transform to the $(x, t)$ frame of reference, something different happens. $x$ and $t$ are both bent inwards keeping the speed of light constant at $c$! The Lorentz transformation has succeeded to keep the speed of light equal to $c$ in *all* frames of reference.

How does it work? A generalization of distance for spacetime called the [Spacetime Interval](https://en.wikipedia.org/wiki/Spacetime#Spacetime_interval) was defined as:

$$
\begin{equation}
    d_L(\vec x_1, \vec x_2) = c^2(t'_2 - t'_1)^2 - (x'_2 - x'_1)^2 = c^2(t_2 - t_1)^2 - (x_2 - x_1)^2 = (\Delta s)^2
\end{equation}
$$

And then the Lorentz transformation was designed to keep $(\Delta s)^2$ invariant while going from $(x, t)$ to $(x', t')$ as measured from the origin:

$$
\begin{flalign*}
    t'^2 - x'^2 &= \frac{(t - vx)^2}{1 - v^2} - \frac{(x - vt)^2}{1 - v^2} \\
    &= \frac{t^2 + v^2x^2 - 2vtx}{1 - v^2} - \frac{x^2 + v^2t^2 - 2vtx}{1 - v^2} \\
    &= \frac{t^2 + v^2x^2}{1 - v^2} - \frac{x^2 + v^2t^2}{1 - v^2} \\
    &= \frac{t^2 - v^2t^2}{1 - v^2} - \frac{x^2 - v^2x^2}{1 - v^2} \\
    &= t^2 - x^2 \\
    &= s^2
\end{flalign*}
$$

There is a long list of interesting consequences from the Lorentz transformation & the resulting invariant spacetime interval:
- [Time dilation & Length contraction](https://en.wikipedia.org/wiki/Spacetime#Time_dilation_and_length_contraction)
- [Twin paradox](https://en.wikipedia.org/wiki/Spacetime#Twin_paradox)
- [Simultaneity of events](https://en.wikipedia.org/wiki/Spacetime#Relativity_of_simultaneity)
- [Mass-energy relationship](https://en.wikipedia.org/wiki/Spacetime#Mass%E2%80%93energy_relationship) (i.e. $E=mc^2$)

And many great courses (like the [Theoretical Minimum](https://theoreticalminimum.com/courses/special-relativity-and-electrodynamics/2012/spring)) that explain these in detail.

### Lorentz vs. Galilean
Here are plots for both the Lorentz and Galilean transformations so you can see them side by side (and in either direction):

**Lorentz**:
{% include html/distances/lorentz_transformation_grid.html%}

**Galilean**:
{% include html/distances/galilean_transformation_grid.html%}

While the Galilean transformation operates as a shear map, the Lorentz transformation operates as a rotation. But not like the [rotation matrix]({% link _posts/2024-06-14-distances-part-1.md %}) from before, but a [hyperbolic rotation](https://en.wikipedia.org/wiki/Lorentz_transformation#Coordinate_transformation):

$$
\begin{equation}
    \begin{bmatrix}
        x' \\
        t'
    \end{bmatrix} = 
    \begin{bmatrix}
        -\sinh{\xi} & \cosh{\xi} \\
        \cosh{\xi} & -\sinh{\xi}
    \end{bmatrix}
    \begin{bmatrix}
        x \\
        t
    \end{bmatrix}
\end{equation}
$$

Where $\xi = \tanh^{-1}{\frac{v}{c}}$ is the rapidity of the hyperbolic rotation. You can see the outline of this hyperbolic rotation from the dashed line following the last time point on the $t'$ axis. This is what keeps the transformed $(x', t')$ from asymptotically increasing, but never reaching, the speed of light $c$. This is unlike the shear map of the Galilean transform, which can reach and exceed the speed of light $c$ as we saw before.

Both the Galilean transformation and Lorentz transformation preserve area as shown by the determinant:

$$
\begin{equation}
    \begin{vmatrix}
        1 & -v \\
        0 & 1
    \end{vmatrix}
    = 1 \cdot 1 - (-v \cdot 0) = 1
\end{equation}
$$

And for Lorentz:

$$
\begin{equation}
    \begin{vmatrix}
        \gamma(v)
        \cdot
        \begin{bmatrix}
            1 & -v \\
            -v & 1
        \end{bmatrix}
    \end{vmatrix}
    = \gamma(v)^2(1 - v^2) = 1
\end{equation}
$$

We can see from either transformation that the area of the grid of points representing space is preserved.