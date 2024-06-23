---
title: Measuring Space & Invariants (Part 1)
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
- **Part 1**: [Euclidean distance]({% link _posts/2024-06-14-distances-part-1.md %})
- Part 2: [Mahalanobis distance]({% link _posts/2024-06-14-distances-part-2.md %})
- Part 3: [Spacetime interval]({% link _posts/2024-06-14-distances-part-3.md %})

## Cartesian Coordinates
To motivate invariant quantities, let's consider rotating the cartesian coordinate frame by a rotation matrix $R$:

$$
\begin{equation}
    R = \begin{bmatrix}
        \cos(\theta) & -\sin(\theta) \\
        \sin(\theta) & \cos(\theta)
    \end{bmatrix}
\end{equation}
$$

Which looks like:

{% include html/distances/euclidean_distance.html%}

The rotational transformation assigns different $(x, y)$ values to the original point at $(1, 1)$. This leads to an ambiguity; does the system behave differently with this new assignment of points? **Invariant quantities** avoid such questions by measuring things that are *independent* of their frame of reference. With invariant quantities, we can develop theories that hold in *any reference frame*.

### Euclidean Distance
An example of an invariant quantity is the [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between two points $p$ and $q$:
$$
\begin{equation}
d(p, q) = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}
\end{equation}
$$
Or, in vector notation:
$$
\begin{equation}
d(\vec p, \vec q) = ||\vec p - \vec q||
\end{equation}
$$

We can visualize an $xy$ plane rotating underneath this distance calculation, and calculate the distance for each frame:
{% include html/distances/square_rotation.html%}

Despite the $xy$ plane rotating, the euclidean distance stays the same (up to floating point precision). The measurement is *invariant* under the rotational transformation. Consider:

$$
\begin{flalign*}
d(\vec p', \vec q') &= ||R(\vec p - \vec q)|| \\
&= (\vec p - \vec q)^T R^T R(\vec p - \vec q) \\
&= (\vec p - \vec q)^T \begin{bmatrix}
        \cos(\theta) & \sin(\theta) \\
        -\sin(\theta) & \cos(\theta)
    \end{bmatrix}
    \begin{bmatrix}
        \cos(\theta) & -\sin(\theta) \\
        \sin(\theta) & \cos(\theta)
    \end{bmatrix}
    (\vec p - \vec q) \\
&= (\vec p - \vec q)^T 
    \begin{bmatrix}
        \cos^2(\theta) + \sin^2(\theta) & -\cos(\theta)\sin(\theta) + \cos(\theta)\sin(\theta) \\
        -\cos(\theta)\sin(\theta) + \cos(\theta)\sin(\theta) & \cos^2(\theta) + \sin^2(\theta)
    \end{bmatrix} (\vec p - \vec q) \\
&= (\vec p - \vec q)^T I (\vec p - \vec q) \\
&= ||\vec p - \vec q||
\end{flalign*}
$$

This is the definition of an [orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix); having $R^TR = I$ and being an [isometry](https://en.wikipedia.org/wiki/Isometry) of Euclidean space. For the $xy$ plane, the result is straightforward and intuitive. We can generalize this idea to [Part 2: Mahalanobis distance]({% link _posts/2024-06-14-distances-part-2.md %}) and [Part 3: Spacetime interval]({% link _posts/2024-06-14-distances-part-3.md %}) in the next posts.