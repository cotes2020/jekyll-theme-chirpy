---
title: Measuring Space & Invariants (Part 2)
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
- **Part 2**: [Mahalanobis distance]({% link _posts/2024-06-14-distances-part-2.md %})
- Part 3: [Spacetime interval]({% link _posts/2024-06-14-distances-part-3.md %})

## Coordinate Frames Relative To Distributions
Now instead of the $xy$ plane transforming underneath points, let's consider probability distributions moving beneath points. For example, let's say we are modeling the spatial distribution of stars in a (2D) galaxy. We may want to model a galaxy where stars can stretch uniformly:
{% include html/distances/isotropic_distribution.html%}

And another where stars on an ellipse can rotate:
{% include html/distances/anti_isotropic_distribution.html%}

We want a quantity that can measure distance *with respect to the distribution of stars*. That is:
- If there are two points located on the opposite extremes of the galaxy, this should be a far distance. 
- As the galaxy expands outwards, forcing two points to the center of the galaxy, the distance should update and become smaller. 

[Euclidean distance]({% link _posts/2024-06-14-distances-part-1.md %}) *will not* meet this criteria because it will stay the same (independent) as the distribution transforms.

### Mahalanobis Distance
We can update our notion of distance to something called [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance):
$$
\begin{equation}
d_M(\vec x, \vec y; \mathcal D) = \sqrt{(\vec x - \vec y)^TS^{-1}(\vec x - \vec y)}
\end{equation}
$$
Where $S$ is a [positive-definite](https://en.wikipedia.org/wiki/Definite_matrix) [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) from distribution $\mathcal D$. We can see that as the distribution transforms, the mahalanobis distance updates to reflect the change:
{% include html/distances/isotropic_distribution_distance.html%}

Since $S^{-1}$ decomposes as $S^{-1} = W^TW$ (by the [spectral theorem](https://en.wikipedia.org/wiki/Spectral_theorem)), then we can actually relate this back to euclidean distance:
$$
\begin{equation}
d_M(\vec x, \vec y; \mathcal D) = ||W(\vec x - \vec y)||
\end{equation}
$$
Which reveals why mahalanobis distance works; it measures the euclidean distance between two points with respect to the geometry of the distribution (as measured by $W$).

### Whitening Transformation
The [whitening transformation](https://en.wikipedia.org/wiki/Whitening_transformation) generalizes Mahalanobis distance to a random vector $X$:
$$
\begin{equation}
Y = W X
\end{equation}
$$
If $X \sim \mathcal D(\vec 0, S)$, then $Y \sim \mathcal D(\vec 0, I)$ where $I$ is the identity matrix. 

The proofs starts with the definition of $Cov[Y]$:

$$
\begin{flalign*}
Cov[Y] &= Cov[WX]\\
&= W S W^T
\end{flalign*}
$$

And shows that $W^T W = S^{-1}$ is sufficient for $Cov[Y] = I$:

$$
\begin{flalign*}
W S W^T & = I \\
\iff W S W^T W &= W \\
\iff W^T W &= S^{-1}
\end{flalign*}
$$

If we use the [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) of the positive-definite matrix $S = UDU^{T}$ and $W = D^{-1/2}U^{T}$ then its a direct equality:

$$
\begin{flalign*}
Cov[Y] &= W S W^T \\
&= (D^{-1/2}U^{T}) (UDU^T) (UD^{-1/2}) \\
&= D^{-1/2} (U^TU)D(U^TU)D^{-1/2} \\
&= D^{-1/2} D D^{-1/2} \\
&= I
\end{flalign*}
$$

Check out [this paper](https://arxiv.org/pdf/1512.00809) to see more choices of $W$. Instead of operating on just two points (like in Mahalanobis distance), we can sphere the entire space (meaning turn the space into a unit sphere) with respect to $W$ *and then* compute the euclidean distance:
{% include html/distances/whitened_transformation.html%}

After completing the whitening transformation, the resulting euclidean distance computations are invariant (and equal to $d_M$). Regardless of which distribution $\mathcal D$ we started from, $d_M$ is a scale-free measure of distance. This allows us to directly compare distances across distributions. 

[Linear Discriminant Analysis (LDA)](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) and [Gaussian Mixture Models (GMMs)](https://en.wikipedia.org/wiki/Mixture_model) use whitening to amortize repeated computations of expensive/high dimensional distance calculations with respect to [Multivariate Normal Distributions](https://en.wikipedia.org/wiki/Multivariate_normal_distribution). Other use cases are [anomaly detection](https://arxiv.org/abs/2003.00402) and [image processing](https://www.nature.com/articles/s42256-020-00265-z).

### Whitening in Higher Dimensions
Whitening doesn't need to be limited to only two dimensions and distance calculations. For example, we can whiten a three dimensional distribution and measure the invariant length of a ring:
{% include html/distances/3d_whitened_transformation.html%}

Transforming entire spaces (not just distances or rings) will be an important feature of the [next post on Spacetime intervals]({% link _posts/2024-06-14-distances-part-3.md %}).