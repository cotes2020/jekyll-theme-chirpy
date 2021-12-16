---
title: "Algebra Done Tensorially: Part 3.1 (Clifford Algebras)"
description: "Constructing some Clifford algebras using tensors"
categories: [abstract algebra, representation theory, tensor algebra]
tags: [Clifford algebras, tensors]
---
Welcome to Part 3.1 of '_Algebra done Tensorially_'. It's been a while since [Part 1 (Bilinear Products)]({% post_url 2021-10-18-bilinear-products %}) and [Part 2 (Algebras Over Fields)]({% post_url 2021-10-23-algebras-over-fields %}), so let us resume our investigation of algebras without further ado.

## Recap: vector algebras

In the [previous post]({% post_url 2021-10-23-algebras-over-fields %}), we had seen that algebras on rank $$1$$ vector spaces are the only ones that are non-redundant in terms of the degrees of freedom of the elements of the vector space (revisit the argument [here](}{% post_url 2021-10-23-algebras-over-fields %}#the-jacobian)). This makes things much simpler, as we don't have to bother about general higher-dimensional algebras (for them, the [Gamma tensor]{% post_url 2021-10-23-algebras-over-fields %}#gamma-tensor) is richer in information than the degrees of freedom).

Therefore, we are left with the notion of bilinear vector products which act on a vector space as a linear transformation by making the other argument implicit:

$$\phi^k B^{i^\prime}_{\phantom{i^\prime} ki} = \Lambda^{i^\prime}_{\phantom{i^\prime} i}$$

According to [the way we had defined the term 'algebra']({% post_url 2021-10-23-algebras-over-fields %}#definition-1), the ordered pair $$\left( \phi^k, B^{i^\prime}_{\phantom{i^\prime} ki} \right)$$ forms an algebra. But the above relation tells us there is an invertible map $$\left( \phi^k, B^{i^\prime}_{\phantom{i^\prime} ki} \right) \mapsto \Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ . The map is invertible as there exists an implicit relationship between $$\phi^k$$ and $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ . Namely, $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ is constrained to have the same degrees of freedom as $$\phi^k$$ .

Due to the existence of such a map from an algebra to a Jacobian, an algebra can be said to have an underlying, isomorphic Jacobian. Therefore, a Jacobian, in principle, is all one needs to construct or describe its vector algebra, which is the reason [representation theory]({% post_url 2021-10-23-algebras-over-fields %}#representation-theory) works!

## Complex numbers in Cartesian coordinates

Before we actually get into the family of algebras called 'Clifford algebras', let us see some of its examples in this post in order to understand the motivation behind the broader family.

### Complex numbers and the orthogonal group

Complex numbers are one of the most well-known algebras, in use ever since people tried to solve 'unsolvable' polynomial equations, such as $$i^2 = -1$$ . But complex numbers are hardly about 'inventing' solutions for such apparently bizarre equations. What's much more interesting is that by inventing these solutions, we've stumbled upon a [field](https://en.wikipedia.org/wiki/Field_(mathematics)), which is a mathematical structure obeying the [field axioms](https://mathworld.wolfram.com/FieldAxioms.html).

One way to understand why the above happens is to construct a vector algebra isomorphic to the algebra of complex numbers and study corresponding structures. It turns out that the precise Jacobian characterizing the vector algebra belongs to the two-dimensional orthogonal group $$O \left( 2, \mathbb{R} \right)$$ , which dictates a rotation and scaling on vector spaces isomorphic to $$\mathbb{R}^2$$ .

### Derivation

To keep things simple, let us first derive the vector algebra for complex numbers by working in a Cartesian coordinate system, where an orthogonal Jacobian $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ is always of the form,


$$\Lambda^{i^\prime}_{\phantom{i^\prime} i} = \begin{pmatrix} a & -b \\ b & a \end{pmatrix}$$

As required, the above Jacobian has only $$2$$ degrees of freedom, the same as the dimension of the vector space it acts on. Now, the idea is to 'pack' the degrees of freedom $$\left( a, b \right)$$ into a vector with those coordinates. In other words, every vector in the vector space represented by the variable ordered pair $$\left( a, b \right)$$ represents the corresponding Jacobian of the form seen above. Here, 'representation' implies a one-to-one map.

$$\phi^k = \begin{pmatrix} a \\ b \end{pmatrix}$$

Therefore, our problem becomes that of determining the coefficients $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ . Recall that $$\phi^k B^{i^\prime}_{\phantom{i^\prime} ki}$$ produces a rotation and scaling described by $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ that acts on a vector on the right. For this reason, it is necessary to place $$\phi^k$$ on the left of $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ as a _row_ matrix, even though it is a vector (typically represented as column matrices in some basis). This notational inconsistency will not emerge when we repeat this exercise in arbitrary coordinates (using tensors, as expected), but for now, we'll have to keep this little caveat in mind.

In the relation $$\phi^k B^{i^\prime}_{\phantom{i^\prime} ki} = \Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ , we get two matrix equations, one for each $$i^\prime$$ (corresponding to rows, as $$i^\prime$$ is upstairs),

$$
\begin{align}
\begin{pmatrix} a & b \end{pmatrix} \begin{pmatrix} B^0_{\phantom{0}00} & B^0_{\phantom{0}01} \\ B^0_{\phantom{0}10} & B^0_{\phantom{0}11} \end{pmatrix} & = \begin{pmatrix} a & -b \end{pmatrix} \\
\begin{pmatrix} a & b \end{pmatrix} \begin{pmatrix} B^1_{\phantom{1}00} & B^1_{\phantom{1}01} \\ B^1_{\phantom{1}10} & B^1_{\phantom{1}11} \end{pmatrix} & = \begin{pmatrix} b & a \end{pmatrix}
\end{align}
$$

The only solution for $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ which holds for arbitrary $$\left( a, b \right)$$ are immediately found to be,

$$
\begin{align}
\begin{pmatrix} B^0_{\phantom{0}00} & B^0_{\phantom{0}01} \\ B^0_{\phantom{0}10} & B^0_{\phantom{0}11} \end{pmatrix} & = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \\
\begin{pmatrix} B^1_{\phantom{1}00} & B^1_{\phantom{1}01} \\ B^1_{\phantom{1}10} & B^1_{\phantom{1}11} \end{pmatrix} & = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
\end{align}
$$

At this point, you might be thinking: _what did we just do_? So, it's a good time to recall what the [components of a bilinear product]({% post_url 2021-10-23-algebras-over-fields %}#structure) really mean:

$$\pmb{\mathcal{B}} \left( \pmb{e}_k, \pmb{e}_i \right) = B^{i^\prime}_{\phantom{i^\prime} ki} \pmb{e}_{i^\prime}$$

Armed with the above knowledge, let's find the products $$\pmb{\mathcal{B}} \left( \pmb{e}_k, \pmb{e}_i \right)$$ for all $$\left( k, i \right)$$ from the components $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ we just found:

$$
\begin{align}
\pmb{\mathcal{B}} \left( \pmb{e}_0, \pmb{e}_0 \right) & = B^0_{\phantom{0}00} \pmb{e}_0 + B^1_{\phantom{1}00} \pmb{e}_1 = \pmb{e}_0 \\
\pmb{\mathcal{B}} \left( \pmb{e}_0, \pmb{e}_1 \right) & = B^0_{\phantom{0}01} \pmb{e}_0 + B^1_{\phantom{1}01} \pmb{e}_1 = \pmb{e}_1 \\
\pmb{\mathcal{B}} \left( \pmb{e}_1, \pmb{e}_0 \right) & = B^0_{\phantom{0}10} \pmb{e}_0 + B^1_{\phantom{1}10} \pmb{e}_1 = \pmb{e}_1 \\
\pmb{\mathcal{B}} \left( \pmb{e}_1, \pmb{e}_1 \right) & = B^0_{\phantom{0}11} \pmb{e}_0 + B^1_{\phantom{1}11} \pmb{e}_1 = - \pmb{e}_0 \\
\end{align}
$$

To better see the 'magic' here, let us tabulate the above products,

$$
\begin{align}
\pmb{\mathcal{B}} && \pmb{e}_0 && \pmb{e}_1 \\
\pmb{e}_0 && \pmb{e}_0 && \pmb{e}_1 \\
\pmb{e}_1 && \pmb{e}_1 && - \pmb{e}_0
\end{align}
$$

Do you recognize the similarity of the above table with the multiplication table for complex numbers? :)

$$
\begin{align}
\times && 1 && i \\
1 && 1 && i \\
i && i && - 1
\end{align}
$$

Bingo! We have derived the algebra of complex numbers purely from their underlying geometry, which has to do with the orthogonal group.

Here are some observations which can be drawn from above:

1. The algebra we obtained is _unital_, i.e. there is a unity element $$\pmb{e}_0$$ which satisfies $$\pmb{\mathcal{B}} \left( \pmb{e}_0, \pmb{e}_i \right) = \pmb{e}_i \: \forall \: i$$ . Or, in the component form,

$$B^{i^\prime}_{\phantom{i^\prime}0i} = \delta^{i^\prime}_{\phantom{i^\prime}i}$$

2. The algebra is _commutative_, i.e. the bilinear product is symmetric in its arguments (and hence, in its lower indices),

$$B^{i^\prime}_{\phantom{i^\prime}ki} = B^{i^\prime}_{\phantom{i^\prime}ik}$$

Furthermore, due to unitarity, $$B^{i^\prime}_{\phantom{i^\prime}0i} = B^{i^\prime}_{\phantom{i^\prime}i0} = \delta^{i^\prime}_{\phantom{i^\prime}i}$$ .

3. The algebra is _associative_, i.e. $$\pmb{\mathcal{B}} \left( \pmb{\mathcal{B}} \left( \pmb{e}_i, \pmb{e}_j  \right) , \pmb{e}_k \right) = \pmb{\mathcal{B}} \left( \pmb{e}_i, \pmb{\mathcal{B}} \left( \pmb{e}_j, \pmb{e}_k \right) \right)$$ . A simple expansion of this equation, followed by the application of bilinearity and linear independence reveals,

$$B_{\phantom{j^{\prime}}ij}^{j^{\prime}} \: B_{\phantom{k^{\prime\prime}}j^{\prime}k}^{k^{\prime\prime}} = B_{\phantom{k^{\prime\prime}}ik^{\prime}}^{k^{\prime\prime}} \: B_{\phantom{k^{\prime}}jk}^{k^{\prime}}$$

The double primes remind us that applying $$\pmb{\mathcal{B}}$$ twice maps the primed vector space obtained through $$\pmb{\mathcal{B}}$$ to another vector space. The geometric significance of the above condition will be seen soon.

Any algebra which satisfies the first and third properties — unitarity and associativity — is called a _Clifford algebra_. But we have a lot more ground to cover before beholding the full glory of Clifford algebras, so hold on tight!

## Complex numbers in arbitrary coordinates

Let us now derive the algebra of complex numbers, encoded in $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ , in coordinate-independent form. But in doing so, we can never find the components of $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ , as they would depend on the coordinate system (and hence, so would the components of bilinear products of the basis vectors). Instead, we can relate $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ in the tensor form to other known tensors.

By the way, note that when we say 'complex numbers in arbitrary coordinates', we refer to the _vector_ algebra of $$O \left( 2, \mathbb{R} \right)$$ when it acts on a vector space isomorphic to $$\mathbb{R}^2$$ . As an artefact of dealing with vectorial objects, we cannot trust components in any form, as they are not coordinate-independent.

By contrast, in the _scalar_ treatment of complex numbers, where the basis is a set of scalars, 'vectors' and their components are informationally equivalent, implying invariance of the components themselves.

### Strategy

Like before, our procedure to find $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ is to write down $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ representing $$O \left( 2, \mathbb{R} \right)$$ , pack its degrees of freedom into $$\phi^k$$ and solve for $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ .

But now that we have freed ourselves of specific coordinate systems like the Cartesian system, how do we proceed with the above? For instance, how can we describe orthogonal Jacobians $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ in a coordinate-invariant manner?

To answer the above questions, let us make a few observations to devise a suitable strategy:

1. In $$n$$ dimensions, an orthogonal matrix contains $$n$$ independent numbers. An informal proof for this is: in $$n$$-dimensional spherical coordinates, any rotation and scaling can be represented as a composition of $$n$$ mutually independent rotations with the same scale factor, each assignable to an [Euler angle](https://en.wikipedia.org/wiki/Euler_angles). Therefore, we have $$n$$ independent numbers characterizing the rotation in spherical coordinates, which must be true in any coordinate system.

2. If we think of the $$n$$ independent numbers describing an $$n$$-dimensional rotation and scaling as its 'components', we should be able to construct an $$n$$-dimensional basis for the same.

The above points motivate us to lay out the following plan to achieve our goal:

1. Instead of using an $$n^2$$ dimensional basis for the rank $$2$$ tensors that dictate rotation and scaling, we will use an $$n$$ dimensional basis, as other degrees of freedom are redundant. We will not specify the basis as we are not interested in any particular coordinate system.

2. Using the arbitrary basis, we will represent orthogonal matrices as linear combinations of basis orthogonal matrices.

3. Next, we will construct an operator which maps every basis Jacobian to the basis vector belonging to the vector space of $$\phi^k$$ . This will allow us to set up an equation where we can solve for $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ .

Let us get started!

### Pauli matrices

#### Involution

We want to represent $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ as a linear combination of the form:

$$
\begin{align}
\pmb{\Lambda} & = \Lambda^{i^\prime}_{\phantom{i^\prime} i} \: \pmb{e}_{i^\prime} \otimes \pmb{\theta}^i \\
 & = \phi^j \: \sigma^{i^\prime}_{\phantom{i^\prime} ij} \: \pmb{e}_{i^\prime} \otimes \pmb{\theta}^i \\
 & = \phi^j \pmb{\sigma}_j
\end{align}
$$

where $$\left\{ \pmb{\sigma}_j \right\}$$ is a basis for $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ and $$\left\{ \pmb{\theta}^i \right\}$$ is the covector basis in the space of $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ .

We also want to map each $$\pmb{\sigma}_j = \sigma^{i^\prime}_{\phantom{i^\prime} ij} \: \pmb{e}_{i^\prime} \otimes \pmb{\theta}^i$$  to $$\pmb{e}_j$$ , the corresponding basis vector.
