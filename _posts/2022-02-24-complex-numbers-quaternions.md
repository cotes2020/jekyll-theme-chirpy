---
title: "Algebra Done Tensorially: Part 3 (Complex Numbers and Quaternions)"
description: "Constructing the algebra of complex numbers and quaternions, from their multiplicative groups"
categories: [representation theory]
tags: [complex numbers, quaternions, algebras, tensors]
---

Welcome to Part 3 of 'Algebra Done Tensorially'. It's been a while since the previous posts, so let us resume our investigation of algebras without further ado. For readers who haven't read earlier posts in this series yet, I'd recommend you to read Part 1 and Part 2 first :)

| Parts | Topics |
| :-- | :-- |
| [Part 1 (Bilinear Products)]({% post_url 2021-10-18-bilinear-products %})|tensors, bilinear products |
| [Part 2 (Algebras Over Fields)]({% post_url 2021-10-23-algebras-over-fields %}) |linear maps, algebra, degrees of freedom |
| [Part 3 (Complex Numbers and Quaternions)]({% post_url 2022-02-24-complex-numbers-quaternions %}) | complex numbers, quaternions, gamma matrices |
| [Part 4 (Clifford Algebras)]() | in progress |
| [Part 5 (Lie Algebras)]() | in progress |

## Recap: vector algebras

In the [previous post]({% post_url 2021-10-23-algebras-over-fields %}), we had seen that algebras on rank $$1$$ vector spaces are the only ones that are non-redundant in terms of the degrees of freedom of the elements of the vector space (revisit the argument [here]({% post_url 2021-10-23-algebras-over-fields %}#the-jacobian)). This makes things much simpler, as we don't have to bother about general higher-dimensional algebras (for them, the [J tensor]({% post_url 2021-10-23-algebras-over-fields %}#j-tensor) is richer in information than the degrees of freedom).

Therefore, we are left with the notion of bilinear vector products which act on a vector space as a linear transformation by making the other argument implicit:

$$\phi^k B^{i^\prime}_{\phantom{i^\prime} ki} = \Lambda^{i^\prime}_{\phantom{i^\prime} i}$$

According to [the way we had defined the term 'algebra']({% post_url 2021-10-23-algebras-over-fields %}#definition-1), the ordered pair $$\left( \phi^k, B^{i^\prime}_{\phantom{i^\prime} ki} \right)$$ forms an algebra. But the above relation tells us there is an invertible map $$\left( \phi^k, B^{i^\prime}_{\phantom{i^\prime} ki} \right) \mapsto \Lambda^{i^\prime}_{\phantom{i^\prime} i}$$. The map is invertible as there exists an implicit relationship between $$\phi^k$$ and $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$. Namely, $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ is constrained to have the same degrees of freedom as $$\phi^k$$.

Due to the existence of such a map from an algebra to a Jacobian, the latter is all one needs to construct its vector algebra, which is the reason [representation theory]({% post_url 2021-10-23-algebras-over-fields %}#representation-theory) works conveniently!

## Complex numbers

Let us begin by applying the ideas covered so far to one of the most important structures in mathematics, that of complex numbers.

### 2-dimensional orthogonal group

Complex numbers are one of the most well-known algebras, in use ever since people tried to solve 'unsolvable' polynomial equations, such as $$i^2 = -1$$. But they are now understood from a more fundamental perspective: group theory. It turns out that the complex numbers are much more than the set $$\mathbb{C}$$. Coupled with the notions of addition and multiplication, this set forms the additive and multiplicative group of complex numbers, respectively, $$\left( \mathbb{C}, + \right)$$ and $$\left( \mathbb{C}, \times \right)$$. These two groups are isomorphic to the two-dimensional translation group $$\text{T} \left( 2 \right)$$ and the orthogonal group $$\text{O} \left( 2 \right)$$, respectively. The latter itself comprises the group of all scalars and rotations on $$\mathbb{R}^2$$, i.e. $$\left( \mathbb{R}, \times \right)$$ and $$\text{SO} \left( 2 \right)$$.

The algebra of complex numbers deals with their multiplicative group. Under multiplication, the action of complex numbers is that of scaling and rotating $$\mathbb{C}$$. As the name 'orthogonal group' suggests, this transformation leaves initially orthogonal elements of $$\mathbb{C}$$ orthogonal. Two complex numbers $$u, v \in \mathbb{C}$$ are said to be orthogonal when $$\left\langle u, v \right\rangle = \Re \left( u^* v \right) = \Re \left( u v^* \right) = 0$$. Equivalently, for a vector space isomorphic to $$\mathbb{C}$$, such as $$\mathbb{R}^2$$, orthogonality is defined analogously using the appropriate inner product.

<a name="invariant_quantity"></a>

However, the elements of $$\text{O} \left( 2 \right)$$ can be defined in a manner which will be useful for later generalization. Namely, when some $$\pmb{\Lambda} \in \text{O} \left( 2 \right)$$ is characterized by a $$\pmb{\phi} \in \mathbb{R}^2$$, the following quantity is invariant:

$$\frac{\left\langle \pmb{\phi}, \pmb{\phi} \right\rangle}{\det \left( \pmb{\Lambda} \right)}$$

Now, we will investigate the action of complex numbers on vectors belonging to $$\mathbb{R}^2$$.

### Algebra

In the convenient Cartesian coordinate system, an orthogonal Jacobian $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ is always of the form,


$$\Lambda^{i^\prime}_{\phantom{i^\prime} i} = \begin{pmatrix} a & -b \\ b & a \end{pmatrix}$$

Indeed,

$$
\begin{align}
\frac{\left\langle \left( a, b \right), \left( a, b \right) \right\rangle}{\det \left( \Lambda^{i^\prime}_{\phantom{i^\prime} i} \right)} & = \frac{a^2+b^2}{a^2+b^2} \\
 & = 1 \\
 & = \text{constant}
\end{align}
$$

As required, the above Jacobian has only $$2$$ degrees of freedom, the same as the dimension of the vector space it acts on. Now, the idea is to pack the degrees of freedom $$\left( a, b \right)$$ into a vector with those coordinates. In other words, every vector in the vector space represented by the variable ordered pair $$\left( a, b \right)$$ represents the corresponding Jacobian of the form seen above. Here, 'representation' implies a one-to-one map.

$$\phi^k = \begin{pmatrix} a \\ b \end{pmatrix}$$

Therefore, our problem becomes that of determining the coefficients $$B^{i^\prime}_{\phantom{i^\prime} ki}$$. Recall that $$\phi^k B^{i^\prime}_{\phantom{i^\prime} ki}$$ produces a rotation and scaling described by $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ that acts on a vector on the right. For this reason, it is necessary to place $$\phi^k$$ on the left of $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ as a _row_ matrix, even though it is a vector (typically represented as column matrices in some basis). This notational inconsistency will not emerge when we repeat this exercise in arbitrary coordinates (using tensors, as expected), but for now, we'll have to keep this little caveat in mind.

In the relation $$\phi^k B^{i^\prime}_{\phantom{i^\prime} ki} = \Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ , we get two matrix equations, one for each $$i^\prime$$ (corresponding to rows, as $$i^\prime$$ is upstairs),

$$
\begin{align}
\begin{pmatrix} a & b \end{pmatrix} \begin{pmatrix} B^{0^\prime}_{\phantom{0^\prime}00} & B^{0^\prime}_{\phantom{0^\prime}01} \\ B^{0^\prime}_{\phantom{0^\prime}10} & B^{0^\prime}_{\phantom{0^\prime}11} \end{pmatrix} & = \begin{pmatrix} a & -b \end{pmatrix} \\
\begin{pmatrix} a & b \end{pmatrix} \begin{pmatrix} B^{1^\prime}_{\phantom{1^\prime}00} & B^{1^\prime}_{\phantom{1^\prime}01} \\ B^{1^\prime}_{\phantom{1^\prime}10} & B^{1^\prime}_{\phantom{1^\prime}11} \end{pmatrix} & = \begin{pmatrix} b & a \end{pmatrix}
\end{align}
$$

The only solution for $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ which holds for arbitrary $$\left( a, b \right)$$ are immediately found to be,

$$
\begin{align}
\begin{pmatrix} B^{0^\prime}_{\phantom{0^\prime}00} & B^{0^\prime}_{\phantom{0^\prime}01} \\ B^{0^\prime}_{\phantom{0^\prime}10} & B^{0^\prime}_{\phantom{0^\prime}11} \end{pmatrix} & = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \\
\begin{pmatrix} B^{1^\prime}_{\phantom{1^\prime}00} & B^{1^\prime}_{\phantom{1^\prime}01} \\ B^{1^\prime}_{\phantom{1^\prime}10} & B^{1^\prime}_{\phantom{1^\prime}11} \end{pmatrix} & = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
\end{align}
$$

At this point, you might be thinking: _what did we just do_? So, it's a good time to recall what the [components of a bilinear product]({% post_url 2021-10-23-algebras-over-fields %}#structure) really mean:

$$\pmb{\mathcal{B}} \left( \pmb{e}_k, \pmb{e}_i \right) = B^{i^\prime}_{\phantom{i^\prime} ki} \pmb{e}_{i^\prime}$$

Armed with the above knowledge, let's find the products $$\pmb{\mathcal{B}} \left( \pmb{e}_k, \pmb{e}_i \right)$$ for all $$\left( k, i \right)$$ from the components $$B^{i^\prime}_{\phantom{i^\prime} ki}$$ we just found:

$$
\begin{align}
\pmb{\mathcal{B}} \left( \pmb{e}_0, \pmb{e}_0 \right) & = B^{0^\prime}_{\phantom{0^\prime}00} \pmb{e}_0 + B^{1^\prime}_{\phantom{1^\prime}00} \pmb{e}_1 = \pmb{e}_0 \\
\pmb{\mathcal{B}} \left( \pmb{e}_0, \pmb{e}_1 \right) & = B^{0^\prime}_{\phantom{0^\prime}01} \pmb{e}_0 + B^{1^\prime}_{\phantom{1^\prime}01} \pmb{e}_1 = \pmb{e}_1 \\
\pmb{\mathcal{B}} \left( \pmb{e}_1, \pmb{e}_0 \right) & = B^{0^\prime}_{\phantom{0^\prime}10} \pmb{e}_0 + B^{1^\prime}_{\phantom{1^\prime}10} \pmb{e}_1 = \pmb{e}_1 \\
\pmb{\mathcal{B}} \left( \pmb{e}_1, \pmb{e}_1 \right) & = B^{0^\prime}_{\phantom{0^\prime}11} \pmb{e}_0 + B^{1^\prime}_{\phantom{1^\prime}11} \pmb{e}_1 = - \pmb{e}_0 \\
\end{align}
$$

To better see the 'magic' here, let us tabulate the above products,

$$
\begin{matrix}
\pmb{\mathcal{B}} & \pmb{e}_0 & \pmb{e}_1 \\
\pmb{e}_0 & \pmb{e}_0 & \pmb{e}_1 \\
\pmb{e}_1 & \pmb{e}_1 & - \pmb{e}_0
\end{matrix}
$$

Do you recognize the similarity of the above table with the multiplication table for complex numbers? :)

$$
\begin{matrix}
\times & 1 & i \\
1 & 1 & i \\
i & i & - 1
\end{matrix}
$$

Bingo! We have derived the algebra of complex numbers purely from their underlying geometry, which has to do with the orthogonal group.

Let us now look at the 4-dimensional extension of complex numbers: quaternions.

## Quaternions

### 4-dimensional orthogonal group

Let $$\mathbb{H}$$ be the set of quaternions. The multiplicative group of quaternions, $$\left( \mathbb{H}, \times \right)$$ is isomorphic to the 4-dimensional orthogonal group, $$\text{O} \left( 4 \right)$$.

Let us generalize [the invariant quantity](#invariant_quantity) with regard to $$\text{O} \left( 2 \right)$$. For a member of $$\pmb{\Lambda} \in \text{O} \left( 4 \right)$$ characterized by some $$\pmb{\phi} \in \mathbb{R}^4$$, we have the invariant,

$$
\begin{align}
\frac{\left\langle \pmb{\phi}, \pmb{\phi} \right\rangle}{\det \left( \Lambda^{i^\prime}_{\phantom{i^\prime} i} \right)} & = 1 \\
\implies \det \left( \Lambda^{i^\prime}_{\phantom{i^\prime} i} \right) & = \left\langle \pmb{\phi}, \pmb{\phi} \right\rangle = \sum_{k=0}^3 \left( \phi^k \right)^2
\end{align}
$$

Which $$4 \times 4$$ matrix characterized by 4 numbers, say $$a, b, c, d$$, necessarily has the determinant $$a^2+b^2+c^2+d^2$$? Instead of doing trial and error over the rather large space of $$4 \times 4$$ matrices, let us cheat and contract the 4 real numbers to 2 complex numbers $$a+ib, \: c+id$$. These numbers implicitly contain their $$2 \times 2$$ matrices of the form,

$$a + ib \equiv \begin{pmatrix} a & -b \\ b & a \end{pmatrix}$$

Now, we can ask, which $$2 \times 2$$ matrix characterized by the two complex numbers has a determinant of:

$$a^2 + b^2 + c^2 + d^2 = \left( a + ib \right) \left( a - ib \right) + \left( c + id \right) \left( c - id \right)$$

The simplest solution is to form a $$2 \times 2$$ matrix with the above factors so that the usual expression for the determinant is the same as the above,

$$\Lambda^{i^\prime}_{\phantom{i^\prime} i} \equiv \begin{pmatrix} \left( a + ib \right) & - \left( c+id \right) \\ \left( c - id \right) & \left( a - ib \right) \end{pmatrix}$$

Expanding each complex entry in the above matrix as $$2 \times 2$$ matrices, we get,

$$\Lambda^{i^\prime}_{\phantom{i^\prime} i} = \begin{pmatrix} a & -b & -c & -d \\ b & a & d & -c \\ c & d & a & b \\ -d & c & -b & a \end{pmatrix}$$

### Algebra

Just as before, we want to characterize the above Jacobian using its degrees of freedom, as,

$$\phi^k = \begin{pmatrix} a \\ b \\ c \\ d \end{pmatrix}$$

Now, we have to find the set of four $$4 \times 4$$ matrices, $$B^{i^\prime}_{\phantom{i^\prime} k i}$$ so that $$\phi^k B^{i^\prime}_{\phantom{i^\prime} k i} = \Lambda^{i^\prime}_{\phantom{i^\prime} i}$$,

$$
\begin{align}
\begin{pmatrix} a & b & c & d \end{pmatrix} \: \pmb{B}^{0^\prime}_{\phantom{0^\prime} ki} & = \begin{pmatrix} a & -b & -c & -d \end{pmatrix} \\
\begin{pmatrix} a & b & c & d \end{pmatrix} \: \pmb{B}^{1^\prime}_{\phantom{1^\prime} ki} & = \begin{pmatrix} b & a & d & -c \end{pmatrix} \\
\begin{pmatrix} a & b & c & d \end{pmatrix} \: \pmb{B}^{2^\prime}_{\phantom{2^\prime} ki} & = \begin{pmatrix} c & d & a & b \end{pmatrix} \\
\begin{pmatrix} a & b & c & d \end{pmatrix} \: \pmb{B}^{3^\prime}_{\phantom{3^\prime} ki} & = \begin{pmatrix} -d & c & -b & a \end{pmatrix}
\end{align}
$$

The appropriate $$B^{i^\prime}_{\phantom{i^\prime} k i}$$ can be found to be,

$$
\begin{align}
B^{0^\prime}_{\phantom{0^\prime} k i} & = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix} \\
B^{1^\prime}_{\phantom{1^\prime} k i} & = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & -1 \\ 0 & 0 & 1 & 0 \end{pmatrix} \\
B^{2^\prime}_{\phantom{2^\prime} k i} & = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix} \\
B^{3^\prime}_{\phantom{3^\prime} k i} & = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & -1 & 0 \\ 0 & 1 & 0 & 0 \\ -1 & 0 & 0 & 0 \end{pmatrix}
\end{align}
$$

Lastly, let us compute the bilinear products of every pair of basis vectors as $$\pmb{\mathcal{B}} \left( \pmb{e}_k, \pmb{e}_i \right) = B^{i^\prime}_{\phantom{i^\prime} k i} \pmb{e}_{i^\prime}$$ and tabulate the results,

$$
\begin{matrix}
\pmb{\mathcal{B}} & \pmb{e}_0 & \pmb{e}_1 & \pmb{e}_2 & \pmb{e}_3 \\
\pmb{e}_0 & \pmb{e}_0 & \pmb{e}_1 & \pmb{e}_2 & \pmb{e}_3 \\
\pmb{e}_1 & \pmb{e}_1 & - \pmb{e}_0 & \pmb{e}_3 & - \pmb{e}_2 \\
\pmb{e}_2 & \pmb{e}_2 & - \pmb{e}_3 & - \pmb{e}_0 & \pmb{e}_1 \\
\pmb{e}_3 & \pmb{e}_3 & \pmb{e}_2 & - \pmb{e}_1 & - \pmb{e}_0
\end{matrix}
$$

This corresponds to the Hamilton product table:

$$
\begin{matrix}
\times & 1 & i & j & k \\
1 & 1 & i & j & k \\
i & i & - 1 & k & - j \\
j & j & - k & - 1 & i \\
k & k & j & - i & - 1
\end{matrix}
$$

## Gamma matrices

### Definition

From the relation $$\phi^k B^{i^\prime}_{\phantom{i^\prime} k i} = \Lambda^{i^\prime}_{\phantom{i^\prime} i}$$, we see that $$\left\{ B^{i^\prime}_{\phantom{i^\prime} k i} \right\}_k$$ forms a basis for $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$ in the space of $$\phi^k$$. This basis may be written as a set of dual vectors, which in the Cartesian basis are represented as the _gamma matrices_:

$$\pmb{\gamma}_k = B^{i^\prime}_{\phantom{i^\prime} k i} \: \pmb{\theta}^i \otimes \pmb{e}_{i^\prime}$$

Note that even though the gamma matrices above look like Jacobians, they transform like dual vectors. In the above equation, a gamma matrix (or more appropriately, tensor) has only one free index, downstairs.

### Complex numbers

For complex numbers, the components of the [dual] gamma matrices are found to be:

$$
\begin{align}
\pmb{\gamma}_0 & = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \\
\pmb{\gamma}_1 & = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
\end{align}
$$

### Quaternions

For quaternions, using the components of $$B^{i^\prime}_{\phantom{i^\prime} k i}$$ and the definition of the gamma matrices, we find,

$$
\begin{align}
\pmb{\gamma}_0 & = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \\
\pmb{\gamma}_1 & = \begin{pmatrix} 0 & -1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & -1 & 0 \end{pmatrix} \\
\pmb{\gamma}_2 & = \begin{pmatrix} 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix} \\
\pmb{\gamma}_3 & = \begin{pmatrix} 0 & 0 & 0 & -1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ -1 & 0 & 0 & 0 \end{pmatrix} \\
\end{align}
$$

### From any parameterized Jacobian

Given a Jacobian parameterized by a vector space as $$\Lambda^{i^\prime}_{\phantom{i^\prime} i} \left( \phi^k \right)$$, we have,

$$\Lambda^{i^\prime}_{\phantom{i^\prime} i} \left( \phi^k \right) = \phi^k B^{i^\prime}_{\phantom{i^\prime} ki} = \phi^k \gamma_{k \phantom{i^\prime} i}^{\phantom{k} i^\prime}$$

where $$\gamma_{k \phantom{i^\prime} i}^{\phantom{k} i^\prime} = \pmb{e}_{i^\prime} \otimes \pmb{\theta}^{i^\prime} \left( \pmb{\gamma}_k \right)$$.

Suppose we want to find the particular $$k^\text{th}$$ gamma matrix instead of summing over $$k$$. To distinguish such dummy indices, let us enclose them in parentheses as $$\left( k \right)$$. Now,

$$\phi^{\left( k \right)} \gamma_{\left( k \right) \phantom{i^\prime} i}^{\phantom{\left( k \right)} i^\prime} = \Lambda^{i^\prime}_{\phantom{i^\prime} i} \left( \phi^j \right)$$

such that,

$$\phi^j = \begin{cases} \phi^{\left( k \right)} & j=k \\ 0 & j \neq k \end{cases}$$

i.e., $$\phi^j = \delta^j_{\phantom{j} \left( k \right)} \phi^{\left( k \right)}$$. Now, we have,

$$\phi^{\left( k \right)} \gamma_{\left( k \right) \phantom{i^\prime} i}^{\phantom{\left( k \right)} i^\prime} = \Lambda^{i^\prime}_{\phantom{i^\prime} i} \left( \delta^j_{\phantom{j} \left( k \right)} \phi^{\left( k \right)} \right)$$

Note that here, $$\phi^{\left( k \right)}$$ acts as a scalar. Since scaling a tensor scales its components and $$\phi^k$$ are constrained to be the components of $$\pmb{\Lambda}$$,

$$
\begin{align}
\phi^{\left( k \right)} \gamma_{\left( k \right) \phantom{i^\prime} i}^{\phantom{\left( k \right)} i^\prime} & = \phi^{\left( k \right)} \Lambda^{i^\prime}_{\phantom{i^\prime} i} \left( \delta^j_{\phantom{j} \left( k \right)} \right) \\
\gamma_{k \phantom{i^\prime} i}^{\phantom{k} i^\prime} & = \Lambda^{i^\prime}_{\phantom{i^\prime} i} \left( \delta^j_{\phantom{j} k} \right)
\end{align}
$$

## Towards generalization

Through the two popular examples of complex numbers and quaternions, we have seen how to deduce the algebra generated by a group, using its representation as a Jacobian of some form.

Some problematic aspects of our methods used so far are:

1. The matrix representation for the action of $$\text{O} \left( 4 \right)$$ on $$\mathbb{R}^4$$ was found by contracting its 4 degrees of freedom into 2 independent complex numbers, without justification. Is there a more 'honest' method to find the form of $$\pmb{\Lambda} \left( \pmb{\phi} \right)$$? Moreover, how can we find $$\pmb{\Lambda}$$ in arbitrary coordinates, as opposed to the convenient Cartesian coordinates?

2. We had to find the components of $$B^{i^\prime}_{\phantom{i^\prime} k i}$$ by brute force. Is there a more efficient, symbolical method to directly express the whole tensor $$B^{i^\prime}_{\phantom{i^\prime} k i}$$ in terms of known objects such as the given Jacobian $$\Lambda^{i^\prime}_{\phantom{i^\prime} i}$$?

The solution for the above problems is briefly to generalize the algebra of complex numbers and quanternions to a family of algebras called Clifford algebras. These can be defined in a coordinate-independent manner motivated by rotations. We shall elaborate on this in the next post, [Algebra Done Tensorially: Part 4 (Clifford Algebras)]()

To sum it all up, what we've learnt so far is that given a group and its representation via a parameterizing vector space, we can figure out the algebra generated by the elements of the group, which are Jacobians acting on the vector space. This gives us a geometric picture of algebra.
