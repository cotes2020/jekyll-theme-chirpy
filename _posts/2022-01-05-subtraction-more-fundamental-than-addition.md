---
title: "Vector Subtraction Is More Fundamental Than Addition"
description: "Why subtraction is more general than addition in the sense of vectors"
categories: [geometry]
tags: [affine space]
---

When we first learn mathematics in school, we are taught the four fundamental operations of arithmetic: addition, subtraction, multiplication and division. It is a common practice to teach addition first. After all, we're adding things all the time in real life: apples to apples, oranges to oranges and so on. Only then are we taught that the reverse operation is necessary to model the removal of apples from apples: subtraction. [^1]

[^1]: Although not directly related, pedagogy in computer science works similarly. We're first taught to add elements to data structures such as arrays. The idea of popping (i.e. removing) elements from structures is not only taught later, but also programatically more intricate. For example, popping elements often requires functions, whereas concatenating lists can almost universally be performed using the inbuilt `+` operator.

As a result, the idea that addition is perhaps more fundamental than subtraction is induced in us (it did, at least for me). But it turns out that geometrically speaking, subtraction is a more fundamental concept than addition! Let's see how.

## Euclidean spaces

Euclidean spaces are the most basic geometric spaces used in classical geometry. Without going into generalized dot products (i.e. inner products), a _Euclidean n-space_ is a vector space isomorphic to $$\mathbb{R}^n$$, equipped with a symmetric function called the _distance function_ $$d : \mathbb{R}^n \mapsto \mathbb{R}$$ defined according to Pythagoras' theorem in $$n$$ dimensions. For $$\pmb{u}, \pmb{v} \in \mathbb{R}^n$$ in Cartesian coordinates,

$$d \left( \pmb{u}, \pmb{v} \right) = \sqrt{ \sum_{i=1}^n \left( v_i - u_i \right)^2}$$

Every Euclidean space has an _origin_ $$\pmb{0}$$, represented by the coordinates $$\left( 0, 0, \overset{n \text{ times}}{\dots}, 0 \right)$$. Then, the _norm_ of a vector in Euclidean space is defined as $$\lvert \pmb{u} \rvert = d \left( \pmb{u}, \pmb{0} \right) = d \left( \pmb{0}, \pmb{u} \right)$$. I.e., in Cartesian coordinates,

$$\lvert \pmb{u} \rvert = \sqrt{\sum_{i=1}^n u_i^2}$$

Furthermore, a Euclidean space is equipped with a symmetric bilinear form, the _dot product_ $$\cdot : \mathbb{R^n} \times \mathbb{R}^n \mapsto \mathbb{R}$$, defined using the [polarization identity](https://en.wikipedia.org/wiki/Polarization_identity) as,

$$\pmb{u} \cdot \pmb{v} = \frac{1}{2} \left( \lvert \pmb{u} + \pmb{v} \rvert^2 - \lvert \pmb{u} \rvert^2 - \lvert \pmb{v} \rvert^2 \right)$$

In Cartesian coordinates,

$$\pmb{u} \cdot \pmb{v} = \sum_{i=1}^n u_i v_i$$

## A 'shift' in perspective

An example of a Euclidean space used for over 2000 years is the geocentric picture of the Cosmos, which treats the earth as the origin of space. Copernicus' heliocentric model shifted the origin to the sun. Finally, Newton made the 'origin' of the universe indefinite, but he did believe it to exist somewhere out there. All these beliefs emerge from the assumption that space is an absolute structure underlying objects. However, this is incorrect. And this was realized by Galileo Galilei, who lived between the lifespans of Copernicus and Newton.

Galileo realized that the space we live in is not endowed with an absolute 'origin' whose location is irrefutable. For instance, the laws of science, as he realized, are the same in all inertial frames, and different inertial frames have different origins. He exposes his realization beautifully in [a thought experiment](https://en.wikipedia.org/wiki/Galileo%27s_ship#The_proposal) narrated by _Salviati_, his alter ego in _Dialogue Concerning the Two Chief World Systems_.

Therefore, we realize that physical space does not distinguish between coordinate systems centred at different origins. Or, the parameters living in physical space are invariant under shifting the origin of the concerned reference frame.

This idea not only has important implications in physics, but also in mathematics. The purpose of mathematics is to study structures which are internally consistent. Since physical space forms such a consistent structure, it is the job of mathematics to build on it, using a formal approach. This is why the idea of affine spaces becomes important.

## When addition is removed

Galileo's enlightenment may be summarized as physical space simply not having a unique notion of _addition_. This is motivated by the idea that adding a vector to an entire vector space corresponds to translating the vector space along the vector. This, in turn, implies shifting the origin, which Galileo found to do nothing to physical laws.

Therefore, physical space is a structure that remains invariant under translations. More generally, we consider the idea of an affine n-space, which is a vector space $$\mathbb{A}^n$$ that remains invariant under the action of the additive group of $$\mathbb{R}^n$$, $$\left( \mathbb{R}^n, + \right)$$. Thus, adding vectors in an affine space makes no sense — we can repeatedly add arbitrary vectors without changing the meaning of the original vector.

Stated differently, in $$\mathbb{A}^n$$, one can _identify_ all points, $$\pmb{u} \leftrightarrow \pmb{u} + \pmb{v} : \pmb{u}, \pmb{v} \in \mathbb{A}^n$$. Here, 'identification' is a kind of equivalence relation on the original, Euclidean space $$\mathbb{R}^n$$.

Some immediate consequences of there being no notion of addition in $$\mathbb{A}^n$$ are:

1. Suppose for a given vector $$\pmb{u} \in \mathbb{A}^n$$, one finds a vector $$- \pmb{u}$$ such that $$\pmb{u} + \left( - \pmb{u} \right) = \pmb{0}$$. Since we can add any vector $$\pmb{v}$$ to this result without changing its meaning, there cannot be a unique $$\pmb{0}$$, i.e. origin. Likewise, $$- \pmb{u}$$ is not defined.

2. The norm of a vector is not defined for similar reasons: $$\lvert \pmb{u} \rvert \leftrightarrow \lvert \pmb{u} + \pmb{v} \rvert$$.

3. Even the dot product of two vectors is not defined, as each vector can be added to a fixed third vector $$\pmb{v} \in \mathbb{R}^n$$, so that,

$$
\begin{align}
\pmb{u} \cdot \pmb{v} & \leftrightarrow \left( \pmb{u} + \pmb{w} \right) \cdot \left( \pmb{v} + \pmb{w} \right) \\
 & = \pmb{u} \cdot \pmb{v} + \left( \pmb{u} + \pmb{v} \right) \cdot \pmb{w} + \pmb{w} \cdot \pmb{w}
\end{align}
$$

The above statements quantitatively state and generalize Galileo's enlightening intuition.

## The sole survivor

Despite affine spaces rejecting most important properties of Euclidean spaces, one important notion survives: that of subtraction (and consequently, a distance function). This is because when an element of $$\left( \mathbb{R}^n, + \right)$$ acts on $$\mathbb{A}^n$$, the _same_ element is added to all vectors in the affine space $$\mathbb{A}^n$$. Therefore, subtracting any vector from another in $$\mathbb{A}^n$$ cancels out the factor added via the additive group $$\left( \mathbb{R}^n, + \right)$$. I.e. for $$\pmb{u}, \pmb{v} \in \mathbb{A}^n$$ and $$\pmb{w} \in \mathbb{R}^n$$,

$$
\begin{align}
\pmb{v} - \pmb{u} & \leftrightarrow \left( \pmb{v} + \pmb{w} \right) - \left( \pmb{u} + \pmb{w} \right) \\
 & = \pmb{v} - \pmb{u} + \pmb{w} - \pmb{w} \\
 & = \pmb{v} - \pmb{u}
\end{align}
$$

The key here is that since $$\pmb{w} \in \mathbb{R}^n$$ and _not_ $$\mathbb{A}^n$$, $$\pmb{w} - \pmb{w} = \pmb{0}$$ is unique. As a result, the $$-$$ operation in $$\mathbb{A}^n$$ is defined!

## A refinement

We've so far learnt that subtraction is immune to the generality introduced by choosing affine spaces over Euclidean spaces. However, in the process, an idea has been used informally without making it more rigorous for application to affine spaces: that of subtraction itself!

If you read this post till here, you might've asked: if there isn't a unique notion of $$- \pmb{u}$$ given some $$\pmb{u} \in \mathbb{A}^n$$, what does the expression $$\pmb{v} - \pmb{u}$$ mean? Well, it doesn't mean adding $$- \pmb{u}$$ to $$\pmb{v}$$ anymore! Rather, $$-$$ is an operator on its own, $$- : \mathbb{A}^n \times \mathbb{A}^n \mapsto \mathbb{R}^n$$, defined as,

$$\left( \pmb{v} - \pmb{u} \right)_i = v_i - u_i$$

It must be made clear that the subtraction operator appearing on the right hand side is scalar subtraction, while the one on the left is vector addition. The difference between the two is that for a scalar, $$-k = \left( -1 \right) k$$, but for a vector in an affine space, $$- \pmb{u} \neq \left( -1 \right) \pmb{u}$$. That's because in an affine space, since adding vectors doesn't yield a unique value, neither does repeatedly adding a vector to itself (which is nothing but scalar multiplication of the vector).

## Conclusion

The previous paragraph may give rise to a further question: why must we restrict ourselves to a scalar version of subtraction which works as $$-k = \left( -1 \right) k$$? If we don't, the components of vectors in affine space are no longer elements of $$\mathbb{R}$$, but $$\mathbb{A}$$. But there is no unique bilinear map $$\mathbb{R}^n \mapsto \mathbb{A}$$. As a result, we cannot say, for instance, that the first component of a vector in $$\mathbb{R}^n$$ with coordinates $$\left( x, y, z \right)$$ is indeed $$x$$ and not $$x+h$$ for some arbitrary $$h$$. As a consequence, the following fact about affine spaces, expressed in Cartesian coordinates, is destroyed,

$$\pmb{u} = \sum_{i=1}^n u_i \pmb{e}_i$$

where $$\pmb{e}_i$$ is the tuple with the entry $$1$$ in the $$i^{\text{th}}$$ place, and $$0$$ elsewhere.

With the above identity being destroyed, the idea of choosing basis vectors to represent any vector is also destroyed. Or, the information about a vector is no longer conserved between different coordinate systems! But information must be a coordinate-independent entity, as coordinates only reflect a particular perspective.

All-in-all, we have a neat picture of subtraction: it is more fundamental than addition, upto vectors. By 'fundamental', we mean 'well-defined in a large number of scenarios'.

So, that's it, folks! Thanks a lot for reading up to here! If you like reading my posts, do leave a thumbs up in the comments section, it greatly motivates me to write more posts like these! :)
