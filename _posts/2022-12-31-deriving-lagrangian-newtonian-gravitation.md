---
title: "Deriving the Lagrangian Density for Newtonian Gravitation"
description: "How to apply the Klein-Gordon theory to gravity"
categories: [classical field theory]
tags: [gravitation, Klein-Gordon theory, energy-momentum tensor]
---

## Some thoughts

Hi everyone! It's been a few months since the last post here. A lot (of good) has happened since then. Before coming to that, I would like to thank all my old readers for sticking by, and new readers for visiting this blog :) I hope you all are doing great and had a wonderful year ... Happy New Year!

In this period of absence, I've been working on university life in a beautiful, new country. It's been a transforming, memorable and educational experience and I'm excited about the journey up ahead!

It's taken a while balancing time between academics and personal life. This has given me the chance to mingle with and learn from very amiable and knowledgeable individuals. Such experiences have motivated me to keep exploring mathematical physics, and plan a layout for future blog posts.

In the upcoming year, I plan to expand the contents of this blog both in breadth and depth. For instance, I hope to employ deeper mathematical tools, intuition in physics, and dedicated posts on philosophy. But it is the time that shall judge this resolution — assuming it exists (half-kidding)!

Let us begin this journey into the yet-to-be-known with a little problem in classical field theory. Before asking and answering the question, we will quickly highlight some aspects of Newtonian gravitation. The main exploration begins in [Constructing the Lagrangian](#constructing-the-lagrangian).

## The problem

### Gravitation is a field theory

The theory of Newtonian gravitation is a field theory in the framework of Newtonian mechanics. This means that the Newtonian gravitational field obeys certain fundamental properties, namely,

- Principle of stationary action

- Galilean invariance

Unfortunately, the Newtonian gravitational field does not obey locality and local Lorentz invariance, making it a non-relativistic classical field theory. These limitations are overcome by Einstein's general theory of relativity, where the gravitational 'field' is the metric of spacetime itself. As this is a substantially different model, we will stick to the non-relativistic classical field theory for the moment.

Since we assume that the gravitational field, henceforth written as the gravitational potential $$\phi$$, obeys the principle of stationary action, it must also obey the Euler-Lagrange equation, given by,

$$\frac{\partial \mathcal{L}}{\partial \phi} - \nabla_i \pi^i = 0$$

where $$\mathcal{L}$$ is an appropriate Lagrangian density for the theory and $$\pi^i$$ is the conjugate momentum tensor, $$\displaystyle{ \pi^i = \frac{\partial \mathcal{L}}{\partial \left( \partial_i \phi \right)} }$$.

Furthermore, to incorporate manifest Galilean invariance, we will use tensor notation with spacelike indices, as above.

### Poisson's equation

From Gauss's law for Newtonian gravitation, we have,

$$\nabla_i g^i = - 4 \pi G \rho$$

where $$g^i$$ is the acceleration due to gravity.

The above is the differential form of Gauss's law for Newtonian gravitation, which can be derived from the integral form using the divergence theorem.

Since the gravitational force field is conservative, i.e. the change in gravitational potential energy along a trajectory only depends on its endpoints, the acceleration due to gravity can be written as the negative gradient of the gravitational potential energy,

$$g^i = - \partial^i \phi$$

Putting the two above equations together, we get what is known as Poisson's equation for the gravitational field i.e. gravitational potential,

$$\nabla_i \partial^i \phi - 4 \pi G \rho = 0$$

### What is a suitable Lagrangian?

We have discussed how Newtonian gravitation is a non-relativistic classical field theory with a Lagrangian description and with the field equation being that of Poisson's equation. This means that the Lagrangian density (called Lagrangian for brevity) for Newtonian gravitation must yield Poisson's equation as the equation of motion.

This begs the question, which Lagrangian is suitable for the above requirements? And how can we construct it using reasonable assumptions?

It turns out that an interesting way to answer the above questions is to apply a modified form of the Klein-Gordon theory to the field theory of gravity. Let us see how.

## Constructing the Lagrangian

### Restricting the Lagrangian

An unexpectedly useful way to restrict the form of the Lagrangian for a scalar field such as the gravitational potential, is to assert the symmetry of the field's energy-momentum tensor. We did this in [Scalar Field Lagrangian From Symmetry Considerations]({% post_url 2021-12-01-scalar-field-lagrangian-symmetry-considerations %}) to find that the form of the Lagrangian must be severely restricted to that of the Klein-Gordon Lagrangian.

Much of the reasoning and mathematical machinery remains the same in this case. The only difference is that here, we have Galilean invariance instead of Lorentz invariance. This translates to asserting the symmetry of the 3-dimensional stress-energy tensor instead of the full 4-dimensional energy-momentum tensor. Thus,

$$
\begin{align}
T^{i j} & = \pi^i \partial^j \phi - g^{i j} \mathcal{L} \\
T^{i j} & = T^{j i}
\end{align}
$$

which is true in the most general case only when,

$$
\begin{align}
\pi^i \partial^j \phi & = \pi^j \partial^i \phi \\
\pi^i & = \partial^i \phi \\
\frac{\partial \mathcal{L}}{\partial \left( \partial_i \phi \right)} & = \partial^i \phi
\end{align}
$$

which is further generally true when the Lagrangian is of the form,

$$\mathcal{L} = \frac{1}{2} \partial_i \phi \partial^i \phi - V \left( \phi \right)$$

The above is essentially a Klein-Gordon Lagrangian in a potential, but with spacelike indices instead of spacetimelike indices.

### Poisson meets Poisson

The equation of motion obtained by varying the above Lagrangian with respect to the field, unsurprisingly, resembles the Klein-Gordon equation but with spacelike indices only,

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \phi} - \nabla_i \pi^i & = 0 \\
\implies \nabla_i \partial^i \phi + \frac{\partial V}{\partial \phi} & = 0
\end{align}
$$

This equation has the form of — drum rolls please — the general Poisson's equation relating the Laplacian $$\nabla_i \partial^i$$ of a function to another function. It is, therefore, no coincidence that the field equation for the gravitational potential is called Poisson's equation — it is a special case of the former, general case (this nomenclature is a little tricky, to be fair).

All we now have to do is figure out which scalar potential $$V \left( \phi \right)$$ makes Poisson's general equation reduce to Poisson's equation for Newtonian gravitation. Then, we can find the suitable Lagrangian for the theory by plugging this potential into the modified Klein-Gordon Lagrangian we deduced.

### The scalar potential

Mapping the two forms of Poisson's equation, one from the world of mathematics and the other from that of physics (in a certain sense), we find,

$$\frac{\partial V}{\partial \phi} = - 4 \pi G \rho$$

Obviously, $$V \left( \phi \right)$$ is of the form below, up to linearity,

$$V \left( \phi \right) = - 4 \pi G \rho \phi$$

### The Lagrangian

Plugging in the above into the form of the Lagrangian we argued for the scalar gravitational potential, we get,

$$\mathcal{L} = \frac{1}{2} \partial_i \phi \partial^i \phi + 4 \pi G \rho \phi$$

Notice that scaling the Lagrangian does not change the equation of motion obtained (which engenders the physics) and scaling by a factor of $$\displaystyle{\frac{1}{4 \pi G}}$$ yields a Lagrangian seen frequently in the physics literature,

$$\mathcal{L} = \frac{1}{8 \pi G} \partial_i \phi \partial^i \phi + \rho \phi$$

## Conclusion (Second Law of Epistemological Thermodynamics)

Just like the Second Law of Thermodynamics states that the entropy of the universe can only increase with time, we propose the Second Law of Epistemological Thermodynamics:

> The number of questions regarding the universe can only increase with time.

An immediate result of the above law is that we are not yet done when it comes to deriving a Lagrangian for Newtonian gravitation. 

The Lagrangian surely yields the correct equation of motion (Poisson's equation) when varied with respect to the gravitational potential $$\phi$$. However, notice what happens when we vary it with with respect to the mass density field $$\rho$$,

$$\phi = 0$$

Oops! We now have two possible paths ahead of us:

- Promote mass density $$\rho$$ to a non-relativistic classical field and modify the Lagrangian to yield Poisson's equation when varied with respect to both the fields $$\phi$$ and $$\rho$$.

- Justify if $$\rho$$ is not a classical field and elaborate on its role in Poisson's equation and the associated Lagrangian, rigorously.

We will take one of these paths in a new post, soon. For now, which path will be adopted in the post will be akin to a stochastic process as I am not very sure of the outcome myself. However, I'm studing the topic in my free time and am beginning to see interesting connections between both the possibilities. In fact, they may be simultaneously workable, if we generalize classical fields to include distributions! And this seems to yield Newtonian particle dynamics for free solely from field-theoretic constructs.

I'm excited to write posts on the above topics soon. Until then, Bonne Année once more and take care! Thanks for reading and hope you enjoyed :)