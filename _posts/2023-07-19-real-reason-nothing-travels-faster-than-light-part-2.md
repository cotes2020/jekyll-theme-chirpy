---
title: "The Real Reason Nothing Travels Faster Than Light: Part 2 (Dynamical Interpretation)"
description: "Interpreting the dynamics of the well-known statement"
categories: [relativity]
tags: [4-momentum]
---

In [The Real Reason Nothing Travels Faster Than Light]({% post_url 2022-04-16-real-reason-nothing-travels-faster-than-light %}), we analytically proved that faster-than-light motion is logically impossible within the framework of special relativity. In doing so, we disproved the very kinematics of the situation $$v>c$$. This, as we discussed, is logically stronger than disproving the dynamics of the situation, i.e. the question of how such a velocity can be achieved through, say, acceleration. In this post, we will specifically look at the dynamical aspect, and how it agrees with the stronger kinematical argument.

When it comes to this topic, there are multiple layers of common misconception. The first is the supposition that disproving the dynamics also disproves the kinematics, i.e. if one can show that faster-than-light velocity cannot be achieved via any physical process, it must be theoretically impossible given the working of spacetime itself. While such an implication would be nice, it is not always true. After all, no object can be accelerated to the speed of light, and yet, light travels at, well, the speed of light! This makes sense in special relativity because things which are travelling at the speed of light cannot be accelerated in the first place, by virtue of having an invariant speed. In other words, the argument 'impossible dynamics implies impossible kinematics' implicitly assumes $$v < c$$, which in turn assumes that we are dealing with objects carrying positive mass. Thus, the argument fails for $$v \geq c$$ and we are forced to base kinematical arguments solely on the kinematics as opposed to the dynamics.

The second layer of popular misconception is the notion that theoretically, one would strictly need infinite energy to accelerate a body of positive mass from $$v<c$$ to $$v = c$$. It turns out that this is not true! In general, a body would need to convert all its mass to relativistic kinetic energy in order to approach the speed of light — but that means it would cease to exist as matter by the time it reaches the speed of light! [^1]

[^1]: That said, it is indeed true that if the body were to be accelerated to the speed of light _without_ a change in its mass, it would require infinite energy. However, this case leaves out the possibility of the body propelling itself, which does not necessarily involve infinite energy.

## Propulsion

### Conservation laws

Suppose an accelerating body has its mass as a function of proper time, $$m \left( \tau \right)$$ (which is a Lorentz invariant). Assuming the body's dynamics are invariant under linear translation, [Noether's theorem]({% post_url 2022-03-30-conservation-mass-classical-mechanics %}#noethers-theorem) implies the [conservation of total linear momentum]({% post_url 2022-03-30-conservation-mass-classical-mechanics %}#form-of-linear-momentum) of all the constituent parts of the body. [^2]

[^2]: Though the linked post talks about linear momentum in classical mechanics, special relativity works in an analogous way, at least in the above context.

In the case of a body accelerating by virtue of changing mass, there are two distinct parts:

- The 'main' part of the body whose accelerating motion is being tracked. The mass of this part evolves with proper time as $$m \left( \tau \right)$$.

- The part of the body thrust out from the main part. The mass of this evolves with proper time as $$m \left( 0 \right) - m \left( \tau \right)$$ assuming, without loss of generality, that the main body starts losing mass at $$\tau = 0$$.

Let the 4-velocity of the first part in the frame of an observer tracking the body be labelled as $$U^\mu \left( \tau \right)$$. Let the corresponding 3-speed be $$u \left( \tau \right) = u_i \left( \tau \right) u^i \left( \tau \right)$$ such that $$U^i \left( \tau \right) = \gamma \left( u_i \left( \tau \right) u^i \left( \tau \right) \right) u^i \left( \tau \right)$$. Similarly, let the 4-velocity and linked 3-speed of the ejection being thrust behind be denoted as $$V^\mu \left( \tau \right)$$ and $$v \left( \tau \right)$$, respectively.

Last but not the least, without loss of generality, let the initial 4-velocity of the body be zero.

Then, by conservation of 4-momentum of the main and ejected parts of the body combined,

$$m \left( \tau \right) U^\mu \left( \tau \right) + \left[ m \left( 0 \right) - m \left( \tau \right) \right] V^\mu \left( \tau \right) = 0$$

We will now focus only on the timelike part of the above equation,

$$
\begin{align}
\gamma \left( u \left( \tau \right) \right) \left[ m \left( \tau \right) \right]^2 c + \gamma \left( v \left( \tau \right) \right) \left[ m \left( 0 \right) - m \left( \tau \right) \right]^2 c & = 0 \\
\gamma \left( u \left( \tau \right) \right) \left[ m \left( \tau \right) \right]^2 + \gamma \left( v \left( \tau \right) \right) \left[ m \left( 0 \right) - m \left( \tau \right) \right]^2 & = 0
\end{align}
$$

It is worth mentioning that multiplying both sides of the above equation by $$c^2$$ turns it to a statement about the conservation of total relativistic kinetic energy. More on relativistic kinetic energy can be found on the [previous post]({% post_url 2023-07-04-reconstructing-relativistic-kinetic-energy %}). Briefly, at the end of the linked post, we found that for a body of mass $$m$$ moving with a 3-speed $$v$$, its relativistic kinetic energy is given by $$E^2 = \left( mc^2 \right) + \left( \gamma mvc \right)^2$$. For bodies moving slower than the speed of light (or equivalently, those with positive mass), the equation can be transformed to yield,

$$
\begin{align}
E^2 & = m^2 c^2 \left( c^2 + \gamma^2 v^2 \right) \\
& = m^2 c^4 \left( 1 + \frac{v^2}{c^2} \frac{1}{1 - \frac{v^2}{c^2}} \right) \\
& = m^2 c^4 \left( 1 + \frac{v^2}{c^2 - v^2} \right) \\
& = m^2 c^4 \left( \frac{c^2}{c^2 - v^2} \right) \\
& = m^2 c^4 \left( \frac{1}{1 - \frac{v^2}{c^2}} \right) \\
& = \gamma^2 m^2 c^4 \\
\therefore E & = \gamma mc^2
\end{align}
$$

Armed with these ideas, let us see how the conservation of total 4-momentum applies to an speeding particle on a diet.

### Solving for 3-speed

Using the timelike part of the equation for the conservation of the added 4-momenta of an accelerating particle and the mass ejected,

$$\gamma \left( u \left( \tau \right) \right) \left[ m \left( \tau \right) \right]^2 + \gamma \left( v \left( \tau \right) \right) \left[ m \left( 0 \right) - m \left( \tau \right) \right]^2 = 0$$

$$
\begin{align}
c^{-1} \gamma \left( u \left( \tau \right) \right) & = - \left[ m \left( \tau \right) \right]^{-2} c^{-1} \gamma \left( v \left( \tau \right) \right) \left[ m \left( 0 \right) - m \left( \tau \right) \right]^2 \\
\left( c^2 - \left( u \left( \tau \right) \right)^2 \right)^{-1/2} & = - \left[ m \left( \tau \right) \right]^{-2} \left[ c^2 - \left( v \left( \tau \right) \right)^2 \right]^{-1/2} \left[ m \left( 0 \right) - m \left( \tau \right) \right]^2 \\
c^2 - \left( u \left( \tau \right) \right)^2 & = \left[ m \left( \tau \right) \right]^4 \left[ c^2 - \left( v \left( \tau \right) \right)^2 \right] \left[ m \left( 0 \right) - m \left( \tau \right) \right]^{-4} \\
\left( u \left( \tau \right) \right)^2 & = c^2 - \left[ m \left( \tau \right) \right]^4 \left[ c^2 - \left( v \left( \tau \right) \right)^2 \right] \left[ m \left( 0 \right) - m \left( \tau \right) \right]^{-4} \\
u \left( \tau \right) & = c \sqrt{1 - \frac{\left[ m \left( \tau \right) \right]^4}{\left[ \gamma \left( v \left( \tau \right) \right) \right]^2 \left[ m \left( 0 \right) - m \left( \tau \right) \right]^4}} \\
& = c \sqrt{1 - \kappa \left( m \left( \tau \right), v \left( \tau \right) \right)} & [\text{say}]
\end{align}
$$

Clearly, the above 3-speed is:

- Lesser than $$c$$ **iff**, $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right) > 0$$.

- Equal to $$c$$ **iff**, $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right) = 0$$.

- Greater than $$c$$ **iff**, $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right) < 0$$.

By construction, $$m \left( \tau \right) \in \left[ 0, \infty \right)$$ and $$v \left( \tau \right) \in \mathbb{R} \implies \gamma \left( v \left( \tau \right) \right) \in \left( 1, \infty \right) \cup i \left( 0, \infty \right)$$. Therefore, $$\left[ \gamma \left( v \left( \tau \right) \right) \right]^2 \in \left( - \infty, 0 \right) \cup \left( 1, \infty \right)$$. The below follows from these notions.

#### Subluminal ejection

If $$v \left( \tau \right) \in \left( -c, c \right)$$, i.e. $$\gamma \left( v \left( \tau \right) \right) \in \left( 1, \infty \right)$$,

- $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right) > 0$$ **iff**, $$m \left( \tau \right) > 0$$ **and** $$\left[ m \left( 0 \right) - m \left( \tau \right) \right]^4 \in \left( 0, \infty \right)$$.

- $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right) = 0$$ **iff**, $$m \left( \tau \right) = 0$$ **or** $$\left[ m \left( 0 \right) - m \left( \tau \right) \right]^4 = \infty$$.

- $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right) < 0$$ **iff**, $$m \left( \tau \right) > 0$$ **and** $$\left[ m \left( 0 \right) - m \left( \tau \right) \right]^4 \in \left( - \infty, 0 \right)$$.

The first situation describes how when the body's mass is positive and decreasing, its velocity is always subluminal and vice-versa.

On the other hand, the second case suggests that the body will reach the speed of light only when it entirely loses its mass (or has an infinite initial mass, which is unreasonable).

Last but not the least, by construction, the third case is not possible as the masses involved are strictly real (and positive).

#### Light-speed ejection

When $$v \left( \tau \right) = \pm c$$, we have, $$\gamma \left( v \left( \tau \right) \right) = \infty$$, yielding $$u \left( \tau \right) = c$$.

Hence, if any part of the accelerating body with non-zero mass is thrust out at the speed of light, the main part will also shoot off at the speed of light.

#### Superluminal ejection

Given $$v \left( \tau \right) \in \left( - \infty, c \right) \cup \left( c, \infty \right)$$ i.e. $$\left[ \gamma \left( v \left( \tau \right) \right) \right]^2 \in \left( - \infty, 0 \right)$$,

- $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right) > 0$$ **iff**, $$m \left( \tau \right) > 0$$ **and** $$\left[ m \left( 0 \right) - m \left( \tau \right) \right]^4 \in \left( - \infty, 0 \right)$$.

- $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right) = 0$$ **iff**, $$m \left( \tau \right) = 0$$ **or** $$\left[ m \left( 0 \right) - m \left( \tau \right) \right]^4 = \infty$$.

- $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right) < 0$$ **iff**, $$m \left( \tau \right) > 0$$ **and** $$\left[ m \left( 0 \right) - m \left( \tau \right) \right]^4 \in \left( 0, \infty \right)$$.

These are identical to the situations in the case of subluminal ejection, except the first and third of those scenarios have been exchanged. Then, by similar reasoning as before, we find the following.

The first situation is not possible due to the fact that mass is not allowed to be non-real, let alone negative.

The second situation is reasonable only when the body loses its entire mass (as opposed to having infinite initial mass), thereby travelling at the speed of light.

The third case entails superluminosity of the object when its rate of losing mass is positive.

To summarize, decreasing mass is a general way to make an object approach the speed of light. If the ejection is subluminal, so is the body, while if the ejection occurs at the speed of light, then the body also travels at the speed of light and so on. In short, the nature of the speed of the object (subluminosity) is tied to that of the ejected part, via conservation of relativistic kinetic energy.

## Dynamics of propulsion

### Decreasing mass

From the above observations, as $$m \left( \tau \right) \to 0^+$$ for some $$\tau \in \mathbb{R}$$, no matter the velocity of the ejected part, $$u \left( \tau \right)$$ approaches $$c$$. It can be shown that in particular, when $$\left\lvert v \left( \tau \right) \right\rvert < c$$, $$u \left( \tau \right) \to c^-$$ while when $$v \left( \tau \right) = c$$, $$u \left( \tau \right) \to c$$ and lastly, when $$\left\lvert v \left( \tau \right) \right\rvert > c$$, $$u \left( \tau \right) \to c^+$$, at least assuming that $$\displaystyle{\exists \: \widetilde{\tau} \in \mathbb{R} : \lim_{\tau \to \widetilde{\tau}} m \left( \tau \right) = 0^+}$$.

### Increasing ejection velocity

As $$v \left( \tau \right) \to c$$ for some $$\tau \in \mathbb{R}$$, we see that $$u \left( \tau \right)$$ approaches $$c$$. Specifically, for $$v \left( \tau \right) \to c^-$$, $$u \left( \tau \right) \to c^-$$. Similarly, when $$v \left( \tau \right) \to c^+$$ around some $$\tau \in \mathbb{R}$$, $$u \left( \tau \right) \to c^+$$. Last but not the least, at $$v \left( \tau \right) = c$$, $$u \left( \tau \right) = c$$.

The above properties also demonstrate that $$\kappa \left( m \left( \tau \right), v \left( \tau \right) \right)$$ is continuous with respect to both its arguments.

## Summary + Desmos playground

To summarize the above ideas, a body with parts not moving at the speed of light, cannot reach the speed of light unless it eventually thrusts out all its parts!

This neatly obeys the idea that a body with positive mass can only approach the speed of light but never reach it and that only massless objects travel at the speed of light. Such statements are kinematical in nature and as discussed, logically severe and it is nice to have verified that they hold even in extreme situations such as objects thrusting out their entire mass or ejecting parts with speeds approaching that of light (even though these facts are guarunteed from abstract arguments based on the nature of spacetime).

As they say, a picture is worth a thousand words. So, here is a little Desmos graph showing the relationship between a self-propelling body's mass $$m \left( \tau \right)$$, its speed $$u \left( \tau \right)$$ and the speed of the ejected material, $$v \left( \tau \right)$$,

<iframe src="https://www.desmos.com/calculator/o754r2mx5z?embed" width="500" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>

Feel free to play around with the quantities shown in the iframe by clicking the '[edit graph on desmos](https://www.desmos.com/calculator/o754r2mx5z)' button!

And with that, folks, comes an end to our little two-post discussion on why nothing can travel faster than light :)