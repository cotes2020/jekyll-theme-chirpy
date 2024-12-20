---
title: Time Dilation
author: jake
date: 2024-12-19 12:00:00 +0800
categories: [Physics]
tags: []
math: true
mermaid: false
image:
  path: assets/img/custom/time_dilation.png
  alt: Time dilation on a moving train from the perspective of a stationary observer
---

A derivation of time dilation from [Modern Physics 3rd Edition](https://www.cengage.com/c/modern-physics-3e-serway-moses-moyer/9780534493394/) Section 1.5. 

## Experimental Setup
- A moving observer on a train moving with speed $v$.
- A stationary observer that is *not* on the train.
- The moving observer reflects a light beam off of a mirror moving with the train that is $d$ units away.
- The stationary observer also observes the reflected light beam, but with the interaction of the train's velocity.

> We will be using the two postulates of [special relativity](https://en.wikipedia.org/wiki/Special_relativity);
> - laws of physics are invariant in all inertial frames of reference (principle of relativity).
> - all observers see light travel at the same speed, $c$.
{:.prompt-tip}

Although the light beam's origin will be the same for both observers, due to the principle of relativity, we can "shift" the stationary observers origin to form a right triangle (which is helpful for the math).

## Solution
Our goal will be to relate a change in time as measured in $\Delta t$ to the change in time as observed on the moving train $\Delta t'$. First, we can compute $\Delta t'$ using $v = \Delta x / \Delta t$:

$$
\begin{equation}
\Delta t' = \frac{2d}{c}
\end{equation}
$$

In the stationary frame of reference, we see the moving observer move a distance of $v \Delta t / 2$ and we see the light travel diagonally a distance of $c \Delta t / 2$ for a total vertical height of $d$. We can relate these quantities using the pythagorean theorem for this right triangle:

$$
\begin{flalign*}
(c\Delta t / 2)^2 &= d^2 + (v \Delta t / 2)^2 \\
c^2 \Delta t^2 / 4 - v^2 \Delta t^2 / 4 &= d^2 \\
\Delta t^2 \cdot (\frac{c^2 - v^2}{4}) &= d^2 \\
\Delta t^2 &= \frac{4d^2}{c^2 - v^2} \\
\Delta t &= \frac{2d}{\sqrt{c^2 - v^2}} \\
&= \frac{2d}{\sqrt{c^2(1 - v^2 / c^2)}} \\
&= \frac{2d}{c \sqrt{(1 - v^2 / c^2)}} \\ 
&= \frac{2d}{c}\cdot \frac{1}{\sqrt{(1 - v^2 / c^2)}} \\
& = \Delta t' \cdot \frac{1}{\sqrt{(1 - v^2 / c^2)}} \\
& = \gamma \Delta t'
\end{flalign*}
$$

Since $\gamma >= 1$, we know that $\Delta t > \Delta t'$ and that the stationary observer's time has *dilated* (i.e. increased) when viewed from the moving observer's train car. 

## Interpretation
Another interpretation is that since **the distance the light travels in the stationary frame is strictly greater than the moving frame** (hypotenuse vs. leg of a right triangle):

$$
\begin{flalign*}
c\Delta t / 2 &> d \\
c\Delta t &> 2d 
\end{flalign*}
$$

and **the speed of light $c$ has to be the same in each reference** then **the time in the stationary frame *has* to be longer** to account for this extra distance. For example, for $\Delta t$ to double with respect to $\Delta t'$, the train would need a velocity of:

$$
\begin{flalign*}
\gamma &= 2 \\
\frac{1}{\sqrt{(1 - v^2 / c^2)}} &= 2 \\
\frac{1}{(1 - v^2 / c^2)} &= 4 \\
\frac{1}{4} &= 1 - v^2 / c^2 \\
v^2 / c^2 &= 3/4 \\
v &= \frac{\sqrt{3}}{2} c \\
&=0.87 \cdot c
\end{flalign*}
$$

but both observers will still see light moving at speed, $c$!