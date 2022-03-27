---
title: "Thomson's Lamp and Multivaluedness"
description: "The famous lamp will keep you in the dark, no more"
categories: [analysis]
tags: [supertasks]
---

## A Supertask
A supertask is a countably infinite sequence of tasks or events, which occurs in a finite amount of time. The word 'supertask' was coined by the twentieth-century philosopher James F. Thomson. He went on to provide an example of a supertask that soon became his namesake philosophical puzzle.

The problem may be stated as: we are given a hypothetical lamp with a timer. When the timer is started, the lamp is turned on. After the passage of $$1$$ minutes, the lamp is turned off. After $$\frac{1}{2}$$ minutes, the lamp is turned back off, then again to on after $$\frac{1}{4}$$ minutes, and so on.

These time intervals can be added as a converging infinite series, which adds up to precisely $$2$$ minutes:

$$
\begin{align}
\text{Let } S & = \sum_{n=0}^\infty \frac{1}{2^n} \\
\therefore \frac{1}{2} S & = \frac{1}{2} \sum_{n=0}^\infty \frac{1}{2^n} \\
 & = \sum_{n=0}^\infty \frac{1}{2} \cdot \frac{1}{2^n} \\
 & = \sum_{n=0}^\infty \frac{1}{2^{n+1}} \\
 & = \sum_{n=1}^\infty \frac{1}{2^n} \\
 \implies S - \frac{1}{2} S & = \sum_{n=0}^\infty \frac{1}{2^n} - \sum_{n=1}^\infty \frac{1}{2^n} \\
 \frac{1}{2} S & = \frac{1}{2^0} \\
 \implies S & = 2
\end{align}
$$

The question is, what is the state of the lamp exactly when two minutes have elapsed?

## Discrete state update

### Diagrams

As there are many plots for this post, I thought of showing them together as [Desmos](https://www.desmos.com/calculator) embeds below. The two embeds are for the discrete and continuous approaches, respectively:

<iframe src="https://www.desmos.com/calculator/yyaqqboxun?embed" width="500" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>

<iframe src="https://www.desmos.com/calculator/na17prww8s?embed" width="500" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>

### Reparameterization

Firstly, let us restate the problem of Thomson's lamp in mathematical terms. We are given a discrete system (the lamp) $$L$$ with a degree of freedom or state, say $$\sigma$$. $$\sigma$$ has exactly $$2$$ possible values, corresponding to 'off' and 'on'. Let us label these states as $$\sigma = 0$$ and $$\sigma=1$$ respectively. Therefore $$\sigma \in \left\{ 0, 1 \right\}$$.

Since $$L$$ updates its state discretely, we can count its updates using a 'state parameter' $$n$$ which increments with each state update as $$n=0$$, $$n=0$$, $$n=2$$ and so on. For any step $$n$$, we have a state $$\sigma \left( n \right)$$. We are given the initial condition $$\sigma \left( 0 \right) = 1$$.

Now, we have a continuously-updating system $$T$$, namely the timer. It returns a parameter $$t$$, measured in minutes. At $$t=0$$, $$n=0$$. When $$t$$ becomes $$1$$, $$n$$ updates to $$1$$. After the passage of $$\frac{1}{2}$$ minutes, i.e. at $$t = 1 + \frac{1}{2}$$, $$n=2$$. Similarly, at $$t = 1+ \frac{1}{2} + \frac{1}{4}$$, $$n = 3$$. We see a pattern emerging which can be summarized as:

$$\left[ t \left( n \right) \right]_{n-1 \to n} = \sum_{k=0}^{n} \frac{1}{2^k}$$

The subscript $$n-1 \to n$$ in L.H.S. indicates that precisely when the step parameter updates from a previous value ($$n-1$$) to $$n$$, $$t$$ will be measured to be its specified value. However, until the next state update, $$n$$ does not update, whereas $$t$$ continuously increases as it is a continuously-changing parameter. As a consequence, $$t \left( n \right)$$ is not an injective function, i.e. for a given $$n$$, there are many $$t$$. For this reason, it is better to express any parameter in terms of $$t$$ and not $$n$$. Let us begin by finding $$n \left( t \right)$$.

Firstly, recall the floor function $$\left\lfloor \: \right\rfloor : \mathbb{R} \mapsto \mathbb{Z}$$ which for a real argument $$x$$ returns the greatest integer less than or equal to $$x$$. For example, $$\left\lfloor 3.141 \right\rfloor = 3$$ and $$\left\lfloor -2.718 \right\rfloor = -3$$.

Now, we will find $$n \left( \tau \right)$$ for any $$t = \tau$$ such that at that instant, $$n-1 \to n$$. Then, we will find $$n \left( t \right)$$ for any $$t$$ by employing the floor function.

$$
\begin{align}
\tau \left( n \right) & = \sum_{k=0}^{n} \frac{1}{2^k} \\
\frac{1}{2} \tau \left( n \right) & = \sum_{k=0}^n \frac{1}{2^{k+1}} = \sum_{k=1}^{n+1} \frac{1}{2^k} \\
\therefore \tau \left( n \right) - \frac{1}{2} \tau \left( n \right) & = \sum_{k=0}^{n} \frac{1}{2^k} - \sum_{k=1}^{n+1} \frac{1}{2^k} \\
\frac{1}{2} \tau \left( n \right) & = \frac{1}{2^0} - \frac{1}{2^n} = 1 - \frac{1}{2^n} \\
\tau \left( n \right) & = 2 - \frac{1}{2^{n-1}} \\
\therefore \frac{1}{2^{n-1}} & = 2 - \tau \left( n \right) \\
2^{1-n} & = 2 - \tau \left( n \right) \\
1-n & = \log_2 \left( 2 - \tau \left( n \right) \right) \\
n \left( \tau \right) & = 1 - \log_2 \left( 2 - \tau \right)
\end{align}
$$

In the time interval $$t \in \left( \tau \left(n \right), \tau \left( n+1 \right) \right)$$, $$n$$ does not update at all. Similarly, $$\left\lfloor \log_2 \left( 2 - t \right) \right\rfloor$$ does not update too. The reason is that the mentioned interval excludes the times $$t = \tau \left( n \right)$$ and $$t = \tau \left( n+1 \right)$$, and at these instants,

$$
\begin{align}
\left\lfloor \log_2 \left( 2 - \tau \left( n \right) \right) \right\rfloor & = \left\lfloor \log_2 \left( 2 - \left( 2 - 2^{1-n} \right) \right) \right\rfloor \\
 & = \left\lfloor \log_2 \left( 2^{1-n} \right) \right\rfloor \\
 & = \left\lfloor 1-n \right\rfloor \\
 & = 1-n \\
\left\lfloor \log_2 \left( 2 - \tau \left( n+1 \right) \right) \right\rfloor & = \left\lfloor \log_2 \left( 2 - \left( 2 - 2^{1-n-1} \right) \right) \right\rfloor \\
 & = \left\lfloor \log_2 \left( 2^{-n} \right) \right\rfloor \\
 & = \left\lfloor -n \right\rfloor \\
 & = -n
\end{align}
$$

As no integer lies between $$-n$$ and $$1-n$$, $$\left\lfloor \log_2 \left( 2 - \tau \right) \right\rfloor$$ does not update in the open interval $$\left( n, n+1 \right)$$. Due to the way the floor function is defined, the function $$\left( 1-\left\lfloor \log_2 \left( 2 - \tau \right) \right\rfloor \right)$$ remains at the same value as at $$t = \tau \left( n \right)$$, namely $$\left( 1 - \log_2 \left( 2 - \tau \right) \right)$$. At $$t = \tau \left( n+1 \right)$$, however, $$\left( 1-\left\lfloor \log_2 \left( 2 - \tau \right) \right\rfloor \right)$$ updates to $$\left( 1- \log_2 \left( 2 - \tau \left( n+1 \right) \right) \right) = n \left( \tau \right) + 1 = n \left( t \right)$$. Thus, we find that in general,

$$n \left( t \right) = 1-\left\lfloor \log_2 \left( 2 - t \right) \right\rfloor$$

Thus, we have parameterized the step parameter in terms of $$t$$. Now let us parameterize the state function $$\sigma$$ similarly.

### State function

When Thomson's lamp is first turned on along with the timer, $$n = 0$$ and $$\sigma = 1$$. In the next state update, $$n=1$$ and the lamp is switched off, giving $$\sigma = 0$$. We notice that $$\sigma \left( n \right)$$ keeps alternating between $$1$$ and $$0$$. Such a function can be written as a modulo operation,

$$\sigma \left( n \right) = \text{mod} \left( n+1, 2 \right)$$

Where $$\text{mod} \left( a, b \right)$$ is defined as the remainder obtained on long division of $$a$$ by $$b$$. In conjunction with [Donald Knuth's definition of the modulo operation](https://en.wikipedia.org/wiki/Modulo_operation#Variants_of_the_definition), since we have the strict relation $$a = b \left\lfloor \frac{a}{b} \right\rfloor + \text{mod} \left( a, b \right)$$, we can extend this relation implicitly into the reals and define the modulo operation as,

$$\text{mod} \left( a, b \right) = a - b \left\lfloor \frac{a}{b} \right\rfloor$$

Therefore, we have,

$$\sigma \left( n \right) = n + 1 - 2 \left\lfloor \frac{n+1}{2} \right\rfloor$$

Substituting $$n = n \left( t \right) = 1-\left\lfloor \log_2 \left( 2 - t \right) \right\rfloor$$ will now give us $$\sigma \left( t \right)$$ explicitly:

$$
\begin{align}
\sigma \left( t \right) & = 1-\left\lfloor \log_2 \left( 2 - t \right) \right\rfloor + 1 - 2 \left\lfloor \frac{1-\left\lfloor \log_2 \left( 2 - t \right) \right\rfloor + 1}{2} \right\rfloor \\
\sigma \left( t \right) & = 2 - \left\lfloor \log_2 \left( 2 - t \right) \right\rfloor - 2 \left\lfloor \frac{2-\left\lfloor \log_2 \left( 2 - t \right) \right\rfloor}{2} \right\rfloor
\end{align}
$$

Not the prettiest-looking function, but it does serve our purpose, judging by its plot in the first [diagram](#diagrams).

At long last, we can state the problem of Thomson's lamp in terms of mathematical elements: what is the value of $$\sigma \left( 2 \right)$$? The expression $$\log_2 \left( 2 - t \right)$$ appears frequently in the formula for $$\sigma \left( t \right)$$, and at $$t=2$$, it is undefined. Therefore, it does not make sense to ask what $$\sigma \left( 2 \right)$$ is!

<a name="closer_look"></a>

### A 'closer' look

Even though $$\sigma \left( 2 \right)$$ is undefined, we may still want to ask what happens to $$\sigma \left( n \right)$$ as $$n \to \infty$$. As for every $$\sigma \left( n \right)$$ there is a $$\sigma \left( n+1 \right)$$, $$\sigma$$ must update countably infinite times up to $$t = 2$$. This, in fact, raises a contradiction. In James Thomas' own words,

> It seems impossible to answer this question. It cannot be on, because I did not ever turn it on without at once turning it off. It cannot be off, because I did in the first place turn it on, and thereafter I never turned it off without at once turning it on. But the lamp must be either on or off. This is a contradiction.

Here's how I like to interpret the above result: at $$t=2$$, $$\sigma \left( 2 \right)$$ breaks down. Since we are fitting a countably infinite number of state updates right up to $$t=2$$, _at_ $$t=2$$, the Thomson's lamp updates at an infinite rate, manifested in $$\log_2 \left( 2-t \right)$$ shooting to $$- \infty$$. But if infinite update events are squeezed into the same instant $$t=2$$, it no longer makes sense to ask what particular value $$\sigma$$ has, because it is multivalued! In order to be able to ask what value $$\sigma$$ has at a given time, $$\sigma$$ must return one and only one value at that instant, i.e. $$\sigma \left( t \right)$$ must be injective.

But this isn't where the story ends. To demonstrate the infinite rate of state update at $$t=2$$ a bit more rigorously, we need to find the derivative of $$\sigma \left( t \right)$$ and its limit at $$t=2$$. However, the expression for $$\sigma \left( t \right)$$ makes things rather messy. Finding its discrete derivative is no easy task; on the other hand, differentiating it is impossible as it is not a continuous function for $$t \in \mathbb{W}$$.

A workaround solution is to slightly broaden the problem, but keep the nature of the singularity at $$t=2$$ the same, so that both cases essentially involve the same mechanics. Namely, we will upgrade $$\sigma \left( t \right)$$ to a continuous parameter $$\sigma:\mathbb{R} \mapsto \left[ 0,1 \right]$$, but for $$t \in \mathbb{W}$$, $$\sigma \left( t \right) \in \left\{ 0, 1 \right\}$$. In other words, we will replace $$\sigma \left( t \right)$$ with a smooth and possibly, differentiable, function which passes through the same points as the original wherever state update is involved in the latter. Therefore, the nature of asking what $$\sigma \left( 2 \right)$$ is, remains the same as $$t=2$$ is common to both the domains.

## Real analytic solution

### A smoother path

We will 'smoothen' $$\sigma \left( t \right)$$ in a number of steps. These steps arise as a consequence of $$\sigma$$ _not_ remaining static between state updates, rather transitioning smoothly between the states fixed by the discrete evolution:

1. We will eliminate the floor function from $$n \left( t \right)$$ by the above reasoning. Therefore,

$$n \left( t \right) = 1 - \log_2 \left( 2 - t \right)$$

2. Since $$\sigma \left( n \right)$$ uses the $$\text{mod} \left( \cdot, 2 \right)$$ function, we will smoothen the latter first. As we will only be concerned with retaining the values concerned with integral preimages of the function, we simply need to find a smooth function which satisfies:

$$
\begin{align}
\text{mod} \left( 0, 2 \right) & = 0 \\
\text{mod} \left( 1, 2 \right) & = 1 \\
\text{mod} \left( 2, 2 \right) & = 0 \\
\vdots
\end{align}
$$

A suitable candidate is a sinusoidal function which oscillates between $$0$$ and $$1$$ with the initial value $$0$$ at $$t=0$$. The advantage of using a sinusoidal function is that it is differentiable. A little bit of playing around shows that under this approach, the simplest possible choice is,

$$\text{mod} \left( x, 2 \right) = \frac{1}{2} \left( 1 - \cos \left( \pi x \right) \right)$$

Substituting the above into the equation $$\sigma \left( n \right) = \text{mod} \left( n+1, 2 \right)$$, we get,

$$\sigma \left( t \right) = \frac{1}{2} \left[ 1 + \cos \left( \pi \left( 1 - \log_2 \left( 2 - t \right) \right) \right) \right]$$

### Rate of update

The rate of continuous update is simply its derivative. Using the standard rules of differentiation, we obtain the rate of update as,

$$\dot{\sigma} \left( t \right) = - \frac{\pi}{2 \ln \left( 2 \right)} \frac{1}{2-t} \sin \left( \pi \left( 1 - \log_2 \left( 2-t \right) \right) \right)$$

Since $$\cos : \mathbb{R} \mapsto \left[ -1, 1 \right]$$, we may say that,

$$\displaystyle{ \lim_{t \to 2} \sin \left( \pi \left( 1 - \log_2 \left( 2-t \right) \right) \right) }$$

is undefined, but not diverging. On the other hand, $$\displaystyle{ \lim_{t \to 2} \frac{1}{2-t} }$$ is diverging. Therefore, $$\displaystyle{ \lim_{t \to 2} \dot{\sigma} \left( 2 \right) }$$ is diverging. From this, we can justify the intuition that at $$t=2$$, $$\sigma \left( t \right)$$ is a vertical line with respect to the time-axis, thereby exhibiting multivaluedness.

## Conclusion

Now that we have solved the problem analytically, it is important to know why we used mathematics here at all. Firstly, we didn't _have_ to use mathematics to investigate this puzzle. It is a logical puzzle, and the role of mathematics here is to simply construct an analysis using tools we already know to be logically consistent.

However, the approach of employing mathematical machinery is enlightening in that the logical problem translates into an analytical one, involving how functions behave and when they break down. Since the calculus of how functions change is a familiar language today, it could be a helpful exercise to express our logical reasoning in this language. It potentially gives us insight into both the mathematics and the logic.
