---
title: 전수I-templete
author: nian
date: 2024-01-01 00:00:00 +0900
categories: [수학, 선형대수학]
tags: [전산수학I, 선형대수학]
math: true
---




<details open>
  <summary>Hello</summary>
  World!
</details>

> The **posts**' _layout_ has been set to `post` by default, so there is no need to add the variable _layout_ in the Front Matter block.
{: .prompt-tip }

> testing
{: .notice}
> hello world

> <details open>
    <summary>Hellow</summary>
    World!
    </details>
{: .prompt-tip}

> <details open>
>   <summary>Hellow</summary>
>   World!
> </details>
{: .prompt-tip}


$$

\uppercase\expandafter{\romannumeral1}
$$

$$
\begin{align*}
  &\vec{a}, \mathbf{a}, \underline{a} \\
  &\mathbb{R}^n
\end{align*}
$$

$$
\begin{split}
  &\vec{a}, \mathbf{a}, \underline{a} \\
  &\mathbb{R}^n
\end{split}
$$

$$
\begin{equation}
  -\Delta\phi=4\pi k\rho.
\end{equation}
$$


$$
\begin{aligned}
  ||\mathbf{x} - \mathbf{y}||_2 = ||\mathbf{y} - \mathbf{x}||_2 \\\\
  \cos\theta = \frac{< \mathbf{x}, \mathbf{y} >}{||\mathbf{x}||_{2}||\mathbf{y}||_{2}}
\end{aligned}
$$

$$
\begin{align*}
  &\vec{a}, \mathbf{a}, \underline{a} \\
  &\mathbb{R}^n
\end{align*}
$$


$$
\forall \exists \therefore \geq \in \subset \subseteq 
$$
$$a^2$$

$$A_{m,n} =
 \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n}
 \end{pmatrix}$$

 <!-- Block math, keep all blank lines -->

$$
LaTeX_math_expression
$$

<!-- Equation numbering, keep all blank lines  -->

$$
\begin{equation}
  LaTeX_math_expression
  \label{eq:label_name}
\end{equation}
$$

Can be referenced as \eqref{eq:label_name}.

<!-- Inline math in lines, NO blank lines -->

"Lorem ipsum dolor sit amet, $$ LaTeX_math_expression $$ consectetur adipiscing elit."

<!-- Inline math in lists, escape the first `$` -->

1. \$$ LaTeX_{m_{a}}th_expression $$
2. \$$ LaTeX{_m}_ath_expression $$
3. \$$ LaTeX_math_expression $$