---
title: "유클리드 호제법 (Euclidean algorithm)"
author: kwon
date: 2022-11-22T23:00:00 +0900
categories: [boj, koi]
tags: [math, number theory, Euclidean algorithm]
math: true
mermaid: false
---

유클리드 호제법은 2개의 자연수에 대해 최대공약수를 구하는 알고리즘이며 다음과 같은 성질을 통해 알고리즘을 진행한다.

> $a, b \in \mathbb{Z}$이고 $a$를 $b$로 나눈 나머지를 $r$이라 하자. ($b \leq a, 0 \leq r \leq b$)
> 
> 
> $a, b$의 최대 공약수를 $(a, b)$라고 하면, 다음이 성립한다.
> 
> $(a, b)=(b, r)$
> 

출처: wikipidia

$r$이 0이 될 때 알고리즘을 멈추며, 이 때의 $b$가 최대공약수가 된다.

예를 들어 1460과 1037에 대해 알고리즘을 진행해보면 다음과 같다.

$$\begin{flalign*}
(1460, 1037)\\=(1037, 323)\\=(323, 68)\\=(68, 52)\\=(52, 16)\\=(16, 4)\\=(4,0)
\end{flalign*}$$

$r$이 0일때 $b$가 4이므로 1460과 1037의 최대공약수는 4이다.