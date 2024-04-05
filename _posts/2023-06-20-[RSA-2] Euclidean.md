---
title: "[RSA - 2] 확장 유클리드 호제법 (Extended Euclidean Algorithm)"
date: 2023-06-20 01:06:00 +0900
author: aestera
categories: [Cryptography, RSA]
tags: [Cryptography, Euclidean Algorithm]
math: true
---

# 유클리드 호제법 (Euclidean Algorithm)

지난 포스팅에서는 모듈러 연산과 모듈러 역원에 대해 알아봤다. 이번 포스팅에서는 **유클리드 호제법과** 이를 확장시킨 **확장 유클리드 호제법**에 대해 알아보자.

## - 유클리드 호제법이란?

**확장 유클리드 호제법**에 대해 알아보기 전 **유클리드 호제법**에 대해 알아보자.
<br><br>
유클리드 호제법은 자연수 $ a $, $ b $가 주어졌을 때 $ \gcd(a, b) $
즉 a와 b의 최대공약수를 구하는 방법이다. 
<br><br>
1. 자연수 a에 대하여 $ \gcd(a, 0) = a $이다. 
2. $ a = bq + r $ 일때 $ \gcd(a, b) = \gcd(b, r) $이다. 

위 두개의 식으로 최대공약수를 쉽게 구할 수 있다.
<br><br>
예시로 $ \gcd(270, 192) $를 구해보자.
<br>

$$  \displaystyle \begin{aligned} 270 &= 192 \times 1+78 \\ 
                                   192 &= 78 \times 2+36 \\ 
					               78  &= 36 \times 2+6 \\ 
								   36  &= 6 \times 6+0 
\end{aligned} $$

$$ \gcd(270, 192) = \gcd(192, 78) = \gcd(78, 36) = \gcd(36, 6) = \gcd(6, 0) = 6 $$

코드로는 간단하게 구현할 수 있다.

```c
int gcd(int a, int b) {
	return b ? gcd(b, a%b) : a;
}
```
b가 0이 될 때 까지 연산을 수행하고 b가 0이 되면 a를 반환해준다.
<br><br>

## - 유클리드 호제법의 증명

$ a = bq + r $ 일때 $ \gcd(a, b) = \gcd(b, r) $를 귀류법을 통해 증명해봤다.

<details>
<summary>유클리드 호제법 증명</summary>

<!-- summary 아래 한칸 공백 두어야함 -->
<br>
$ A = Ga $, 
$ B = Gb $<br>
($a$, $b$는 서로소)
<br><br>
$ A = Bq + R $<br>
$ R = A - Bq = Ga - Gbq = G(a - bq) $<br>
<br>
$a$와 $b$가 서로 서로소이면 $A$와 $B$의 최대공약수는 $G$<br> 
$\gcd(A, B) = \gcd(B, R)$
을 증명하기 위해서는 $b$와 $a - bq$가 서로소임을 증명해야 한다.
<br><br>
귀류법 (결론 부정 → 모순)<br>
$b = mk$<br>
$a - bq = mk'$<br><br>
$a = bq + mk'$<br>
$a = mkq + mk'$<br>
$a = m(kq + k')$<br><br>

i) $m \ne 1$<br>
$a$ 와 $b$는 1이상의 공약수를 가지게 되어 $a$와 $b$는 서로소라는 전제가 모순된다.
<br><br>
ii) $m = 1$<br>
$b = k$<br>
$a - bq = k'$<br>
따라서 $b$와 $a - bq$가 서로소가 된다.<br>
→ $b$와 $a - bq$가 서로소가 아니라는 귀류법의 전제에 모순<br>
$$ \gcd(A, B) = \gcd(B, R) = G $$

</details>

<br><br>

****

# 확장 유클리드 호제법 (EEA, Extended Euclidean Algorithm)

확장 유클리드 호제법은 베주 항등식 (Bezout's Identity) 의 명제를 기반으로 해를 구하는 알고리즘이다. <br><br>
배주 항등식을 간단히 설명하면<br>

$ \gcd(a, b) = d $ 라고 할 때 $ ax + by = d $ 를 만족하는 정수 $x$, $y$가 존재한다.
<br><br>
확장 유클리드 호제법은 배주 항등식에 대하여 유클리드 호제법을 역으로 타고올라가 정수해를 만족하는 x, y를 찾아낼 수 있는 알고리즘이다.
<br>
배주 알고리즘에 대한 [증명](https://baeharam.github.io/posts/algorithm/extended-euclidean/)은 해당 사이트를 참고하자
<br><br>
말로 해봤자 더 헷갈리니 위에서 사용했던 같은 숫자를 사용해 예시를 들어보자
<br><br>
```270과 192의 최대공약수를 c 라고 할때 부정방정식 270x + 192y = c 의 정수해와 c를 구해보자 ```
<br><br>

$270 = 192 \times 1 + 78$ 이므로<br>

$$78 = 270 - 192 \tag{i}$$
<br>

$192 = 78 \times 2 + 36$ 이므로<br>

$$ \displaystyle \begin{align*} 36 &= 192 - 78 \times 2 \\ 
                                   &= 192 - (i) \times 2 \\
                                   &= 192 - (270 - 192) \times 2 \\
                                   &= 192 - 270 \times 2 + 192 \times 2 \\
								   &= 192 \times 3 - 270 \times 2 \tag{ii} \\
\end{align*}$$

$78 = 36 \times 2 + 6$ 이므로<br>

$$ \displaystyle \begin{align*} 6 &= 78 - 36 \times 2 \\
                                  &= (i) - (ii) \times 2 \\
								  &= (270 - 192) - (192 \times 3 - 270 \times 2) \times 2 \\
								  &= 270 - 192 - 192 \times 6 + 270 \times 4 \\
								  &= 270 \times 5 - 192 \times 7 \\
								  &= 270 \times 5 + 192 \times (-7) \tag{iii}
\end{align*}$$


$36 = 6 \times 6 + 0$ 이므로 정지
<br><br>

$ (iii) $을 보면  $270$과 $192$의 최대공약수 $c$는 $6$이고 <br>
$ 270x + 192y = 6 $ 을 만족하는 $(x, y)$ 는 $(5, -7)$ 임을 알 수 있다. 
<br><br>
이제 확장 유클리드 호제법을 사용해서<br>
$ \gcd(a, b) = d $ 라고 할 때 $ ax + by = d $ 를 만족하는 정수 $x$, $y$를 구할 수 있다.

<br><br>

****

## 확장 유클리드 호제법으로 모듈러 역원 구하기

잠시 지난 포스팅을 복습해보자.<br><br>
모듈러 연산에서 $ (A \times B) \equiv 1 \bmod\,N  $이 성립하는 **정수** B를 **모듈러 역수**라고 했었다.<br>
즉 $\gcd(a,b) = 1$ 이 되는 수를 구하면 되는 것이었다.<br>
이렇게만 설명해도 감이 잡히는 사람들도 있겠지만 아직 잘 모르겠는 사람들도 있을 것이다.<br><br> 
$3 \times x \equiv 1 \bmod\,26 $ 모듈러 연산을 예로 들어보자.

$$ 3 \times x \equiv 1 \bmod\,26 $$

$$ 3 \times x = 1 + 26 \times y $$

$$ 3x + 26y = 1 $$

식을 배주 항등식의 형태로 바꿀 수 있다! (여기서 $y$의 부호는 의미 없다.)<br><br>
이제 우리의 목표인 3의 모듈러 역원 $x$를 구할 수 있다.

$26 = 3 \times 8 + 2$ 이므로<br>

$$2 = 26 - 3 \times 8 \tag{i}$$
<br>

$3 = 2 \times 1 + 1$ 이므로<br>

$$ \displaystyle \begin{align*} 1 &= 3 - 2 \times 1 \\ 
                                  &= 3 - (i) \times 1 \\
								  &= 3 - (26 -3 \times 8) \\
								  &= 3 -26 + 3 \times 8 \\
								  &= 3 \times 9 + 26 \times (-1)
\end{align*}$$

이렇게 모듈러 26에 대한 3의 역원 **9**를 구하는데 성공했다.
<br><br>

이제 RSA를 복호화하기 위해 필요한 **모듈러 역원**을 구하는 방법을 알게됐다.
