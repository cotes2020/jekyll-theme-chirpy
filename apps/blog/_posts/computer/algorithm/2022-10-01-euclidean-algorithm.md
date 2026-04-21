---
title: "Euclidean Algorithm | 유클리드 알고리듬/호제법"
# description: ""
categories: [컴퓨터, 알고리즘]
tags: [알고리듬]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true
use_math: true

date: 2022-10-01. 08:42
# last_modified_at: 2023-10-31. 14:57
# last_modified_at: 2024-08-29. 22:18
last_modified_at: 2024-11-13. 05:12 # 정리
---

## Euclidean Algorithm | 유클리드 알고리듬/호제법

---

r = A % B (이때, A > B)  
(A, B 최대공약수) == (B, r 최대공약수) 를 이용해  
(A, B 최대공약수)를 구하는 알고리즘  

\+ (A, B 최대공약수)를 알면 (A, B 최소공배수)도 알 수 있음  

## 코드

---

C++ STL 내장 함수로도 존재 gcd, lcm  

```cs
int GCD(int A, int B)
{
    int r = 0;
    while (A % B != 0)
    {
        r = A % B;
        A = B;
        B = r;
    }

    return B;
}

int LCM(int A, int B)
{
    return A * B / GCD(A, B);
}
```

## 증명

---

```plaintext
(A, B, r ∈ ℕ), (0 ≤ r ＜ B ＜ A) 일 때,  
gcd(A, B) ⇔ gcd(B, r)  
A = qB + r ⇔ B = qr + r₂  

pf.  
assume gcd(A, B) = g, gcd(B, r) = g  
  
A = ga, B = gb  
a, b는 서로소 ⇔ gcd(a, b) = 1 (∵ if gcd(a, b) = k ⇔ k|A, k|B ⇔ gcd(A, B) = gk ≠ gcd(A, B) = g)  

A = qB + r ⇔ ga = qgb + r (∵ A = ga, B = gb)  
r = ga - qgb = g(a - qb)  
∴ g|r  

α ≔ (a - qb)  
r = gα  

gcd(B, r) = g 를 증명하기 위해서  
B = gb, r = gα 를 가지고  
gcd(b, α) = 1 을 증명해보자 (∵ if gcd(b, α) = k ⇔ k|B, k|r ⇔ gcd(B, r) = gk ≠ gcd(B, r) = g)  

귀류법  
gcd(b, α) = k 이라면?  

A = qB + r ⇔ ga = qgb + gα (∵ A = ga, B = gb, r = gα)  
b = kβ, α = kγ  
gcd(β, γ) = 1 (∵ gcd(b, α) = k)  

A = ga = qgkβ + gkγ = gk(qβ + γ)  
∴ gk | A  
B = gb = gkb  
∴ gk | B  
이러면 A, B의 최대공약수가 gk 이므로 gcd(A, B) = g 와 모순

∴ gcd(β, γ) ≠ 1, gcd(b, α) ≠ k  
∴ gcd(b, α) = 1  
∴ gcd(B, r) = g  
∴ gcd(A, B) = gcd(B, r)  
```

이를 통해 큰 수 (A, B) 를 작은 수 (B, r) 로 계산할 수 있다.  

## 약수, 인수

---

[約, 묶여있는](https://hanja.dict.naver.com/#/entry/ccko/ecc0d50d850b485a8361cb39d3995490){: target="_blank"} 수, Divisor  
[因, 원인, 원소, 이유가 되는](https://hanja.dict.naver.com/#/entry/ccko/cec202238f684acf86d21bf343b79aa8){: target="_blank"} 수, Factor  

(A, B ∈ ℕ), A ≠ 0 일 때,  
A가 B의 약수 ⇔ B = A * k (k ∈ ℕ)  
A가 B의 약수 ⇔ A가 B를 나눈다 ⇔ A|B  

## 공약수, 최대공약수

---

### Common Divisor (Factor) | 공약수

A와 B의 공통된 약수  

### GCD | Greatest Common Divisor (Factor) | 최대 공약수

A와 B의 공통된 약수 중에서 가장 큰 수  
gcd(A, B) ⇔ A, B의 최대공약수  

### A, B 의 최대공약수의 약수는 A, B의 공약수

- 12의 약수: 1, 2, 3, 4, 6, 12
- 18의 약수: 1, 2, 3, 6, 8, 18

- 12와 18의 공약수: 1, 2, 3, 6
- 12와 18의 최대공약수: 6
- 6의 약수: 1, 2, 3, 6

## 서로소

---

[素](https://hanja.dict.naver.com/#/entry/ccko/d16e6665e5f943be80491da2e2d0f3d4){: target="_blank"}, Coprime  
서로 묶이지 않는 수, 서로가 순수한(공통이 없는 수) 수  

A, B가 서로소다 ⇔ 공약수(최대공약수)가 1이다 ⇔ 1을 제외한 공약수가 없다 ⇔ 공약수의 개수가 1개이다  
gcd(A, B) = 1  

## 공배수, 최소 공배수

---

### Common Muliple | 공배수

A와 B의 공통된 배수  

### LCM | Least/Lowest Common Multiple | 최소 공배수

A와 B의 공통된 배수 중 가장 작은 수  
lcm(A, B) ⇔ A, B의 최소공배수  

### 최대 공약수로 최소 공배수 구하기

```plaintext
LCM = A * B / GCD
A * B = GCD * LCM
a = A / GCD, A = a * GCD
b = B / GCD, B = b * GCD
A * B = GCD * a * GCD * b
LCM = a * b * GCD
```

## 기록

---

- 가장 오래된 알고리듬.
- [백준 알고리즘 분류](https://www.acmicpc.net/problemset?sort=ac_desc&algo=26)
