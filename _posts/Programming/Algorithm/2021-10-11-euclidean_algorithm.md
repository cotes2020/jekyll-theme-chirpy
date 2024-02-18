---
title : 최대공약수, 최소공배수 [GCD, LCM]
categories: [Programming, Algorithm]
tags : [유클리드, 최대공약수, 최소공배수]
---

## 유클리드
<hr style="border-top: 1px solid;"><br>

```
a=qb+r
gcd(a,b)=gdc(b,r) -> r == 0 : break

gcd(a, b) = gcd(b, a%b)이고, gcd(k, 0)일 때 답은 k ( 단, a, b, k가 양의 정수일 때)

ex) gcd(35, 25) = gcd(25, 10) = gcd(10, 5) = gcd(5, 0) = 5.
```

<br>

```
lcm(a,b) = a/gcd(a,b) * b

-> lcm(a,b) : AB = LG, (G : gcd, L : lcm) -> A*B = 최소 * 최대공약수
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Code
<hr style="border-top: 1px solid;"><br>

```python
a,b=map(int,input().split())
c=a*b
while(1) :
    q=a//b
    r=a%b
    if(r == 0) : break
    a=b
    b=r
print(b)
print(c//b)
```

<br>

```python
a,b = map(int,input().split())

def euc(a,b) :
    if a% b == 0 :
        return b
    else :
        return euc(b,a%b)
        
ans_d = euc(a,b)
ans_m = euc(a,b)*(a/euc(a,b))*(b/euc(a,b))
        
print(ans_d)
print(int(ans_m))
```

<br>

```python
a, b = map(int, input().split())
def gcd(a, b):
  while b:
    a, b = b, a % b
  return a

print(gcd(a, b))
print(a * b // gcd(a, b))
```

<br>

참고 
: <a href="https://dimenchoi.tistory.com/46" target="_blank">dimenchoi.tistory.com/46</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
