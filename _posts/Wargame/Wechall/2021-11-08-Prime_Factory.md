---
title : "Wechall - Prime Factory"
categories : [Wargame, Wechall]
---

## Prime Factory
<hr style="border-top: 1px solid;"><br>

```
Your task is simple:
Find the first two primes above 1 million, whose separate digit sums are also prime.
As example take 23, which is a prime whose digit sum, 5, is also prime.
The solution is the concatination of the two numbers,
Example: If the first number is 1,234,567
and the second is 8,765,432,
your solution is 12345678765432
```

<br>

백만이 넘는 소수 중 각 자리의 숫자 합도 소수가 되는 값을 2개 구하라.

정답 제출 시 두 값을 이어 붙여서 제출해라.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>


```python
from sys import stdin
from math import sqrt
input=stdin.readline

n=1000000
cnt=0
while cnt < 2 :
    prime=True
    for i in range(2, round(sqrt(n))+1) :
        if n % i == 0 :
            prime=False
            break
    if prime :
        check=0
        for i in str(n) :
            check+=int(i)
        for i in range(2,round(sqrt(check))+1) :
            if check % i == 0:
                prime=False
                break
    if prime : 
        cnt+=1
        print("Prime : {}".format(n))
    n+=1
```
```
Prime : 1000033
Prime : 1000037
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
