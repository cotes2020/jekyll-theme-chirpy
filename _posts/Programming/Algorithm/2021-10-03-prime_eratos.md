---
title : 에라토스테네스의 체 [Sieve of Eratosthenes]
categories: [Programming, Algorithm]
tags : [에라토스테네스의 체, 소수 구하기]
---

## 에라토스테네스의 체
<hr style="border-top: 1px solid;"><br>

**특정 숫자의 배수는 소수가 아니라는 법칙에 착안하여 2 ~ N까지의 숫자에서 숫자들의 배수를 모두 제거한 뒤 제거되지 않은 숫자를 소수로 판별하는 방식임.**

즉, 2부터 시작해서 2는 남겨두고 2의 배수를 모두 없앰. (소수는 1과 자기 자신만을 약수로 하니까)

그 다음 3은 남겨두고 3의 배수를 모두 없앰. 동일한 방식을 계속 적용해서 남는 수들은 모두 소수만 남게됨.

소수를 구하는 방식에는 다른 방식도 있지만 에라토스테네스의 체 방식은 **특정 숫자가 소수인지 판별할 때 사용하면 비효율적이나 특정 범위 내에서 소수를 찾을 때 가장 빠른 속도로 찾을 수 있어서 효율적임.**

<br>


만약 k라는 숫자가 소수인지 판별하고자 하면 소수를 구하는 다른 방식에는 아래와 같음.

1. 2부터 k-1까지 나눠서 나머지가 0인 값을 찾음.
   + 가장 원초적인 방법, O(n)
 

2. 2부터 k의 절반까지만 나눠서 찾음. 
   + k를 제외한 절반을 초과한 값에서 나눴을 때 나머지가 0인 숫자는 존재 x.


3. k의 제곱근까지의 숫자로 나눠서 찾음.
   + 2번 방법을 응용하여 소수를 판별하는 가장 효율적인 알고리즘.
   + 해당 숫자의 √N 까지 확인하는 방법으로 이 원리는 약수의 중심을 구하는 방법임.

<br>

```
ex3) 

80의 약수 -> 1, 2, 4, 5, 8, 10, 16, 20, 40, 80

1:80, 2:40, 4:20, 5:16, 8:10 -> 약수 간의 곱이 80이 되는 약수 쌍

약수의 중간값(√N)을 구하여 2부터 √N 까지로 나누어서 0이 되는 값이 없으면 소수임.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 에라토스테네스의 체 구현
<hr style="border-top: 1px solid;"><br>

```python
# k는 범위 ex) 100 보다 작은 소수 구하고자 하면 k=100

def eratos() :
  checks = [False,False] + [True]*(k-1)
  primes=[]

  for i in range(2,k):
      if checks[i]:
          primes.append(i)
          for j in range(2*i, k, i):
              checks[j] = False
  return primes
```

<br>

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main(){
	int n; cin >> n;
	vector<int> primes;
	vector<bool> check(n+1,true);
	check[0]=false;
	check[1]=false;
	for(int i=2; i<=n; i++) {
		if(check[i]) {
			primes.push_back(i);
			for(int j=i*2;j<=n;j+=i) {
				check[j]=false;
			}
		}
	}
	for(int i=0; i<primes.size(); i++) {
		cout << primes[i] << ' ';
	}
	cout << '\n';
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://myjamong.tistory.com/139" target="_blank">https://myjamong.tistory.com/139</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 백준 - 1929번 : 소수 구하기
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://www.acmicpc.net/problem/1929" target="_blank">acmicpc.net/problem/1929</a>

<br>

```python
n,m=map(int, input().split())
arr=[False,False]+[True]*m

def eratos() :
    primes=[]
    for i in range(2, m+1) :
        if arr[i] :
            if i >= n :
                primes.append(i)
            for j in range(i*2,m+1,i) :
                arr[j]=False
    return '\n'.join(map(str,primes))

print(eratos())
# 143012 KB / 196 ms
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 백준 - 1978번 : 소수 찾기
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://www.acmicpc.net/problem/1978" target="_blank">acmicpc.net/problem/1978</a>

<br>

```python
from sys import stdin
n=int(stdin.readline())
cnt=0
for i in stdin.readline().split() :
    i=int(i)
    check=True
    j=2
    if i != 1 :
        while j**2 <= i :
            if i%j == 0 :
                check=False
                break
            j+=1  
        if check : cnt+=1
print(cnt)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
