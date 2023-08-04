---
title : 다이나믹 프로그래밍 [Dynamic Programming]
categories: [Programming, Algorithm]
tags : [Dynamic Programming, 다이나믹 프로그래밍, 피보나치 수열, 백준 2775번]
---

## Dynamic Programming
<hr style="border-top: 1px solid;"><br>

동적 프로그래밍이란?
: 하나의 문제는 단 한 번만 풀도록 하는 알고리즘

<br>

즉, 알고리즘을 짤 때 분할정복 기법을 많이 사용하는데, 큰 문제를 작은 문제로 분할하여 풀 때 같은 문제를 반복해서 푸는 경우가 발생함.

그 문제들을 매번 재계산하지 않고 값을 저장해두었다가 재사용하는 기법이 동적 프로그래밍임.

일반적으로 상당수의 분할 정복 알고리즘은 같은 문제를 반복적으로 해결하는 비효율적인 문제점을 가지고 있음.

<br>

정의를 좀 더 자세히 쓰면 다음과 같음.
: 어떤 문제가 반복적이고 최적 하위구조로 이루어질때, 하위 구조에 있는 부분 문제의 답을 기반으로 전체 문제의 답을 구하는 방법.

이 때, Memoization 기법을 사용함.

Memoization 
: 프로그램 실행 시 이전에 계산한 값을 저장하여, 다시 계산하지 않도록 하여 전체 실행 속도를 빠르게 하는 기술
: 시간 비용으로 인해 발생하는 문제를 공간 비용(Cache)을 투입해 해결하는 기법

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 조건
<hr style="border-top: 1px solid;"><br>

DP가 적용되기 위해서는 **2가지 조건을 만족해야 함.**  

<br>

### Overlapping Subproblems(겹치는 부분 문제)
<hr style="border-top: 1px solid;"><br>

DP는 기본적으로 문제를 나누고 그 문제의 결과 값을 재활용해서 전체 답을 구함. 

**그래서 동일한 작은 문제들이 반복하여 나타나는 경우에 사용이 가능함.**

즉, DP는 부분 문제의 결과를 저장하여 재계산하지 않을 수 있어야 하는데, 해당 부분 문제가 반복적으로 나타나지 않는다면 재사용이 불가능하니 **부분 문제가 중복되지 않는 경우에는 사용할 수 없음.**

<br><br>

### Optimal Substructure(최적 하위 구조)
<hr style="border-top: 1px solid;"><br>

전체 문제의 답이 부분 문제의 답으로부터 만들어지는 구조. 

예를 들어, 어떤 문제를 7개의 하위문제로 나눌 수 있을때, 7개의 하위문제의 답을 모두 얻어야 이 문제의 답을 구할 수 있다면 이 문제는 최적 하위구조를 갖추었다고 할 수 있음.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 분할정복과의 공통점, 차이점
<hr style="border-top: 1px solid;"><br>

분할정복과 비슷해 보이지만, 분할 정복은 문제를 큰부분에서 작은부분으로 나눔. (Top-Down)

반면, 동적 계획법(DP)은 제일 작은 부분부터 큰 문제로 풀어 올라감. (Bottom-Up)

<br>

또한, **분할정복은 나눈 문제들을 완전히 새로운 하나의 독립된 새로운 문제로 보는 반면, 동적 계획법(DP)은 이전 단계의 답에 의존적임.**

그래서 DP는 한번 푼 적 있는 문제는 다시 푸는 일이 없도록 테이블 등에 저장해둠. 이를 Memoization이라고 했었음.

<br>

```
공통점 : 문제를 가장 작은 단위로 분할

차이점 :
  - DP 
      : 부분 문제가 중복되며, 상위 문제 해결 시 재활용됨. (의존적)
      : Memoization 기법 사용
  
  - 분할정복 
      : 부분 문제가 서로 중복되지 않음. (독립적)
      : Memoization 기법 사용 X
```

<br>

**대표적으로 피보나치 수열이 있음.**  

아래 링크는 피보나치 수열을 다양한 알고리즘으로 구현한 것을 정리한 사이트임. 백준에도 정리한 게시글이 있었음. 
: <a href="https://cupjoo.tistory.com/20" target="_blank">cupjoo.tistory.com/20</a>  
: <a href="https://shoark7.github.io/programming/algorithm/피보나치-알고리즘을-해결하는-5가지-방법.html" target="_blank">shoark7.github.io/programming/algorithm/</a>  
: <a href="https://www.acmicpc.net/blog/view/28" target="_blank">acmicpc.net/blog/view/28</a>  

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 동적 프로그래밍 예제
<hr style="border-top: 1px solid;"><br>

백준 2775번 - 부녀회장이 될테야
: <a href="https://www.acmicpc.net/problem/2775" target="_blank">acmicpc.net/problem/2775</a>

```python
from sys import stdin
input=stdin.readline

case=int(input())
r=[[x for x in range(1,15)]]
idx=1

def resident(k,n,r,idx) :
    if idx <= k :
        for _ in range(k) :
            r.append([1]*14)
            for i in range(1,n+1) :
                r[idx][i]=r[idx-1][i]+r[idx][i-1]
            idx+=1
    return r[k][n]

for _ in range(case) :
    k=int(input())
    n=int(input())
    print(resident(k,n-1, r,idx))
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://www.zerocho.com/category/Algorithm/post/584b979a580277001862f182" target="_blank">zerocho.com/category/Algorithm/post/584b979a580277001862f182</a>  
: <a href="https://janghw.tistory.com/entry/알고리즘-Dynamic-Programming-동적-계획법?category=646973" target="_blank">janghw.tistory.com/Dynamic-Programming</a>  
: <a href="https://syujisu.tistory.com/147" target="_blank">syujisu.tistory.com/147</a>  
: <a href="https://hongjw1938.tistory.com/47" target="_blank">hongjw1938.tistory.com/47</a>  

<br><br>
<hr style="border: 2px solid;">
<br><br>
