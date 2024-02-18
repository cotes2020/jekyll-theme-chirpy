---
title : 매개변수 탐색 [Parametric Search]
categories: [Programming, Algorithm]
tags : [Parametric Search, 매개변수 탐색]
---

## 매개변수 탐색
<hr style="border-top: 1px solid;"><br>

조건을 만족하는 최대값을 구하는 알고리즘. 

이진 탐색을 이용하는데 여기선 배열 내에서 특정한 값을 찾는게 아니라 조건을 만족하는 최대값을 찾아야 된다는 점을 유의해야 함. 

<br>

중요한 점은 start, end 변수의 정의를 설정하는 것임. 

start는 조건을 무조건 충족하는 값, end는 조건을 무조건 충족 못하는 값

start는 0, end는 데이터의 최대값으로 설정하고 탐색 시작.

mid가 조건을 충족하는지 검사를 하고 
  1. 충족하면 mid 오른쪽 탐색 -> start=mid+1
  2. 충족하지 않으면 mid 왼쪽 부분을 탐색 -> end=mid-1

<br>

탐색이 끝나는 조건은 start 가 end보다 커질때 (start <= end)탐색 종료.

따라서 모든 탐색이 끝난 뒤 start 값이 조건을 만족하는 최대 값이 되는 것.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 예제
### 백준 1654번: 랜선 자르기
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://www.acmicpc.net/problem/1654" target="_blank">www.acmicpc.net/problem/1654</a>  

<br>

조건은 k개의 랜선에 대해 일정한 길이 cm로 잘라서 각각의 개수의 총합이 최소 n개가 되는가

왜 최소인가면 문제에 이렇게 써있음. ```"N개보다 많이 만드는 것도 N개를 만드는 것에 포함된"```

<br>

```python
k, n = map(int,input().split())
data=[int(input()) for _ in range(k)]

start=1 // k는 1이상으로 최소길이는 1로 설정
end=max(data)
while start <= end :
    count=0
    mid=(start+end)//2
    for i in data :
        count += i//mid
    if count >= n :
        start=mid+1
    else :
        end=mid-1

print(start-1)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

### 백준 2805번 : 나무 자르기
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://www.acmicpc.net/problem/12805" target="_blank">www.acmicpc.net/problem/2805</a> 

<br>


조건 : N개의 나무들을 H미터 높이에서 잘라서 자른 나무들의 합이 최소 M미터가 되는가

<br>

```python
from sys import stdin
input=stdin.readline

n,m=map(int,input().split())
trees=list(map(int, input().split()))

def parametic_search(start, end) :
    while start <= end :
        sumh=0
        mid=(start+end)//2
        for i in trees :
            if i > mid : 
                sumh+=i-mid
        if sumh >= m :
            start=mid+1
        else :
            end=mid-1
    return start-1

print(parametic_search(1, max(trees)))
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://kosaf04pyh.tistory.com/95?category=1046922" target="_blank">kosaf04pyh.tistory.com/95?category=1046922</a>  

<br><br>
<hr style="border: 2px solid;">
<br><br>
