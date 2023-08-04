---
title : 분할 정복 [Divide & Conquer]
categories: [Programming, Algorithm]
tags : [Divide & Conquer]
---

## 분할 정복
<hr style="border-top: 1px solid;"><br>

분할정복법은 주어진 문제를 작은 사례로 나누고(Divide) 각각의 작은 문제들을 해결하여 정복 (Conquer)하는 방법.

문제의 사례를 2개 이상의 더 작은 사례로 나눔. 이 작은 사례는 주로 원래 문제에서 따옴. 

나눈 작은 사례의 해답을 바로 얻을 수 있으면 해를 구하고 아니면 더 작은 사례로 나눔.

해를 구할 수 있을 만큼 충분히 더 작은 사례로 나누어 해결하는 방법임.

<br>

분할 정복법은 하향식(top-down) 접근 방법으로 최상위 사례의 해답은 아래로 내려가면서 작은 사례에 대한 해답을 구함으로써 구함.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 분할 정복 처리 과정
<hr style="border-top: 1px solid;"><br>


1. 분할(divide)
  + 주어진 문제를 여러 개의 작은 문제로 분할
  + 문제를 제대로 나누면 conquer 하는 것은 간단하기 때문에 divide가 중요.


2. 정복(conquer)
  + 작은 문제들을 순환적으로 분할하고 작은 문제가 더 이상 분할되지 않을 정도로 크기가 충분히 작다면 순환호출 없이 작은 문제에 대한 해를 구함

3. 결합(combie/merge)
  + 작은 문제에 대해 정복된 해를 결합하여 원래 문제의 해를 구함.
  + 보통 재귀 알고리즘이 사용되는데 이 부분에서 효율성을 깎을 수 있음.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 조건
<hr style="border-top: 1px solid;"><br>

분할 정복을 적용하기 위해서는 문제에 다음과 같은 몇 가지 특성이 성립해야 함.

1. 부분 문제로 나누는 자연스러운 방법이 있어야 함.
  + 만약 분할될 때마다 분할된 부분문제의 입력 크기의 합이 기존의 입력 크기보다 매우 커진다면 분할 정복 알고리즘을 적용하는게 부적절할 수 있음.
  + 예를 들어, n번째 피보나치 수를 구할 때 재귀 호출을 사용하면, 분할 후 입력 크기가 거의 2배씩 늘어나므로 반복문을 사용하는게 효율적임.


2. 부분 문제의 답을 조합해 원래 문제의 답을 계산하는 효율적인 방법이 있어야 함. 
  + 분할 정복을 사용한다고 무작정 효율이 좋아지는것이 아님.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 특징
<hr style="border-top: 1px solid;"><br>

분할된 작은 문제는 원래 문제와 성격이 동일함  
: 입력 크기만 작아짐

분할된 문제는 서로 독립적임(중복 제거 X) 
: 순환적 분할 및 결과 결합 가능
: 동적 프로그래밍과의 차이점(동적 프로그래밍은 의존적임)

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 분할 정복을 이용한 거듭제곱
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://hbj0209.tistory.com/43" target="_blank">hbj0209.tistory.com/43</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://blog.naver.com/qpghnv/221580612451" target="_blank">blog.naver.com/qpghnv/221580612451</a>  
: <a href="https://loosie.tistory.com/237" target="_blank">loosie.tistory.com/237</a>  
: <a href="https://data-make.tistory.com/232" target="_blank">data-make.tistory.com/232</a>   

<br><br>
<hr style="border: 2px solid;">
<br><br>
