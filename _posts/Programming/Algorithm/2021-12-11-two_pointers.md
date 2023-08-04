---
title : 투 포인터 알고리즘 [Two Pointers Algorithm]
categories: [Programming, Algorithm]
tags : [Two Pointers Algorithm]
---

## 투 포인터
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/165762471-116784ba-b136-41fb-aa3c-97a2474b40fb.png)

<br>

**투 포인터 알고리즘은 1차원 배열에서 각자 다른 원소를 가리키고 있는 2개의 포인터를 조작하는 알고리즘임.**

대표적인 예로 부분 배열의 합이 특정한 값을 만족하는 부분 배열을 찾는 문제가 있음.

아래는 투 포인터를 이용해 푸는 문제임.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 예제
<hr style="border-top: 1px solid;"><br>

백준 2467번 - 용액
: <a href="https://www.acmicpc.net/problem/2467" target="_blank">acmicpc.net/problem/2467</a>

<br>

용액의 특성 값이 들어있는 일차원 배열이 있고 이 배열은 이미 오름차순임.

**오름차순 배열 속에서 두 용액의 특성 값의 합이 0에 가장 가까운 값을 만족하는 두 용액을 찾는 문제.**

즉, 두 개의 용액을 나타내는 두 개의 포인터(화살표)가 필요함.

처음 문제를 보았을 때 한 요소를 다른 나머지 요소 전부와 일대일로 비교하여 풀려했으나, 총 N!번을 비교하게 되므로 시간 초과가 나게 됨.

하지만 투 포인터를 이용하면 N번만 비교하면 되므로 O(n)

<br><br>
<hr style="border: 2px solid;">
<br><br>
