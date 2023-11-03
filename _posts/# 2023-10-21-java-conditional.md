---
title: Section 06 이클립스 소개 - 첫번째 자바 프로젝트
date: 2023-10-21
categories: [blog]
tags: [java]
---

## If Else 문 연습문제 

만약 여러분이 C나 C++언어 프로그래머라면 값이 안 나오리라 생각할 것입니다

```sh

jshell> int i = 0;
i ==> 0

jshell> if (i){
   ...> System.out.println('i');
   ...> }
|  Error:
|  incompatible types: int cannot be converted to boolean
|  if (i){
|      ^

```

하지만 사실은 컴파일 에러가 납니다

왜냐하면 조건의 자리에 정수를 사용할 수 없기 때문이죠

C 와 C++ 언어에서는 정수, 0이 아닌 정수면 참으로 나오지만,

자바의 경우 정수를 조건의 자리에 쓸 수 없습니다.



i = 1 은 assignment 이지 comparison 이 아닙니다.

```sh
jshell> if (i=1){
   ...> System.out.println('i');
   ...> }
|  Error:
|  incompatible types: int cannot be converted to boolean
|  if (i=1){
|      ^-^
```

## 삼항 조건 연산자
삼항 조건 연산자를 사용할땐 결과값의 타입이 동일해야합니다.

int i = 2;
boolean isEven = (i % 2 == 0 ? true : false);
String isEvenStr = (i % 2 == 0 ? "Yes" : "No");

조직에 따라 쓰지 말라는 데도 있으니 잘 알아보세영.