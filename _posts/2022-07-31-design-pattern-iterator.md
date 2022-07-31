---
title: "디자인 패턴 - iterator 패턴에 대해 알아보자"
date: 2022-07-31 00:56:00 +0900
categories: [디자인패턴]
tags: [디자인패턴, iterator패턴]
---

# 디자인 패턴 - iterator 패턴에 대해 알아보자

## 0. 소개

소프트웨어 마에스트로에서 멘토님 그리고 팀원들과 디자인패턴 스터디를 시작하였다. 이번 주는 iterator 패턴에 대해 진행하였고 이를 짧게나마 블로그에 기록하고자 한다.

## 1. UML

우선 UML이 무엇인지 간단하게 알아보자. UML은 Undefined Modeling Language의 약자로, 시스템이나 프로세스의 동작 및 구조를 시각적으로 보여주는 방법이다. 디자인패턴을 공부하다보면 대부분의 책에서 클래스 다이어그램으로 해당 패턴을 설명하곤 한다.
<br><br>
(아래 첨부된 사진에 나와있는 내용만 알고 있어도 우선 ok) <br>
<img width="475" alt="스크린샷 2022-08-01 오전 1 50 37" src="https://user-images.githubusercontent.com/64428916/182036936-fa8edaa7-c4d3-45ba-8416-2841eff24bae.png"> <br><br>
이번 포스팅에서 다룰 주 내용은 아니므로 자세한 내용은 생략하겠다.

## 2. Iterator 패턴이란?

Iterator는 무엇인가를 **반복한다**라는 의미이며, **반복자**라고도 한다.

```java
for (int i = 0; i < arr.length; i++) {
  System.out.println(arr[i]);
}
```

위에서 사용된 루프 변수 i는 배열의 첫 요소부터 마지막 요소까지 차례대로 처리해 간다. 여기에서 사옹되고 있는 변수 i의 기능을 추상화하여 일반화한 것을 **iterator 패턴**이라고 한다.

이를 클래스 다이어그램으로 나타내면 아래와 같다.
<img width="792" alt="스크린샷 2022-08-01 오전 2 07 16" src="https://user-images.githubusercontent.com/64428916/182037531-359cd4ae-bb41-45e3-bb0b-a53207d46379.png">

여기서 질문 한가지!
어떻게 Aggregate interface가 interface를 생성할 수 있을까?

Aggregate의 추상 메소드인 iterator의 return type이 iterator interpace이기 때문이다.

## 3. Q&A

Q. 데이터의 처음부터 끝까지 훑는 것이 아닌, 중간의 일부부만을 참조하려면 어떻게 해야하나?

- 중간만 참조하는 iterator를 만들어야 한다. 하나의 ConcreteAggregate에 대해 여러 ConcreteIterator를 만들 수 있다.

Q. 자료구조 원본의 index 접근이 더 편리하지 않을까? 왜 원본을 0부터 끝까지 직접 참조하지 않을까?

- 원본 노출을 하지 않기 위해서이다. 즉, Iterator 패턴은 자료구조의 원본을 지킬 수 있는 보안 패턴이다. 원본은 감춘채 거기에 있는 데이터를 탐색할 수 있게 된다.

## 참고문헌

[Java언어로 배우는 디자인 패턴 입문](http://www.yes24.com/Product/Goods/2918928)
