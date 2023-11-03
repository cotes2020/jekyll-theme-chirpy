---
title: 혼자 공부하는 컴퓨터구조 & 운영체제
date: 2023-08-20
categories: [blog]
tags: [os]
---

![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/6.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/6.png)

> 라즈베리파이에서 봤던 구성물들이 바로 운영체제 부품들이었다.

## 프로세서

프로세서 = 중앙처리장치 = CPU = Central Processor Unit

싱글코어, 듀얼코어, 멀티코어 등 코어 관련 용어로 말하던 부품이 바로 프로세서이다.

도커 이미지를 빌드할때 나를 괴롭혔던 친구이다^^

<div markdown="block" style="width: 45%;">
![https://cdn.dribbble.com/users/229527/screenshots/3127016/tunnelcpu.gif](https://cdn.dribbble.com/users/229527/screenshots/3127016/tunnelcpu.gif)
</div>

<br/>

<div markdown="block" style="width: 45%; float: left; padding: 0 2% 0;">

#### 💡 x86

고성능 강점

인텔, AMD 에서 개발한다.

주로 개인용 컴퓨터(PC)에서 사용되며, 대부분의 Windows 및 일부의 리눅스 운영체제가 이 아키텍처를 지원합니다.

![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/3.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/3.png)

</div>

<div markdown="block" style="width: 45%; float: left; clear: right; padding: 0 2% 0;border-left: 1px solid lightgray; ">

#### 💡 ARM

저전력 강점

모바일 기기에서 많이 사용

애플 m1, 라즈베리파이 프로세서들이 arm 이다.

![ARM architecture 1](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/1.gif)

![ARM architecture 2](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/2.jpg)

</div>

<div markdown="block" style="clear: both;">

<br/>

## 메인보드

<div markdown="block" style="width: 80%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/4.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/4.png)
</div>

메인보드 = 마더보드 = 보드

여러 부품들을 연결해주는 통로이다.

**CPU** : 계산해야 되니까 데이터 달라고 램한테 명령함 🤖

**Control Bus :** 데이터를 읽으라는 명령어를 보내는 통로

**Address Bus** : 메모리 어느 주소를 읽을건지 보내는 통로

**메모리(램)** : 담고 있던 데이터를 줌 💾

**Data Bus :** 메모리에서 데이터를 받아서 CPU 안에 담으러 가는 통로

</div>

<br/>

## 메모리

![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/5.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-08-20-operating-system/5.png){:width="30%"}

메모리 = 램 = RAM = **R**andom **A**ccess **M**emory = 주기억장치

컴퓨터가 켜지는 순간부터 cpu 가 여기에 기록하기 시작한다.

메인 메모리에 주로 사용되는 RAM은 일반적으로 전원이 차단되면 내용이 지워지는 **휘발성** 기억 장치이다.

[CPU](https://namu.wiki/w/CPU)에서 이뤄진 연산을 메모리에 READ, WRITE한다. 수학 계산 연습장 같은 역할을 한다.
예를 들면 포토샵을 쓰려고 실행하면 그 때부터 램에다가 작업 중인 내용을 저장한다.

그러다가 포토샵을 종료하면 하드에 저장하지 않은 내용은 주기억장치에서 삭제한다. 다시 컴퓨터를 켜서 포토샵을 실행하면 새로운 세션에서 시작하게 됩니다.

<details markdown="block"><summary>상세설명</summary>
RAM 이란 사용자가 자유롭게 내용을 읽고 쓰고 지울 수 있는 [기억장치](https://namu.wiki/w/%EA%B8%B0%EC%96%B5%EC%9E%A5%EC%B9%98)이다. 컴퓨터가 켜지는 순간부터 CPU는 연산을 하고 동작에 필요한 모든 내용이 전원이 유지되는 내내 이 기억장치에 저장된다. '주기억장치'로 분류되며 보통 램이 많으면 한번에 많은 일을 할 수 있기에 '책상'에 비유되곤 한다.
램의 용량이 클수록 그 용량만큼 동시에 기록하고 연산하는 것이 가능하다는 것이며 고용량 램일수록 컴퓨터의 성능이 올라가고 가격이 비싸진다.
이런 특성으로 인해 속도는 느리지만 전원이 끊어져도 정보를 저장할 수 있는 [자기 테이프](https://namu.wiki/w/%EC%9E%90%EA%B8%B0%ED%85%8C%EC%9D%B4%ED%94%84), [플로피 디스크](https://namu.wiki/w/%ED%94%8C%EB%A1%9C%ED%94%BC%20%EB%94%94%EC%8A%A4%ED%81%AC), [하드 디스크](https://namu.wiki/w/%ED%95%98%EB%93%9C%20%EB%94%94%EC%8A%A4%ED%81%AC%20%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C) 같은 보조 기억 장치가 나오게 되었다.
HDD 등의 기억장치와 같이 어느 위치에나 직접 접근할 수 있으나 데이터의 물리적 위치에 따라 읽고 쓰는 시간에 차이가 발생하게 되는 기억장치들은 Direct Access Memory 또는 Direct Access Data Storage라고 부른다.
어느 위치에든 똑같은 속도로 접근하여 읽고 쓸 수 있다.
</details>

## 📎 Related articles

| 이슈명                              | 링크                                                                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 혼자 공부하는 컴퓨터구조 & 운영체제 | [ https://product.kyobobook.co.kr/detail/S000061584886](https://product.kyobobook.co.kr/detail/S000061584886) |
