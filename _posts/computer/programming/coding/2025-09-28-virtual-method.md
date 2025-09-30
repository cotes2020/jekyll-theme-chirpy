---
title: "가상 메서드 (Virtual Method)"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-09-28. 12:50 # Init
# last_modified_at: 2025-09-28. 12:50
---

## 말머리

---

### 가상 메서드 (Virtual Method)

- `virtual`로 선언된 메서드
  - 자식 클래스에서 `override`로 재정의 가능
  - 자식 클래스에서 `sealed`로 이제 다시 재정의하지 못하도록 막을 수 있음
  - 동적 디스패치 (Dynamic Dispatch):
    - 컴파일 시간에 호출될 함수가 결정되지 않고,
    - 런타임에 **가상 함수 테이블(vtable)**을 통해 실제 객체 타입에 맞는 함수 호출.
    - --> 부모 클래스 포인터(변수)로 자식 객체를 다룰 수 있음.

## 디스패치 (Dispatch)

---

어떤 메서드(함수)를 호출할지 결정하고, 실제로 그 코드를 실행하도록 연결해주는 과정

### 정적 디스패치 (Static Dispatch)

- **컴파일 타임**에 호출할 함수가 결정됨.
- 변수의 **선언된 타입**을 기준으로 결정되며, 실행 속도가 상대적으로 빠름.
- 대상: 일반 함수들

### 동적 디스패치 (Dynamic Dispatch)

- **런타임**에 호출할 함수가 결정됨.
- 포인터가 **실제로 가리키는 객체의 타입**을 기준으로 결정됨.
- 대상: `virtual` 함수
  - **가상 함수 테이블(vtable)**을 통해 실제 호출될 함수를 찾음
- 다형성을 가능하게 하는 핵심 원리

## 메모

---

- 키워드
  - Virtual (가상함수)
  - VTable
  - Dispatch (디스패치)
- 참고
  - ['예제로 배우는 C# 프로그래밍': 'C# Virtual Table의 구조와 Polymorphism에 관한 이해'](https://www.csharpstudy.com/DevNote/Article/28)
