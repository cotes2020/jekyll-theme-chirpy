---
title: "오버라이딩, 오버로딩"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-09-28. 12:10 # Init
# last_modified_at: 2025-09-28. 12:10
---

## 머리말

---

- 목적
  - 다형성을 구현하거나, 코드의 편의성 개선
- 효과
  - 동일 기능은 하나의 인터페이스로
    - 타입 별로 동일 기능 필요한 경우: LogInt, LogFloat, LogString
    - 하나 이름으로 통일시켜 제공할 수 있음: Log
- 요약
  - 오버라이딩: 상속으로 함수를 재정의
  - 오버로딩: 매개변수 차이로 여러 형태 구현

## 오버라이딩

---

부모 클래스로부터 상속받은 메서드를, 자식 클래스에서 **재정의**하는 것.  
\# 클래스의 다양화  

### 오버라이딩 대상

- 부모 클래스의 메서드

### 가상 함수

- [가상 함수](/posts/virtual-method) [](/_posts/computer/programming/coding/2025-09-28-virtual-method.md)

## 오버로딩

---

동일한 이름의 메서드를 매개변수의 타입이나 수를 다르게 하여 여러 개 정의하는 것.  
\# 메서드의 다양화  

### 오버로딩 대상

- 부모 클래스의 메서드
- 자신 클래스의 메서드
- 자신 클래스의 생성자
- 자신 클래스의 연산자

## 메모

---

- 키워드
  - Override, Overriding (오버라이드, 오버라이딩)
  - Overload, Overloading (오버로드, 오버로딩)
