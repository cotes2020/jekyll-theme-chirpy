---
title: "Property (프로퍼티)"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2025-09-29. 07:26 # Init
# last_modified_at: 2025-09-29. 07:26
---

## 말머리

---

## Property (프로퍼티)

---

- 클래스 속성
  - 흔히 get, set을 정의 해주는 간편한 예약어이다.
- 프로퍼티는 근본적으로 메서드 ?

## 응용

---

```cs
public int SomeValue0 { get; set; }
public int SomeValue1 { get; private set; }
[field: SerializeField] public int SomeValue2 { get; private set; }
```

`get`, `set` 마다 접근제한자를 따로 지정해줄 수 있다.  

위처럼 `get`은 `public`, `set`은 `private`로 만들어게 되면, 해당 `property`는 선언된 클래스 안에서는 `set`이 가능하지만, 외부에서는 `get`만 가능하도록 만들어줄 수 있다.  

Unity에서는 `[field: SerializeField]`를 통해 인스펙터에서는 값 설정이 가능하도록 만들 수 있다. 물론 외부 클래스에서는 여전히 `set`이 불가능하다.  

## 메모

---

- 키워드
  - Property (프로퍼티)
- TODO: 꼬리질문
  - How `SerializeField` works
  - 가장 먼저 나올 수 있는 의문점은, 그래서 일반 `public` 필드랑 어떤 차이가 있냐는 것이다.
  - Variable (변수)와 Field (필드)
