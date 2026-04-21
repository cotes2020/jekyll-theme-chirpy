---
title: "Functional Programming"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-08-29. 20:54
last_modified_at: 2025-05-28. 21:53 # +Q: 왜 OOP
---

{% include custom/computer/programming-paradigm.html %}

## Q

---

- 왜 Functional

## Functional Programming \| 함수형 프로그래밍

---

- 과정을 해석할 필요 없이, 있는 코드의 목적을 명확하게 알 수 있는
- 순수 함수와 불변성을 기반으로한 선언형 프로그래밍?
- 고차함수를 이용하는 프로그래밍 스타일로 데이터를 처리?
- 코드의 가독성을 높힌다?

### 순수 함수 (독립적으로 수행되는)

- 동일한 입력에 대해 항상 동일한 출력을 반환한다
- 함수 외부의 상태를 변경하지 않는다
- 사이드 이펙트 (부작용)이 없다

### 불변성

- 데이터는 변경되지 않는다
- 데이터 변경이 필요하면 새로운 데이터를 생성한다
- -> 이는 상태 변화로 인한 버그를 줄이는 데 도움이 된다

### 고차함수

- 함수를 인자로 받거나 함수를 반환하는 함수
- i.e. Where Select Sum

### 일급 객체 (로서의 함수)

- 함수가 변수에 할당될 수 있고(델리게이트),
- 다른 함수의 인자로 전달되거나
- 반환값으로 사용할 수 있다.
- 데이터 구조에 저장할 수 있다 (객체를 배열, 리스트, 맵등의 데이터 구조에 저장할 수 있다)

### 선언형 프로그래밍

- 무엇을 할 것인지에 집중하며, 어떻게 할 것인지는 명시하지 않는다
- Against 명령형 프로그래밍

## 메모

---

### 참고

- [Lambda](/posts/lambda/)
- [Delegate](/posts/delegate/)
- [LINQ](/posts/linq/)
