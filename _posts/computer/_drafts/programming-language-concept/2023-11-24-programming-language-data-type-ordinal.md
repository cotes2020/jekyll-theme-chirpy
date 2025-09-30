---
title: "프로그래밍 언어 - Ordinal Data Type"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: [Data-Type]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-11-24. 09:20
last_modified_at: 2023-12-01. 10:20
---

## 사용자 정의 순서 타입

---

## 순서 타입 - Ordinal type

---

- 가능한 값들의 범위가 양의 정수 집합과 연계되는 타입
- 기본 순서 타입 - Java의 integer, char 혹은 C의 int, short, char

- 사용자-정의 순서 타입
  - 열거
  - 부분 범위

→ 사용하면 코드가 간결해지고 가독성이 좋아진다.  

## 열거 타입 - Enumeration Type

---

- 열거 상수(enumeration constants)들의 모임을 정의하고 그룹핑한 타입
  - C#, Enum

- 열거 상수에는 전형적으로 정수 값 0, 1, … 등이 암묵적으로 할당되나 임의의 값을 명시적으로 할당하는 것도 가능

### 설계 고려사항

열거형이 정수형으로 강제 형변환 가능한가?  
열거형과 정수이외의 타입과의 관계를 허용할 것인가?  

- 열거 타입이 제공되지 않는 언어에서는 정수를 이용하여 열거형을 흉내냄

```c
#define MALE 0
#define FEMALE 1
// or
int const MALE = 0;
int const FEMALE = 1;

if (gender == MALE) // ~
```

- 열거 타입을 이용하면 새로운 타입을 정의하게 되고 가독성을 향상시킨다.

```c
enum Gender
{
    MALE = 0,
    FEMALE = 1
};

if (gender == MALE) // ~
```

@ C, Pascal은 열거형을 처음으로 도입하여 널리 쓰인 언어  
@ C는 열거 타입 enum을 기본 타입으로 제공  

- 열거 타입을 제공하는 대부분의 언어에서 정수 타입과의 형변환은 허용하지 않음
  - 열거 타입은 열거 타입이지 정수형은 아님, 단 열거형의 값을 정수형으로 참조는 가능
  - C++, Java, Python, ...
- 반면 C에서는 허용
  - i.e. `gender++, gender = 0;`

- Java, Java 5.0(2004)에 java.lang.Enum 기본 타입으로 제공
  - 모든 열거 타입은 묵시적으로 java.lang.Enum에 상속을 받음 (정수형 X)
  - 따라서 toString, ordinal, values 등 몇몇 메소드들을 사용 가능

- 흥미롭게도 스크립트 언어 중 열거 타입을 기본 타입으로 지원하는 것은 없음
  - 표준 라이브러리에 제공
  - JavaScript, Python, Ruby, Lua, PHP, ...

### 평가

열거형은 가독성과 신뢰성을 향상시킨다.  

열거형은 산술연산을 허용하지 않고(열거형은 정수형이 아님), 정의된 범위 밖의 값을 할당 받을 수 없기 때문에.  

C, 열거형을 정수형처럼 취급, 이는 가독성을 향상시키지만 신뢰성 문제를 야기할 수 있다.  
C++, 열겨형 비교연산 시 정수형 타입처럼 다뤄 질 수 있다.  

## 부분 범위 타입

---

- 순서 타입의 연속된 부분 순서열(subsequence)
  - 예) 12..14

가독성과 신뢰성을 향상시지만, 대부분 언어에서 차용하지 않음.  
