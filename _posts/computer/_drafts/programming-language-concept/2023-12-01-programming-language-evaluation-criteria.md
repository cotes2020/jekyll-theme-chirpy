---
title: "프로그래밍 언어 - 언어 평가 기준"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-12-01. 12:39
last_modified_at: 2023-12-08. 10:03
---

{% include custom/computer/programming-language.html %}

## 언어 평가 기준

---

[참고](https://www.geeksforgeeks.org/language-evaluation-criteria/)  

프로그래밍 언어가 SW 개발 및 유지보수에 대해 미치는 요소  
`가독성 Readability`, `작성력 Writability`, `신뢰성 Realiability`, `비용 Cost`  

## 가독성 Readability

---

- 코드 이해 난이도 척도 → 유지보수 난이도
- 언어 특징: 단순성, 직교성, 제어문, 데이터 타입과 구조, 구문 고려사항

(언어 자체에 대하여, 잘짜고 못짜고의 문제가 아니라)  

### 단순성 (전반적인 단순성 Overall Simplicity)

- 단순성 = 가독성
- 복잡한 언어 기본 구조 → 배우기 어려움, 가독성 X
  - 특징 다중성, 심한 연산자 중복 (Feature Multiplicity, Operator Overloading)
- But, 극단적인 단순성 != 가독성 → i.e. Assembly

### 직교성 Orthogonality

@ U 중간고사 출제: 직교성이 무엇이고, 언어에 어떤 영향을 주는가?  

- 기본 요소들을 조합하여, 새로운 제어나 데이터 구조를 생성할 수 있는 능력
  - 기본 데이터 타입 x 타입 연산자 (배열, 포인터)

- 이해/읽기 쉽게,
- 의미 문맥 독립적이게 (서로 다른 표현은 서로 다른 의미),
- 모든 조합 합법적이게

- 직교성은 단순성에 영향을 준다
  - 많은 기본 구조 \< 일반적 규칙 (적은 기본 구조 x 조합 규칙 = 직교성)
- But, 극단적인 직교성 != 단순성 → i.e. Algol 68

### 제어문

- 충분한 제어문 제공
  - i.e. for가 적합한 상황에서, while만 제공된다면

### 데이터 타입과 구조

- 충분한 데이터 타입, 구조 제공
  - i.e. C, 0/1 Boolean 표현 (가독성 낮음)

### 구문 설계 Syntax Design

- 특수어 - Special Word
  - 특수어 = 가독성
    - 적은 특수어 → 단순성
      - i.e. C/C++, ;
    - 많은 특수어 → 가독성
      - i.e. Ada, end if, end loop
  - 특수어로 식별자를? (Identifier)
    - i.e. Fortran, 가능 (가독성 낮음)

- 형식과 의미
  - 그 자체로 목적이 나타나는, 문장의 형태
  - 같은 형식, 다른 의미는 가독성(직교성) 낮춤
    - i.e. C/C++, 문맥에 따른 static

## 작성력 Writability

---

- 프로그램 목적에 따른, 언어 작성 난이도
  - i.e. Visual Basic, GUI 작성에 적합
  - i.e. C, System SW 작성에 적합
- 가독성과 비례

- But, 극단적 직교성 != 작성력
  - 모든 조합이 합법적이면, 컴파일러가 탐지하기 어려운 오류 가능성

### 추상화 지원

- 복잡한 데이터 구조, 연산 정의 → 세부 사항 무시
  - i.e. (클래스, 자료구조, 함수) 추상화 없이, 코드를 짠다면?
- 프로세스 추상화 - 부프로그램 (함수) 지원
- 데이터 추상화 - 미리 구현된 자료구조, 쉬운 자료구조 구현
  - i.e. Binary Tree, Fortran VS C++/Java

### 표현력

- 문제 표현에 있어 편리한 방법 제공 여부
  - i.e. Counting loop, for VS while
  - i.e. (i = i + 1) VS (i++)
  - i.e. Ada, and then Boolean Operator로 Short-Circuit

## 신뢰성 Reliability

---

- 모든 조건/상황에 대해, 주어진 명세 수행 여부
- 언어 특징
  - 타입 검사, 예외 처리, 별칭, 가독성과 작성력

### 타입 검사

- CompileTime / Runtime 타입 검사
  - CompileTime, Runtime 타입 검사에 비해 Cost 적음
    - i.e. int a = 3.14 → Error
- i.e. 1978 C, 매개변수 타입 검사 X
  - 타입 검사하는 lint
  - 지금은 타입 검사

### 예외 처리

- Runtime 오류를 수정하고, 실행을 계속할 수 있는 능력
- 논리 오류 말고도, 외부적인 요인 (Like 네트워크 이슈)

### 별칭

@ TODO:

- 동일한 기억 장소에 대해 (?), 여러 참조 방법을 갖는 것
- 신뢰성을 저해하나, 대부분 언어에서 제공
  - 왜 Why, 부족한 데이터 추상화 ↑
  - i.e. type def

## 비용

---

- 교육
- 언어 구현 (컴파일러), 작성, 컴파일, 실행
- 신뢰성 부족, 유지보수

## ETC

---

- 이식성: 언어의 표준화, 다른 언어와의 호환성
- 일반성: 응용 분야에 대한 적용성
- 분명성: Docu 수준
