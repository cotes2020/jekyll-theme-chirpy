---
title: "Spring Boot Validation"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: [Spring-Boot]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-04-11. 18:16
# last_modified_at: 2024-04-11. 18:16
---

{% include embed/youtube.html id='J_7xasdHBZI' %}

## Validation

---

- 서비스의 비즈니스 로직이 올바르게 동작하기 위해 사용되는 데이터에 대한 사전 검증하는 작업이 필요함
- 유효성 검사 혹은 데이터 검증이라고 부르는데, 흔히 Validation이라고 부름
- 데이터의 검증은 여러 계층에서 발생하는 흔한 작업
- Validation은 들어오는 데이터에 대해 의도한 형식의 값이 제대로 들어오는지 체크하는 과정을 뜻함

### 일반적인 Validation의 문제점

- 어플리케이션 전체적으로 분산되어 존재
- 코드의 중복이 심함 (코드가 복잡해짐)
- 비즈니스 로직에 섞여 있어 검사 로직 추적이 어려움

### Bean Validation / Hibernate Validator

Java  

- Bean Validation: 어노테이션을 통해 다양한 데이터를 검증할 수 있게 기능을 제공
- Hibernate Validator: Bean Validation의 명세에 대한 구현체

- SpringBoot의 유효성 검사 표준은 Hibernate Validator를 채택

- `@Size`: 문자열의 길이를 검증
- `@NotNull`: null이 아닌지 검증
- `@NotEmpty`: null이 아니고 값이 비어있지 않은지 검증
- `@NotBlank`: null이 아니고 값이 비어있지 않고 공백이 아닌지 검증

- `@Past`: 과거 날짜인지 검증
- `@PastOrPresent`: 과거나 현재 날짜인지 검증
- `@Future`: 미래 날짜인지 검증
- `@FutureOrPresent`: 미래나 현재 날짜인지 검증

- `@Pattern`: 정규표현식을 사용하여 값이 패턴에 맞는지 검증

- `@Max`: 최대값을 검증
- `@Min`: 최소값을 검증
- `@AssertTrue`: true인지 검증
- `@AssertFalse`: false인지 검증

- `@Valid`: 객체의 내부 필드에 대한 검증을 수행
  - `@Valid @RequestBody User user`: User 객체의 필드에 대한 검증을 수행
