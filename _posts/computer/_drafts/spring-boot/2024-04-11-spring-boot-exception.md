---
title: "Spring Boot Exception"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: [Spring-Boot]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-04-11. 19:09
# last_modified_at: 2024-04-11. 19:09
---

{% include embed/youtube.html id='nyN4o9eXqm0' %}  
{% include embed/youtube.html id='5XHhAhN-9po' %}  

## Exception

---

스프링 부트의 예외 처리 방식은 크게 2가지  

- `@ControllerAdvice` 를 통한 모든 Controller에서 발생할 수 있는 예외 처리
- `@ExceptionHandler` 를 통한 특정 Controller에서 발생할 수 있는 예외 처리

- `@ControllerAdvice` 는 모든 컨트롤러에서 발생할 예외를 정의하고,
- `@ExceptionHandler` 를 통해 발생하는 예외 마다 처리할 메소드를 정의

### 예외 클래스

- 모든 예외 클래스는 Throwable 클래스를 상속받는다.
- `Exception` 은 많은 자식 클래스가 있음
- `RuntimeException` 은 `Unchecked Exception`, 그 외는 `Checked Exception`

- `Checked Exception`: 컴파일 시점에 예외 처리를 강제하는 예외
  - 처리 여부: 반드시 예외 처리 필요
  - 확인 시점: 컴파일 단계
  - 예외발생 시 트랜잭션: 롤백하지 않음
  - 대표 예외: IOException, SQLException
- `Unchecked Exception`: 컴파일 시점에 예외 처리를 강제하지 않는 예외
  - 처리 여부: 명시적 처리를 강제하지 않음
  - 확인 시점: 런타임 단계
  - 예외발생 시 트랜잭션: 롤백
  - 대표 예외: NullPointerException, ArrayIndexOutOfBoundsException, ...

롤백 여부는 설정에 따라 달라질 수 있으며, 기본적으로 `Unchecked Exception`은 롤백이 일어남  

### @ControllerAdvice, @RestControllerAdvice

- `@ControllerAdvice`: 스프링에서 제공하는 어노테이션
- 둘 다 발생하는 예외를 한 곳에서 관리하고 처리할 수 있게 하는 어노테이션
- 설정을 통해 범위 지정이 가능하며, Default 설정은 모든 예외 처리를 관리함
  - `@RestControllerAdvice(basePackages = "com.example.controller")`와 같이 패키지 범위를 지정할 수 있음
- 예외 발생 시 json 형태로 결과를 반환하기 위해서는 `@RestControllerAdvice` 를 사용

### @ExceptionHandler

- 예외 처리 상황이 발생하면 해당 Handler로 처리하겠다고 명시하는 어노테이션
- 어노테이션 뒤에 괄호를 붙여 어떤 ExceptionClass를 처리할지 설정할 수 있음
  - `@ExceptionHandler(~~Exception.class)`
- `Exception.class`는 최상위 클래스로 하위 세부 예외 처리 클래스로 설정한 핸들러가 존재하면, 그 핸들러가 우선 처리하게 되며, 처리 되지 못하는 예외 처리에 대해 ExceptionClass에서 핸들링함
- `@ControllerAdvice`로 설정된 클래스 내에서 메소드로 정의할 수 있지만, 각 Controller안에 설정도 가능
- 전역 설정 (`@ControllerAdvice`)과 지역 설정 (`@ExceptionHandler`)이 동시에 존재할 경우, 지역 설정이 우선순위를 가짐

### 우선순위

- `@ExceptionHandler(Exception.class)` < `@ExceptionHandler(RuntimeException.class)` < `@ExceptionHandler(NullPointerException.class)`
- = 최상위 클래스 < 중간 클래스 < 하위 클래스

- `ControllerAdvice`/`RestControllerAdvice` < `Controller`/`RestController` < `Method`
- = 전역 설정 < 지역 설정 < 메소드 설정

## Custom Exception

---

### 구현 예시

- ErrorType: HttpStatus의 ReasonPhrase
- ErrorCode: HttpStatus의 Value
- Message: 상황별 디테일 Message

### HttpStatus

- `HttpStatus` 는 HTTP 상태 코드를 나타내는 Enum 클래스
- `BAD_REQUEST(400, Series.CLIENT_ERROR, "Bad Request")`와 같이 상태 코드, 상태 코드 시리즈, 상태 코드 메시지를 가짐
