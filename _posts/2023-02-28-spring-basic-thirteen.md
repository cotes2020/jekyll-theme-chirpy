---
title: Spring Basic[빈 스코프]
date: 2023-02-23 21:11:00 +0800
categories: [Spring-Basic, 빈 스코프]
tags: [Spring]
---

# 빈 스코프
## 빈 스코프란?
스프링 컨테이너가 빈(Bean)객체를 생성하고 유지하는 범위를 나타냅니다.
또한, 빈의 스코프는 빈이 언제 생성되고 얼마나 오랫동안 유지되는지에 대한 규칙을 정의합니다.

## 빈 스코프 등록 방법
다음과 같이 지정이 가능 합니다.
### 컴포넌트 스캔 자동 등록
```java
@Scope("prototype")
@Component
public class ABean {}
```

### 수동 등록
```java
@Scope("prototype")
@Bean
public class ABean {
    return new ABean();
}
```

## 다양한 스코프
일반적으로 사용되는 빈 스코프에는 여러가지가 있습니다.
1. 싱글톤(Singleton) 스코프
    - 스프링 컨테이너의 시작과 종료까지 유지됩니다.
    - 기본 스코프이며, 가장 넓은 범위의 스코프 입니다.
2. 프로토타입(Prototype) 스코프
    - 스프링 컨테이너는 프로토타입 빈의 생성과 의존관계 주입 까지만 관여합니다.
    - 매우 짧은 범위의 스포크 입니다.
3. 웹 관련 스포크
    1. request
        - 웹 요청이 들오오고, 나갈때 까지 유지되는 스코프 입니다.
    2. session
        - 웹 세션이 생성되고, 종료될 때 까지 유지되는 스코프 입니다.