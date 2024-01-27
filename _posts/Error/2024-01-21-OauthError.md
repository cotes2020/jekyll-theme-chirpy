---
title: 소셜 로그인 구현 시 oauth client 관련 오류
date: 2024-01-21 12:26:00 +0900
categories: [Error, oauth2.0]
tags: [oauth2.0]
math: true
mermaid: true

---

## google 로그인 인증 구현 중 oauth client 관련 오류

@[스프링 부트 3 백엔드 개발자 되기(자바편)](https://goldenrabbit.co.kr/product/springboot3java/)으로 oauth2를 사용한 google 로그인 인증 구현 중 에러 발생

![](https://velog.velcdn.com/images/fhazlt303/post/d5912768-df15-4db8-b79f-ea6cf0df4278/image.png)

```
Description:

Method filterChain in me.choigu.springbootdeveloper.config.WebOAuthSecurityConfig required a bean of type 'org.springframework.security.oauth2.client.registration.ClientRegistrationRepository' that could not be found.


Action:

Consider defining a bean of type 'org.springframework.security.oauth2.client.registration.ClientRegistrationRepository' in your configuration.
```

> org.springframework.security.oauth2.client.registration.ClientRegistrationRepository 

이 빈은 OAuth 2.0 클라이언트 등록 정보를 제공하는 데 사용되는데 해당 빈을 찾을 수 없다고 하니 Google 클라이언트의 정보를 제공하는 곳에 문제가 있을 확률이 높다.

## 해결
### 방법 1) `application.yml`을 하나로 통합해서 사용하는 경우.

`application.yml` 파일에서 `client:`를 추가하여 Google 클라이언트의 정보를 제공.
이렇게 설정을 추가함으로써 Spring Security가 `ClientRegistrationRepository` 빈을 알아서 생성하게 된다.

```yml
spring:
  jpa:
    show-sql: true
    properties:
      hibernate:
        format_sql: true
    defer-datasource-initialization: true
  datasource:
    url: jdbc:h2:mem:testdb
    username: sa
  h2:
    console:
      enabled: true
  security:
    oauth2:
      client:
        registration:
          google:
            client-id: YOUR_GOOGLE_CLIENT_ID
            client-secret: YOUR_GOOGLE_CLIENT_SECRET
            scope:
              - email
              - profile
```

![](https://velog.velcdn.com/images/fhazlt303/post/fb027eaa-260c-430e-98bb-ba910d21c52e/image.png)

수정 후 잘 동작된다.

### 방법 2) `application.yml` 설정파일을 여러 개 사용하는 경우

❶ OAuth 관련 설정을 하는 파일(ex : `application-oauth.yml`)에 `client:`가 잘 작성되어 있는지 확인.

❷ 기본 설정 파일에 OAuth 관련 설정 파일을 include 했는지 확인

`application.yml`
```yml
spring:
  profiles:
    active: default

...(생략)...
```
`application-oauth.yml`
```yml
security:
  oauth2:
    client:
      registration:
        google:
          client-id: YOUR_GOOGLE_CLIENT_ID
          client-secret: YOUR_GOOGLE_CLIENT_SECRET
          scope:
            - email
            - profile
...(생략)...
```
`pring.profiles.active` 속성이 `default`로 설정되어 있기 때문에 기본적인 설정이 활성화되며, `spring.profiles.include=oauth`를 추가하거나
`spring.profiles.active=oauth`를 추가하여 OAuth 관련 설정을 활성화 해주면 된다.
