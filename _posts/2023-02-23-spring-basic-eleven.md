---
title: Spring Basic[빈 생명주기 콜백 방법 2]
date: 2023-02-23 20:39:00 +0800
categories: [Spring-Basic, Spring-MainPoint]
tags: [Spring]
---

# 빈 생명주기 콜백 방법 2

## 방법 : 빈 등록 초기화, 소멸 메서드 지정
설정 정보에 @Bean(initMethod = "init", destroyMethod = "close")처럼 초기화, 소멸 메서드를 지정할 수 있습니다.

## 설정 정보 사용하도록 변경
```java
public class BeanClient {
 private String url;
 public BeanClient() {
 System.out.println("생성자 호출, url = " + url);
 }
 public void setUrl(String url) {
 this.url = url;
 }
 //서비스 시작시 호출
 public void connect() {
 System.out.println("connect: " + url);
 }
 public void call(String message) {
 System.out.println("call: " + url + " message = " + message);
 }
 //서비스 종료시 호출
 public void disConnect() {
 System.out.println("close + " + url);
 }
 public void init() {
 System.out.println("BeanClient.init");
 connect();
 call("초기화 연결 메시지");
 }
 public void close() {
 System.out.println("BeanClient.close");
 disConnect();
 }
}

```
설정 정보를 사용하도록 변경하는 코드입니다.

## 초기화, 소멸 메서드 지정
```java
@Configuration
static class LifeCycleConfig {
 @Bean(initMethod = "init", destroyMethod = "close")
 public BeanClient beanClient() {
 BeanClient beanClient = new BeanClient();
 beanClient.setUrl("http://test.dev");
 return beanClient;
 }
}
```

## 결과
```java
생성자 호출, url = null
BeanClient.init
connect: http://test.dev
call: http://test.dev message = 초기화 연결 메시지
13:33:10.029 [main] DEBUG 
org.springframework.context.annotation.AnnotationConfigApplicationContext - 
Closing BeanClient.close
close + http://test.d
```

## 설정 정보 사용시 얻을 수 있는 특징
- 메서드 이름을 원하는대로 자유롭게 지정이 가능합니다.
- 스프링 빈이 스프링 코드에 의존하지 않습니다.
- 설정 정보를 사용하기 때문에 코드를 고칠 수 없는 외부 라이브러리에도 초기화, 종료 메서드 적용이 가능합니다.

## 종료 메서드 추론
- @Bean의 destroyMethod 속성을 사용시, 특별한 기능을 활용할 수 있습니다.
- 라이브러리의 종료 메서드가 일반적으로 close, shutdown이라는 이름을 사용하는데, destroyMethod는 이러한 이름의 메서드를 자동으로 호출합니다.

## 결론
설정 정보를 사용하면 메서드 이름을 자유롭게 정의할 수 있고, 스프링 빈이 스프링 코드에 의존하지 않으며, 외부 라이브러리의 초기화 및 종료 메서드를 편리하게 적용할 수 있습니다. 종료 메서드의 추론 기능을 이용하면 일반적인 라이브러리의 종료 메서드를 간편하게 등록할 수 있습니다.