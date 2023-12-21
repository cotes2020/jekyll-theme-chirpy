---
title: Spring Basic[빈 생명주기 콜백 방법 3]
date: 2023-02-23 21:11:00 +0800
categories: [Spring-Basic, 빈 생명주기 콜백 방법 3]
tags: [Spring]
---
# 빈 생명주기 콜백 방법 2

## 방법 : @PostConstruct, @PreDestroy
어노테이션을 활용하는 방법입니다.

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

 @PostConstruc
 public void init() {
 System.out.println("BeanClient.init");
 connect();
 call("초기화 연결 메시지");
 }

 @PreDestroy
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
 @Bean
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

## @PostConstruct, @PreDestroy 어노테이션 특징
@PostConstruct ,@PreDestroy 어노테이션은 최신 스프링에서 권장되는 초기화 메서드 정의 방법으로, 편리하고 가독성이 좋습니다. 또한 스프링에 종속되지 않은 표준 기술이므로 다양한 환경에서 사용 가능하며, 컴포넌트 스캔과 잘 어울립니다. 다만, 외부 라이브러리에는 적용할 수 없는 단점이 있습니다.

## 결론
되도록이면 @PostConstruct, @PreDestroy 어노테이션을 사용해야 합니다.        
코드를 고칠 수 없는 외부라이브러리를 초기화, 종료해야 하는 상황이 발생하면,         
@Bean의 initMethod, destroyMethod를 사용하도록 합니다.