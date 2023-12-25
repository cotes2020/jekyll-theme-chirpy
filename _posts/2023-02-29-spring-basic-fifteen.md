---
title: Spring Basic[스코프와 Provider]
date: 2023-02-29 17:35:00 +0800
categories: [Spring-Basic, Spring-MainPoint]
tags: [Spring]
---

# Provider
Provider는 스프링 프레임워크에서 제공하는 인터페이스 중 하나로,         
빈을 동적으로 가져오는 기능을 제공하고 있습니다.        
주로 런타임 시에 의존성을 주입하는 상황에서 사용됩니다. 아래는 간단한 예제 코드 입니다.
## Service
```java
@Service
@RequiredArgsConstructor
public class LogDemoService {
 private final ObjectProvider<MyLogger> myLoggerProvider;
 public void logic(String id) {
 MyLogger myLogger = myLoggerProvider.getObject();
 myLogger.log("service id = " + id);
 }
}
```

## Controller
```java
@Controller
@RequiredArgsConstructor
public class LogDemoController {
 private final LogService logService;
 private final ObjectProvider<MyLogger> LoggerProvider;
 @RequestMapping("log-demo")
 @ResponseBody
 public String logDemo(HttpServletRequest request) {
    String requestURL = request.getRequestURL().toString();
    LogerScope logerscope = LoggerProvider.getObject();
    logerscope.setRequestURL(requestURL);
    logerscope.log("controller test");
    logService.logic("testId");
    return "OK";
 }
}
```
- 접속은 http://localhost:8080/log-demo의 주소로 접속이 가능합니다.

<br/>

> **@RequiredArgsConstructor**
> - Lombok의 어노테이션 중 하나로, 자주 사용되는 어노테이션입니다.
> - 주로 생성자를 자동으로 생성하는데 사용됩니다.
>     - final로 선언된 필드, @NonNull 어노테이션이 붙은 필드에 사용됩니다.
> - **예시:**
> ```java
> @RequiredArgsConstructor
> public class UserClass {
>     private final String userID;
>     private final int phonenumber;
>     
>     // @RequiredArgsConstructor 어노테이션을 사용시
>     // 아래의 생성자를 작성 안해도 된다는 편의성이 존재합니다.
>     // public UserClass(String userID, String phonenumber) {
>     //     this.userID = userID;
>     //     this.phonenumber = phonenumber;
>     // }
> }
> ```

## 실행 결과
```java
[d06b992f...] request scope bean create
[d06b992f...][http://localhost:8080/log-demo] controller test
[d06b992f...][http://localhost:8080/log-demo] service id = testId
[d06b992f...] request scope bean close
```
- ObjectProvider.getObject() 를 호출하는 시점까지 request scope 빈의 생성을 지연할 수 있습니다.
- ObjectProvider.getObject() 를 호출하는 시점에는 HTTP 요청이 진행중이므로 request scope 빈의 생성이 정상 처리됩니다.
- ObjectProvider.getObject() 를 LogDemoController , LogDemoService 에서 각각 한번씩 따로 호출해도 같은 HTTP 요청이면 같은 스프링 빈이 반환됩니다.

다음 [포스트](https://ljw22222.github.io/posts/spring-basic-fifteen/)에서는 이보다 더 개선된 Proxy에 대해서 설명 드리겠습니다. 