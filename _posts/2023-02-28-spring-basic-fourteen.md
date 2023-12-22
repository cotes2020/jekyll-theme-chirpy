---
title: Spring Basic[웹 스코프]
date: 2023-02-28 23:08:00 +0800
categories: [Spring-Basic, 웹 스코프]
tags: [Spring]
---

# 웹 스코프
## 웹 스코프란
프로토 타입과는 다르게 스프링이 해당 스코프의 종료 시점까지 관리하는 스코프 입니다.         
그로 인해 종료 메서드가 호출됩니다, 또한 웹 환경에서만 동작합니다.          

## 웹 스코프의 종류
### Request
- HTTP 요청 하나가 들어오고, 나갈 때 까지 유지되는 스코프입니다.
- 각각의 HTTP 요청마다 별도의 빈 인스턴스가 생성되고 관리 됩니다.
### Request 예시 코드
```java
@Component
@Scope("request")
public class RequestBean {
    //...
}
```

### session
- HTTP Session과 동일한 생명주기를 가지는 스코프 입니다.
- 세션이 시작되고 종료시점에 빈이 생성 및 소멸 됩니다.
- 사용자의 세션과 관련된 데이터를 유지하고자 할 때 유용합니다.
### session 예시 코드
```java
@Component
@Scope("session")
public class SessionBean {
    //...
}
```
        
## application
- 서블릿 컨텍스트와 동일한 생명주기를 가지는 스코프 입니다.
    - 즉, 애플리케이션이 시작될때 생성되고, 종료시점까지 유지되는 빈 입니다.
- 전역에서 공유되는 상태를 가진 빈을 사용하고자 할 때 유용합니다.
### session 예시 코드
```java
@Component
@Scope("application")
public class ApplicationBean {
    //...
}
```
<br/>

# request 스코프 예제
1. build.gradle에 web 라이브러리를 추가합니다. ( 있으면 건너 뛰어도 됨 )
```java
implementation 'org.springframework.boot:spring-boot-starter-web'
```
2. main메서드를 실행하면 웹 어플리케이션이 실행되는 것을 확인하실 수 있습니다.
- 위의 라이브러리를 추가하면, Spring Boot는 내장 톰켓 서버를 활용해서 웹 서버와 스프링을 함께 실행 시켜 줍니다.
- 만약 실행시에 포트 오류가 발생하면 포트를 바꿔주면 됩니다
        - main -> resource -> application.properties 에 아래의 설정을 추가합니다
        ```java
        server.port=[사용하고싶은 포트 번호]
        ```

3. request 스코프 예제 코드
- 이 코드는 동시에 여러 HTTP 요청이 오면 요청이 남긴 로그인지 구분하기 어려울때 사용하는 request 스코프 예제 입니다.
### LogerScope Class
```java
@Component
@Scope(value = "request")
public class LogerScope {
    private String uuid;
    private String requestURL;
    public void setRequestURL(String requestURL) {
        this.requestURL = requestURL;
    }
    public void log(String message) {
        System.out.println("[" + uuid + "]" + "[" + requestURL + "] " +
        message);
    }
    @PostConstruct
    public void init() {
        uuid = UUID.randomUUID().toString();
        System.out.println("[" + uuid + "] request scope bean create:" + this);
    }
    @PreDestroy
    public void close() {
        System.out.println("[" + uuid + "] request scope bean close:" + this);
    }
}
```
- 로그를 출력하기 위한 Class 입니다.
- 이 request 스코프 빈은 HTTP 요청당 하나씩 생성되고, 요청이 끝나는 시점에 소멸 됩니다.
- 빈이 소멸되는 시점에 @PreDestory를 사용해서 종료 메세지를 남깁니다.
### Service
```java
@Service
@RequiredArgsConstructor
public class LogService {
 private final LogerScope logerscope;
 public void logic(String id) {
    logerscope.log("service id = " + id);
 }
}
```
- 비즈니스 로직이 있는 서비스 계층입니다.
### Controller
```java
@Controller
@RequiredArgsConstructor
public class LogDemoController {
 private final LogService logService;
 private final LogerScope logerscope;
 @RequestMapping("log-demo")
 @ResponseBody
 public String logDemo(HttpServletRequest request) {
    String requestURL = request.getRequestURL().toString();
    logerscope.setRequestURL(requestURL);
    logerscope.log("controller test");
    logService.logic("testId");
    return "OK";
 }
}
```
- 로거 확인용 테스트용 컨트롤러입니다.
- 여기서 HttpServletRequest을 통해 요청 URL을 받아옵니다.
- 이렇게 받은 URL값을 logerscope에 저장해 둡니다.
- logerscope는 HTTP 요청 당 각각 구분되므로 다른 HTTP 요청 때문에 값이 섞이는 걱정은 안해도 됩니다.
- Controler에서 controller test라는 로그를 남깁니다.

하지만 이렇게 하면 오류가 발생합니다. 그이유는 다음 [포스트](https://ljw22222.github.io/posts/spring-basic-fifteen/)에서 다루도록 하겠습니다.
