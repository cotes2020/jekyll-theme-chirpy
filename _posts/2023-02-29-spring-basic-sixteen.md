---
title: Spring Basic[스코프와 Proxy]
date: 2023-02-29 21:15:00 +0800
categories: [Spring-Basic, Spring-MainPoint]
tags: [Spring]
---

# Proxy
앞서 Provider방식에 대해 알아 보았습니다. 이번에는 Proxy방식을 사용 해보겠습니다.

## Proxy (Controller, Class, Service)예제 코드
### Class
```java
Component
@Scope(value = "request", proxyMode = ScopedProxyMode.TARGET_CLASS)
public class Logger {
}
```
- **proxyMode = ScopedProxyMode.TARGET_CLASS** 이부분이 핵심 입니다.
    - 적용 대상이 인터페이스가 아닌 클래스면 TARGET_CLASS 를 선택합니다.
    - 만약 인터페이스면 INTERFACES를 선택합니다.

이렇게 하면 Logger의 가짜 프록시 클래스를 만들어두고, HTTP Request와 상관없이 가짜 프록시 클래스를 다른 빈에 주입이 가능합니다. 

### Controller
```java
@Controller
@RequiredArgsConstructor
public class LogDemoController {
 private final LogService logService;
 private final Logger logger;
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

### Service
```java
@Service
@RequiredArgsConstructor
public class LogService {
 private final Logger logger;
 public void logic(String id) {
    logger.log("service id = " + id);
    System.out.println("myLogger = " + logger.getClass());
 }
}
```

## 웹 스코프와 프록시의 동작 원리
### print 출력 결과
```java
myLogger = class hello.core.common.MyLogger$$EnhancerBySpringCGLIB$$b68b726d
```
- CGLIB라는 라이브러리로 내 클래스를 상속 받은 가짜 프록시 객체를 만들어서 주입합니다.<br/>

 ![Spring-basic-DI-Container-png](/assets/img/spring/spring-basic-DI-container.png){: width="700" height="600" }<br/>
가짜 프록시객체는 요청이 들어오면, 그떄 내부에서 진짜 빈을 요청하는 위임로직이 들어있습니다.

- 가짜 프록시 객체는 내부에 진짜 Proxy(사진)를 찾는 방법을 알고 있습니다.
- 클라이언트가 Proxy.proxy()을 호출하면 사실은 가짜 프록시 객체의 메서드를 호출한 것입니다.
- 가짜 프록시 객체는 request 스코프의 진짜 Proxy.proxy()를 호출 합니다.
- 가짜 프록시 객체는 원본 클래스를 상속 받아서 만들어졌기 때문에 이 객체를 사용하는 클라이언트 입장에서는 사실 원본인지 아닌지도 모르게 동일하게 사용할 수 있습니다. ( 다형성 )

## 결론
CGLIB라는 라이브러리를 활용하여 내 클래스를 상속받은 가짜 프록시 객체를 동적으로 생성하고 주입합니다. 이 프록시 객체는 요청이 발생할 때 내부에서 실제 빈을 요청하는 위임 로직을 포함하고 있습니다. 주목할 점은 이 객체가 실제로 request scope와 직접적인 관련이 없으며, 가짜로 생성된 것이기 때문에 내부에는 간단한 위임 로직만 존재하며, 싱글톤처럼 동작합니다.
하지만 싱글톤을 사용하는것 같지만, 다르게 동작은 다르기 때문에 주의하면서 사용하여야 합니다.       
그리고 이런 특별한 스코프는 꼭 필요한 곳에만 최소하하여 사용하여야 합니다.          
무분별하게 사용하였다가는 유지보수가 어려워질 수 있습니다.