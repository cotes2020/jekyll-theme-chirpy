---
title: Spring Basic[스코프와 Proxy]
date: 2023-02-29 21:15:00 +0800
categories: [Spring-Basic, Scope And Proxy]
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
- proxyMode = ScopedProxyMode.TARGET_CLASS 이부분이 핵심 입니다.
    - 적용 대상이 인터페이스가 아닌 클래스면 TARGET_CLASS 를 선택합니다.
    - 만약 인터페이스면 INTERFACES를 선택합니다.
- 이렇게 하면 Logger의 가짜 프록시 클래스를 만들어두고, HTTP Request와 상관없이 가짜 프록시 클래스를 다른 빈에 주입이 가능합니다. 

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
 }
}
```

## 웹 스코프와 프록시의 동작 원리