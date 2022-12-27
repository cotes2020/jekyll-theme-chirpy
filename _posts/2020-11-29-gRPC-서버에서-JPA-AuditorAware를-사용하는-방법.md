---
title: gRPC 서버에서 ThreadLocal을 올바르게 사용하는 방법 (gRPC Context, gRPC + JPA AuditorAware)
authors: Jongin Kim
date: 2020-11-29 00:00:00 +0900
categories: [grpc]
tags: [jpa, java, grpc]
---
### Spring Web + JPA 에서 AuditorAware 구현
- grpc 서버라고 가정하지 않고 Spring Web 기반의 API 서버라고 생각해보면
- 우리는 JPA AuditorAware를 보통 아래와 같이 구현할 것이다. (SpringSecurity를 사용했을 때를 가정하고 작성한 코드다.)
```java
@Component
public class CurrentAuditorAware  implements AuditorAware<String> {
    @Override
    public Optional<String> getCurrentAuditor() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (null == authentication || !authentication.isAuthenticated()) {
            return null;
        }
        Member member = (Member) authentication.getPrincipal();
        return Optional.of(member.getMemberId());
    }
}
```

- SecurityContext정보는 ThreadLocal에 저장되고 있고. SecurityContextHolder를 통해 ThreadLocal에서 SecurityContext정보를 가져온 것 이다.(SecurityContextHolder전략을 ThreadLocal로 설정한 경우)
- 여기서 말하고자 하는 점은 ThreadLocal에서 값을 가져왔다는 것 이다.
- SpringSecurity를 사용하든 안하든 결국 AuditorAware 구현체의 getCurrentAuditor 메소드는 ThreadLocal에서 가져온 값을 리턴하는 방법으로 보통 작성한다.

### grpc server + JPA 에서 AuditorAware 구현
- 우선 grpc server에서 ThreadLocal에 요청자 정보를 담고, AuditorAware를 구현하는 소스를 작성한다고 하면 대부분이 아래와 같이 생각 할 것이다.
	- grpc 헤더(메타데이터)에 token 데이터를 담아서 rpc요청 -> grpc interceptor에서 해당 token을 decode해서 유저정보 얻기 -> ThreadLocal에 유저정보 담기 -> AuditorAware getCurrentAuditor 구현시 ThreadLocal에 유저정보 응답

- 하지만 이렇게 ThreadLocal에서 그대로 값을 가져와 AuditorAware를 구현하는건 grpc 서버에서는 꽤나 위험한 행위이다.
- 일반 웹 요청 같은 경우를 생각해보자. Servlet 같은 경우엔 일반적으로 HTTP 요청 하나당 하나의 스레드를 사용하고 응답을 할 때까지는 그 스레드는 블로킹 되어있다. 또 서블릿의 생명주기와 스레드의 생명주기가 동일하게 처리된다. 이런 이유들 때문에 요청당 memberId (ThreadLocal에서 가져온 회원정보) 값이 요청자의 값인 게 의심의 여지가 없다.
- **하지만 grpc 요청은 그렇지 않을 수 있다.**
- `gRPC의 모든 콜백은 다른 스레드에서 발생할 수 있고, 여러 grpc에 대한 콜백은 같은 스레드에서 발생할 수 있다..` 즉, 요청당 memberId (ThreadLocal에서 가져온 회원정보) 값이 요청자의 값이 아닐 수 있는 상황이 발생할 수 있다는 것이다.
- JPA AuditorAware를 구현해 단순히 뭐 `@CreatedBy`, `@LastModifiedBy`의 값을 채우는 게 아닌 정말 중요한 개인 정보에 대한 로그를 쌓는다든지, 인증/인가 정보에 활용한다든지 하는 경우엔 심각한 문제가 발생할 수 있다.

### 해결책 (java기준)
- 그래서 grpc-java에서는 `Context`라는 클래스와 `Contexts`라는 유틸성 클래스를 제공해준다. 이들을 사용하면 쉽게 해결이 가능하다.
- 결론부터 말하자면 사실. Context도 ThreadLocal를 사용한다. 그렇다면 어떻게 이 문제를 해결할까? 막상 grpc-java 소스를 보면 엄청 간단한 컨셉이다.
- 아주 간략하게 설명하면 Context는 Storage라는 ThreadLocal 스토리지에 attach / detach 할 수 있게 만들어 둔 클래스다.
- 그리고 더 중요한 클래스는 Contexts이 유틸 클래스인데 이 함수에 Contexts.interceptCall() 메서드를 보면 grpc 요청으로 인해 일어나는 이벤트를 listening 해서 메시지 요청 시점과 요청 종료 시점 혹은 사실상 종료에 준하는 시점에 맞춰 Context를 detach 하는 것을 알 수 있다.
- 그리고 grpc interceptor(ServerCall.Listener)에서 `next.startCall(call, headers)` 대신 `Contexts.interceptCall(context, call, headers, next)`를 return 해주면 된다. [소스예시](https://github.com/grpc/grpc-java/blob/master/examples/example-jwt-auth/src/main/java/io/grpc/examples/jwtauth/JwtServerInterceptor.java#L43)
- 즉, grpc 메시지 호출 시점과 종료시점에 맞춰 ThreadLocal을 제어해주는 그런 컨셉이다. 
- 간단하게 말했지만 아래 참고링크를 보면 위에 소개한 것 처럼 간단한 기능만 있는건 아니다.
- 물론 아직 Context.Storage class는 실험단계라고 되어있긴 한데 어떤 부분을 실험하는지도 명확하지 않고.. 오랫동안 변경점도 없고.. 제가 1달정도 실무에 사용해보니 큰 문제는 없는 것 같다. [참고링크](https://github.com/grpc/grpc-java/issues/2462)

>참고링크
> - [ThreadLocalContextStorage](https://github.com/grpc/grpc-java/blob/3811ef3d22f90ae8da8200964d178cbd829ee9f8/context/src/main/java/io/grpc/ThreadLocalContextStorage.java#L25)
> - [Context](https://github.com/grpc/grpc-java/blob/3811ef3d22f90ae8da8200964d178cbd829ee9f8/context/src/main/java/io/grpc/Context.java#L99)
> - [Contexts](https://github.com/grpc/grpc-java/blob/3811ef3d22f90ae8da8200964d178cbd829ee9f8/api/src/main/java/io/grpc/Contexts.java#L25)
