---
title: "#02_Sprig Boot의 원리(gradle 지시어, @SpringBootApplication, Tomcat)"
categories: [Java, Spring Boot]
tags: [Java, Spring Boot, complieOnly, SpringBootApplication, Tomcat]
---

# 2. 스프링 부트 원리

### gradle dependencies 이해

- gradle 기본 참고 : https://velog.io/@sa1341/Gradle-%EA%B8%B0%EB%B3%B8-%EB%8B%A4%EC%A7%80%EA%B8%B0-1%ED%8E%B8

- dependencies 에 정의되는 각 지시어 의미

  > - Compile classpath - JDK가 java code를 .class files로 컴파일 할때 필요로 하는 의존 리스트이다.
  > - Runtime classpath - 컴파일된 자바 코드가 실제로 실행할때 필요로 하는 의존 리스트이다.

  - `complieOnly` - 컴파일 패스에만 설정

  - `runtimeOnly` - 런타임 패스에만 설정

  - `implementation` - 위 두개의 패스에 둘다 설정 + 해당 의존성을 직/간접적으로 의존하고 있는 모든 의존성 재빌드

  - `complie` - 위 두개의 패스에 둘다 설정 + 해당 의존성을 직접적으로 의존하고 있는 의존성만 재빌드

  - 💡 위 네 개의 지시어는 앞에 `test`를 붙여 테스트 환경에서만 사용하게 할 수도 있다.

  - ➕ `annotationProcessor` - lombok 사용 시 필수 추가 지시어

### 자동 설정 이해

`@SpringBootApplication` 구성

- @SpringBootConfiguration

- @EnableAutoConfiguration

  - ==> 두 단계로 다뉘어 읽힘

    - 1단계: @ComponentScan

    - 2단계: @EnableAutoConfiguration

- @ComponentScan

  - @Component 어노테이션을 가진 클래스를 스캔하여, 빈으로 등록

  - ex) @Configuration @Repository @Service @Controller @RestController

- @EnableAutoConfiguration

  - spring.factories
  - org.springframework.boot.autoconfigure.EnableAutoConfiguration
    - @Configuration
    - @ConditionalOnXxxYyyZzz

> ❗web application 서버로 사용하지 않는다면 아래 두 개의 설정으로도 띄울 수 있다.
>
> > - @Configuration
> >
> > - @ComponentScan
>
> => 별도의 `ServletWebServerFactory` 생성 필요 (=> `@EnableAutoConfiguration` 에서 자동 생성해줌)

### 독립적으로 실행 가능한 JAR

- gradle의 tasks의 build로 들어가보면 아래 항목들이 존재

  - build : executable jar, plain jar 두 개로 빌드

  - bootJar : executable jar만 빌드

  - clean : build된 jar파일을 삭제

- jar 파일?

  - 모든 클래스 (의존성 및 어플리케이션)를 하나로 압축하는 방법

  - 참고 : [springboot gradle 배포 (war방식, jar방식)](<https://git.bwg.co.kr/gitlab/finlab/archi/work/-/wikis/springboot-gradle-%EB%B0%B0%ED%8F%AC-(war%EB%B0%A9%EC%8B%9D,-jar%EB%B0%A9%EC%8B%9D)>)

# 3\_내장-웹-서버

### 자동 설정으로 생성되는 Tomcat

<!-- prettier-ignore -->
> Tomcat이란 ?
>
> -  자바 기반 웹 애플리케이션을 구동하기 위한 서버 소프트웨어
> - JSP/서블릿(HTTP 요청을 받고, 응답을 되돌려주는 자바 웹 기술) 등의 동적 요청을 처리할 뿐 아니라, 빌드된 정적 리소스(HTML, JS, CSS 등)의 서버 역할을 해준다.
> - Spring Boot는 내부적으로 이 Tomcat을 내장해 두어, 별도 설정 없이 단일 애플리케이션으로 별도 웹 서버 구축 필요 없이, 쉽게 구동 및 배포할 수 있다.
{: .prompt-info }

- `build.gradle`에 springboot를 의존성에 주입해두면, 프로젝트의 External Libraries에 `org.springframework.boot:spring-boot-autoconfigure` 가 생성된다.

- 하위의 META-INF > spring.factories에 여러 자동 설정들을 볼 수 있다.

  - 자세한 정보는 `spring-autoconfigure-metadata.properties` 파일에서 볼 수 있다.

  - `ServletWebServerFactoryAutoConfiguration` 에 웹 서버(톰캣)과 컨텍스트가 정의되어 있다.

    - 내부에 `TomcatServletWebServerFactoryCustomizer`로 서버 커스터마이징도 한다.

  - `DispatcherServletAutoConfiguration`에 서블릿이 정의되어 있다.
