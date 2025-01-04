---
title: "#01_Spring Boot 프로젝트 처음 시작하기"
categories: [Java, Spring Boot]
tags: [Java, Spring Boot]
---

# 1. Start Spring Boot

### Spring Boot?

- Spring기반의 gradle 프로젝트의 독립적인 application 생성을 쉽게 하도록 해준다.

- Spring framework의 최적화된 기본 설정을 제공한다.

  - third-party libararies 제공 => ex) tomcat WAS 8080 포트 기본 제공

- code generation 및 xml 설정을 사용하지 않는다.

---

### Create Spring Boot(+gradle) Project

##### 1. Spring boot 생성기로 생성

- https://start.spring.io/

  - Project / language / boot 버전 등 설정

  - project meta 데이터 설정

  - 필요 시 의존성 생성 후 generate 하여 수행

##### 2. IDE 에서 생성

version

> - spring boot : 3.2.0
>
> - jdk : 17

1. intelliJ 에서 `new` > `Project` 선택

2. 설정

   - Language : `java`

   - build system : `Gradle`

   - JDK : 17 version jdk 디렉토리 선택

   - Gradle : DSL

3- `build.gradle` 에서 sprint boot 3 추가 ([참고](https://docs.spring.io/spring-boot/docs/current/reference/htmlsingle/#getting-started.first-application.gradle))

```groovy
plugins {
    id 'java'
    id 'org.springframework.boot' version '3.2.0'
}

apply plugin: 'io.spring.dependency-management'

group = 'com.example'
version = '0.0.1-SNAPSHOT'
sourceCompatibility = '17'

repositories {
    mavenCentral()
}

dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
}
```

4. scr > main > java 하위 디렉토리 및 클래스 생성하여 다음과 같이 작성

   ```java
   package yewool.study;

   import org.springframework.boot.SpringApplication;
   import org.springframework.boot.autoconfigure.SpringBootApplication;

   @SpringBootApplication
   public class Application {
       public static void main(String[] args) {
           SpringApplication.run(Application.class, args);
       }
   }
   ```

5. 실행

   1. IDE(intelli J)에서 실행
      - `ctrl` + `shift` + `F10` 을 통해 프로젝트 실행 후 웹에서 `localhost:8080` 소로 무언가 띄워져 있으면 성공

   2- 터미널에서 실행

   - bootJar 빌드

     ```
     gradlew bootJar
     ```

   - `{프로젝트 root 경로}\build\libs` 로 이동하여 jar 파일 생성 확인 후 수행

     ```
     java -jar {settings.gradle에 설정한 project name}.jar
     ```

---

### Spring Boot 프로젝트 구조

- 프로젝트 구조

  - 소스 코드 (src\main\java)

  - 소스 리소스 (src\main\resource)

  - 테스트 코드 (src\test\java)

  - 테스트 리소스 (src\test\resource)

- 메인 어플리케이션 위치

  - 메인 어플리케이션 ? `@SpringBootApplication` 이 붙은 최상위 클래스

  - default 패키지 바로 아래에 생성하는 것을 추천
