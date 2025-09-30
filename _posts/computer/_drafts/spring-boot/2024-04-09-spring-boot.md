---
title: "Spring Boot"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: [Spring-Boot]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-04-09. 12:38
# last_modified_at: 2024-04-11. 16:46
last_modified_at: 2024-08-29. 21:22
---

{% include embed/youtube.html id='AalcVuKwBUM' %}

## 스프링

---

### 스프링 프레임워크

자바에서 가장 많이 사용되는 프레임워크  

어떻게 하면 객체지향적으로 프로그래밍을 할 수 있을까?  
어떻게 하면 코드를 재사용(짧고, 간결하게) 할 수 있을까?  

- 의존성 주입 (DI, Dependency Injection)
- 제어 역전 (IoC, Inversion of Control)
- 관점 지향 프로그래밍 (AOP, Aspect Oriented Programming)

위 세 가지를 중점적으로 만들어진 프레임워크  
위 세 가지를 통해 느슨한 결합 (Loose Coupling)을 이루어낼 수 있다.  

느슨한 결합으로 개발하면 단위 테스트를 쉽게 할 수 있다.  

### 스프링 프레임워크의 대표적 모듈

- Spring JDBC
- Spring MVC
- Spring AOP
- Spring ORM
- Spring Test
- Spring Expression Language (SpEL)
- ...

이하 스프링.  

## 스프링 부트

---

### 스프링 부트

스프링에서 제공하는 한 프로젝트  
큰 범주로 스프링이 있고, 이를 편하게 사용할 수 있게 만든 것이 스프링 부트.  

### 왜 Why

스프링도 의존성 주입 등을 편하게 쓰려고 만들어졌는데,  
그것도 좀 복잡해서 나온게 스프링 부트.  

스프링 부트는 단지 실행만 하면 되는 스프링 기반의 어플리케이션을 쉽게 만들 수 있다.  
실행만 하면 가동 가동이 되는 어플리케이션을 만들 수 있는 템플릿을 제공해주는 것이 목적  

스프링은 `Hibernate` 관련 설정을 직접 해줘야 하는데,  
스프링 부트는 `Hibernate` 설정을 자동으로 해준다.  

### 제공하는 기능

#### 자동 설정 (Auto Configuration)

- 앱 개발에 필요한 모든 디펜던시(Dependency)를 프레임워크에서 관리.
- jar파일이 클래스 패스에 있으면, Dispatcher Servlet을 자동으로 구성한다.
- 미리 설정되어 있는 Starter 프로젝트를 제공
- xml 설정 없이 자바 코드를 이용해 설정 가능

#### 디펜던시 버전 관리

- 앱을 개발에 쓰는 디펜던시들은 호환되는 버전으로 관리해줘야 함
- SpringBoot-Starter를 제공하여 자동으로 버전 관리

### Starter 디펜던시

- `spring-boot-starter-web`: RESTful 응용 프로그램
- `spring-boot-starter-test`: 단위 테스트, 통합 테스트
- `spring-boot-starter-security`: 스프링 시큐리티와 OAuth2를 사용한 보안 (인증, 권한 부여)
- `spring-boot-starter-jdbc`: 기본적인 JDBC를 사용한 데이터 액세스
- `spring-boot-starter-data-jpa`: JPA를 사용한 데이터 액세스 (Hibernate)
- `spring-boot-starter-cache`: 스프링의 캐시 추상화를 사용한 캐시
- ...

## 의존성 주입

---

DI (Dependency Injection)  

### 의존성 주입을 쓰지 않은 코드

```java
@RestController
public class HelloController {
 private HelloService service = new HelloServiceImpl();

 @GetMapping("/hello")
 public String getHello() {
  return service.getHello();
 }
}
```

- `HelloController`는 `HelloService` 객체에 의존하고 있다.
- 객체의 인스턴스를 얻게 되면 객체 간의 결합이 강해진다.
- 이런 코드 작성은 단위테스트를 위해 `Mock` 객체를 사용할 수 없다.

### 의존성 주입을 쓴 코드

```java
@Service
public class HelloServiceImpl implements HelloService {
 @Override
 public String getHello() {
  return "hello";
 }
}

@RestController
public class HelloController {
 private final HelloService service;

 @Autowired
 public HelloController(HelloService service) {
  this.service = service;
 }

 @GetMapping("/hello")
 public String getHello() {
  return service.getHello();
 }
}
```

- `@Service`, `@Autowired` 어노테이션을 통해 `HelloServiceImpli` 객체를 주입받는다.
- 위와 같이 코드를 작성하면, 단위 테스트를 위해 `Service` 객체를 `Mock` 객체로 대체하여 쉽게 테스트할 수 있다.

## 관점 지향 프로그래밍 (AOP, Aspect Oriented Programming)

---

스프링 프레임워크에서 제공하는 강력한 기능 중 하나  

쉽게 말해, OOP를 보완하는 수단으로,  
공통 기능을 모듈화(분리)하여 유지보수/재사용성을 높이는 기술.  

AOP를 통해 기존 프로젝트에 다향한 기능을 로직 수정 없이 추가할 수 있다.  

이런 개발 방식을 통해 결합도를 낮추고, 유지보수성을 높일 수 있다.  

- 알아보면 좋은 것
  - 프록시 디자인 패턴
  - 핵심적인 관점
  - 부가적인 관점
  - 흩어진 관심사(Crosscutting Concerns)

## Hello World

```java
@GetMapping({"/hello"})
public String getHello()
{
 return "Hello Around Hub Studio";
}
```

## pom.xml, build.gradle

---

- `pom.xml`: Maven 프로젝트 설정 파일
- `build.gradle`: Gradle 프로젝트 설정 파일

- `Project Object Model (POM)` 정보를 담고 있음
  - 프로젝트 정보: 프로젝트 이름, 개발자 목록, 라이센스, ...
  - 빌드 설정 정보: 소스, 리소스, 라이프 사이클, 실행할 플러그인, ...
  - POM 연관 정보: 의존 프로젝트(모듈), 상위 프로젝트, 하위 모듈, ...

## application.properties, application.yml

---

- `application.properties`: 프로퍼티 설정 파일
- `application.yml`: 프로퍼티 설정 파일

`application.yml`이 많이 사용된다.  
둘 다 사용할 경우, 우선순위는 `application.properties`가 더 높다.  

## H2 DB 설정

---

`build.gradle.kts`

```gradle
dependencies {
 runtimeOnly("com.h2database:h2")
}
```

`application.yml`

```yml
spring:
  application:
    name: GraphNovel
  h2:
    console:
      enabled: true
      path: /h2-console
  datasource:
    url: jdbc:h2:~/test
    driver-class-name: org.h2.Driver
    username: sa
    password:
  jpa:
    database-platform: org.hibernate.dialect.H2Dialect
    properties:
      hibernate:
        dialect: org.hibernate.dialect.H2Dialect
        format_sql: true
        show_sql: true
    hibernate:
      ddl-auto: create-drop
      # 크게 의미는 없다, H2를 끄면 같이 날라가서
```

[localhost:8080/h2-console](http://localhost:8080/h2-console)  

`show tables;`  
`select * from user;`  

## 구현 구성

---

### Package

특정 `Domain`에 대한 클래스들을 모아놓는다.  

### Controller

`Input`을 받아서 `Output`을 내보내는 역할을 한다.  

```java
// @Controller 라는 것도 있는데, @RestController가 더 많이 쓰인다.
@RestController
@RequiredArgsConstructor // 생성자를 직접 만들어서 주입해주는 방법 대신
public class MemberController { 

 private final MemberService memberService; // @RequiredArgsConstructor를 통해 자동으로 주입

 @Autowired
 public MemberController(MemberService memberService) {
  this.memberService = memberService;
 }

 @PostMapping("/join")
 public String join(@RequestBody JoinRequest joinRequest)
 {
  String id = joinRequest.getId();
  String name = joinRequest.getName();
  String phoneNumber = joinRequest.getPhoneNumber();

  // String result = memberService.join(id, name, phoneNumber);
  String result = memberService.join(joinRequest);

  // if (result.equalsIgnoreCase("success"))
  if ("success".equalsIgnoreCase(result)) // Null Exception 방지
  {
   return "success";
  }
  else
  {
   return "fail";
  }

  // Talend API Tester, 확장 프로그램
  // http://localhost:8080/join
  // { "id": "feature", "name": "asd", "phoneNumber": "010-0000-0000" }
 }
}
```

`controller/dto/JoinRequest.java`

```java
@Data // @Getter, @Setter, @ToString, @EqualsAndHashCode, @RequiredArgsConstructor를 한번에 ?
public class JoinRequest {
 private String id;
 private String name;
 private String phoneNumber;
}
```

- `@RestController`: `@Controller`와 `@ResponseBody`를 합친 어노테이션
  - `@Controller`: View를 반환하는 컨트롤러
  - `@ResponseBody`: View를 반환하는 것이 아닌, HTTP Response Body에 직접 작성하는 컨트롤러

- `@RequiredArgsConstructor`: `final`이 붙은 필드를 생성자로 만들어준다.

- DTO (Data Transfer Object): 데이터를 전송하는 객체
  - `@RequestBody`: HTTP 요청의 body 내용을 자바 객체로 매핑하는 역할
  - `@ResponseBody`: 자바 객체를 HTTP 응답의 body 내용으로 매핑하는 역할
  - `@RequestBody`를 통해 `JoinRequest` 객체를 받아서 `MemberService`로 전달한다.
  - `@ResponseBody`를 통해 `MemberService`의 반환값을 HTTP 응답의 body 내용으로 반환한다.

### Service

`Controller`로부터 요청을 받아서 비즈니스 로직을 처리하는 영역  

```java
public interface MemberService {
 String join(JoinRequest joinRequest);
}
```

```java
@Service
@RequiredArgsConstructor
public class MemberServiceImpl implements MemberService {

 private final MemberRepository memberRepository;

 @Override
 public String join(JoinRequest joinRequest) {
  Member member = Member.builder()
    .id(joinRequest.getId())
    .name(joinRequest.getName())
    .phoneNumber(joinRequest.getPhoneNumber())
    .build();
  memberRepository.save(member);
  return "success";
 }
}
```

### Repository

`Service`로부터 받은 데이터를 DB에 저장하거나, `Service`로 데이터를 전달하는 영역  

```java
public interface MemberRepository extends JpaRepository<Member, Long /*ID Type*/ > { }
```

### Test

`Controller`, `Service`, `Repository`에 대한 테스트를 작성하는 영역  

```java
@SpringBootTest
public class MemberRepositoryTest {
 @Autowired MemberRepository memberRepository;
 @Test
 public void crudTest()
 {
  // 예시를 위한 코드, 테스트 코드는 이렇게 짜는 것이 아님

  Member member = Member.builder()
    .id("feature")
    .name("asd")
    .phoneNumber("010-0000-0000")
    .build();

  // create test
  memberRepository.save(member);

  //get test
  Member foundMember = memberRepository.findByID(1L).get();

  // save, findByID 같은 메소드는 JpaRepository > ListCrudRepository > CrudRepository 에서 제공하는 메소드
  // 클래스와 ID 타입을 주면, 대부분의 쿼리를 자동으로 만들어준다.
 }
}
```

## REST API

---

[REST API](/posts/rest-api/)  

`Controller`로 요청을 받는다? `REST API` 통신을 한다.  

## CRUD의 표현

---

- `@GetMapping`: Read, 서버에 있는 리소스를 가져올 때 사용
- `@PostMapping`: Create, 서버에 리소스를 추가할 때 사용
- `@PutMapping`: Update
  - 리소스가 존재하면 갱신하고, 없으면 새로 생성, Update
  - 보통 잘 안쓰고, `@PostMapping`을 많이 쓴다.
- `@DeleteMapping`: Delete, 서버를 통해 리소스를 삭제

## 생각의 흐름

---

1. 어떤 기능을 만들것인가?
2. 어떤 기능을 만들기 위해 어떤 데이터가 필요한가?
   - = 어떤 데이터를 저장할 것 인가?
   - `Entity`
   - DB와 직접 연결되는

## 메모

---

### `@Entity`

- `@Entity`를 사용하면 클래스를 데이터베이스 테이블과 매핑할 수 있다.
- `@Entity`가 붙은 클래스는 JPA가 관리하는 클래스가 된다.
- `@Entity`가 붙은 클래스는 `@Id`가 붙은 필드를 기본키로 사용한다.
- `@Entity`가 붙은 클래스는 `@NoArgsConstructor` 어노테이션을 사용하여 기본 생성자를 만들어준다.
- `@Entity`가 붙은 클래스는 `@AllArgsConstructor` 어노테이션을 사용하여 모든 필드를 매개변수로 받는 생성자를 만들어준다.
- `@Entity`가 붙은 클래스는 `@Builder` 어노테이션을 사용하여 빌더 패턴을 사용할 수 있다.
- `@Entity`가 붙은 클래스는 `@Getter`, `@Setter` 어노테이션을 사용하여 getter, setter 메소드를 만들어준다.

### `@Builder`

- `@Builder`를 사용하면 클래스의 필드를 기반으로 데이터 컬럼을 구성을 해준다. (?)
- 생성자를 통해 객체를 생성할 수 있다.

### `@GeneratedValue`

- Like auto_increment

- `@GeneratedValue`를 사용하면 자동으로 값을 생성해준다.
- `@GeneratedValue(strategy = GenerationType.IDENTITY)`: 자동 증가
- `@GeneratedValue(strategy = GenerationType.SEQUENCE)`: 시퀀스
- ...

### _

- 유효성 검사
  - @valid, @validated
  - exceptionhandler, controlleradive

- restTemplate, webClient을 구현해보는 연습
- 탈퇴, 휴먼 기능을 넣어보는 연습

- `@PathVariable`: GET 요청에서 파라미터를 전달하기 위해 URL에 값을 담아 요청하는 방법

```java
@GetMapping("/hello/{name}")
public String hello(@PathVariable String name)
{
 return "hello " + name;
}

@GetMapping("/hello/{some}")
public String hello(@PathVariable("some") String name)
{
 return "hello " + name;
}
```

- `@RequestParam`: GET 요청에서 쿼리 문자열을 전달하기 위해 URL에 값을 담아 요청하는 방법
- `?`를 기준으로 우측에 `{Key}={Value}`의 형태로 전달되며, 복수 형태로 전달할 경우 `&`로 구분
- `http://localhost:8080/hello?name=feature&age=20`

```java
@GetMapping("/hello")
public String hello(
 @RequestParam String name,
 @RequestParam int age)
{
 return "hello " + name + " " + age;
}

// 어떤 값이 들어올지 모를 때
@GetMapping("/hello")
public String hello(
 @RequestParam Map<String, String> param)
{
 StringBuilder sb = new StringBuilder();

 param.forEach((key, value) -> {
  sb.append(key).append(": ").append(value).append("\n");
 });

 return sb.toString();
}
```

- DTO 사용: `Key`와 `Value`가 정해져있지만, 받아야할 값이 많을 때 사용

```java
public class MemberDTO {
 private String name;
 private int age;
}

@GetMapping("/hello")
public String hello(MemberDTO memberDTO)
{
 // return "hello " + memberDTO.getName() + " " + memberDTO.getAge();
 return "hello " + memberDTO.toString();
}
```

- Class 내에서 공통된 URL을 사용할 때, `@RequestMapping`을 사용하여 URL을 지정할 수 있다.

```java
@RequestMapping("/hello")
public class HelloController {
 @GetMapping("/world") // /hello/world
 public String world()
 {
  return "world";
 }
}
```

- `@PostMapping`: POST 요청을 받을 때 사용

```java
@PostMapping("/member")
public String hello(@RequestBody Map<String, Object> postData)
{
 StringBuilder sb = new StringBuilder();

 postData.entrySet().forEach(entry -> {
  sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
 });

 return sb.toString();
}
```

- DTO 사용: `Key`와 `Value`가 정해져있지만, 받아야할 값이 많을 때 사용

```java
public class MemberDTO {
 private String name;
 private int age;
}

@PostMapping("/member")
public String hello(@RequestBody MemberDTO memberDTO) // @RequestBody를 꼭 사용해야 한다.
{
 return "hello " + memberDTO.toString();
}
```

- Swagger: API 문서를 만들어주는 도구, 협업을 위해 필요한 라이브러리
- 서버로 요청되는 API리스트를 HTML 화면으로 문서화하여 테스트 할 수 있는 라이브러리
- 서버가 가동되면서 `@RestController`가 붙은 클래스를 찾아서 문서화를 해준다.

2.9.2  

- 필요한 이유
  - Restapi의 스펙을 문서화 하는것은 중요
  - API를 변경할 때마다 Referernce문서를 계속 업데이트 해야하는 번거로움을 줄여준다.

- 설정 방법
  - `@Configuration`: 어노테이션 기반의 환경 구성을 돕는 어노테이션, IoC Containter에게 해당 클래스를 Bean 구성 Class 임을 알려줌
  - `@Bean`: 개발자가 직접 제어가 불가능한 외부 라이브러리 등을 Bean으로 만들 경우 사용

Bean?  

- `@PutMapping`

```java
@PutMapping("/member")
public String hello(@RequestBody MemberDTO memberDTO)
{
 return "hello " + memberDTO.toString();
}

@PutMapping("/member/")
public String hello(@RequestBody MemberDTO memberDTO)
{
 return "hello " + name + " " + memberDTO.toString();
}

@PutMapping("/member/")
public MemberDTO hello(@RequestBody MemberDTO memberDTO)
{
 return memberDTO;
}

@PutMapping("/member/")
public ResponseEntity<MemberDTO> hello(@RequestBody MemberDTO memberDTO)
{
 return ResponseEntity.status(HttpStatus.ACCEPTED).body(memberDTO);
}
```

- ResponseEntity: 스프링에서 제공하는 클래스 중 HttpEntity라는 클래스를 상속받아 사용하는 클래스
- 사용자의 HttpRequest에 대한 응답 데이터를 포함
- 포함하는 클래스
  - HttpStatus
  - HttpHeaders
  - HttpBody
- 400,200 이런 응답을 좀 더 자세하게 할 수 있다.

- Lombok
- 반복되는 메소드를 어노테이션을 사용하여 자동으로 생성해주는 라이브러리
- 일반적으로 VO, DTO, Model, Entity 등의 데이터 클래스에서 주로 사용됨
- 대표적으로 많이 사용되는 Annotation
  - `@Getter`: Getter 메소드 생성
  - `@Setter`: Setter 메소드 생성
  - `@ToString`: toString 메소드 생성
    - exclude: 제외할 필드
  - `@NoArgsConstructor`: 파라미터가 없는 생성자 생성
  - `@AllArgsConstructor`: 모든 필드를 파라미터로 받는 생성자 생성
  - `@RequiredArgsConstructor`: `final`이나 `@NonNull`이 붙은 필드를 파라미터로 받는 생성자 생성
  - `@EqualsAndHashCode`: equals, hashCode 메소드 생성
    - `equals`: 객체의 내용이 같은지 동등성(equality)를 비교
    - `hashCode`: 두 객체가 같은 객체인지 동일성(identity)를 비교
    - callSuper: 부모 클래스의 필드까지 감안하여 equals, hashCode 메소드를 생성
  - `@Data`: `@Getter`, `@Setter`, `@ToString`, `@EqualsAndHashCode`, `@RequiredArgsConstructor`를 한번에 사용할 수 있다.

- 스프링부트 서비스 구조
  - Client -(DTO)-> Controller -(DTO)-> Service/ServiceImpl -(Entity)-> DAO(Repository)/DAOImpl -(Entity)-> DB
  - Controller부터 Entity 쓰는 경우도 있고, 전부 DTO 쓰는 경우도 있고, 캐바캐

- Entity(Domain)
  - DB에 쓰일 칼럼과 여러 엔티티간의 연관 관계를 정의
  - DB의 테이블을 하나의 엔티티로 생각해도 무방함
  - 실제 데이터베이스의 테이블과 1:1로 매핑됨
  - 이 클래스의 필드는 각 테이블 내부의 칼럼(Column)을 의미

- Repositoty
  - Entity에 의해 생성된 DB에 접근하는 메소드를 사용하기 위한 인터페이스
  - Service와 DB를 연결하는 고리의 역할을 수행
  - DB에 적용하고자하는 CRUD를 정의하는 영역

- DAO (Data Access Object)
  - DB에 접근하는 객체를 의미 (Persistence Layer)
  - Service가 DB에 연결할 수 있게 해주는 역할
  - DB를 사용하여 데이터를 조회하거나 조작하는 기능을 담당
  
- DTO (Data Transfer Object)
  - DTO는 VO(Value Object)와 같은 개념
  - 계층간 데이터 교환을 위한 객체를 의미
  - VO의 경우 Read Only의 개념을 가지고 있음

<https://youtu.be/7t6tQ4KV37g?si=Kvs8iSEVvYqODZp1&t=21054>
