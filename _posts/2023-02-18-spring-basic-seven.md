---
title: Spring Basic[컴포넌트 스캔]
date: 2023-02-18 18:52:00 +0800
categories: [Spring-Basic, 컴포넌트 스캔]
tags: [Spring]
---

# 컴포넌트 스캔
## 컴포넌트 스캔이란?
스프링 프레임워크에서 사용되는 기능 중 하나로,      
애플리케이션에서 사용할 **빈(Bean)**들을 찾아서 등록하는 프로세스를 말합니다.       
스프링이 관리하는 빈은 주로 `@Component` 및 그와 관련된 어노테이션을 사용하여 정의합니다.       

## 사용 방법
주로 컴포넌트 스캔은 이름 그대로 `@Component` 어노테이션이 붙은 클래스를 스캔해서 스프링 빈으로 등록합니다.        
그러므로 등록할 클래스에 @Component를 붙여 주면 됩니다.
@Congiguration이 컴포넌트의 스캔 대상이 될 수 있습니다.

## 등록 예시
```java
@Component
public class MemoryBeanRepository implements BeanRepository {
    ...
}
```

```java
@Component
public class BeanServiceImpl implements BeanService {
    private final BeanRepository beanrepository;
    @Autowired
    public BeanServiceImpl(BeanRepository beanrepository){
        ....
    }
    ...
}
```
- @Autowired는 의존관계를 자동으로 주입해준다.

## 컴포넌트 스캔, 자동 의존관계 주입 동작
1. @ComponentScan
[사진]
- @ComponentScan은 @Component가 붙은 모든 클래스를 스프링 빈으로 등록한다.      
- 스프링 빈의 기본 이름은 클래스명을 사용하면서, 맨 앞글자만 소문자를 사용한다.        


2. @Autowired 의존관계 자동 주입
[사진]
- 생성자에 @Autowired를 지정하면, 스프링 컨테이너가 자동으로 해당 스프링 빈을 찾아서 주입한다.      
- 기본 조회 전략은 타입이 같은 빈을 찾아서 주입한다.
[사진]
- 생성자에 파라미터가 많아도 다 찾아서 자동으로 주입 한다.

# 컴포넌트 스캔 기본 대상
## 대상
1. @Component
    - 컴포넌트 스캔에서 사용
2. @Controller
    - MVC Controller에서 사용
3. @Service
    - 비즈니스 로직에서 사용
4. @Repository
    - 데이터 접근 계층에서 사용
5. @Configuration
    - 스프링 설정 정보에서 사용

## 결론
이처럼 컴포넌트 스캔은 스프링에서 빈을 자동으로 찾아 등록하는 기능으로, 주로 @Component 어노테이션을 사용하여 클래스를 스프링 빈으로 등록합니다.        
의존성 주입은 @Autowired 어노테이션을 통해 자동으로 이루어집니다.      
컴포넌트 스캔은 @ComponentScan을 통해 특정 패키지 내에서 빈을 찾아 등록하며, @Controller, @Service, @Repository, @Configuration 등의 어노테이션도 대상으로 합니다.      
이를 통해 애플리케이션의 구성을 유연하게 관리하고, 의존성 주입을 자동화하여 코드의 가독성과 유지보수성을 향상시킬 수 있습니다.      