---
title: Spring Basic[스프링 컨테이너]
date: 2023-02-06 22:12:00 +0800
categories: [Spring-Basic, 컨테이너]
tags: [Spring]
---

# Spring Container

## Spring Container?
애플리케이션의 객체(Bean)를 생성, 관리하며 제공하는 환경을 제공하는 역할을 합니다.      
즉, 스프링 컨테이너는 애플리케이션 개발자가 작성한 코드에서 객체의 생성 및 관리를 대신 맡아주는 것입니다.        
또한 스프링의 핵심 기능 중 하나인 **의존성 주입(DI), 제어의 역전(IoC)**을 구현하고 있습니다.        

## Spring Container의 생성 코드
     ```bash
     ApplicationContext applicationContext = new AnnotationConfigApplicationContext(AppConfig.class);
     ```
 ApplicationContext를 스프링 컨테이너라고 부릅니다.<br/>

## Spring Container의 생성 과정
 ![Spring Container png](/assets/img/spring/springcontainer.png){: width="350" height="200" }<br/>      
 스프링 컨테이너를 생성하고, 등록을 하면, 스프링 빈 저장소에 하나씩 등록이 된다.      

# Spring Bean
## Spring Bean 등록
 ![Spring Bean Update png](/assets/img/spring/springbean.png){: width="350" height="200" }        
 Spring에서 빈(Bean)을 생성하면 해당 빈은 Spring 컨테이너에 등록되고 관리하게 됩니다.       

## Spring Bean Update Code
    ```java

     @Bean
     public BeanAService beanaservice() 
     {
        return new BeanAServiceImpl(beanaservicerepository);
     }

     @Bean
     public BeanARepository beanaservicerepository() 
     {
        return new BeanAMemoryRepository();
     }
    ```
    빈 이름에는 메서드 이름을 사용하고,      
    빈 이름에 직접 부여 할 수도 있다. ( 예시 : @Bean(name="Example"))       
    위와 같이 Bean들을 생성하게 되면, 위의 Bean들은 스프링 컨테이너에 등록되고, 관리를 받게 된다.       

## Spring bean DI
 ![Spring Bean DI png](/assets/img/spring/springbeanupdate.png){: width="350" height="200" }      
 Spring container에서 등록된 설정 정보를 참고하여,     
 Spring 컨테이너에서 등록된 설정 정보를 참고하여 **DI(의존성 주입)**이 이루어집니다.
 
## BeanFactory
 ![Spring BeanFactory png](/assets/img/spring/SpringBeanFactory.png){: width="350" height="200" }       
 Spring Container의 최상위 인터페이스 입니다.
 Spring Bean을 관리 및 조회하는 역할을 담당하고 있습니다.
 
## ApplicationContect 부가 기능 종류
 ApplicationContext에서는 여러가지의 부가기능들을 제공해 주는데, 몆가지를 말해보면, 아래의 기능들이 존재 합니다.     
 1. 메시지소스를 활용한 국제화 기능
 2. 환경변수
 3. 애플리케이션 이벤트
 4. 편리한 리소스 조회


# 요약
***스프링 컨테이너***는 애플리케이션에서 사용하는 객체들을 생성하고 관리하는 역할을 합니다.     
스프링 컨테이너는 ApplicationContext라는 인터페이스를 구현한 클래스로 만들 수 있습니다. ApplicationContext는 AppConfig라는 클래스를 인자로 받아서, 그 안에 있는 @Bean 어노테이션을 붙인 메서드들을 호출해서 스프링 빈으로 등록합니다.       

***스프링 빈***은 스프링 컨테이너에 의해 관리되는 객체입니다. 스프링 빈은 이름, 타입, 스코프 등의 속성을 가집니다.      
스프링 빈은 다른 스프링 빈에 의존할 수 있습니다. 이때 스프링 컨테이너가 자동으로 의존성을 주입해줍니다.     
이를 ***DI(Dependency Injection)***이라고 합니다.        

스프링 컨테이너의 최상위 인터페이스는 ***BeanFactory***입니다.        
BeanFactory는 스프링 빈을 생성하고, 조회하고, 관리하는 기본적인 기능을 제공합니다.      
ApplicationContext는 BeanFactory의 하위 인터페이스입니다. ***ApplicationContext***는 ***BeanFactory***의 기능에 더하여, 국제화, 환경변수, 애플리케이션 이벤트, 리소스 조회 등의 부가 기능을 제공합니다.     