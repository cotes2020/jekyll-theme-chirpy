---
title: Spring Basic[스프링 컨테이너]
date: 2023-02-06 22:12:00 +0800
categories: [Spring-Basic, 컨테이너]
tags: [Spring]
---

# 스프링 컨테이너

## 스프링 컨테이너란 ?
애플리케이션의 객체(Bean)를 생성,관리하며 제공하는 환경을 제공하는 역할을 합니다.<br/>
즉 스프링 컨테이너는 애플리케이션 개발자가 작성한 코드에서 객체의 생성 및 관리를 대신 맡아주는 것입니다.<br/>
또한 스프링의 핵심 기능 중 하나인 **의존성 주입(DI), 제어의 역전(IoC)**을 구현하고 있습니다.<br/>


## 스프링 컨테이너의 생성 코드
     ```bash
     ApplicationContext applicationContext = new AnnotationConfigApplicationContext(AppConfig.class);
     ```
 ApplicationContext를 스프링 컨테이너라고 부릅니다.<br/>

## 스프링 컨테이너의 생성 과정
 ![Spring Container png](/assets/img/spring/springcontainer.png){: width="350" height="200" }<br/>      
 스프링 컨테이너를 생성하고, 등록을 하면, 스프링 빈 저장소에 하나씩 등록이 된다.      

# 스프링 빈 등록
 ![Spring Container png](/assets/img/spring/springbean.png){: width="350" height="200" }        
 Spring에서 빈(Bean)을 생성하면 해당 빈은 Spring 컨테이너에 등록되고 관리하게 된다.
 
 ## Spring Bean Update Code
    ```bash

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
 ![Spring Container png](/assets/img/spring/springbeanupdate.png){: width="350" height="200" }      
  Spring container에서 등록된 설정 정보를 참고하여,     
  **DI( Dependency injection )**이 되는 모습을 확인할 수 있다.
 