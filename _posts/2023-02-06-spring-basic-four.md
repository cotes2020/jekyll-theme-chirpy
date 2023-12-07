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

스프링 컨테이너 생성 코드는 아래와 같습니다.
     ```bash
     ApplicationContext applicationContext = new AnnotationConfigApplicationContext(AppConfig.class);
     ```