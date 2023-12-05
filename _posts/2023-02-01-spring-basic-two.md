---
title: Spring 기초 [프로젝트 설정]
date: 2023-02-01 19:43:00 +0800
categories: [Spring-basic]
tags: [Spring]
---

오늘은 Spring 프로젝트를 생성하고 설정하는 방법에 대해 설명하겠습니다. <br/>
Spring 프로젝트를 생성하는 방법은 총 2가지가 있습니다.<br/>
1. Tools(InteliJ, VsCode 등)에서 직접 프로젝트를 생성<br/>
2. [Spring](https://start.spring.io/) 사이트에서 간편하게 프로젝트를 생성<br/>

이 포스트에서는 사이트에서 프로젝트를 생성하는 기준으로 설명하겠습니다. <br/>

## 프로젝트 생성
![Spring Start Site](/assets/img/spring/projectsetting.png){: width="700" height="400" }<br>
> 1. 프로젝트에서 Groovy 또는 Maven을 선택합니다.
> 2. 언어 섹션에서 Java를 선택합니다.
> 3. Spring Boot의 최신 버전을 사용합니다. (다만, SNAPSHOT 버전은 제외)
>> 제외 이유: 안정화되지 않은 버전이기 때문입니다.
> 4. 프로젝트의 그룹을 결정하고 기록합니다.
> 5. 프로젝트의 Artifact 이름을 정하고 기록합니다.
> 6. 프로젝트에 대한 설명이 있으면 설명 섹션에 작성하고, 그렇지 않으면 건너뛸 수 있습니다.
> 7. 패키지 이름은 그룹과 Artifact를 결합하여 자동으로 작성되므로 건너뛸 수 있습니다.
> 8. Jar와 War 중에서 빌드 출력 형식을 선택합니다.
> 9. 사용할 Java 버전을 선택합니다.
> 10. 프로젝트를 생성할 때 필요한 라이브러리와 종속성을 선택하여 종속성을 추가할 수 있습니다.<br/>
>> 지금 선택하지 않아도 나중에 따로 추가할 수 있습니다.

## 다운로드 및 압축 해제
1. 위의 10가지 설정이 완료되면 아래의 GENERATE 버튼을 클릭하여 생성된 zip 파일을 다운로드합니다.<br/>
2. 다운로드한 zip 파일을 원하는 위치에 압축 해제합니다.<br>
3. 압축 해제한 프로젝트를 프로젝트 생성 도구를 사용하여 엽니다 (여기서 IntelliJ를 사용).<br>
4. 파일을 열고 나면 아래와 같은 형태가 나오게 됩니다.
![Spring Start Site](/assets/img/spring/springprojectsetting.png){: width="200" height="30" }<br>

## 프로젝트 설정
1. 파일 구조에서 build.gradle 파일을 우클릭하고 디스크에 로드시킵니다.<br/>
> 이부분에서 프로젝트 생성시 Dependencies 부분에서 추가를 못했다면 build.gradle 파일에서 따로 설정이 가능합니다.<br/>
> ![Spring Start Site](/assets/img/spring/dependencie.png){: width="700" height="400" }<br>
![Spring Start Site](/assets/img/spring/reload.png)<br/>
2. 리로드 버튼을 눌러 build.gradle에 있는 설정 내용을 프로젝트에 적용 시켜줍니다.

