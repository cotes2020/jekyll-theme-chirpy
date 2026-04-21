---
title: "Logback"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: [Spring-Boot]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-04-11. 17:38
# last_modified_at: 2024-04-11. 17:38
---

{% include embed/youtube.html id='fkwb8coxBJM' %}

## Logback

---

- Log4J를 기반으로 개발된 로깅 (Logging) 라이브러리
- Log4J에 비해 약 10배 정도 빠른 퍼포먼스, 메모리 효율성 증대
- 출시 순서: Log4J -> Logback -> Log4J2

### 특징

- 로그에 특정 레벨을 설정할 수 있음
  - `Trace`: 디버깅보다 더 디테일한 메시지
  - `Debug`: 디버깅을 위한 정보성 메시지
  - `Info`: 상태변경과 같은 정보성 메시지
  - `Warn`: 시스템 에러의 원인이 될 수 있는 경고 레벨, 처리 가능한 사항
  - `Error`: 로직 수행 중에 오류가 발생한 경우, 시스템적으로 심각한 문제가 발생하여 작동이 불가한 경우
- 실운영과 테스트 상황에서 각각 다른 출력 레벨을 설정하여 로그를 확인할 수 있음
- 출력 방식에 대해 설정할 수 있음
- 설정 파일을 일정 시간마다 스캔하여 어플리케이션 중단 없이 설정 변경 가능
- 별도의 프로그램 없이 자체적으로 로그 압축을 지원
- 로그 보관 기간 설정 가능

### 설정

- 일반적으로 Classpath에 logback.xml 파일을 참조
  - Java Legacy, Spring의 경우에는 logback.xml 파일을 참조
  - Spring Boot의 경우에는 logback-spring.xml 파일을 참조
  - 바꿀 수 있음

### 구조

#### Appender

Log의 형태 및 어디에 출력할지 설정하기 위한 영역

- `ConsoleAppender`: 콘솔에 출력
- `FileAppender`: 파일에 출력
- `RollingFileAppender`: 파일에 출력하되, 용량이 넘어가면 새로운 파일로 생성
- `DailyRollingFileAppender`: 일별로 파일 생성
- `SMTPAppender`: 이메일로 전송
- `DBAppender`: DB에 저장

#### Encoder

Appender 내에 포함되는 항목, Pattern을 사용하여 원하는 형식으로 로그를 표현할 수 있음  

#### Pattern

로그의 출력 형식을 지정하는 영역  

- `%d{yyyy-MM-dd HH:mm:ss.SSS}`: 날짜 및 시간
- `%Logger{length}`: Logger 이름, length를 설정하면 패키지명 줄여서 출력
- `%-5level`: 로그 레벨, -5는 출력의 고정폭 값
- `%thread`: 스레드 이름
- `%msg`: 로그 메시지 (== `%message`)
- `%n`: 줄바꿈
- ...

#### Root

설정한 Appender를 참조하여 로그의 레벨을 설정할 수 있음  
Root는 전역 설정이며, 지역 설정을 하기 위해서는 Logger를 사용  

#### Filter

로그를 출력할지 말지 결정하는 영역

- `LevelFilter`: 로그 레벨에 따라 출력 여부 결정
- `ThresholdFilter`: 특정 레벨 이상의 로그만 출력
- ...
