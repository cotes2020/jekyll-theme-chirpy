---
title: "Spring Boot RestTemplate"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: [Spring-Boot]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-04-11. 20:07
# last_modified_at: 2024-04-11. 20:07
---

{% include embed/youtube.html id='PfJQnbyAAhY' %}

## RestTemplate

---

- 스프링에서 제공하는 HTTP 통신 기능을 쉽게 사용할 수 있게 설계되어 있는 템플릿  
- HTTP 서버와의 통신을 단순화하고 RESTful 원칙을 지킴  
- 동기 방식으로 처리되며, 비동기 방식으로는 AsyncRestTemplate이 있음  
- RestTemplate 클래스는 REST 서비스를 호출하도록 설계되어 HTTP 프로토콜의 메소드에 맞게 여러 메소드를 제공  

### RestTemplate 메소드

- `getForObject`: GET, GET 형식으로 요청하여 객체로 결과를 반환 받음
- `getForEntity`: GET, GET 형식으로 요청하여 ResponseEntity로 결과를 반환 받음
- `postForObject`: POST, POST 형식으로 요청하여 객체로 결과를 반환 받음
- `postForEntity`: POST, POST 형식으로 요청하여 ResponseEntity로 결과를 반환 받음
- `delete`: DELETE, DELETE 형식으로 요청
- `put`: PUT, PUT 형식으로 요청
- `patchForObject`: PATCH, PATCH 형식으로 요청
- `exchange`: any, HTTP 헤더를 생성하여 추가할 수 있고 어떤 형식에서도 사용할 수 있음