---
title: "세션 & 쿠키"
# description: ""
categories: [컴퓨터, 인터넷]
tags: [Web]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-06-05. 11:11
last_modified_at: 2023-06-05. 14:15
---

## Cookie, 쿠키

---

Client(Browser)에 저장되는, Key와 Value를 가지는 작은 Data File

- 특징
  - User Auth의 Expires/Max-Age 명시 가능
  - Expires/Max-Age가 설정되면, Browser가 종료되어도 Auth가 유지 됨
  - Client의 State Info를 Local에 저장했다가 참조
  - Client에 최대 300개, 개당 최대 4KB 저장 가능
    - Domain 당 최대 20개
  - Response Header, Set-Cookie 속성으로 생성 가능
  - User가 따로 요청하지 않아도, Browser가 Request시 Request Header를 넣어서 자동으로 Server에 전송

- 구성 요소
  - 이름: Cookie 구별 시 사용
  - 값
  - 유효시간
  - 도메인: Cookie를 전송할 도메인
  - 경로: Cookie를 전송할 요청 경로

- 동작 방식
  1. Client가 Page 요청
  2. Server에서 Cookie 생성
  3. HTTP Header에 Cookie를 포함시켜 Response
  4. Browser가 종료되어도 Cookie Expires/Max-Age가 있다면 Client에서 보관
  5. 같은 Request를 할 경우, HTTP Header에 Cookie를 함께 보냄
  6. Server에서 Cookie를 읽어 이전 State Info를 변경 할 필요가 있을 때, Cookie를 업데이트 하여 변경된 Cookie를 HTTP 헤더에 포함시켜 응답

Info를 Client의 Local(Browser)에 저장  
→ 변질되거나 Request시 제 3자가 Sniffing 가능성
→ 보안에 취약

- 사용 예: 제 3자에게 공개되거나 조작되어도 크게 문제 없는 정보들
  - 로그인 시 ID/PW 저장 여부, 자동 로그인 여부
  - 팝업 "오늘 더 이상 이 창을 보지 않음" 여부
  - 쇼핑몰 장바구니
  - 자주 보는 웹툰 목록
  - 다크 모드 여부

## Session 세션

---

- 세션
  - (통신에서) 사용자와 컴퓨터 또는 두 대의 컴퓨터 간의 활성화된 접속
  - 프로그램 사물과 관련해서 한 응용프로그램의 가동을 시작해서 종료할 때까지 시간

Cookie를 기반하고 있지만, 사용자 정보 파일을 Server 측에서 관리  
(Cookie는 Browser에 저장)  

Client가 Request를 보내면,  
Server에서는 Client를 구분하기 위해 유일 ID인 Session ID(Session Key)를 부여하며,  
Browser가 Server에 접속했을 때부터, Browser를 종료할 때까지 Auth State를 Server Memory에 유지  

Like 영화 티켓,  
Server가 User Info와 Session ID를 만들고, 티켓 뜯어주듯 Session ID만 Client에게 전달  

Cookie처럼 기간 지정하여, 지워지지 않게 설정 가능  

정보를 Server에 두기에 보안이 Cookie보다 좋지만, 사용자가 많아질수록 Server 메모리를 많이 차지  
동접자 수가 많으면, Server에 과부하를 주게 되므로 성능 저하의 요인이 되기도  

- 동작 방식
  - Client가 Server 접속 시 Session ID를 발급 받음
  - Client는 Session ID를 Cookie로 저장하고 가지고 있음
  - Client는 Server에 요청할 때, 이 Cookie의 Session ID를 같이 Server에 전달해서 요청
  - Server는 Session ID로 Session에 있는 Client 정보를 가져와서 사용
  - Client 정보를 가지고 Server 요청을 처리하여 Client에게 응답

- 특징
  - 각 Client에게 고유 ID를 부여
  - Session ID로 Client를 구분해서 Client의 요구에 맞는 서비스를 제공
  - 보안 면에서 Cookie보다 우수
  - 사용자가 많아질수록 Server 메모리를 많이 차지하게 됨

- 사용 예
  - 로그인 같이 보안상 중요한 작업을 수행할 때 사용

## 💫

---

- Cookie와 Session의 차이
  - Session이 Cookie 기반이기에, 비슷한 역할과 동작원리를 가짐
  - User State Info 저장s 위치
    - Cookie: Browser(Local,) Server 자원 X
    - Session: Server 자원
  - 보안, Cookie < Session
    - Cookie: Client의 Local(Browser)에 저장, = 변질되거나 Request 시 스니핑 가능성, = 보안에 취약
    - Session: Cookie로 Session ID만 저장, Session ID만으로 Server에서 처리, = 보안성 비교적 좋음
  - 요청 속도, Cookie > Session
    - Cookie: Cookie 자체에 정보가 있기 때문에, Server에 Request 시 처리 속도가 빠름
    - Session: 정보가 Server에 있기 때문에, 처리가 요구되어 속도 비교적 느림
  - Life Cycle @
    - Cookie :
      - 만료시간이 있지만 파일로 저장되기 때문에, 브라우저를 종료해도 계속해서 정보가 남아 있을 수 있다.
      - 만료기간을 넉넉하게 잡아두면 Cookie 삭제를 할 때 까지 유지 가능
    - Session :
      - 만료시간을 정할 수 있지만 브라우저가 종료되면 만료시간에 상관없이 삭제됩니다.
      - 예를 들어, 크롬에서 다른 탭을 사용해도 Session을 공유됩니다. 다른 브라우저를 사용하게 되면 다른 Session을 사용할 수 있습니다.

- Session을 사용하면 좋은데 왜 Cookie를 사용할까?
  - Session이 Server 자원을 사용하기 때문에, Server 속도/Memory 감당량 이슈 발생 가능

- Cookie/Session VS Cache
  - Cache는 image, css, js File 등을 Browser나 Server Front-End에 저장해놓고 사용하는 것
  - 한 번 Cache에 저장되면 Browser를 참고, 때문에 Server에서 변경이 되어도 User(Local)는 변경되지 않게 보일 수 있음
  - → Cache를 지워주거나, Server에서 Client로 응답을 보낼 때 Header에 Cache Expires/Max-Age를 명시하는 방법 등

## 메모

---

### 키워드

- Token
- Cache

### 참고

- [Cookie와 세션 개념](https://interconnection.tistory.com/74)
- [데이터를 저장하는 5가지 개념](https://hongong.hanbit.co.kr/%EC%99%84%EB%B2%BD-%EC%A0%95%EB%A6%AC-%EC%BF%A0%ED%82%A4-%EC%84%B8%EC%85%98-%ED%86%A0%ED%81%B0-%EC%BA%90%EC%8B%9C-%EA%B7%B8%EB%A6%AC%EA%B3%A0-cdn/)
