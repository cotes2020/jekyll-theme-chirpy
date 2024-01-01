---
title: NetWork_Basic[HTTP-1]
date: 2023-03-07 22:34:00 +0800
categories: [CS, HTTP]
tags: [Network_Basic,HTTP]
---

## HTTP
Hypertext Transfer Protocol의 약자로, 웹 상에서 정보를 주고 받을수 있는 프로토콜 입니다.<br/>

## HTTP 역사
 - HTTP/0.9
    - 1991년 초기 버전으로 다순 요청과 응답을 위한 프로토콜입니다.
 - HTTP/1.0
    - 1996년 많은 기능이 추가된 버전입니다. 기능에는 헤더, 상태코드 요청 메서드등 기능이 도입되었습니다.
    - TCP 연결을 맺어야 하는데, 이로인해 RTT가 증가하는 단점이있습니다.
        - RTT : 패킷의 왕복 시간
 - HTTP/1.1
    - 가장 널리사용되는 버전으로, 지속적인 연결, 파이프라인, 가상호스팅 등 많은 기능이 도입되고, 성능이 향상된 버전입니다.
    -TCP를 한 번 초기화한 후, Keep-Alive 옵션을 통해 여러 개의 파일을 송수신할 수 있지만, 대기시간이 길어지는 단점이 존재합니다.
 - HTTP/2.0
    - 2015년에 발표된 버전으로, 멀티플렉싱, 헤더압축, 우선순위 지정등으로 속도와 효율성이 개선된 버전입니다.
        - 멀티플렉싱 : N개의 스트림을 사용하여 송수신하는 방법입니다. 특정 스트림이 손실되어도 나머지 스트림은 정상 동작합니다.
    - 클라이언트 요청없이 서버가 바로 리소스를 푸시할수 있습니다.
 - HTTP/3.0
    - QUIC프로토콜을 기반으로하는 기반으로 하는 2018년에 새로 등장한 버전입니다. 
    - TCP대신 UDP를 사용하여 3-웨이 핸드세이크 과정을 거치지 않아 연결이 빠르고, 이로인해 RTT가 감소하는 장점이 있고, 다중화 및 흐름제어를 개선한 버전입니다.

## HTTP의 특징
1. 클라이언트 서버 구조
 ![http-network-basic-client-server-png](/assets/img/spring/http-network-basic-client-server.png){: width="700" height="600" }<br/>
    - 클라이언트는 서버에 요청을 보내고, 응답을 대기합니다.
    - 서버는 요청에 대한 결과를 만들어 클라이언트에 응답합니다.
2. 무상태 프로토콜 (stateless )
 ![http-network-basic-clien-server-stateful-png](/assets/img/spring/http-network-basic-clien-server-stateful.png){: width="700" height="600" }<br/>
- 서버가 클라이언트 상태를 보존하지 않습니다.
    - 장점 : 서버 확장성이 높습니다. ( 스케일 아웃 )
    - 단점 : 클라이언트가 추가로 데이터를 전송해야합니다.
![http-network-basic-client-server-stateful-fail-png](/assets/img/spring/http-network-basic-client-server-stateful-fail.png){: width="700" height="600" }<br/>
- 서버에 장애가 발생하면 응답이 불가능할 수 있습니다.

3. 상태 유지
 ![http-network-basic-client-server-stateless-png](/assets/img/spring/http-network-basic-client-server-stateless.png){: width="700" height="600" }<br/>
- 항상 같은 서버가 유지되어야 합니다.
 ![http-network-basic-client-server-stateless-fail-png](/assets/img/spring/http-network-basic-client-server-stateless-fail.png){: width="700" height="600" }<br/>
- 중간에 서버에 장애가 발생하여도, 다른 서버를 이용하여 응답이 가능합니다.

## 결론
이처럼 HTTP는 웹상에서 정보를 주고받을수 있는 프로토콜입니다.<br/>
HTTP는 여러버전을 거쳐 성능이 향상되고, 기능이 추가되고 있으며, 현재도 진행중입니다.<br/>
HTTP는 클라이언트-서버 구조를 가지고 있습니다.<br/>
상태유무에 따라 무상태, 상태유지 두 가지 모델로 나눌 수 있습니다.<br/>
무상태 프로토콜은 상태를 보존하지 않는다는 특징을 가지고 있어, 웹의 간단한 조회, 검색 그리고 RestFul API를 만들떄 사용하기에 좋습니다.<br/>
하지만, 상태유지는 상태를 보존하고 있다는 특징이 있습니다. 그러므로 로그인 같은 세션 관리가 필요한경우나 트랜잭션 처리에 유용하게 쓰일 수 있습니다.<br/>
