---
title: NetWork_Basic[Network-2]
date: 2023-03-02 22:24:00 +0800
categories: [CS, InternetNetwork]
tags: [Network_Basic,Network]
---

## TCP 3 way handshake
TCP의 연결 과정인데, 신뢰성을 확보할 때 하는 작입니다.<br/>

### 3 way handshake
TCP의 연결 과정은 3단계의 과정이 필요합니다.
![http-network-3-way-handshake-png](/assets/img/spring/http-network-3-way-handshake.png){: width="700" height="600" }<br/>
위의 그림처럼 3단계의 과정이 필요합니다.<br/>
1. SYN 단계 - 클라이언트가 서버쪽으로 ISN을 담아 SYN을 보내줍니다.
    - SYN : 연결요청 플래그입니다.
    - ISN : 초기 네트워크 연결시, 할당된 32비트의 고유 시퀸스 번호 입니다.
2. SYN + ACK 단계 - 서버는 SYN을 수신하고 서버의 ISN을 보내며 승인번호로 클라이언트의 ISN + 1 을 전송
3. ACK 단계 - 서버의 ISN + 1 한 값인 승인번호를 담아 ACK를 서버로 전송

이와 반대로 TCP의 연결 해제 과정 또한 존재하는데, 이 과정은 4 way handshake라고 합니다<br/>

### 4 way handshake
TCP 연결 해제 과정은 4단계의 과정을 거칩니다.
![http-network-3-way-handshake-png](/assets/img/spring/http-network-4-way-handshake.png){: width="700" 
height="600" }<br/>
1. 클라이언트가 연결을 닫으려고 할 때, FIN으로 설정된 세그먼트를 보냅니다.<br/>
그리고 클라이언트는 FIN_WAIT_1 상태로 들어가게 되고, 서버의 응답을 기다립니다.
2. 서버는 ACK라는 승인 세그먼트를 보냅니다. 그리고 CLOSE_WAIT 상태에 들어갑니다. 그리고 클라이언트가 세그먼트를 받으면, FIN_WAIT_2 상태에 들어갑니다.
3. 서버는 ACK를 보내고 일정시간 이후에 클라이언트에 FIN이라는 세그먼트를 보냅니다.
4. 클라이언트는 TIME_WAIT상태가 되고 다시 서버로 ACK를 보내서 서버는 CLOSED 상태가 됩니다. 이후 클라이언트는 어느 정도의 시간을 대기한 후 연결이 닫히고 클라이언트와 서버의 모든 자원의 연결이 해제 됩니다.

## 결론
TCP의 연결과정은 **3-way-handshake**라고 불리고, 연결 해제 과정은 **4-way-handshake**라고 불리고 있습니다.<br/>
연결과정은 
1. SYN 단계
2. SYN + ACK 단계
3. ACK 단계
의 3단계의 과정을 거칩니다.<br/>
연결 해제 과정은 아래의 4단계 과정을 거치게 됩니다.<br>
1. FIN
2. ACK
3. FIN
4. ACK
연결 해제 과정에서 주의깊게 봐야할 부분이 바로 TIME_WAIT입니다.<br/> 
서버를 닫지않고, TIME_WAIT이 필요한 이유는 지연 패킷이 발생할 경우를 대비하기 위함입니다.<br/>
또한 두 장치가 연결이 제대로 닫혔는지를 확인하기 위한 이유도 있습니다.<br/>