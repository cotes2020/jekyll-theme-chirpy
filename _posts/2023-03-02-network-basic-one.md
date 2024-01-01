---
title: NetWork_Basic[Network-1]
date: 2023-03-02 20:24:00 +0800
categories: [CS, InternetNetwork]
tags: [Network_Basic,Network]
---
## 컴퓨터와 컴퓨터의 통신은 어떻게?
컴퓨터 간 통신은 주로 인터넷을 통해 이루어집니다. 인터넷은 많은 컴퓨터와 네트워크로 구성된 거대한 국제적인 네트워크로, 컴퓨터들은 이를 통해 서로 통신합니다.

## ARPANET
인터넷의 시초인 ARPANET은 미국 국방부의 ARPA가 주도한 프로젝트로, 1969년에 최초로 가동된 패킷 스위칭 네트워크입니다. ARPANET은 TCP/IP 프로토콜을 사용하여 컴퓨터 간의 효율적인 통신을 실현했습니다.


## 인터넷(Internet)이란?
 ![http-network-mang-png](/assets/img/spring/http-network-mang.png){: width="700" height="600" }<br/>
 인터넷은 많은 컴퓨터와 네트워크로 구성된 거대한 국제적인 네트워크로,  컴퓨터들은 이를 통해 정보를 주고받고 서비스를 이용합니다. TCP/IP  프로토콜을 기반으로 하며, 웹 브라우징, 이메일, 파일 전송 등 다양한  서비스를 제공합니다.
 ### NetWork?
 네트워크는 컴퓨터 등의 장치들이 통신 기술을 이용하여 구축하는 연결망을 지칭하며, 노드와 링크가 서로 연결되어 있어 리소스를 공유하는 집합체를 의미합니다.


## IP
 ![http-network-ip-png](/assets/img/spring/http-network-ip.png){: width="700" height="600" }<br/>
컴퓨터 간의 효율적인 통신을 위해 각 컴퓨터는 네트워크에서 고유한 식별자로 사용되는 IP 주소를 할당받습니다. IP 주소는 데이터의 출발지와 목적지를 지정하여 특정 컴퓨터로 데이터를 안전하게 전송할 수 있도록 합니다.


### 패킷이란?
 ![http-network-packet-png](/assets/img/spring/http-network-packet.png){: width="700" height="600" }<br/>
작게 나눠진 작은 데이터 뭉치를 말합니다. 각 패킷에는 목적지와 출발지 IP 주소, 전송 제어 정보, 그리고 실제 데이터가 포함되어 있습니다. 이 작은 패킷들은 네트워크를 통해 전송되며, 수신 컴퓨터에서는 이를 다시 조합하여 전체 데이터를 복원합니다. 패킷 기반의 통신은 효율성과 안정성을 높여주며, 다양한 종류의 데이터를 효과적으로 전송할 수 있도록 합니다.


## 전송 과정
### 클라이언트 -> 서버
 ![http-network-sendpacket-png](/assets/img/spring/http-network-sendpacket.png){: width="700" height="600" }<br/>
1. 요청을 패킷으로 만듭니다.
2. 클라이언트는 패킷에 담긴 목적지 IP를 참고해, 목적지를 설정합니다.
3. 서버에게 전송하려는 데이터를 패킷에 채웁니다. 패킷을 목적지 IP에 전송합니다.
4. 패킷은 네트워크를 통해 목적지 IP에 맞는 서버로 도착하게 됩니다.

### 서버 -> 클라이언트
 ![http-network-backpacket-png](/assets/img/spring/http-network-backpacket.png){: width="700" height="600" }<br/>
1. 클라이언트로부터의 요청 패킷을 수신합니다.
2. 서버는 클라이언트의 요청에 따라 필요한 처리를 수행합니다.
3. 처리결과를 포함한 응답 패킷을 생성합니다.
4. 목적지 IP를 설정합니다.
5. 데이터를 패킷에 채우고 전송합니다. 

## IP Protocol 한계
- 비 연결성
     ![http-network-send-fail-png](/assets/img/spring/http-network-send-fail.png){: width="700" height="600" }<br/>
    - 패킷 받을 대상이 없거나 서비스 불능 상태여도 패킷을 전송합니다.
- 비 신뢰성
    - 패킷 소실문제
    ![http-network-packet-fail-png](/assets/img/spring/http-network-packet-fail.png){: width="700" height="600" }<br/>
    - 패킷 순서문제
    ![http-network-packet-list-png](/assets/img/spring/http-network-packet-list.png){: width="700" height="600" }<br/>
- 프로그램 구분
    - 같은 IP를 사용하는 서버에서 통신하는 애플리케이션 둘 이상 문제

## TCP
### TCP/IP 4계층
![http-network-tcp-ip-4-png](/assets/img/spring/http-network-tcp-ip-4.png){: width="700" height="600" }<br/>
애플리케이션 계층에서 데이터를 생성하고, SOCKET 라이브러리를 통해 전달합니다. 이후 전송계층에서 TCP 정보를 생성하며, 이 정보에는 메시지 데이터가 포함됩니다. 다음으로 인터넷 계층에서 IP 패킷을 생성하고, 이 IP 패킷은 네트워크 인터페이스 계층의 LNA 카드를 통해 인터넷으로 전송되어 서버에 전달됩니다.    
### TCP/IP Packet
TCP/IP Packet은 IP Packet과는 다르게 TCP 세그먼트가 포함됩니다.   

## TCP / UDP      
### UDP
UDP는 순서를 보장하지 않고, 데이터그램 패킷 교환 방식을 사용합니다.
### TCP
TCP는 연결지향 ( TCP 3 way handshake ) 방식을 사용, 그리고 가상회선 패킷 교환 방식을 사용합니다.
### TCP, UDP 교환 방식
 - 데이터 그램 교환 방식 ( **UDP** )
    - 순서를 보장하지 않고, 수신여부 확인 X, 단지 데이터만 주는 방식
    ![http-network-tcp-data-gram-png](/assets/img/spring/http-network-udp-data-gram.png){: width="700" height="600" }<br/>
    - 장점
        - 방식이 단순하고, 빠르다는게 장점입니다.

 - 가상 회선 패킷 교환 방식 ( **TCP** )
    - 각 패킷에 가상회선 식별자가 포함되며, 패킷들이 전송된 순서대로 도착하는 방식
    ![http-network-tcp-virtual-line-png](/assets/img/spring/http-network-tcp-virtual-line.png){: width="700" height="600" }<br/>
    - 장점
        - 데이터 순서를 보장하고, 데이터 전달을 보증이 가능합니다.

양이 많아, 다음 포스터에서 이어서 설명하도록 하겠습니다.
