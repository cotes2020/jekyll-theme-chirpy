---
title: OSI 7 Layer Model
date: 2023-09-10 23:45:55 +0900
author: kkankkandev
categories: [Network]
tags: [osi 7 layer, Network, cloud, osi, ip, udp, tcp, osi, router, switch]     # TAG names should always be lowercase
comments: true
---

OSI 7 Layer Model은 컴퓨터 네트워크와 통신 시스템에서 사용되는 표준화된 계층 모델로, 네트워크 프로토콜과 통신을 이해하고 설명하기 위한 틀을 제공합니다.
     
# OSI 각 계층 별 역할과 주요 프로토콜, 기능, 장비

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/68aefb99-690e-4a33-ab42-f7e6dbf6053a)

<!-- |  계층  |  역할  |  주요 프로토콜  |  기능  |  장비  |
| ----- | ----- | ----- | ----- | ----- |
| L7 Application | 응용 프로그램과 통신 프로그램 사이에서 인터페이스 제공 | HTTP, FTP, SMTP, DNS, HTTP | 응용 프로세스와 직접 관계하여 일반적인 응용 서비스 수행 | |
| L6 Presentation | 데이터의 형식 변환, 인코딩, 암호화 및 해독 | ASCII, MPEG, SSL, JPEG | 코드 간의 번역, 인코딩, 암호화, 해당 데이터의 확장자 구분 | |
| L5 Sesstion | 세션의 시작 및 종료 제어 | TCP session setup | 데이터가 통신하기 위한 논리적인 연결 |
| L4 Transport | 종단 프로그램 사이의 데이터 전달 (컴퓨터사이의 데이터 전송, 수신) <br> TCP(연결형) UDP(비연결형) | TCP, UDP | 시퀀스 넘버 기반의 오류 제어 <br> | 포트 |
| L3 Network | 종단 장비 사이의 데이터 전달 (IP 주소 기반) <br> [Packet 단위] | IP, ICMP, ARP | 라우팅, Packet의 경로 설정, 흐름제어, 세그맨테이션, 오류제어, 인터네트워킹(주소부여, 경로설정) | Router, Switch(L3) |
| L2 DataLink | 직접 연결된 노드 간 데이터 전송 <br> (MAC 주소 기반) <br> (Frame 단위) | Ethernet, Wi-Fi, PPP, ATM, HDLC | 맥 주소를 통해 프레임단위로 통신 <br> (에러검출, 재전송, 흐름제어) | 브리지, Switch, Ethernet |
| L1 Physical | 데이터를 전기 신호, 광 신호 또는 무선 신호로 변환하여 전송 <br> [Bit 단위] | 100Base-TX, V.35| 데이터를 전기적인 신호로 변환 후 주고받는 기능 | 케이블, 리피터, 허브 | -->

- 상위 계층(Application)에서 하위 계층(Physical) 계층으로 내려올 때 Header가 계속해서 추가됩니다.  
**(Packaging, Encapsulation)**
- 반대로 하위 계층(Physical)에서 상위 계층(Application)으로 데이터가 전달될 때는 각 계층에 해당하는 헤더가 하나씩 사용됩니다.  
**(Depackaging, Decapsulation)**

Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
