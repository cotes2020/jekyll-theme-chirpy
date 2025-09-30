---
title: "Ping Of Death"
# description: ""
categories: [컴퓨터, 인터넷]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-05-10. 09:18
last_modified_at: 2023-05-10. 09:18
---

[참고 - Ping Of Death](https://run-it.tistory.com/52)  
[참고 - Ping Of Death](https://ko.wikipedia.org/wiki/%EC%A3%BD%EC%9D%8C%EC%9D%98_%ED%95%91)  
[참고 - Ping Of Death](https://www.cloudflare.com/ko-kr/learning/ddos/ping-of-death-ddos-attack/)  

## Ping Of Death

---

규정 크기 이상의 ICMP 패킷을 보내 Victim의 시스템을 마비시키는 공격  

Ping을 실행하면, ICMP Echo Request 패킷을 원격 IP 주소에 송신하고 ICMP 응답을 기다림  
Ping을 이용하여, ICMP 패킷의 Header를 정상 크기보다 크게 만들어 공격 대상에게 보내는 공격  
I.E. Google 8.8.8.8 Server에 크기를 65500 BYTE 늘린 Ping 전송  

크기가 큰 패킷은 네트워크 상에서 한 번에 보낼 수 없기에, 분할되어 목적지로 전송  
I.E. (65500 / Router의 MTU) 만큼의 패킷으로 분할 Fragment  

분할되어진 수많은 패킷을 받은 공격 대상 Victim은, 나눠진 Ping을 조립하는 과정에서 일반적인 Ping보다 부하가 발행 (버퍼 크기, IP 스택을 넘치게 하는 것)  

Ping 간격을 줄이고, 패킷의 크기를 늘리고, DDoS를 활용하여 수많은 컴퓨터를 이용해 Ping을 보낸다면?  

## Ping Of Death 실습

---

[참고 - Ping Of Death 실습](https://blog.naver.com/skyclad1975/220366329285)  

타 호스트로의 공격은 Jail 이슈  
→ VM을 이용해 자신의 IP로 공격  

필자는 Virtual Box, Kali, WireShark를 이용  
→ Kali Default ID/PW = kali  

Operation not permitted  
→ sudo -s  
→ sudo [Commend]  

Unable to locate package [PackageName]  
→ sudo apt update  

APT, Advance Packing Tools  
apt update: 설치 가능한 패키지 '리스트' 최신화  
apt list: 현재 리스트를 불러오기  
apt list --upgradable: 현재 리스트 중 업그레이드 필요한 패키지 불러오기  
apt upgrade: 설치 가능한 패키지 최신화  

VM에서 192.168.0.1 HostComputer  
원래 192.168.0.1 은 Router, VM 은 HostComputer 내부에 있으므로  

Router에 Ping을 보내듯, VM에서 HostComputer로 Ping Of Death 공격을  

교수님께서 알려주신 명령어는  
hping3 --icmp --rand-source 192.168.0.18 -d 6000 -S -flood  
hping3 --rand-source 192.168.0.18 -p 21 -S -flood  

--rand-source: 공격자 IP 주소를 랜덤하게 생성  
-d 1: 전송 패킷 크기를 1으로 설정  
-S: 지속적으로 전송  
-flood: 빠른 속도로 전송  

## Ping Of Death 대응 방안

---

1. 일반적인 ICMP 패킷은 분할하지 않으므로, 패킷 중 분할이 일어난 패킷을 공격으로 의심/탐지
2. ICMP 패킷 자체를, 서버 앞단 혹은 서버에 ICMP 패킷을 블로킹 해주는 설정
3. 같은 IP에서 일정 시간 내에 ICMP 패킷이 여러 개 전송될 경우 차단
4. 일정 크기 이상의 Ping 패킷이 전송되면 차단

## 학교 과제 양식

---

1. 어떠한 공격이 진행되었는가?
   - 공격 대상에게 결과적으로 수많은 패킷을 보내 시스템에 부하를 일으키는 공격
2. 사용된 공격 명령어는 어떤 것인가?
   - hping3 --icmp --rand-source 192.168.0.18 -d 6000 -S -flood
   - hping3 --icmp 192.168.0.18 -d 2500
   - 등 ...
3. 공격의 판단 근거/탐지 방법은 무엇인가?
   - Data 영역이 전부 58 (X)로 채워진 패킷 다수가 수신됨
4. 공격자의 IP/MAC 주소를 판단할 수 있는가? With 근거
   - --rand-source를 통해 IP를 무작위로 가리고 보내기 때문에, 공격자의 IP/MAC 주소는 판단하기 어려울 것이다
5. 공격에 대한 대응방법은 무엇인가?
   - [Ping Of Death 대응 방안](#-ping-of-death-대응-방안)
