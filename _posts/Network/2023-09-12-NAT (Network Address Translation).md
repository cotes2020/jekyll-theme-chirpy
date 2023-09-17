---
title: NAT (Network Address Translation) [개념, 설정 법]
date: 2023-09-12 19:05:24 +0900
author: kkankkandev
categories: [Network]
tags: [network, nat, pat, dnat, router, ip, cisco, packet tracer, ospf]     # TAG names should always be lowercase
comments: true
image: 
   path: https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/c769758b-c9a9-40aa-ae94-644be8b50c09
---

NAT란 IP Packet에 있는 출발지 및 목적지의 IP 주소와 TCP/UDP 포트 숫자 등을 바꿔가면서 네트워크 트래픽을 주고 받을 수 있게하는 기술입니다.

## 1. NAT의 사용 목적

### 1.1 IP 주소 절약
NAT 기술을 사용하면, 하나의 공인 IP 주소를 사용하여 여러 대의 호스트가 인터넷에 접속할 수 있습니다. 가정에서 공유기를 사용하여 하나의 인터넷 회선을 사용해 여러 PC와 모바일 기기를 연결하여 사용 할 때, 우리가 인터넷 사용이 가능한 이유는 인터넷 공유기에 NAT 기능이 탑재되어 있기 때문입니다.

### 1.2 보안
NAT의 특성을 사용하면 IP를 숨길 수 있습니다. 라우터 외부로 트래픽이 나갈 때는 사설 IP가 공인 IP 주소로 바뀌므로 공격자가 라우터 안 쪽에 있는 사설 IP를 알 수 없습니다.
따라서 공격자는 공격 대상의 최종 목적지를 알아 내기가 어렵습니다. 이러한 이유로 인해 NAT기술을 사용하여 내부 네트워크 및 호스트들을 보호할 수 있습니다. 
(보안 측면 ↑)

## 2. NAT의 동작 원리
집에서 사용하는 인터넷 공유기를 통해 외부에 있는 웹 서버에 접근할 경우, 해당 요청 패킷은 반드시 해당 공유기(Gateway)를 거치게 되어 있습니다. 이 때, 출발자의 Private IP 주소가 그대로 외부 인터넷에 송신될 경우, 수신측(Webserver)는 알 수 없는 사설망의 IP 주소이므로 최종적으로 Packet을 어디로 보내줘야 할 지 알 수 없습니다.
### 2.1 동작 순서
1. 패킷 헤더에 출발지와 목적지 주소를 기록 (출발지주소 => Private IP)
```
- PC(호스트) 에서 출발
# 출발지 IP 주소 : 10.0.0.1
# 목적지 IP 주소 : 200.100.10.1
```
2. 기본 게이트웨이(공유기 등) 에서는 외부로 나가는 패킷을 인식 (PAT)
   => 출발지 IP 주소를 게이트웨이 자신의 공인 IP주소로 변경. (별도의 NAT Table을 보관)
```
- Default Gateway에서 다시 출발
# 출발지 IP 주소 : 10.0.0.1 -> 150.150.0.1 (재기록하여 변경)
# 목적지 IP 주소 : 200.100.10.1S
```



      | 프로토콜 | 사설 IP 주소 | 출발지 IP 주소 | 목적지 IP 주소 |
      |:--------:|:------------:|:--------------:|:--------------:|
      |   TCP    | 10.0.0.1     | 150.150.0.1    | 200.100.10.1   |



1. 웹 서버에서 수신한 데이터를 처리한 후, 응답하여 보내는 패킷에 출발지와 목적지의 IP 주소를 포함시켜 출발지로 재전송  
   (목적지 IP => Host의 Default Gateway 공인 IP 주소)
```
- 웹 서버에서 출발
# 출발지 IP 주소 : 200.100.10.1
# 목적지 IP 주소 : 150.150.0.1
```
1. 호스트의 기본 게이트웨이에서 웹 서버가 보낸 패킷을 받으면, 기록해두었던 NAT 테이블을 참조하여 최종 목적지인 호스트의 사설 IP 주소로 변경하여 해당 호스트로 패킷을 전달
```
- 기본 게이트웨이에서 다시 출발
# 출발지 IP 주소 : 200.100.10.1
# 목적지 IP 주소 : 150.150.0.1  -> 10.0.0.1 (재기록하여 변경 Public IP => Private IP)
```

>  * 사설 네트워크에 한 대의 호스트가아닌 여러 대의 호스트가 같은 목적지와 통신하고자 할 때, 되돌아오는 패킷의 최정 목적지가 어디가 되어야 하는지 혼선이 생길 수 있습니다. 
>    
>    이러한 문제를 해결하기 위해 별도의 추가 포트를 설정하여 패킷을 구분하는 PAT방식 또는 NAPT 방식을 사용합니다.
>    
>  * NAT이 작동하는 과정은 네트워크의 성능에 영향을 미칩니다.   
>    =>  패킷에 변화가 생기기 때문 (IP, TCP/UDP의 체크섬이 다시 계산되어 재기록해야 하기 떄문)

## 3. 주소 할당 방식에 따른 NAT 종류 구분

### 3.1 StaticNAT (1:1 NAT)  
공인 IP 주소와 사설 IP 주소가 1:1로 매칭되는 방식
=> 주로 사설 IP 주소를 사용하는 서버가 여러 가지 역할을 할 때, 포트포워딩을 목적으로 사용

* 포트포워딩(Port forwarding) =>특성 서비스에 임의의 포트를 지정하여 해당 포트를 통해 특정 서비스의 경로를 지정해주는 기능

### 3.2 Dynamic NAT (N:N NAT)
여러 개의 공인 IP 주소 대비 사설 IP 갯수가 더 많을 경우, 즉 내부 네트워크에 호스트의 숫자가 많을 경우 사용

### 3.3 PAT(Port Address Translation, NAPT : Network Address Port Translation) (1:N)
- 공인 IP 주소 1개에 사설 IP 주소 여러개가 매칭이 되는 방식.
- **외부 인터넷에서 들어오는 패킷을 내부 네트워크 내에 있는 목적지에 올바로 전달 해주기 위해 사용**
- 사설 네트워크 내 각 호스트에 임의의 포트번호를 지정하여 사설 IP와 해당 포트번호를 공인 IP 주소와 해당 포트번호로 매칭/치환하는 방식

## 4. 패킷 방향에 따른 NAT 종류 구분

### 4.1 SNAT (Source Network Address Translation)
- 내부 IP(Source)를 게이트웨이의 공인 IP 주소로 바꾸어주는 방식
  ex) 인터넷 공유기


### 4.2 DNAT (Destination Network Address Translation)
- 외부에서 내부로 들어오는 패킷에 있는 목적지 (Destination)IP 주소를 변경하여 최종적으로 내부에 잇는 호스트에 패킷이 도달할 수 있도록 하는 것.
  ex) 방화벽, 로드 밸런서(Load Balancer)
  
## 5. PAT 방식, DNAT 방식 실습

### 5.1 PAT 방식

1. 내부의 변환될 대상을 지정 (Source)

```
access-list <list num> permit <대상 네트워크의 대표주소> <WildCard>
```

2. 라우터의 각 인터페이스에 내부와 외부를 특정

```
ip nat inside
ip nat outside
```

3. NAT 명령어 사용

```
ip nat inside source list <list name> int f0/1 overload

=> inside의 IP를 인터페이스 f0/1에 해당하는 주소로 변환하겠다.
=> overload (다수의 사설IP가 공인 IP에 과적되는 방식으로 변환)
```

### 5.2 DNAT 방식

- 외부에서 들어오는 Request를 서버 IP로 전달 하는 명령어
   ```
   ip nat inside source static <서버 주소> <외부 인터페이스 주소>
   ```
- 포트를 통해 외부에서 들어오는 Request를 서버 IP로 전달 (80포트)
   ```
   ip nat inside source static tcp <서버 주소> <외부 인터페이스 주소> <포트 번호>>
   ```

### 5.3 ip가 변경되어 들어오고 나가는 것 확인

```
do sh ip net tran
```

### 5.4 결과

#### 토폴로지 구조

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/c769758b-c9a9-40aa-ae94-644be8b50c09)

#### Private PC0(192.168.0.10)에서 Router 3(20.20.20.100)으로 Ping

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/e249fc9a-4977-415e-a18f-f4eec40c9a92)

#### Private IP => Public IP로 변환되는 것 확인 ( IN Router 0 )

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/45dc93e6-2290-415c-94b6-ee37ee4d6958)

<br>

<strong>궁금하신점이나 추가해야할 부분은 댓글이나 아래의 링크를 통해 문의해주세요.</strong>   
Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
