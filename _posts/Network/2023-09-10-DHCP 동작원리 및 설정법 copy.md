---
title: DHCP (Dynaic Host Configuration Protocol) 동작원리 및 설정법
date: 2023-09-10 23:45:55 +0900
author: kkankkandev
categories: [Network]
tags: [network, dhcp, vmware, gns3, router, ip, sub netmask, cisco, packet, tracer, packet tracer]     # TAG names should always be lowercase
comments: true
---

## 1.DHCP란?

> DHCP(Dynamic Host Configuration Protocol)   
> => **DHCP란 Client가 DHCP Server로부터 IP, Subnet Mask, Default Gateway를 동적으로 부여받는 프로토콜입니다.**

<br>

## 2. DHCP를 사용하는 이유
모든 디바이스는 TCP 기반 네트워크에서 IP, MAC Address를 기반으로 통신을 합니다. 따라서 중복되는 IP가 없어야 합니다.  

만약 DHCP를 사용하지 않고 수동으로 IP, MAC Address를 설정하게 된다면 Network Engineer는 해당 네트워크에 대한 모든 IP 주소와 Gateway주소를 파악해야 하며 네트워크에서 제거 된 컴퓨터에 대한 IP주소를 수동으로 회수해야 합니다.  
**(유지보수성 ↓ 에러발생 확률 ↑)**

DHCP Server는 **Pool**이라는 개념을 사용하여 IP를 유지 관리하고, 더 이상 사용하지 않는 주소는 자동으로 재할당을 위해 pool로 반환시킵니다.  
**(안정적인 IP 주소 구성 가능, 네트워크 유지보수의 편리)**


> **< 핵심 Keyword >**
> 
> pool, 임대 기간, 중앙 집중화, IP 자동 할당, IP 주소 변경을 효율적으로 처리
  
<br>
## 3.DHCP 동작 원리


![DHCP 동작 원리](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/66468fb7-45f6-4098-8967-5da9f215c7f5)

```
1. Discover 
    - Client가 Broadcast를 통해 Server와 통신

2. Offer 
    -  Server가 Client에게 IP 제안

3. Request
    - Client가 주소 확인 후 서버에게 해당 주소 요청

4. ACK
    - 서버가 응답
```

## 4. DHCP 설정법 (Server)
> Virtual Machine => **VMware** 사용  
> OS => **CentOS 7 64bit** (DHCP Server, Client)
### 1. dhcp 다운로드
```
yum -y install dhcp
```

### 2. /etc/dhcp/dhcpd.conf 설정

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/77fe2a9e-7ede-4586-9ce0-08f8839b1b20)

```
7: subnet & netmask
  => Network Address, Subnet Mask 설정 (Client에게 부여해줄 Network)

8: range
  => 호스트 주소 범위 설정 (211.183.3.60 ~ 211.183.3.80 범위 안에서 IP 동적 할당)

9: option domain-na9e-servers
  => 도메인 주소 설정

10: option routers
  => Default Gateway 설정

11: default-lease-time
  => IP 유효기간 설정 (기본 임대 시간 => lease time을 지정하지 않을 경우 이 시간으로 임대시간 할당)

12: max-lease-time
  => IP 임대 가능 최대 시간 설정 (임대 시간)
```

## 5. DHCP 설정 (Client - IP 받아오기)

- [dhcp 설정 전 - IP 확인]

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/c8e21e44-1b62-441e-b243-b0acbecb4bd1)

- [DHCP 설정 전 - Network 설정]

```
vi /etc/sysconfig/network-scripts/ifcfg-ens32  

# end32는 상황에 따라 33이 될 수도 있고 다른 숫자가 될 수도 있음
```  

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/aea4c549-4fd2-42d2-b2ec-fbbb2f24eab7)

```
변경할 부분

4: BOOTPROTO="dhcp"
  
    /*
     * static - 고정 IP 할당 
     * dhcp - 유동 IP 할당
     * none - 사용안함
     */

16~19: 수동으로 IP를 지정해 사용할 때 IP Address, Subnet Mask, DNS주소를 입력하는 부분 주석처리
```

- [DHCP 설정 후 - 주소확인]


```
# 네트워크 재시작
systemctl restart network

# 현재 이더넷에 연결된 네트워크 확인
ifconfig
```

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/13702f4d-46c8-4ed8-8767-97cd78e2c668)

IP 211.183.8.100 => 211.183.8.61 

**DHCP 서버가 지정한 Range(60~80)사이의 IP로 IP가 동적할당 된 것을 확인할 수 있습니다.**

<br><br>

<strong>궁금하신점이나 추가해야할 부분은 댓글이나 아래의 링크를 통해 문의해주세요.</strong>   
Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
