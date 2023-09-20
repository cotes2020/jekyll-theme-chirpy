---
title: PxE(Pre-boot eXecution Envirionment) 개념, 사용법 
date: 2023-09-20 20:11:22 +0900
author: kkankkandev
categories: [Linux]
tags: [pxe, linux, centos, dhcp, tftp, ftp]     # TAG names should always be lowercase
comments: true
image:
  path: https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/28066284-64db-4667-bbad-d3ef05316e01
---

만약 퇴근 1시간 전. 내가 사무실 컴퓨터 40대에 OS를 설치 해야하는데, USB가 1개 뿐이라면??  

멘붕이 올 수 있다.. 이 때 사용할 수 있는 기술이 바로 PxE(Pre-boot eXecution Environment)입니다.  

실습환경 => CentOS, VMware

## 1. PxE의 개념

기본적으로 서버나 PC를 켜면, 부팅 우선순위에 따라 디바이스나 디스크를 선택하고, 해당 디바이스나 디스크에서 운영체제를 탐색합니다. 만약 부팅 우선순위에 있는 디바이스나 디스크에서 운영체제를 찾지 못했을 때, 마지막으로 서버나 PC는 운영체제를 설치할 수 있는 네트워크 서버를 탐색하게 됩니다. 이 때 네트워크 상에서 운영체제를 설치하는 서버가 바로 PxE Server입니다. Client는 PxE Server를 통해 ip와 OS를 받을 수 있습니다.

## 2. PxE Server의 구성요소

### 2.1 DHCP Server

	1. IP가 없는 Baremetal Client와 통신을 위해 사용됩니다
### 2.2 TFTP Server

	1. OS설치에 필요한 절차가 담긴 설정 파일 전송을 위해사용
	2. UDP 기반 통신 (부트 로딩)에 사용
### 2.3 FTP Server

	1. 실제 iso파일을 전송할 때 사용
	2. TCP/IP 기반 통신 (신뢰성, 보안성)

## 3. PxE Server 구축하기

### 3.1 방화벽, Selinux, NetworkManager 끄기

CentOS는 Network설정에 NetworkManager와 /etc/sysconfig/network-script/ifcfg-ens32가 사용됩니다.  두 개의 설정이 충돌이 일어날 수 있습니다.  ftp, tftp 통신을 위해 방화벽, Selinux, NetworkManger를 꺼주도록 하겠습니다.

=> **ifcfg-ens32는 상황에 따라 ens33이 될 수도 있고 eth0이 될 수도 있습니다.**

```
#NetworkManager 중단 및 재부팅 후 자동 실행 막기
	systemctl disable --now NetworkManager

#방화벽(firewalld) 중단 및 재부팅 후 자동 실행 막기
	systemctl disable --now firewalld

#Selinux 중단 및 재부팅 후 자동 실행 막기
	sed -i s/SELINUX=enforcing/SELINUX=disabled/g /etc/selinux/config

	만약 위 명령어가 작동하지 않을 시 vi나 nano를 통해 /etc/selinux/config파일에 접근 후
	SELINUX=enforcing에 해당하는 부분을 SELINUX=disabled로 바꿔주세요
```

### 3.2 dhcp 서버, tftp pacakage 다운로드 및 설정

```
# DHCP, FTP, TFTP package 다운로드
	yum -y install dhcp tftp-server vsftpd

# DHCP 설정
	vi /etc/dhcp/dhcpd.conf
	
-------------------------예시----------------------------
	subnet 211.183.3.0 netmask 255.255.255.0 
	{ 
	#GW 
	option routers 211.183.3.2; 
	
	#SM 
	option subnet-mask 255.255.255.0; 
	
	#ip를 받아올 수 없는 상황에서도 ip를 부여받을 수 있도록 
	range dynamic-bootp 211.183.3.240 211.183.3.250; 
	
	#DNS 
	option domain-name-servers 8.8.8.8; 
	
	#부팅 허용. 
	allow booting; 
	
	#PxE 서버의 주소 
	next-server 211.183.3.21; 
	
	#Filename 설정 filename 
	"pxelinux.0"; 
	} 
-------------------------------------------------------

#DHCP 재시작
	systemctl restart dhcpd

```

예시

<p align="center"> 
	<figure > 
		<img src="https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/bfd3195d-71c0-444e-a070-05deba6bda59"> 
</figcaption>
	</figure>
</p>

### 3.3 TFTP로 운영체제 설치파일(iso), 부트로더 전송을 위한 설정  

TFTP로 전송할 파일은 다음과 같습니다.
1. vmlinuz      => 압축된 커널
2. initrd.img   => 임시 저장공간으로 사용할 램디스크
3. pxelinux.0   =>  pxe에 존재하는 부팅에 필요한 파일

- boot-loader - 부팅을 하기 위한 로딩, 사전 절차가 들어있는 파일
- /var/lib/tftpboot => 사용자가 tftp로 서버에 접속할 때 가는 기본 경로

```
#mount /{iso파일이 들어있는 경로} /media
	mount /dev/cdrom /media

# 압축된 커널 복사 (vmlinuz)
	cp /media/images/pxeboot/vmlinuz /var/lib/tftpboot

# 임시 저장공간으로 사용할 램디스크 복사 (OS설치 이전에는 디스크가 존재하기 않기 때문)
	cp /media/images/pxeboot/initrd.img /var/lib/tftpboot

# pxe에 존재하는 부팅에 필요한 파일 복사
	yum -y install syslinux
	cp /usr/share/syslinux/pxelinux.0 /var/lib/tftpboot/

# tftp 서비스의 실행 여부 설정 (No => 실행하겠다)
vi /etc/xinetd.d/tftp

# tftp 재시작
	systemctl status tftp
	systemctl enable --now tftp

```

### 3.4 FTP로 운영체제 파일 전송을 위한 준비

```
cp -r /media/* /var/ftp/pub

mkdir /var/lib/tftpboot/pxelinux.cfg
cd /var/lib/tftpboot/pxelinux.cfg

# PxE 설치를 위한 절차를 /var/lib/tftpboot/pxelinux.cfg/default에 정의

--------------------------default 파일 내용---------------------------
DEFAULT CentOS7_Auto_Install 
LABEL CentOS7_Auto_Install 
	kernel vmlinuz # 커널 정보는 여기 담겨있다 
	APPEND initrd=initrd.img repo=ftp://211.183.3.21/pub 

----------------------------------------------------------------------

# FTP Deamon 재시작
systemctl enable --now vsftpd
```


## PxE를 사용해 Client에 OS 설치하기

### 클라이언트에 CD/DVD가 들어있지 않음.

PxE를 사용하기 위해서는 PxE Server에서 DHCP 주소를 받아야하므로 Network Adapter를 NAT로 설정해주었습니다. NAT 네트워크 대역(211.183.3.0 /24)
![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/9c8214d8-84ea-4e18-bcd3-3e2270cfe35c)

### DHCP Server를 통해 주소 부여 받기

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/28066284-64db-4667-bbad-d3ef05316e01)

### PxE를 통한 OS 설치 가능 여부 확인

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/7dfd87af-7abd-43dc-9ecd-b8d8745f72bd)



<br>

<strong>궁금하신점이나 추가해야할 부분은 댓글이나 아래의 링크를 통해 문의해주세요.</strong>   
Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
