---
title: DNS (Domain Name System) [개념, 설정 법]
date: 2023-09-14 20:45:34 +0900
author: kkankkandev
categories: [Network]
tags: [network, dns, dnat, router, ip]     # TAG names should always be lowercase
comments: true
image:
  path: https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/21874f24-e2e1-4c62-aa82-4f21cccf1eec
---

DNS(Domain Name System)은 IP 주소 및 기타 데이터를 저장하고 이름별로 쿼리할 수 있게 해주는 계층형 분산 시스템입니다

만약 웹 사이트를 접속해야 할 때 웹 서버의 IP주소를 통해 접근해야 한다면 웹 서버의 IP주소를 모두 외워야 합니다.  

```
네이버를 접속 해야 한다 
=>웹 브라우저에 223.130.200.107 IP 주소 입력
```

이러한 불편함을 해소하기 위해 DNS(Domain Name Server)를 사용합니다. DNS란 위와 같이 특정 사이트의 IP 주소(컴퓨터가 이해할 수 있는 IP 주소)를 사람이 쉽게 기억할 수 있는 이름으로 접근 할 수 있게 할 수 있는 서비스를 제공합니다.

## 1. DNS의 동작 원리
![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/21874f24-e2e1-4c62-aa82-4f21cccf1eec)

```
1. 사용자가 DNS Server에 'www.naver.com' 접속 요청

2. DNS Resolver가 'com'에 해당하는 부분을 TLD(Top Level Domain)에 쿼리

3. TLD가 'com'에 해당하는 부분 Query후 DNS 서버에 응답

4. DNS Resolver가 'naver'에 해당하는 부분 SLD(Second Level Domain)에 쿼리

5. SLD가 'naver'에 해당하는 부분 Query후 DNS 서버에 응답

6. DNS Resolver가 'www'에 해당하는 부분 Sub-Domain에 쿼리

7. Sub-Domain에서 'www'또는 그 하위에 해당하는 부분을 찾은 후 반환

8. DNS Resolver가 '200.130.200.107' IP 주소를 웹 브라우저(USER)에게 응답
``````

## 2. CentOS에서 DNS Server 구축하기

### 2.1 **bind, bind-utils** package 다운로드
   
   ```yum -y install bind bind-utils```

### 2.2 /etc/named.conf 파일 설정
   
> Domain Server의 기능적인 설정

   ```vi /etc/named.conf```

   ![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/df2df6c3-2db2-412c-bd52-c2dc154b4012)
   
1. 13행 listen-on port 53 { any; }; 로 변경
2. 21행 allow-query { localhost; }; 에서 { any; }; 로 변경

### 2.3 /etc/named.rfc1912.zones 파일 설정 

> 안내 파일에 대한 정보 명시

   ```vi /etc/named.rfc1912.zones```
   ![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/1d6cb1dc-51aa-42ab-83e9-43dbbcbfc192)



### 2.4 /var/named/ 하위에 안내파일 생성 및 설정

> 서버 목록이 담긴 안내파일 생성
> 영문 주소 => IP 주소 매칭 목록 추가

   ```vi /etc/named/{안내 파일 명}```
   ![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/a27b06d0-f00d-41bc-848e-49ff7edb1d1c)

### 2.5 설정 확인

```
#안내 파일 형식에 문제가 없는지 체크

   named-checkzone {도메인 주소} {안내파일}
      ex) named-checkzone aws1.com /var/named/aws1.com.db

#DNS Server 시작프로그램에 추가 및 실행

   systemctl enable --now named

#DNS Server 변경 (IN Client)

   vi /etc/sysconfig/network-script/ifcfg-ens32 
      => ens32에 해당하는 부분은 사용자의 이더넷에 맞게 설정 변경 필요

#도메인 설정이 올바르게 되었는지 확인 (IN Client)

   #nslookup을 사용하기 위한 Package 설치
   yum -y install bind-utils

   #현재 DNS 서버 확인
   cat /etc/resolv.conf

   #현재 DNS 서버가 올바른 주소로 안내하는지 확인
   nslookup aws1.com
```


<br>

<strong>궁금하신점이나 추가해야할 부분은 댓글이나 아래의 링크를 통해 문의해주세요.</strong>   
Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
