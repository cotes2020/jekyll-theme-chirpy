---
title : Root-me IP restriction bypass
date: 2022-06-13-17:10 +0900
categories : [Wargame, Root-me]
tags : [X-Forwarded-For]
---

## HTTP - IP restriction bypass
<hr style="border-top: 1px solid;"><br>

```
Author
Cyrhades,  23 March 2021

Statement
Dear colleagues,

We’re now managing connections to the intranet using private IP addresses, 
so it’s no longer necessary to login with a username / password 
when you are already connected to the internal company network.

Regards,

The network admin
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

문제에서 내부망에 속해있으면 로그인이 필요 없다고 한다.

따라서 내 IP를 내부망 대역 IP로 변경해줘야 한다.

RFC1918에 따르면 내부망 IP 대역은 아래와 같다.
: ```10.0.0.0        -   10.255.255.255  (10/8 prefix)```
: ```172.16.0.0      -   172.31.255.255  (172.16/12 prefix)```
: ```192.168.0.0     -   192.168.255.255 (192.168/16 prefix)```

<br>

```X-Forwarded-For``` 헤더를 통해 클라이언트 IP를 변경할 수 있다.
: <a href="http://blog.plura.io/?p=6597" target="_blank">blog.plura.io/?p=6597</a>

아래 내용은 위의 블로그에서 가져온 설명이다.

<br>

웹 서버 앞에 Proxy server, caching server 등의 장비가 있을 경우, 웹서버는 Proxy server 이나 장비 IP 에서 접속한 것으로 인식한다.

따라서 웹서버는 실제 클라이언트 IP 가 아닌 앞단에 있는 Proxy 서버 IP 를 요청한 IP 로 인식하고, Proxy 장비 IP 로 웹로그를 남기게 된다.

이때 웹프로그램에서는 ```X-Forwarded-For``` HTTP Hearder 에 있는 클라이언트 IP 를 찾아 실제 요청한 클라이언트 IP 를 알 수 있고, 웹로그에도 실제 요청한 클라이언트 IP 를 남길 수 있다.

```X-Forwarded-For``` 는 다음과 같이 콤마를 구분자로 Client 와 Proxy IP 가 들어가게 되므로 첫번째 IP 를 가져오면 클라이언트를 식별할 수 있다.
: ```X-Forwarded-For: client, proxy1, proxy2```

<br>

따라서 내 헤더에 ```X-Forwarded-For: 10.0.0.1```이라고 추가만 해주면 패스워드가 나온다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
