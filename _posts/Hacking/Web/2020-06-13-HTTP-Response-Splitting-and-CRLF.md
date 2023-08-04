---
title : HTTP Response Splitting and CRLF
categories : [Hacking, Web]
tags : [HTTP Response Splitting]
---

## CRLF
<hr style="border-top: 1px solid;"><br>

+ CR : Carriage Return
  + 현재 위치를 나타내는 커서를 맨 앞으로 이동시킨다 -> \r, %0D

+ LF : Line Feed
  + 커서의 위치를 아랫줄로 이동시킨다 -> \n, %0A

<br><br>
<hr style="border: 2px solid;">
<br><br>

## HTTP Header Injection
<hr style="border-top: 1px solid;"><br>

사용자의 입력이 들어가는 부분에 CRLF(```%0d%0a```)를 삽입하는 공격(CRLF Injection)을 하여 Header를 추가하거나 조작하는 공격

<br>

ex) webhacking.kr - web52
: <a href="https://ind2x.github.io/posts/web52/" target="_blank">ind2x.github.io/posts/web52/</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## HTTP Response Splitting
<hr style="border-top: 1px solid;"><br>

CWE-113

HTTP Request에 있는 파라미터가 HTTP Response의 Header에 그대로 전달되는 경우 파라미터에 CRLF가 존재하면 HTTP 응답이 분리될 수 있다는 취약점을 이용한 공격. 

Response Header에 악의적인 코드를 주입함으로써 XSS 및 캐시를 훼손할 수 있음.

<br>

Link
: <a href="https://www.invicti.com/blog/web-security/crlf-http-header/" target="_blank">invicti.com/blog/web-security/crlf-http-header/</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
