---
title: HTTP Header
date: 2022-07-13 10:10:00 +09:00
categories: [Web]
tags: [Web, HTTP]
author: aestera
---

CTF나 워게임을 풀다 보면 HTTP Header에 많은 정보들이 들어있다.

HTTP Header에 대해서 제대로 공부해 보면 좋을 것 같다. Header를 공부하기 전에 HTTP Message를 공부해보자   


<br><br>

****

## **1\. HTTP Message**



HTTP Message는 클라이언트와 서버 사이 데이터가 교환 되는 방식이다. 클라이언트에서 서버로의 reqeust 와 서버에서 클라이언트로의 response 두 가지 종류가 있다.

![Untitled](/assets/img/post_images/HTTP Header/message.png)

Message는 start line, Message Header field, Header field의 끝을 알려주는 빈 줄, body로 이루어져 있다.

서버는 Request Message의 start line이 들어오기 전까지의 모든 CRLF를 무시한다.

&emsp;※CR(Carraige Return, \\r), LF(Line Feed, \\n)

​

---

​

## **2\. HTTP Header**

​

HTTP Header는 크게 4가지로 분류된다.


- General Header
- Request Header
- Response Header
- Entity Header

​
### **General Header**

request 와 response 에 모두 해당하지만 전송되는 body값과는 관련이 없는 Header

​|Header|Description|
|:------:|------------|
|Cache-Control|request / response 체인의 모든 캐싱 메커니즘 제어&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
|Connection|**close :** 메세지 교환 후 tcp통신 종료 <br>**Keep-Alive :**   메세지 교환 후 tcp통신 유지|

​
###  **Request Header**

Message를 전달하는 client에 관한 정보들을 포함하는 Header

|Header|Description|​
|:------------:|---------------------------------|
|Accept|응답시 허용되는 미디어 유형을 지정. 앞에 있는 미디어 유형일수록 우선적으로 받음 <br>아래 특징들은 Accept header의 모든 부속 속성에서도 사용<br><br> \-  \* 은 쿼리문의 \*처럼 모든 범위를 표현 (\*/\* 은 모든 미디어타입, type/\*은 type의 모든 subtype)<br><br> \- 파라미터 q는 0~1 의 숫자로 해당 media range에 대한 선호도를 나타낸다. 기본값은 1이다 (text/html; q=0.8, \*/\*; q=0.9)|
| Accept-Charset | 클라이언트가 이해할 수 있는 Charset과 선호도 표기. 서버는 Content-Type로 선택된 charset 전달 |
| Accept-Language | 클라이언트가 이해할 수 있는 언어와 선호도를 표기 (Accept-Language: ko-KR,ko;q=0.9, en-US;q=0.8) |
| Accept-Encoding | 클라이언트가 처리할 수 있는 압축 알고리즘과 그 선호도를 표기 (Accept-Encoding: br;q=1.0, gzip;q=0.8) |
| Authorization | 서버에게 허용된 사용자임을 증명 Authorization: \<type\>  \<credentials\> 형태 <br>**\<type\> = basic :**&nbsp; 기본적 타입이며 "id:password"를 base64 인코딩 <br>**\<type\> = bearer :**&nbsp; JWT 또는 OAuth에 대한 토큰 사용 |
| Host | 요청자의 host명 port번호 |
| User-Agent | 요청자의 소프트웨어 정보 |
| Cookie | 서버에 전달되는 Cookie값 |
| Referer | 현재 요청을 보내는 페이지의 주소|

​
###  **Response Header**

서버에 대한 정보와 Request-URI로 식별된 리소스에 대한 추가 액세스에 대한 정보를 제공

|Header|Description|
| :--------: | -------- |
| Server | 서버 소프트웨어 정보 |
| Cache-Control | 캐시 관리에 관한 정보 |
| Set-Cookie | 세션 쿠키 정보 전달 |

​

### **Entity Header**

엔티티 본문에 대한 메타정보 정의

|Header|Descriprion|
|:-------:|-------------|
|Content \- \[부속속성\]|Encoding, Length, Language 등 전달되는 데이터에 관한 다양한 정보 |
|Expires|응답이 오래된 것으로 간주되는 날짜/시간을 제공|

​

기본적으로 자주 보이는 헤더들만 정리했다. 이외의 다양한 헤더들은 필요할 때 구글링을 해보도록 하겠다.



참고 : [https://datatracker.ietf.org/doc/html/rfc2616#](https://datatracker.ietf.org/doc/html/rfc2616#)