---
title : Root-me JSON Web Token (JWT) Introduction
date: 2022-06-15-22:17 +0900
categories : [Wargame, Root-me]
tags : [JSON Web Token, JWT, JWT None algorithm injection, JWT none type injection]
---

## JSON Web Token (JWT) - Introduction
<hr style="border-top: 1px solid;"><br>

```
20 Points
Secure token exchange
Author
Kn0wledge,  21 August 2019

Statement
To validate the challenge, connect as admin.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

```JWT(Json Web Token)```이란?
: <a href="https://brunch.co.kr/@jinyoungchoi95/1" target="_blank">brunch.co.kr/@jinyoungchoi95/1</a>

<br>

관련 자료 중 Attacking JWT Authentication이 있어서 검색해보았고 아래 블로그에서 자세한 설명을 볼 수 있었다.
: <a href="https://dohunny.tistory.com/m/15" target="_blank">dohunny.tistory.com/m/15</a> --> GOOD

<br>

요약하면 JWT는 Header, Payload, Signature로 구성되어 있으며, 각각을 base64-url safe로 인코딩 후 dot으로 구분을 한다. 

즉, Header.Payload.Signature 이며 세부적으로는 ```base64(Header).base64(Payload).HMACSHA256(base64(Header)+"."+base64(payload),key)```이다.

그런데 Header 부분에서 algorithm을 ```None```으로 해주면 Signature 없이 인증을 할 수 있다고 한다. 

<br>

따라서 아래 값을 <a href="https://irrte.ch/jwt-js-decode/" target="_blank">JWT Encoding 사이트</a>에서 토큰을 생성하여 Header, Payload 값을 이용해 인증해주면 된다.

```
HEADER:
{
    "alg": "none",
    "typ": "JWT"
}
PAYLOAD:
{
    "username": "admin"
}
signHMACSHA256(
  base64Url(header) + "." +
  base64Url(payload),
Your-HMAC-secret
)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
