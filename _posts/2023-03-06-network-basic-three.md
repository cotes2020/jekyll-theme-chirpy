---
title: NetWork_Basic[URI,URL,URN]
date: 2023-03-06 21:24:00 +0800
categories: [CS, InternetNetwork]
tags: [Network_Basic,URL]
---
## URI(Uniform Resource Identifier)
URI은 통합 자원 식별자라고도 하는데, 인터넷에 자원을 나타내는 유일한 주소입니다.<br/>
인터넷에서 기본조건으로서 인터넷 프로토콜에 항상 붙어 다닙니다.<br/>
크게 URI는 URL과 URN을 포함하고 있습니다.<br/>

## URN? URL?
URL은 리소스가 있는 위치를 지정하고, URN은 리소스에 이름을 부여합니다.

## URL
URL은 보통 다음과 같이 이루어져 있습니다.
URL 예시
```
schema://[userinfor@]host[:port][/path][?query][#fragment]

https://www.google.com:443/search?q=hello&hl=ko
```
1. 프로토콜 ( https/http )
2. 호스트명 ( www.google.com )
3. 포트 번호 ( 443 )
4. 패스 (/search )
5. 쿼리 파라미터 (q=hello&hl=ko )

### schema
- 주로 프로토콜이 사용됩니다.
- 프로토콜 : 어떤 방식으로 자원이 접근할 것인가 하는 약속 규칙입니다.

### userinfo
- 사용자정보 인증을 위해 사용되지만, 거의 사용하지는 않습니다.

### host
- 도메인명(DNS), 또는 IP 주소를 직접 사용이 가능합니다.

### PORT
- 네트워크 토잇ㄴ에서 프로세스들이 서로를 구분하기 위한 번호입니다. 즉, 접속 포트를 의미합니다. 

### PATH
- 리소스의 경로를 나타냅니다.

### query
- key=value형태이며, ?로시작하고 추가기능이 존재합니다.
간단한 예시를 들면
```
https://www.example.com/search?q=apple&type=fruit
```
다음과 같은 URL이 존재할 때, q=apple이라는 검색어를 통해 apple이라는 과일을 찾고자 하고,<br/>
type=fruit를 통해 서버에게 찾는 것이 과일임을 알려줍니다.<br/>
여기서 key는 q과 type이 해당되고, value는 apple, fruit가 해당됩니다.<br/>
### fragment
- html 내부 북마크 등에 사용되지만, 서버에 전송하는 정보는 아닙니다.