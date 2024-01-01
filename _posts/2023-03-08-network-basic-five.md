---
title: NetWork_Basic[HTTP-2]
date: 2023-03-08 18:34:00 +0800
categories: [CS, HTTP]
tags: [Network_Basic,HTTP]
---

## HTTP Messege
저번 포스트에 이어서 HTTP에 대해 더 알아가보겠습니다.
HTTP의 요청 메세지는 클라이언트가 서버로 요청을 보낼떄와, 서버에서 클라이언트로 응답을 할때 사용되는 메세지 입니다.

## HTTP 요청 메세지
```
GET /search?q=spring&hl=ko HTTP/2.0
Host : www.googole.com
```
위의 요청 메세지를 기반으로 설명 드리겠습니다.
- 상태 라인
   ```
   GET /search?q=spring&hl=ko HTTP/2.0
   ```
   - HTTP 메서드 ( GET )
      - 서버가 수행해야할 동작을 지정합니다.
   - 요청 대상/경로 (/search?q=spring&hl=ko)
      - "/"로 시작하는 경로는 절대 경로입니다.
   - HTTP Version ( HTTP/2.0 )
      - 요청 메세지의 HTTP 버전입니다.
- 헤더
   ```
   Host : www.googole.com
   ```
   - 요청에 대한 부가 정보를 포함하는 부분입니다. ( Host : www.googole.com )
- 본문 ( 예제 메세지에서는 없음 )
   - 요청의 실제 데이터를 포함하는 부분입니다.<br/>


## HTTP 응답 메세지
```
HTTP/2.0 200 OK 
Content-Type: text/html;charset=UTF-8
Content-Length: 3423
<html>
 <body>...</body>
</html
```
위의 응답 메세지를 기반으로 설명 드리겠습니다.
- 상태라인
   ```
   HTTP/2.0 200 OK 
   ```
   - HTTP 버전 ( HTTTP/2.0 )
   - HTTP 상태 코드 ( 200 )
   - HTTP 이유 문구 ( OK )
- 헤더
   ```
   Content-Type: text/html;charset=UTF-8
   Content-Length: 3423
   ```
   - HTTP의 전송에 필요한 모든 부가정보를 표시합니다.
   - 위의 헤더는 Content의 타입과 문자 길이에 대한 정보를 표시하고 있습니다.
- 본문
   ```
   <html>
   <body>...</body>
   </html
   ```
   - 실제로 클라이언트에게 전송할 데이터 입니다.
   - HTML, JSON, 문서, 이미지 등 다양한 데이터가 여기에 포함될 수 있습니다.

## 결론
HTTP 요청 메세지는 클라이언트가 서버에게 어떤 동작을 수행하고 어떤 리소스를 요청하는지를 정의하며, 응답 메세지는 서버가 클라이언트에게 전송한 데이터와 수행한 동작의 결과를 담고 있습니다. 메세지의 구조를 이해하면 웹에서의 데이터 교환 과정을 파악할 수 있습니다.
