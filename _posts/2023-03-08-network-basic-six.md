---
title: NetWork_Basic[HTTP-3]
date: 2023-03-08 20:12:00 +0800
categories: [CS, HTTP-Method]
tags: [Network_Basic,HTTP-Method]
---

## HTTP Method
HTTP-Method는 클라이언트가 서버에게 어떠한 동작을 수행할지를 나타내는 표준화된 방법입니다.<br/>
HTTP-Method에는 다양한 방법이 있으며, 각각의 메서드는 특정 목적을 수행합니다.<br/>

## GET
### 요청 메세지
```
GET /members/100 HTTP/1.1
Host: localhost:8080
```
- 클라이언트가 서버로 요청 메세지를 전송합니다.

### 만들어진 데이터
```
{
 "username": "young",
 "age": 20
}
```
- 서버에서는 GET의 요청 경로를 기반으로 데이터를 만들어줍니다.

### 응답 데이터
```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 34
{
 "username": "young",
 "age": 20
}
```
- 서버에서는 응답 데이터를 응답 메세지에 담아 클라이언트에게 다시 전송합니다.<br/>
- 이때 전달하고 싶은 데이터는 query를 통해서 전달합니다.
- GET 방식은 주로 리소스(데이터)를 조회할 때 사용되며, 요청한 데이터는 URL에 포함됩니다.<br/>

## POST
### 요청 메세지
```
POST /members HTTP/1.1
Content-Type: application/json
{
 "username": "young",
 "age": 20
}
```
- POST방식은 먼저 클라이언트가 요청 메세지를 서버에게 전달합니다.

### 만들어진 데이터

```
{
 "username": "young",
 "age": 20
}
```
- 클라이언트에서 요청한 데이터를 처리합니다.
   - 메세지 바디를 통해 들어온 데이터를 처리하는 모든 기능을 수행합니다.


### 응답 메세지
```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 34
{
 "username": "young",
 "age": 20
}
```
- 요청한 데이터를 처리하면, 응답데이터를 응답 메세지에 담아 클라이언트에게 다시 전송합니다.
- POST방식은 주로 신규 리소스를 등록하거나 프로세스 처리에 사용합니다.

## PUT
- 리소스를 대체하는 메소드입니다.
   - 리소스가 있으면 대체합니다.
   - 리소스가 없으면 생성합니다.

### 요청 메세지
```
PUT /members/100 HTTP/1.1
Content-Type: application/json
{
   "username": "old",
   "age": 50
}
```
- 클라이언트가 서버에게 요청 메세지를 전송합니다.

### 기존 데이터
```
{
   "username": "young",
   "age": 20
}
```
- 리소스가 있는데 위의 요청이 들어오면 기존에 있던 리소스를 대체합니다.
   - username : "young" -> "old"로 대체
   - age : "20" -> "50"로 대체
- 반대로 리소스가 없으면 새로 생성합니다.
   - 기존의 리소스가 없기 때문에, 새로 생성
      - username : "old" -> 새로 생성
      - age : "20" -> 새로 생성
- **주의할점!!**
   ### 요청 메세지
   ```
   PUT /members/100 HTTP/1.1
   Content-Type: application/json
   {
      "age": 50
   }
   ```
   - 요청 메세지가 위처럼 들어오게 된다면, age필드는 있고, username 필드가 없는 상태로 들어왔기 때문에, username이라는 필드는 삭제되고 age만 대체되게 됩니다.
      - username 필드 존재 X -> 필드 삭제
      - age : "20" -> "50"
      ```
      {
      "age": 50
      }
      ```


## PATCH
- 리소스를 부분 변경할떄 사용합니다.
### 요청 메세지
```
PATCH /members/100 HTTP/1.1
Content-Type: application/json
{
 "age": 50
}
```
### 리소스 변경
```
변경 전
{
 "username": "young",
 "age": 20
}
```
```
변경 후
{
 "username": "young",
 "age": 50
}
```

## DELETE
- 리소스 제거시에 사용됩니다.
### 요청 메세지
```
DELETE /members/100 HTTP/1.1
Host: localhost:8080
```
### 리소스 삭제
```
아래의 리소스를 삭제
{
 "username": "young",
 "age": 20
}
```

## 결론
- GET: 리소스 조회를 위한 메서드입니다.
- POST: 신규 리소스 생성 또는 프로세스 처리를 위한 메서드입니다.
- PUT: 리소스를 대체하는 메서드로, 클라이언트가 서버에게 요청 메시지를 전송하여 해당 리소스를 대체하거나 새로 생성합니다. 요청 메시지에 포함된 데이터로 리소스를 대체하며, 부분적인 변경은 불가능합니다.
- PATCH: 리소스를 부분적으로 변경하기 위한 메서드입니다.
- DELETE: 리소스를 제거하는 메서드입니다.