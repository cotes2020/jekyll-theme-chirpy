---
title : Javascript Web API (Incomplete)
categories: [Programming, Javascript]
tags: [XMLHttpRequest]
---

## XMLHttpRequest
<hr style="border-top: 1px solid;"><br>

객체 생성
: ```const xhr = new XMLHttpRequest();```

<br>

요청 전송
: ```xhr.open(Method, URL);```
: ```xhr.open('GET','/url');```

<br>

헤더 설정
: ```xhr.setRequestHeader(Header, Value);```
: ```xhr.setRequestHeader('Content-Type','application/x-www-form-urlencoded');```

<br>

전송은 ```xhr.send()```를 통해 보내는데, POST의 경우에는 보내야할 데이터가 있으므로 다음과 같이 설정하면 된다.

<br>

```javascript
const data = {
  id: 2,
  title: 'abc',
};

xhr.send(JSON.stringify(data));
```

<br>

보낸 요청을 서버에서 응답했을 때, 처리하는 방법은 다음과 같다.

<br>

```javascript
xhr.onload = () => {
  if(xhr.status === 200) {
    const res = JSON.parse(xhr.response);
    console.log(res);
  }
  else {
    console.error(xhr.status, xhr.statusText);
  }
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
