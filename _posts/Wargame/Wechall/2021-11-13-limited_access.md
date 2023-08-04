---
title : "Wechall - Limited Access"
categories : [Wargame, Wechall]
tags : [".htaccess", "Apache Authentication", "Javascript XMLHttpRequest"]
---

## Limited Access
<hr style="border-top: 1px solid;"><br>

```
I try to secure my pages with .htaccess.
Am I doing it right?

To prove me wrong, please access protected/protected.php.
```

<br>

**GeSHi`ed Plaintext code for .htaccess**

```console
AuthUserFile .htpasswd
AuthGroupFile /dev/null
AuthName "Authorization Required for the Limited Access Challenge"
AuthType Basic
<Limit GET>
require valid-user
</Limit>
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

Apache Authentication 
: <a href="https://httpd.apache.org/docs/2.2/ko/howto/auth.html" target="_blank">httpd.apache.org/docs/2.2/ko/howto/auth.html</a>

<br>

서버의 디렉토리를 암호로 보호하는 기본적인 방법은 htpasswd 도구를 이용해 암호 파일을 설정하는 것임.

+ AuthType
  + 사용자를 인증할 방법을 선택. 가장 일반적인 방법은 Basic으로, mod_auth_basic이 구현함. 
  + 그러나 Basic 인증은 브라우저가 서버로 암호를 암호화하지 않고 보냄.

<br>

+ AuthName "영역"
  + 인증에 사용할 영역(realm)을 지정하고 영역은 두 가지 역할을 함. 
    + 첫번째는 클라이언트가 보통 이 정보를 암호 대화창에 보여줌. 
    + 두번째는 영역 정보를 사용하여 클라이언트가 특정 인증구역에 어떤 암호를 보낼지 결정함.

<br>

예를 들어, 일단 클라이언트가 "Restricted Files" 영역에 인증이 성공하였다면, 클라이언트는 자동으로 같은 서버에서 "Restricted Files" 영역으로 표시된 구역에 대해 동일한 암호를 시도함.

그래서 여러 제한 구역이 같은 영역을 공유하면 사용자가 여러번 암호를 입력하지 않아도 됨.

물론 보안상 이유로 클라이언트는 서버의 호스트명이 다르면 항상 새로 암호를 물어봄.

<br>

+ AuthUserFile
  + htpasswd로 만든 암호파일의 경로를 설정함.
  + 문제 상에서는 현재 경로에 있는 htpasswd 에 암호를 걸어둔 것.

<br>

+ AuthGroupFile
  + 여러 사람을 해당 디렉토리로 들여보내고 싶을 때 사용.

<br>

+ Require
  + 서버의 특정 영역에 접근할 수 있는 사용자를 지정하여 권한부여를 함.

<br>

+ Require valid-user
  + 여러 일반 사용자를 들여보내는 다른 방법이 있음.
  + 그룹파일을 만들 필요없이 다음 지시어를 사용하기만 하면 됨.

<br>

+ Limit Directive
  + Restrict enclosed access controls to only certain HTTP methods
  + 문제 상에서 ```<Limit GET>``` 이므로 GET 메소드 제한한다는 것.

<br>

따라서 GET이 아닌 POST 메소드로 들어가면 된다는 것. Burp Suite를 이용하고자 했으나 proxy를 돌리니 wechall에선 막힘.

해서 console창에서 POST로 보내는 방법을 찾아봄.

<br.

```javascript
client=new XMLHttpRequest();
client.open('POST', "https://www.wechall.net/challenge/wannabe7331/limited_access/protected/protected.php", true)
client.setRequestHeader('Content-Type','application/x-www-form-urlencoded');
client.send();
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

XMLHttpRequest
: <a href="https://dongdd.tistory.com/29" target="_blank">dongdd.tistory.com/29</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
