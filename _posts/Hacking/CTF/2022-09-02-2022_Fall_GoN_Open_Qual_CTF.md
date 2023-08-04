---
title: 2022 Fall GoN Open Qual CTF
date: 2022-09-02 10:42  +0900
categories: [Hacking, CTF]
tags: [2022 Fall GoN Open Qual CTF, API PORTAL, sleepingshark]
---

## API PORTAL (web)
<hr style="border-top: 1px solid;"><br>

먼저 공격을 진행하기 전에 중요한 코드를 살펴보면 다음과 같다.

<br>

```php
case "net/proxy/get":
      $param = array($_GET["url"], $_SERVER["REMOTE_ADDR"], urldecode($_SERVER["REQUEST_URI"]));
      break;
case "net/proxy/post":
      $param = array($_GET["url"], $_SERVER["REMOTE_ADDR"], urldecode($_SERVER["REQUEST_URI"])); //TODO: implement POST data
      break;
```

<br>

index.php 페이지에서 action 파라미터로 값을 받는데, 여기에는 입력할 수 있는 다양한 페이지들이 있다.

세부적으로 설명하기에는 많기에 취약점이 터지는 부분은 위의 코드를 참조하여 ```net/proxy/post``` 페이지에서 페이로드를 넣어줄 수 있다.

최종적으로 우리는 flag의 페이지에 접속을 해야 하는데, 이 페이지는 로컬에서 접속해야 하며, 특정한 POST 값을 보내줘야 한다.

이러한 조건이 충족되었을 때, 어느 파일에 flag를 저장해주는데, 이 파일은 우리가 생성해줄 수 있으므로 설명은 생략한다.

<br>

```php
<?php
//TODO: Change to php-curl

$url = "http://".$param[0]; //TODO: support ssl context
$ip = $param[1];
$referer = $param[2];

$header = "User-Agent: API Portal Proxy\r\n";
$header .= "X-Forwarded-For: {$ip}\r\n";
$header .= "X-Api-Referer: {$referer}";

$ctx = stream_context_create(array(
    'http' => array(
        'method' => 'POST',
        "content" => "", //TODO: implement
        'header' => $header
    )
));

die(file_get_contents($url, null, $ctx));
```

<br>

위의 코드가 ```net/proxy/post``` 페이지 소스코드이다.

우리의 입력값이 들어가는 부분은 url 값뿐인데, 이 값이 어디 들어가는가?

우리의 입력값이 들어가는 부분은 url 말고도 ```$_SERVER["REQUEST_URI"]```가 있다.

이 변수는 현재 페이지에서 도메인을 제외한 주소 값을 가져와 주는 변수이다.

이 값이 들어가는 부분은 ```$referer``` 부분이다.

여기서  CRLF 취약점이 발생한다.

<br>

주의할 점은 url 값을 입력할 때, 에러가 나는 것을 방지하기 위해 url 부분에는 response를 받을 드림핵 requestbin 주소를 넣어준 뒤 ```&```으로 구분을 해줌으로써 에러를 없애줄 수 있다.

그래서 확인해보면 ```?url=dreamhack.request.bin&%0d%0aUser-Agent:%20TEST%0d%0a```를 보내주면 User-Agent 값에 TEST가 추가되는 것을 알 수 있다.

<br>

따라서 우리가 얻어야 할 플래그 페이지에 있는 조건에 맞춰서 페이로드를 보내주면 다음과 같다.
: ```net/proxy/post&url=localhost?action=flag/flag&%0d%0aContent-Type:%20application/x-www-form-urlencoded%0d%0aContent-Length:%2027%0d%0a%0d%0amode=write%26dbkey=test%26key=a```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## sleepingshark (forensic)
<hr style="border-top: 1px solid;"><br>

이 문제는 사실상 웹 문제라고 생각되는데, 패킷을 보면 HTTP 스트림에 대해서 가져와서 보면 Blind + Time SQL injection 쿼리를 보내는 것을 알 수 있다.

쿼리는 값이 True일 때, 3초간 Sleep을 한다는 코드를 보냈다.

그래서 패킷을 하나씩 다 보면서 3초 후에 응답을 받은 값들을 찾아내서 모두 찾은 다음 이어서 출력해주면 된다.

<br>

```javascript
flag = String.fromCharCode(71,111,78,123,84,49,109,69,95,66,52,115,51,100,95,53,81,76,95,73,110,106,51,99,55,105,48,110,95,119,73,55,104,95,80,99,52,112,125)

console.log(flag)

/*
3 -> 78
4 -> 123
5 -> 84
6 -> 49
7 -> 109
8 -> 69
9 -> 95
10 -> 66
11 -> 52
12 -> 115
13 -> 51
14 -> 100
15 -> 95
16 -> 53
17 -> 81
18 -> 76
19 -> 95
20 -> 73
21 -> 110
22 -> 106
23 -> 51
24 -> 99
25 -> 55
26 -> 105
27 -> 48
28 -> 110
29 -> 95
30 -> 119
31 -> 73
32 -> 55
33 -> 104
34 -> 95
35 -> 80
36 -> 99
37 -> 52
38 -> 112
39 -> 125
*/
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
