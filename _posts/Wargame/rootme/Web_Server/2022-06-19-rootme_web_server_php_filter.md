---
title : Root-me PHP Filters
date: 2022-06-19-11:13 +0900
categories : [Wargame, Root-me]
tags : [LFI]
---

## PHP - Filters
<hr style="border-top: 1px solid;"><br>

```
FileManager v 0.01
Author
g0uZ,  27 February 2011

Statement
Retrieve the administrator password of this application.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

inc 파라미터가 있는데 여기로 들어갈 페이지를 입력을 받는다.

```../```를 입력해주면 warning이 뜨면서 에러를 출력해주는데, 보면은 입력 값이 include에 포함된다. 

즉, ```include($_GET['inc'])```로 되어 있다.

LFI 공격을 할 때, include 함수에도 wrapper를 사용할 수 있다. 

따라서 ```php://filter/convert.base64-encode/resource=login.php```를 해주면 base64로 된 login.php의 코드가 나온다.

decode 해준 뒤, config.php에 비밀번호가 있어서 config.php도 확인해주면 admin의 비밀번호가 나온다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
