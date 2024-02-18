---
title : Root-me CRLF
categories : [Wargame, Root-me]
tags : [CRLF]
---

## CRLF
<hr style="border-top: 1px solid;"><br>

```
Author
g0uZ,  31 July 2011

Statement
Inject false data in the journalisation log.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

하... CRLF 라길래.. 계속 삽질을 했는데..

로그에 ```admin authenticated.```가 찍히면 되는 거였다..

따라서 ```username=admin%20authenticated.%0d%0a123```를 입력해주면 password가 나온다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
