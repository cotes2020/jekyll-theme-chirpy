---
title: Calculator
date: 2022-05-28 21:37 +0900
categories: [Wargame,h4ckingga.me]
tags: [h4ckingga.me web, ssti]
---

## Calculator
<hr style="border-top: 1px solid;"><br>

```
simple calculating machine!!

http://web.h4ckingga.me:10000/

made by Sechack
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

jinja template의 기본적인 ssti 취약점이다. (dreamhack의 simple ssti 문제와 동일)

값으로 ```2*2```를 넣어보면 4가 출력되는 걸 알 수 있다.

따라서 ```''.__class__.__base__.__subclasses__()```를 해서 ```subprocess.Popen``` 객체를 찾아서 공격을 진행하면 된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
