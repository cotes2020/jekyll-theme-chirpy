---
title : Root-me File upload Null byte
date: 2022-06-13-22:09 +0900
categories : [Wargame, Root-me]
tags : [File upload]
---

## File upload - Null byte
<hr style="border-top: 1px solid;"><br>

```
Author
g0uZ,  26 December 2012

Statement
Your goal is to hack this photo galery by uploading PHP code.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

이 문제는 파일 확장자와 파일 타입을 검사하게 되어 있는데, double extensions가 불가능하다.

하지만 문제 제목이 null byte이므로 파일명에 널 바이트를 넣어주면 된다.

파일명을 ```sh.php%00.png``` 식으로 해주면 비밀번호가 나온다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
