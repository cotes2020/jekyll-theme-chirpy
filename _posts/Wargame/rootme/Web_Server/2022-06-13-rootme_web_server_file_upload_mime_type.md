---
title : Root-me File upload MIME type
date: 2022-06-13-20:30 +0900
categories : [Wargame, Root-me]
tags : [File upload]
---

## File upload - MIME type
<hr style="border-top: 1px solid;"><br>

```
Author
g0uZ,  26 December 2012

Statement
Your goal is to hack this photo galery by uploading PHP code.
Retrieve the validation password in the file .passwd.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

이 문제는 확장자를 검사하는게 아닌 파일 타입을 검사하므로 파일 타입은 ```image/png```로 해준 뒤, 웹셸을 업로드 해주면 된다.

```cd ../../../; cat .passwd```를 해주면 비밀번호가 출력된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
