---
title : Root-me File upload Double extensions
date: 2022-06-13-20:21 +0900
categories : [Wargame, Root-me]
tags : [File upload]
---

## File upload - Double extensions
<hr style="border-top: 1px solid;"><br>

```
Author
g0uZ,  24 December 2012

Statement
Your goal is to hack this photo galery by uploading PHP code.
Retrieve the validation password in the file .passwd at the root of the application.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

웹셸을 작성한 후 파일명을 확장자를 두 개를 해주면 된다. 
: ```webshell.php.png```

업로드를 해준 뒤 ```.passwd``` 파일을 출력해주면 되는데, 이 파일은 확인해보니 현재 경로에서 ```../../../```로 가면 파일이 있다.

따라서 ```?cmd=cd ../../../; cat .passwd```를 해주면 비밀번호가 출력된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
