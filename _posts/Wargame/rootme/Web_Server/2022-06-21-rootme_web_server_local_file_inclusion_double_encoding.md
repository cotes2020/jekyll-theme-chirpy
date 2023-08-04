---
title: Root-me Local File Inclusion Double Encoding
date: 2022-06-21-20:31  +0900
categories: [Wargame, Root-me]
tags: [LFI, LFI Double Encoding]
---

## Local File Inclusion - Double Encoding
<hr style="border-top: 1px solid;"><br>

```
30 Points
Include can be dangerous.
Author
zM_,  13 June 2016

Statement
Find the validation password in the source files of the website.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

Double Encoding을 이용해서 풀라고 문제에서 알려줬으므로 우선 ```../```을 보내보았다.
: ```%252E%252E%252F```

<br>

open_basedir이 걸려 있으며, include 하는 파일을 보니 ```$_GET['page'].inc.php```이다.

문제에서는 home, CV, contact 페이지가 있는데 먼저 home의 코드를 살펴보았다.
: ```php://filter/convert.base64-encode/resource=home```을 Double Encoding을 해준 뒤 보내준 뒤 코드를 확인해보면 ```config.inc.php```를 include 하는 걸 알 수 있다.

따라서 그 다음 ```config.inc.php``` 파일의 코드를 확인하여 플래그를 알아내면 된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
