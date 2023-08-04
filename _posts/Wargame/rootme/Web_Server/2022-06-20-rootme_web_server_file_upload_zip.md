---
title: Root-me File upload ZIP
date: 2022-06-20-14:00  +0900
categories: [Wargame, Root-me]
tags: [file upload, zip file upload, zip slip attack, zip based exploit with symlink]
---

## File upload - ZIP
<hr style="border-top: 1px solid;"><br>

```
File upload - ZIP
Unsafe decompression
Author
ghozt,  3 August 2017

Statement
Your goal is to read index.php file.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

Zip based exploit
: <a href="https://levelup.gitconnected.com/zip-based-exploits-zip-slip-and-zip-symlink-upload-21afd1da464f" target="_blank">levelup.gitconnected.com/zip-based-exploits-zip-slip-and-zip-symlink-upload-21afd1da464f</a>

<br>

위의 블로그에 따르면 zip slip이란 공격 기법을 이용해서 심볼릭 링크를 통해 공격을 진행한다.

우선 zip slip이란, 원래 파일명에 경로를 구분해주는 slash를 사용할 수 없는데 만약 포함이 되어 있을 때, zip에서 archive 파일을 decompressed 할 때 이를 이용해 공격자가 원하는 경로로 파일을 이동시키는 공격 기법이라고 한다.
: <a href="https://www.hahwul.com/cullinan/zip-slip/" target="_blank">hahwul.com/cullinan/zip-slip/</a>

<br>

이를 이용해서 아래와 같이 공격을 진행하면 된다.

```
ln -s ../../../index.php test
zip --symlink test.zip test
```

<br>

그 후 생성된 ```test.zip``` 파일을 업로드 해준 뒤, ```test```로 가주면 ```index.php``` 파일이 보여진다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
