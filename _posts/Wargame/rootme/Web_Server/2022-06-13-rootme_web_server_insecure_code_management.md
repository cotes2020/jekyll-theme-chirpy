---
title : Root-me Insecure Code Management
date: 2022-06-13-23:17 +0900
categories : [Wargame, Root-me]
tags : [.git]
---

## Insecure Code Management
<hr style="border-top: 1px solid;"><br>

```
Author
Swissky,  29 September 2019

Statement
Get the password (in clear text) from the admin account.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

관련 자료로 git-scm 홈페이지를 알려주는데, 이를 토대로 확인해보면 git을 사용할 때, 디렉토리로 ```.git```이라는 디렉토리가 생긴다.

따라서 ```.git```으로 가보면 아래와 같이 나타난다.

<br>

![image](https://user-images.githubusercontent.com/52172169/173374541-a6634452-7bce-44b1-a875-cc3ca20e4cd1.png)

<br>

이걸 다운로드를 해서 git으로 파악을 해줘야 한다. 
: ```wget -r -P ch61 http://challenge01.root-me.org/web-serveur/ch61/.git/```

<br>

위의 명령어로 웹서버에서 파일을 다운로드하여 ```./ch61``` 디렉토리에 저장하였다.

이제 다운로드된 파일을 git으로 확인해준다.

<br>

로그가 중요하므로 ```logs/refs/head``` 파일을 살펴보면 아래와 같이 되어 있다.

<br>

```
0000000000000000000000000000000000000000 5e0e146e2242cb3e4b836184b688a4e8c0e2cc32 John <john@bs-corp.com> 1567674615 +0200	commit (initial): Initial commit for the new HR database access
5e0e146e2242cb3e4b836184b688a4e8c0e2cc32 1572c85d624a10be0aa7b995289359cc4c0d53da John <john@bs-corp.com> 1568279406 +0200	commit: secure auth with md5
1572c85d624a10be0aa7b995289359cc4c0d53da a8673b295eca6a4fa820706d5f809f1a8b49fcba John <john@bs-corp.com> 1569148712 +0200	commit: changed password
a8673b295eca6a4fa820706d5f809f1a8b49fcba 550880c40814a9d0c39ad3485f7620b1dbce0de8 John <john@bs-corp.com> 1569244207 +0200	commit: renamed app name
550880c40814a9d0c39ad3485f7620b1dbce0de8 c0b4661c888bd1ca0f12a3c080e4d2597382277b John <john@bs-corp.com> 1569607805 +0200	commit: blue team want sha256!!!!!!!!!
```

<br>

로그를 보면 비밀번호를 변경한 로그가 있다. 

우리는 이 commit을 복구시켜야 한다. 

commit을 복구시키는 방법은 아래 블로그에서 확인.
: <a href="https://www.letmecompile.com/git-restore-lost-commits/" target="_blank">letmecompile.com/git-restore-lost-commits/</a>

<br>

그 전에 git에서는 ```.git```이 있는 폴더를 working tree로 인식하는데 로그를 확인하려면 work tree로 가야하므로 ch61 디렉토리에서 확인해줘야 한다.

```git reflog```를 한 뒤 ```change password```를 commit한 commit ID를 확인한 뒤, ```git reset --hard a8673b2```으로 복구시켰다.

그랬더니 ```index.php, config.php``` 파일이 복구되었다. ```config.php``` 파일에 비밀번호가 있으므로 인증하면 된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>
