---
title: Regist SSH Connection on Github
date: 2023-06-27 13:08 +0900
category: [Environment Settings]
tag: [Git]
---

깃헙에 SSH 키를 등록하여 인증하는 방법이다.

1. `ssh-keygen`을 실행하고 완료될 때까지 엔터를 누른다.

2. `~/.ssh`경로에 id_rsa, id_rsa.pub 파일이 생성된 것을 확인한다.

3. `cat id_rsa.pub`로 SSH 키를 출력할 수 있다. 이를 복사하여 Github 계정에 등록한다.

> ssh로 인증을 요구할 때 디폴트로 id_rsa 이름을 가진 키를 확인한다.
{: .prompt-tip}

