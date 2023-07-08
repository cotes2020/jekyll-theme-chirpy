---
title: git add commit push 한번에 간편하게 하는 방법
date: 2023-07-09 07:24:00 +0900
author: kkankkandev
categories: [Git & Github]
tags: [git add, git commit, git push, git, github, linux, alias, git alias]     # TAG names should always be lowercase
comments: true
pin: true
---

> git을 사용하여 github에 코드를 올리는 작업을 하던 도중 ```git add```, ```git commit```, ```git push``` 총 3가지 명령어를 계속 입력하는데 번거로움이 있었습니다. 
> 이 세 가지 명령어를 해결하는 방법을 공유합니다
> 
> alias라는 기능을 사용해서 다른 명령어 조합도 간단하게 입력할 수 있습니다


# git add commit push 한번에 하는 방법
> ## **Terminal에 아래의 명령어 입력.**

```
git config --global alias.cmp '!f() { git add -A && git commit -m "$@" && git push; }; f'
```

# 사용법
```
git cmp "Message(커밋 메시지)"
```

# 동작 원리 (git alias)
> **cmp라는 alias를 전역 설정 --global에 추가한다**
> 
> **f() 셸 함수의 중 괄호 안에 git add, git commit, git push를 한번에 할 수 있게 구현 되어 있다.**
   
       
          


Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
