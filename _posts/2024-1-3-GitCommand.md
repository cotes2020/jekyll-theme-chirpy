---
title: Git Command
date: 2024-1-3 03:05:01 +0900
categories: [git]
tags: [git, freq]
math: true
---

자주 쓰는 git 명령어들에 대해 정리해두겠다.

## git switch
```
$ git switch feat/first
'feat/first' 브랜치로 전환한다.
```
```
$ git switch -c feat/new
새로 만든 'feat/new' 브랜치로 전환한다.
```

## git restore
```
$ git restore app.js
app.js 파일 수정한 것을 복원시킨다.
```
```
$ git restore --staged app.js
이미 git add 로 staged 된 app.js 파일 수정사항 들을 복원시킨다.
```

## git merge
```
$ git merge feat/first
현재 브랜치(만약 master)으로 feat/first의 commit 들을 merge 시킨다.
```

## my routine

```
$ git status
$ git add .
$ git commit -m "feat: blahblah"
$ git push origin feat/first(현재 branch)
```

## 원격 저장소 branch들 가지고 오기
```
$ git remote update
$ git branch -a
$ git switch -c (new branch) origin/branchname
origin/branchname의 commit 을 들고있는 로컬 브랜치 (new branch)를 만든다.
```

## 로컬 branch 지우기
```
$ git branch -d (branchname)
```