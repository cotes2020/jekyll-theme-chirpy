---
title: .gitignore 파일 설정 [gitignore.io]
date: 2023-07-09 07:24:00 +0900
author: kkankkandev
categories: [Git & Github]
tags: [git add, git commit, git push, git, github, linux, alias, gitignore, .gitignore, gitignore.io]     # TAG names should always be lowercase
comments: true
---

> **git을 사용해 프로젝트 폴더나, 알고리즘 파일을 업로드 해야 할 때 업로드될 필요가 없는 파일까지 같이 업로드 되는 경우가 있습니다.**
> 
> **이러한 문제를 해결하기 위해 .gitignore파일을 수정하게되는데 IDE마다 gitignore서식을 다르게 해주는데 번거로움이 있었습니다.**
>
> **간단하게 .gitignore파일을 설정 할 수 있는 방법을 찾다가 제가 찾은 방법을 공유합니다.**
    
# 1. [.gitignore] 파일이란?
>  **Git 저장소에서 추적하지 않을 파일 또는 디렉토리를 지정하는 데 사용되는 설정 파일**


# 2. [.gitignore] 파일 규칙 [# => 주석]

```
# 특정 파일 무시
파일명.txt

# 특정 확장자 무시
*.확장자

# 특정 디렉토리 무시
디렉토리명/

# 경로를 기준으로 특정 파일 무시
상위디렉토리/특정파일.txt

# 특정 패턴으로 시작하는 파일 무시
특정패턴*
```

# 3. [gitignore.io] 홈페이지 사용법
> 1.[gitignore.io](https://gitignore.io) 홈페이지에 들어간다.

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/10da090c-58f9-4cba-a431-7449611702e5)
---
> 2.아래 Text를 입력하는 곳에 운영 체제, 개발 환경, 프로그래밍 언어를 기입하고 생성 버튼을 누른다   

![image-1](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/e63013d7-3093-4c9a-bb70-bef2792bfacf)
---
> 3.넘어간 페이지에서 Ctrl + A 혹은 Command + A를 눌러 소스코드를 전체 복사한 뒤 .**gitignore** 파일에 추가해준다

![image-2](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/3fd3da45-c3ce-4fdf-bf24-4c7e3aaedce9)
