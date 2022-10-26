---
title: "[Git Error] fatal: in unpopulated submodule ... 해결하기"
author: cotes
categories: git
tag: [error]
math: true
mermaid: true
---

# Submodule Error

## ❓ 에러 발생 원인
***
> 상위폴더가 있고 상위 폴더 안에 파일들이 있는데 상위폴더에도 `git init`을 하고 
> 안에 파일에서도 `git init`을 하게 되면 파일을 submodule로 인식
> 깃 허브에 화살표가 있는 디렉토리가 생성되고 클릭도 안됨....😥

## 🚀 해결법
***
### 1️⃣ 첫 번째 해결법 - .git 파일 삭제

로컬 저장소로 설정된 디렉토리를 취소(`git init` 취소)해야 한다. `.git`은 숨김폴더이므로 `ls -a`를 통해 확인할 수 있다.

나의 경우에는 다음 명령어로도 지워지지 않아서 `-r` 대신 `-rf`를 붙이니 지워지고 브랜치가 `master`에서 `main`으로 바뀌었다.

```powershell
$ rm -r .git
```

그리고 다시 해보면 여전히 안된다!🙃🔫

```powershell
$ git add README.md
fatal: in unpopulated submodule '2022/202204/20220422'
```

### 2️⃣ 두 번째 해결법 - cache 삭제

그래도 이번엔 에러 메시지가 나와서 찾아보니 외부에 `.git`이 있는데 내부에 또 `.git`이 있으면 서브모듈(submodule)로 인식하고 커밋이 추가가 안된다고 한다.

`.git` 오류를 일으키는 하위 파일들을 모두 삭제하기 위해 해당 디렉토리에서 다음 명령어를 입력한다.

```powershell
$ git rm --cached . -rf
```

그리고 다시 푸시하면 정상적으로 잘 업로드가 된 걸 볼 수 있다!