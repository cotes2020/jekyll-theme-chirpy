---
title: Git 설치하기
date: 2022-07-09 16:05:00 +0900
categories: [Web, Git]
tags: [VCS, 버전 관리, Git, GitHub, 리누스 토르발즈]
---

> 본 포스팅에서는 아래 내용에 대해 소개합니다.
> - 버전 관리 시스템에 대한 이해
> - Git 설치하기

# Git?

**[Git](https://git-scm.com/)**은 리눅스의 아버지인 **[리누스 토발즈](https://ko.wikipedia.org/wiki/%EB%A6%AC%EB%88%84%EC%8A%A4_%ED%86%A0%EB%A5%B4%EB%B0%9C%EC%8A%A4){:target="_blank"}**가 최초로 개발한 버전 관리 시스템(VCS, Version Configuration System)입니다. Git은 가장 널리 쓰이는 VCS로, Git을 이해하기 위해 버전 관리에 대한 이야기를 먼저 해볼까 합니다.

> VCS에 대한 설명을 건너뛰고 바로 Git 설치 가이드를 보시려면 **[여기](#git-설치하기)**를 눌러주세요.
{: .prompt-info }

## 버전 관리의 필요성

개발을 하다 보면 검증 단계에서든 릴리즈 후든 언제든지 이슈가 생길 수 있습니다. 사용중인 패키지가 업데이트 되면서 발생하는 호환성 문제라든지 충분히 검증되지 않은 수정사항이 반영되었다가 출시 후에 시장 이슈가 발생한다든지 말이죠. 재빨리 패치를 해서 문제점을 수정할 수도 있겠지만, 일단은 문제가 없었던 이전 버전으로 돌려놓고 제대로 준비한 뒤에 제품을 업데이트 할 수도 있습니다.

그런데 제품을 급하게 개발하느라 코드를 백업해 놓지 않았다면, 그리고 이전 코드 위에 새로운 코드를 적어버렸다면? 이전 버전으로 되돌아 갈 방법이 없겠죠? 그래서 제품을 출시하기 전에는 변경사항이 생길 때마다 코드를 백업해 두기로 했습니다.

코드를 백업하면서 버전관리를 하니 한 결 낫긴 한데, 문제점을 발견하게 됩니다. 파워포인트로 조별과제를 하는 상상을 해봅시다. 

1. 우리는 신제품 이미지와 시연 영상까지 포함한 100장 짜리 제안서를 쓰고 있습니다. 갑자기 수정할 게 생각나서 장표 하나를 추가하고 새로운 버전의 파일을 만들었습니다.

2. 그리고 여러 사람이 각자 파트를 업데이트 하다보니 어느 순간 각자 버전을 만들게 됩니다. 그래서 누락된 수정 사항이 생겨버렸고, 결국 제가 추가한 장표가 최종 발표에서는 빠져버렸습니다.


![](/assets/img/2022-07-08/2022-07-08-install-git-vcs_bad_example.png)*버전관리의 나쁜 예*


1에서 나타난 문제는 저장 공간의 비효율입니다. 제안서가 1GB였다면, 단 하나의 장표를 업데이트 하기 위해 2GB의 저장 공간을 사용한 셈입니다. 엄청난 낭비죠. 2의 문제는 협업의 어려움입니다. 누가 어떤 내용을 언제 반영했는지 서로에게 잘 공유될 수 없었기 때문에 수정사항이 모두 반영되지 못했습니다. 이런 문제를 해결해 준 시스템이 바로 버전 관리 시스템입니다. 

## 버전 관리 시스템의 기능

버전 관리 시스템은 소프트웨어 개발을 위해 다음 기능들을 제공합니다.

1. **변경 이력 관리**
<br>버전 관리 시스템은 소스 코드의 모든 버전을 파일 형태로 저장하지 않고, 변경사항을 저장합니다. 그래서 스토리지 리소스를 효율적으로 사용할 수 있게 해주며 소스 코드가 변경된 이력을 쉽게 확인할 수 있습니다.

2. **백업 & 복원**
<br>코드를 저장하고 코드에 문제가 발생했을 경우 복구할 수 있습니다.

3. **협업 기능**
<br>협업자끼리 충돌하는 변경 사항에 대해 알려주며 협업자간 변경사항을 공유하기 쉽게 해줍니다.

버전 관리 시스템은 사실 Git만 있는게 아닙니다. 기업 환경에 따라 Apache Subversion, Mercurial 등 다른 VCS를 사용하는 경우도 있습니다. 다만 ~~제가 Git만 써봤기 때문에~~ Git을 가장 보편적으로 사용하기 때문에 Git 설치 방법에 대해 말씀드리겠습니다.

# Git 설치하기

## 1. Git 다운로드

Git은 **[여기](https://git-scm.com/downloads)**에서 무료로 다운로드 할 수 있습니다. <kbd>Download for Windows</kbd>를 눌러 설치파일을 다운로드 받습니다. 윈도우는 32-bit, 64-bit Installer 선택 화면이 나오는데 <kbd>Click here to download</kbd>를 클릭하시면 됩니다.

![](/assets/img/2022-07-08/2022-07-08-install-git-download.png)

## 2. Git 설치파일 실행

Git 설치파일을 실행하시면 설치 전 설정 메뉴들이 나타납니다. 설정이 아주 많은 편이지만(License Agreement부터 설치 완료 화면까지 무려 16개!) 복잡한 프로그램이라서라기보다는 그만큼 개발자가 원하는 설정을 할 수 있게 하기 위한 배려라고 생각하시고 설치를 시작해봅시다.

## 3. Git 설치 설정

결론부터 말씀드리면, **설치 중 아무런 설정도 변경하지 않았습니다.**

Git Installer에 설정된 기본값은 가장 보편적인 설정입니다. Git에 대한 이해가 높지 않은 상황에서 설정을 바꾸는 것은 의도하지 않은 결과를 불러올 수 있습니다. 그러므로 Git을 처음 쓰신다면 기본 설정으로 설치해 사용하시고, 이후 필요하실 때 설정값을 바꾸시길 추천드립니다. 

> 설치중 **(NEW!)**는 새롭게 제안하는 설정으로 Git에 익숙하지 않으시다면 체크하지 않으셔도 무방합니다. 반대로 **(Recommended)**는 체크하는 것을 권장합니다.
{: .prompt-tip }

주요 설정에 대해서는 아래 설명을 참고하세요.

### 3-1. Git 설치 설정 - 구성요소

여기서는 Git 설치시 함께 설치할 구성요소의 범위를 결정합니다.

![](/assets/img/2022-07-08/2022-07-08-install-git-install3.png)

- Additional icons
  - ⬜On the desktop : 바탕화면에 바로가기 아이콘을 생성하지 않습니다.
- Windows Explorer integration
  - ✅**Git Bash Here** : 폴더 우클릭시 <kbd>Git Bash Here</kbd> 메뉴를 추가합니다.
  - ✅**Git GUI Here** : 폴더 우클릭시 <kbd>Git GUI Here</kbd> 메뉴를 추가합니다.
- ✅**Git LFS (Large File Support)** : 대용량 파일을 지원합니다.
- ✅**Associate .git* configuration files with the default text editor** : 기본 텍스트 에디터(메모장, vim, nano 등)에서 `*.git` 파일을 열 수 있도록 연결합니다.
- ✅**Associate .sh files to be run with Bash** : Git Bash에서 쉘 스크립트 파일(`*.sh`)이 동작하도록 연결합니다.
- ⬜Check daily for Git for Windows updates : Git 업데이트 사항을 매일 확인하지 않습니다.
- ⬜(NEW!) Add a Git Bash Profile to Windows Terminal : 윈도우 터미널에 Git Bash 프로파일을 추가하지 않습니다.

> **Git Bash** : CLI(Command Line Interface) 기반의 쉘 프로그램으로 Unix 명령어를 사용할 수 있습니다.
<br>**Git GUI** : 알기 쉬운 그래픽 기반의 인터페이스로 마우스를 이용해 Git을 조작할 수 있습니다.
{: .prompt-info }

### 3-2. Git 설치 설정 - 기본 에디터 설정

여기서는 어떤 에디터를 Git 기본 에디터로 설정할지 결정합니다.

![](/assets/img/2022-07-08/2022-07-08-install-git-install5.png)

 Vim, VS Code, Atom 등 여러 에디터를 설정할 수 있습니다. 사용중인 에디터가 있다면 그 에디터를 기본 에디터로 설정하셔도 되고, 없다면 **Vim**(powerful, ~~can be hard to use~~)을 설정하시면 됩니다.

### 3-3. Git 설치 설정 - 기본 브랜치명 설정

여기서는 새 저장소를 생성할 때 branch 이름을 지정하는 방법을 결정합니다.

![](/assets/img/2022-07-08/2022-07-08-install-git-install6.png)

- ✅**Let Git decide** : 브랜치명을 **master**로 지정합니다.
- ⬜Override the defult branch name for new repositories : 빈 칸에 입력하는 이름으로 브랜치명을 지정합니다. 설정할 경우 보통 **main**을 씁니다.

## 4. Git Bash 실행

설정이 끝나고 Git이 설치되면 아래와 같이 설치 완료 화면이 나옵니다. ✅**Launch Git Bash**를 선택해 Git Bash를 실행합니다.

![](/assets/img/2022-07-08/2022-07-08-install-git-install10.png)

Git bash는 명령 프롬프트와 비슷해 보이지만 Unix 명령어를 사용할 수 있고 Git 조작에 도움을 주는 기능을 포함하고 있습니다. 대표적인 Unix 계열 명령어인 `ls`를 Git Bash와 명령 프롬프트에 각각 입력해보겠습니다.

![](/assets/img/2022-07-08/2022-07-08-install-git-git_bash_and_cmd.png)*Git Bash 실행결과(위), 명령 프롬프트 실행결과(아래)*

Git Bash에서는 `ls`가 실행되어 디렉토리 리스트를 보여주지만 명령 프롬프트에서는 `ls` 명령어를 인식하지 못하는 것을 알 수 있습니다. 

## 5. Git 사용자 이름, 이메일 등록

Git은 협업을 전제로 한 버전 관리 도구입니다. 변경 사항을 누가 반영했는지 기록하기 위해 Git을 본격적으로 사용하기 전 사용자 이름과 이메일을 등록하겠습니다.

```shell
$ git config --global user.name "USERNAME"
$ git config --global user.email "USEREMAIL@domain.com"
```

`USERNAME`은 사용하고 싶은 이름을 입력하시면 됩니다. GitHub와 연동하여 사용할 예정이라면, `USEREMAIL@domain.com`에 GitHub에 회원가입한 이메일을 입력하시면 로컬 PC에서 GitHub로 Push할 때 해당 이메일을 사용하는 GitHub 사용자 계정과 연결됩니다.
