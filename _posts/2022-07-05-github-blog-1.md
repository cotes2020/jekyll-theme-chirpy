---
title: Github 블로그 만들기 - 1. 시작하기
date: 2022-07-05 21:25:00 +0900
lastmod: 2022-07-09 16:45:00 +0900
categories: [Web, GitHub blog]
tags: [GitHub, blog, GitHub blog]
---

> 본 포스팅에서는 아래 내용에 대해 소개합니다.
> - GitHub 블로그가 무엇이며 왜 쓰는지
> - GitHub 블로그를 만들기 위한 GitHub 설정 기초 

# Github 블로그?

인터넷을 검색하다 보면 다양한 블로그를 만나게 됩니다. 네이버 블로그나 티스토리, 브런치 등등 많죠. 그런데 이 친구들과 뭔가 약간 다른 블로그를 보신 적 있지 않으신가요? 잘 설명은 못하겠는데 뭔가 익숙한듯 하면서도 새로운 디자인의(특히 미니멀한) 블로그 말이에요. 혹시 그런 블로그를 지금 보고 계시다면 주소입력창을 다시한 번 보시겠어요? 혹시 `*.github.io` 라고 적혀있나요?

> 네, 그 블로그가 바로 GitHub 블로그입니다.[^1]
{: .prompt-info }

GitHub 블로그는 GitHub 라는 원격 소스코드 저장 플랫폼의 **[GitHub Pages](https://pages.github.com/){:target="_blank"}** 라는 기능을 이용해 만든 블로그를 뜻합니다. 이 기능은 사실 블로그만을 위한 기능은 아니고, 사용하기에 따라 다양한 형태의 웹사이트를 만들고 서비스할 수 있게 해줍니다. 다만 많은 사람들이 블로그 형태로 이용하고 있을 뿐이죠.

# 그걸 왜 쓰는데요?

제가 생각하는 GitHub 블로그의 장점은 쉽게 말하면 기존 블로그 서비스에서 하지 못하는 것을 할 수 있다는 점 입니다. 조금 더 풀어보면 아래와 같습니다.

1. **내 마음대로 만들 수 있다**
<br>무엇보다도 블로그의 테마(스킨) 나 레이아웃부터 폰트, 컬러 등 원한다면(가능하다면) 모든 디자인 요소를 본인의 취향껏 변경할 수 있습니다. 뿐만 아니라 다양한 기능도 추가할 수 있습니다.

2. **GitHub와 연동된다**
<br>GitHub는 개발자들에게는 정말 중요하고 유용한 플랫폼입니다. 본인이 작성한 코드를 관리하는 툴이자 남에게 보여줄 수 있는 포트폴리오이기 때문입니다. 전세계 개발자들이 다 쓰니까 글로벌하기까지 하죠. 때문에 GitHub 블로그를 잘 가꾸는 것도(특히 기술블로그의 경우) 하나의 좋은 포트폴리오가 될 수 있습니다.

3. **광고를 붙일 수 있다**
<br>블로그에 **[Google AdSense](https://www.google.com/adsense/start/){:target="_blank"}**를 링크시킴으로써 광고 수익을 기대할 수 있습니다. 다른 블로그 서비스에서도 일부 지원하지만, GitHub 블로그는 광고 게시 위치를 마음대로 정할 수 있는 등 설정이 더 자유롭습니다.

내 맘대로 꾸미고 돈도 되는데 포트폴리오로서 가치도 있다니 이걸 왜 이제야 알게 되었을까, 혹시 이런 생각이 드셨을 수도 있습니다. 그리고 왜 다들 하지 않지? 라는 생각도요. 저만해도 주변에 블로그를 운영하는 사람은 몇 있지만 GitHub로 만든 블로그를 운영하는 사람은 거의 없습니다. 왜냐하면요.

> **겁.나.어.렵.습.니.다.**
{: .prompt-danger }

내 마음대로 할 수 있다는 건, 마음대로 하기 위해서는 원하는 것을 구현할 수 있어야 한다는 말이기도 합니다. GitHub 블로그에서는 다른 블로그 서비스에서 기본적으로 제공하는 조회수, 댓글 기능들도 모두 셀프로 구현해야 합니다. 그리고 블로그의 생성부터 관리 과정이 모두 GitHub와 연동되기 때문에, 코딩에 대한 이해 뿐만 아니라 git에 대한 이해도 어느 정도 있어야 합니다. 이렇게 공들여 만들고 나서도 검색엔진에 등록하고 **[SEO(Search Engine Optimization)](https://ko.wikipedia.org/wiki/%EA%B2%80%EC%83%89_%EC%97%94%EC%A7%84_%EC%B5%9C%EC%A0%81%ED%99%94){:target="_blank"}**까지 마쳐야만 비로소 사람들에게 읽혀질 수 있습니다.

그럼에도 불구하고, GitHub 블로그가 가진 자유로움에 반해 많은 블로거들이 기존에 사용하던 블로그 서비스 대신 GitHub 블로그를 선택하고 있습니다. 특히 기술적인 장벽이 낮아서 잘 사용하는 소프트웨어 개발자 뿐만 아니라, 전공 특성상 프로그래밍/코딩과 친숙한 분야(빅데이터, 통계, AI 등)에 계신 분들도 GitHub 블로그를 많이 사용합니다.

있어보이지 않나요? 네, 그래서 이번 포스팅의 주제이기도 합니다. 지금부터는 GitHub 블로그를 만들기 전에 필요한 GitHub 설정에 대해 간단히 알아보도록 하겠습니다.

# GitHub 설정하기

블로그건 홈페이지건, 웹사이트를 운영하기 위해서는 인터넷상에 저장 공간이 있어야 합니다. 저장 공간에 웹페이지를 업로드해두고 외부에서 접근할 수 있게 주소(URL)를 지정해주면, 방문자가 주소를 통해 그 저장 공간에 접근해 웹페이지를 다운로드 받아서 Chrome이나 Internet Explorer(R.I.P) 같은 브라우저를 통해 보는 것이죠. 

GitHub는 웹사이트를 개설할 수 있는 저장 공간과 주소를 제공합니다. 아래 순서대로 따라해 보세요.

## 1. GitHub 회원가입하기

**[GitHub](https://github.com/){:target="_blank"}**에서 <kbd>Sign up for GitHub</kbd>를 눌러 회원가입 절차를 시작합니다.

![](/assets/img/2022-07-05/2022-07-05-github-blog-1-github_main.png)*우주여행을 떠날 것만 같은 GitHub 메인*

이메일과 비밀번호 등을 입력하고 나면 Launch code를 입력해야 합니다.

![](/assets/img/2022-07-05/2022-07-05-github-blog-1-github_verification.png)

가입시 입력한 이메일의 받은 메일함에서 Launch code를 확인하셔서 입력하시고 간단한 설문조사를 마치고 나면 회원가입 완료! GitHub Sign up 절차는 언제봐도 깔끔하고 멋지네요.

## 2. Repository 생성하기

웹사이트의 소스 코드를 저장할 공간을 만들어 보겠습니다. Repository는 프로젝트의 소스코드를 관리하기 위한 저장소입니다. 

메인화면 좌측 상단의 <kbd>Create repository</kbd>를 클릭하고 각 항목을 아래와 같이 입력합니다.

![](/assets/img/2022-07-05/2022-07-05-github-blog-1-create_repository.png)

- **Repository name** : `USERNAME.github.io`를 입력합니다.
- **Public/Private** : 저장소를 타인에게 공개할지 비공개할지 설정합니다. 여기서는 Public을 선택합니다. 
- **Add a README file** : 저장소를 생성할 때 `README.md` 파일을 같이 만들지 여부입니다. 여기서는 ✅합니다.

그 외 다른 설정들이 있으나 건드리지 않고 넘어가겠습니다. 화면 맨 아래 <kbd>Create repository</kbd> 버튼을 눌러 저장소를 생성합니다.

## 3. Git clone 하기

저장소를 처음 만들면 `README.md` 파일만 덩그러니 있습니다. 이제 이 저장소에서 GitHub 블로그를 만드는데 필요한 코드를 관리할텐데요, 인터넷 환경보다는 본인의 PC에서 작업하는 게 더 편하기 때문에 원격 저장소와 로컬 PC를 연결하고 원격 저장소의 데이터를 로컬로 복사해오는 과정이 필요합니다. 이를 clone 이라고 합니다.

![](/assets/img/2022-07-05/2022-07-05-github-blog-1-github_clone.png)

GitHub 저장소 화면의 Code 버튼을 누르면 그림과 같이 지금 저장소의 데이터를 복사하기 위한 URL이 표시됩니다. URL을 복사하고 명령 프롬프트를 실행합니다.

> 명령 프롬프트는 <kbd>윈도우 키</kbd>를 누르시고 `cmd`를 입력해 실행하실 수도 있습니다.
{: .prompt-tip }

아마 `C:\Users\사용자명>` 처럼 표시될텐데, 저 경로의 폴더를 열어놓은 상태라고 이해하시면 됩니다. 여기서 화면에 컴퓨터가 이해할 수 있는 명령을 적으면 실행되는 것이죠.(이름이 명령 프롬프트인 이유!) git 폴더를 만들고 `git clone` 명령을 입력해 보겠습니다.

```console
C:\Users\kiyun>mkdir git
C:\Users\kiyun>cd git
C:\Users\kiyun\git>git clone https://github.com/ita-bility/ita-bility.github.io
'git'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는 배치 파일이 아닙니다.
```

git이 무슨 의미인지 모르겠다는 건데요, PC에 **[Git](https://git-scm.com/){:target="_blank"}**이 설치되어 있지 않아서 컴퓨터가 명령을 이해하지 못한 것입니다. 우선 Git을 설치해야겠네요.

> Git을 아직 설치하지 않으셨다면 **[여기](https://tired-o.github.io)**를 참고하셔서 설치 후 이어서 진행해주세요.
{: .prompt-warning }

git이 설치되어 있다면 아래와 같은 내용이 나올 겁니다.
```console
C:\Users\kiyun\git>git clone https://github.com/ita-bility/ita-bility.github.io
Cloning into 'ita-bility.github.io'...
remote: Enumerating objects: 3, done.
remote: Counting objects: 100% (3/3), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
Receiving objects: 100% (3/3), done.
```

그리고 해당 경로로 가보면 파일이 잘 복사되어 있는걸 확인할 수 있습니다.

## 4. 로컬 Git에 새로운 파일 생성하기

로컬 Git에 새로운 파일을 만들어 보겠습니다. 폴더 안에 새 텍스트 파일을 만들고 **[Hello world](https://ko.wikipedia.org/wiki/%22Hello,_World!%22_%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8){:target="_blank"}**를 입력한 뒤 저장합니다. 파일명은 `Hello world.html`로 하겠습니다. 그리고 파일 확장자를 `txt`에서 `html`로 바꿔줍니다. 

![](/assets/img/2022-07-05/2022-07-05-github-blog-1-create_html.png)*확장명을 변경해도 사용할 수 있으니 바꿔주세요.* 

파일을 실행하면 웹브라우저에 Hello world가 떠있는 화면이 보이실 겁니다.

## 5. GitHub에 수정사항 반영하기

`Hello world.html` 파일을 만들었지만 로컬 PC에만 만들어졌을 뿐입니다. 작업한 내용을 GitHub에 저장하기 위해 아래 순서대로 진행해 보겠습니다.

1. 변경사항 저장
2. 변경사항 확정
3. 원격 저장소에 업로드

### 1. 변경사항 저장 

변경사항을 저장하기 위해 명령 프롬프트로 돌아가 아래와 같이 입력합니다.

```console
C:\Users\kiyun\git>cd ita-bility.github.io
C:\Users\kiyun\git\ita-bility.github.io>git add -A
```

변경사항을 staging(저장을 확정하기 전 중간 단계)에 저장하되, 변경된 모든(`-A`, All) 내용을 저장한다는 명령어입니다.

### 2. 변경사항 확정 

앞서 저장한 변경사항을 그대로 올릴 것이니 확정하겠습니다.

```console
C:\Users\kiyun\ita-bility.github.io>git commit -m "first commit"
[main af26d4c] first commit
 1 file changed, 1 insertion(+)
 create mode 100644 Hello world.html
```

`commit`은 변경사항을 확정하는 것으로 변경사항에 대한 커멘트를 남겨(`-m "메시지"`) 나중에 무엇때문에 변경사항이 발생했는지 보기 쉽게 해줍니다. 

### 3. 원격 저장소에 업로드

이제 `push` 명령어로 GitHub 저장소에 업로드 하겠습니다.

```console
C:\Users\kiyun\ita-bility.github.io>git push
info: please complete authentication in your browser...
```

업로드가 바로 될 줄 알았는데, 아래 화면처럼 팝업창이 뜹니다. <kbd>Sign in with your browser</kbd>를 눌러 GitHub에 로그인합니다.

![](/assets/img/2022-07-05/2022-07-05-github-blog-1-git_push_login.png)

로그인을 하고 나니 추가 인증이 필요합니다. <kbd>Authorize GitCredentialManager</kbd>를 눌러 Git Credential Manager에 권한을 줍니다.

![](/assets/img/2022-07-05/2022-07-05-github-blog-1-git_push_oauth.png)

Authorization confirm을 위해 GitHub 비밀번호를 다시 입력하면 Authentication Succeeded 화면이 뜨고 명령 프롬프트에서도 업로드가 완료된 것을 확인하실 수 있습니다.

![](/assets/img/2022-07-05/2022-07-05-github-blog-1-git_push_oauth_success.png)*인증 완료 화면*

```console
C:\Users\kiyun\ita-bility.github.io>git push
info: please complete authentication in your browser...
Enumerating objects: 4, done.
Counting objects: 100% (4/4), done.
Delta compression using up to 4 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 294 bytes | 147.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/ita-bility/ita-bility.github.io.git
a17b2cb..af26d4c  main -> main
```

GitHub 저장소에도 `Hello world.html` 파일이 추가된 것을 확인할 수 있습니다.

![](/assets/img/2022-07-05/2022-07-05-github-blog-1-github_update.png)*Hello world.html 파일이 "first commit" 메시지와 함께 추가됐네요.*

# What's next
이렇게 해서 GitHub에 저장 공간을 만들고, `USERNAME.github.io` 라는 주소를 갖는 블로그를 개설할 준비를 마쳤습니다. 다음 포스팅에서는 블로그에 적용할 테마를 찾아보고 적용하는 방법에 대해 소개해 드리겠습니다.

---

[^1]: Custom domain을 적용한 경우 다를 수 있지만 `github.io` 형식을 그대로 사용하는 경우가 많으므로 이해를 돕고자 위와 같이 표현했습니다.