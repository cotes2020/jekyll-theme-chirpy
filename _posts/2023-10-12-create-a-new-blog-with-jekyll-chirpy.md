---
title: Jekyll Chirpy를 활용한 새 블로그 만들기
author: nahjh2
date: 2023-10-12 20:00:00 +0900
categories: [Blog, Create a new blog]
tags: [jekyll, chirpy, GitHub Pages, github.io, blog, giscus, slack bot]
render_with_liquid: false
---

평소 사이드 프로젝트를 진행 중이던 저희는 가끔식 문제를 어떻게 해결했는지를 공유하곤 했습니다.
하다 보니 좋은 것 같아 여기서 멈추지 않고, 개발을 하면서 겪은 이슈들을 정리해서 저희끼리 발표해 보고, 이를 정리해 블로그로 만들어 보기로 했습니다.

이 글은 저희가 블로그를 만들기로 한 뒤 어떻게 블로그를 세팅했는지를 정리한 글입니다.

# Jekyll의 Chirpy 테마를 사용한 이유

처음 블로그를 작성하려 했을 때 후보군으론 Medium, Notion, Velog, Tistory, Github pages 등이 있었고
이중 디자인이 깔끔하고 글 자체에 포커스를 둔 Medium이 저희의 눈길을 끌었습니다.
하지만 Medium의 Team 계정은 달에 5$를 내야 해 저희끼리 가볍게 운영할 생각이었던 저희는 굳이 지출을 만들고 싶지 않았습니다.

그러던 중 저희는 Jekyll의 Chirpy라는 테마를 보게 되었고, 깔끔한 디자인에 커스텀 마이징이 가능해 Chirpy를 사용하기로 결정했습니다.

## Jekyll 특징

Jekyll은 Ruby로 작성된 정적 사이트 생성기 중 하나로서, 마크다운 기반의 문서를 기반으로 웹사이트를 생성해 줍니다.
정적 사이트이기 때문에 매우 가볍고 SEO에 유리하며, Git을 통해 글을 관리하기 용이합니다.
또한 Github Action을 제공하여 Github pages CI/CD가 쉽습니다.

개발자들은 마크다운과 Git에 익숙해 조금만 해봐도 바로 적응하기 쉽습니다.

# Jekyll Chirpy 시작하기

> There are two ways to create a new repository for this theme:
>
> - Using the Chirpy Starter - Easy to upgrade, isolates irrelevant project files so you can focus on writing.
> - GitHub Fork - Convenient for custom development, but difficult to upgrade. Unless you are familiar with Jekyll and are determined to tweak or contribute to this project, this approach is not recommended.

[공식 문서](https://chirpy.cotes.page/posts/getting-started/#creating-a-new-site)를 보면 Starter를 사용하는 방법과 Fork를 사용하는 2가지 방법을 알려주는데, 이중 블로그를 커스텀하기에 유리한 Fork 방식으로 작업을 해보겠습니다.

> 커스텀 마이징이 필요하지 않다면 Starter 방식을 적극 권장합니다.
> 힘들게 설치하고 스크립트 돌리고 초기화할 필요 없이 바로 글 작성이 가능하기 때문입니다.
> {: .prompt-tip }

## 작업환경 구축하기

일단 시작하기에 앞서 Jekyll을 사용하기 위해 Ruby와 Jekyll을 설치해야 합니다.

> 필자는 우분투 환경을 기준으로 설명하며 자신의 OS에 맞는 설명은 [공식 문서](https://jekyllrb-ko.github.io/docs/installation/)를 참고하면 됩니다.
> {: .prompt-warning }

### Ruby 설치하기

우분투 환경에선 apt-get을 이용해 Ruby와 디펜던시들을 설치할 수 있습니다.

```shell
sudo apt-get install ruby-full build-essential zlib1g-dev
```

### Jekyll 설치하기

Ruby를 설치했다면 gem을 이용해 Jekyll을 설치할 수 있습니다.

```shell
gem install jekyll bundler
```

### Node.js 설치하기

추후 Chirpy의 init 스크립트를 사용하기 위해서 [Node.js](https://nodejs.org/ko)가 설치되어 있어야 합니다.

우분투 환경에서는 [nodesource](https://github.com/nodesource/distributions)를
참고해서 다운받으면 됩니다.

```shell
# Download and import the Nodesource GPG key
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

# Create deb repository
NODE_MAJOR=20
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list

# Run Update and Install
sudo apt-get update
sudo apt-get install nodejs -y
```

> 다른 OS의 경우 https://nodejs.org/ko/download 에서 다운로드 받으시면 됩니다.
> {: .prompt-tip }

## Chirpy fork 받기

작업환경을 구축했다면 이제 Chirpy의 공식 Repo를 Fork 받으면 됩니다.

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/fork-chirpy.png)
_https://github.com/cotes2020/jekyll-theme-chirpy/fork_

이때 Github Pages를 이용해 호스팅할 것이기에 Repo의 이름을 _USERNAME.github.io_ 로 지정해야 합니다.

> https://docs.github.com/en/pages/quickstart#creating-your-website
> {: .prompt-info }

## Chirpy 초기화 하기

Fork 받은 내용들은 Chirpy를 개발하는 데 사용되는 파일들이 전부 들어있으므로 블로그를 운영하기 위한 환경이 아닙니다. 이를 세팅해 주기 위해 init 스크립트를 실행시켜 줘야 합니다.

Init을 하기 위해 먼저 Repo를 Clone 해줍니다.

```shell
git clone https://github.com/vivace-log/vivace-log.github.io
```

> Repo 주소는 본인이 fork 받은 주소로 해야 합니다.
> {: .prompt-warning }

이후 Chirpy에서 제공하는 init 스크립트를 실행시키면 됩니다.

```shell
bash tools/init
```

스크립트를 실행시키면 최신 릴리즈로 코드를 갱신하고, 불필요한 파일들을 제거하며, JS 파일들을 빌드한뒤 커밋을 진행합니다.

### 로컬에서 테스트해 보기

준비가 제대로 됐는지 확인해 보기 위해서 로컬에서 테스트해 보겠습니다.

로컬에서 테스트해 보기 위해선 디펜던시를 설치해 줘야 합니다.

```shell
bundle
```

설치가 다 되었다면 이제 로컬로 서버를 띄어볼 수 있습니다.

```shell
bundle exec jekyll s
```

## 첫 배포 하기

정상적으로 화면이 잘 나온다면 이번에는 배포해 볼 차례입니다.
기본적으로 Github Action이 설정이 되어있어 단순히 Repo에 push만 해도 배포가 됩니다.

배포를 진행하는 Github Action은 `.github/workflows` 에 있는 파일을 통해 확인이 가능합니다.
Chirpy는 `.github/workflows/pages-deploy.yml`에 명시가 되어있는 데로 배포를 진행하게 되는데 간단히 봐서 우리가 로컬에서 띄어보기 위해 진행한 과정을 진행한다고 생각하면 됩니다. 또한 진행 과정을 Repo의 Actions 탭에서 확인이 가능합니다.

```shell
git push --force
```

![Github Action의 작업 결과](/assets/img/create-a-new-blog-with-jekyll-chirpy/github-action.png)
_Github Action의 작업 결과_

![USERNAME.github.io의 주소로 호스팅이 된 모습](/assets/img/create-a-new-blog-with-jekyll-chirpy/first-deploy.png)
_USERNAME.github.io의 주소로 호스팅이 된 모습_

이제 우리만의 블로그가 첫발을 내딛은걸 볼 수 있습니다.
여기에 설정 좀 하고 글을 쓰게 되면 저희만의 블로그가 완성됩니다.

## Jekyll Chirpy 설정하기

Chirpy의 기본적인 설정은 \_config에서 설정할 수 있습니다.
\_config 파일을 확인해 보게 되면 설정값들이 무엇을 의미하는지 주석이 달려 있으며,
timezone, title, tagline, description, url, github, social 등을 설정하면 됩니다.

```yml
lang: en

timezone: Asia/Seoul

title: Vivace Log # the main title

tagline: The record of our journey # it will display as the sub-title

description: >- # used by seo meta and the atom feed
  Vivace log, the record of our journey.

# fill in the protocol & hostname for your site, e.g., 'https://username.github.io'
url: "https://vivace-log.github.io/"

github:
  username: vivace-log # change to your github username

social:
  # Change to your full name.
  # It will be displayed as the default author of the posts and the copyright owner in the Footer
  name: vivace-log
  email: log.vivace@gmail.com # change to your email address
  links:
    - https://github.com/vivace-log # change to your github homepage
```

## 첫 번째 글 작성하기

글을 작성할 때는 파일을 \_posts 디렉토리에 `yyyy-mm-dd-제목.md`의 형식으로 만들어야 합니다.
이때 제목은 영어로, 띄어쓰기는 - 로 작성해야 합니다.

파일을 만들었다면 처음에 글의 정보를 적어준 뒤 `---`로 감싸고, 본문을 마크다운 형식으로 작성해 주면 됩니다.

```
---
title: First Post
date: 2023-10-07 23:25:00 +0900
categories: [Test, First-Post]
tags: [test_tag, first-post]
---

First Post!
```

{: file="\_posts/2023-10-07-first-post.md" }

> 더 자세한 사용법들은 [공식문서](https://chirpy.cotes.page/posts/write-a-new-post/)를 참고하세요
> {: .prompt-tip }

작성후 Github에 push 하면

```shell
git add .
git commit -m "Add comments"
git push
```

![Github Action Summary](/assets/img/create-a-new-blog-with-jekyll-chirpy/github-action.png)

Github Action에서 빌드를 하고 배포까지 해줍니다.

![blog home](/assets/img/create-a-new-blog-with-jekyll-chirpy/first-post-home.png)
![first post](/assets/img/create-a-new-blog-with-jekyll-chirpy/first-post.png)

블로그의 첫 번째 글이 성공적으로 작성되었습니다!

# 댓글 기능 추가하기

이제 블로그의 모양새를 갖췄으니, 독자들의 피드백을 받을 수 있도록 댓글 기능을 추가해 보도록 해봅시다.

아마 \_config을 자세히 봤다면 comments 설정이 있었던 걸 볼 수 있었을 것 입니다.
Chirpy는 disqus, utterances, giscus를 기본적으로 제공하는데 이중 giscus를 이용해 댓글 기능을 추가해 보겠습니다.

## Giscus 란?

Giscus는 깃허브의 Discussions 기능을 댓글처럼 사용하는 앱입니다.

## Giscus 설정하기

1. giscus 앱 설치하기

![giscus.app/ko](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus.app.png)
_https://giscus.app/ko_

에 접속해서 2. giscus 앱이 설치되어 있어야 합니다. 를 클릭하여

![github.com/apps/giscus](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-install-page.png)
_https://github.com/apps/giscus_

설치 페이지로 이동합니다.

![giscus target group](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-install-target.png)

설치할 그룹을 정한 뒤

![giscus target repo](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-install-repo.png)

github page를 호스팅할 repo를 지정합니다.

이제 2번에 나와있는 Repo에 Discussions 기능 활성화 하기 위해, 해당 repo의 설정에 들어가서 discussions를 체크합니다.

![repo discussions setting](/assets/img/create-a-new-blog-with-jekyll-chirpy/set-discussions.png)

2개를 세팅한 다음에

![giscus ready](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-passed.png)

이렇게 모든 조건을 만족합니다가 뜨면 준비 완료입니다.

![giscus-script](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-script.png)

창을 좀더 내려서 data-repo-id와 data-category-id 부분을 복사해서
\_config 파일에 아래처럼 수정해줍니다.

```yml
comments:
  active: giscus
  giscus:
    repo: vivace-log/vivace-log.github.io # <gh-username>/<repo>
    repo_id: R_kgDOKdKNnw # data-repo-id
    category: Announcements # category
    category_id: DIC_kwDOKdKNn84CZ8ZG # data-category-id
```

설정을 마치고 다시 깃허브에 올리면

```shell
git add .
git commit -m "Add comments"
git push
```

![post-with-comments](/assets/img/create-a-new-blog-with-jekyll-chirpy/post-with-comments.png)

이렇게 댓글 기능을 사용할 수 있게 됩니다.

## Slack Bot을 활용한 댓글 알림

Giscus는 깃허브의 discussions을 활용하기에 Github Action을 활용하여 Slack으로 알림을 받아볼 수 있습니다.

### Slack Bot 만들기

https://api.slack.com/apps 에 들어가서 Create an App을 누릅니다.
![craete-slack-bot](/assets/img/create-a-new-blog-with-jekyll-chirpy/craete-slack-bot.png)

From Scratch를 선택합니다.

![craete-slack-bot-from-scratch](/assets/img/create-a-new-blog-with-jekyll-chirpy/create-an-app-from-scratch.png)

원하는 앱 이름과 봇을 사용할 스페이스를 선택합니다.

![app-name-and-workspace](/assets/img/create-a-new-blog-with-jekyll-chirpy/app-name-and-workspace.png)

생성이 되었다면 우측아래 Permissions 설정을 해줘야 합니다.

![app-basic-info](/assets/img/create-a-new-blog-with-jekyll-chirpy/app-basic-info.png)

Permissions 탭에서 조금 내려 Scopes로 이동해 chat:write, chat:write.public, links:write 권한을 줍니다.

![app-permissions](/assets/img/create-a-new-blog-with-jekyll-chirpy/app-permission.png)

설정을 완료했다면 다시 Basic Information 탭으로 이동해 Install to Workspace를 눌러 앱을 설치합니다.

![app-basic-config-fin](/assets/img/create-a-new-blog-with-jekyll-chirpy/app-basic-config-fin.png)

이제 OAuth & Permissions 탭으로 가면 앱의 토큰을 얻을 수 있습니다.
![app-token](/assets/img/create-a-new-blog-with-jekyll-chirpy/app-token.png)

### Github Action 설정하기

[github discussions notifier](https://github.com/ostk0069/github-discussions-notifier)를 사용해 위에서 만든 슬랙봇에 댓글이 추가되면 알림을 보내볼것입니다.

Blog를 올리는 Repo의 Actions 탭에서 New workflow를 선택합니다.

![github-action-choose](/assets/img/create-a-new-blog-with-jekyll-chirpy/github-action-choose.png)

여기서 set up a workflow yourself 를 선택한뒤 아래 코드를 붙여 넣고, 파일 이름을 `github-discussions-notifier.yml`로 설정합니다.

```yml
name: GitHub Discussions Notifier

on:
  discussion:
    types: [created]
  discussion_comment:
    types: [created]

jobs:
  notify-github-discussions:
    runs-on: ubuntu-latest
    steps:
      - uses: ostk0069/github-discussions-notifier@v0.0.2
        with:
          SLACK_CHANNEL_ID: ${{secrets.SLACK_CHANNEL_ID}}
          SLACK_BOT_TOKEN: ${{secrets.SLACK_BOT_TOKEN}}
```

{: file="./github-discussions-notifier.yml" }

이후 commit changes를 눌러 코드를 반영합니다.

![github-discussions-notifier](/assets/img/create-a-new-blog-with-jekyll-chirpy/github-discussions-notifier.png)

### Github Secrets 설정하기

위에 yml 파일을 보면

`${{secrets.SLACK_CHANNEL_ID}}` 와 `${{secrets.SLACK_BOT_TOKEN}}`
라고 되어 있는 부분이 있는데 이는 [Github Secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions)로써 코드에 공개되면 안되는 정보들을 저장하는데 쓰입니다.

이를 설정하기 위해서 Repo의 설정에서 `Actions secrets and variables` 탭을 들어가면 됩니다.

![github-secrets](/assets/img/create-a-new-blog-with-jekyll-chirpy/github-secrets.png)

여기서 New repository secrets를 선택해 SLACK_CHANNEL_ID와 SLACK_BOT_TOKEN를 설정해주면 됩니다.

> SLACK_CHANNEL_ID는 워크스페이스에서의 채널 이름입니다.<br>
> SLACK_BOT_TOKEN은 위에서 만든 봇의 Bot User Oauth Token을 넣으시면 됩니다.
> {: .prompt-tip }

### 알림 확인해보기

이제 테스트로 한번 댓글을 달아보면

![github-discussions-notifier-success](/assets/img/create-a-new-blog-with-jekyll-chirpy/github-discussions-notifier-success.png)

![comment-notify-result](/assets/img/create-a-new-blog-with-jekyll-chirpy/comment-notify-result.png)

이렇게 댓글 알림이 잘 오게 됩니다.

# 마무리

이렇게 해서 기본적인 블로그 설정과 댓글 기능을 사용할 수 있게 되었습니다
만약에 Jekyll Chirpy를 가지고 한번 나만의 블로그를 만들어 보고 싶다면 한번 따라해보셔도 좋을 것 같습니다.

다음번에는 시간이 된다면 이 블로그를 어떻게 커스텀마이징할 수 있는지를 포스트 해보도록 하겠습니다.
