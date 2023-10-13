---
title: Jekyll Chirpy를 활용한 새 블로그 만들기
author: nahjh2
date: 2023-10-12 20:00:00 +0900
categories: [ETC, Blog]
tags: [jekyll, chirpy, GitHub Pages, github.io, blog]
---

평소 사이드 프로젝트를 진행중이던 우린 가끔식 문제를 어떻게 해결했는지를 공유하곤 했습니다.

하다보니 좋은것 같아 여기서 멈추지 않고, 개발을 하면서 겪은 이슈들을 정리해서 발표해보기로 정하고, 이를 블로그로 만들어보기로 했습니다.

처음 블로그를 작성하려 했을 때 후보군으론
Medium, Notion, Velog, Tistory, Github pages 등이 있었고

이중 Medium과 같이 글 자체에 포커스를 둔 블로그를 만들고 싶어 Medium을 사용하려 했었습니다.

하지만 Medium의 Team 계정은 달에 5$를 내야돼서, Medium 스타일의 Github pages를 만들기로 결정했습니다.

우리의 목표는 블로그 글을 작성하는 것이지 블로그를 만드는게 아니였으므로, github pages를 가볍게 만들 수 있는 jekyll을 사용하기로 정했고,
이중 깔끔한 Chirpy 테마를 사용하기로 결정했습니다.

## Jekyll의 특징

ruby로 만들어짐

일단 정적 사이트로서 SEO에 용의하다

MIT 라이센스이다.

Github Action을 제공하여 Github pages 관리가 쉽다.

등등.

# Jekyll Chirpy 시작하기

There are two ways to create a new repository for this theme:

- Using the Chirpy Starter - Easy to upgrade, isolates irrelevant project files so you can focus on writing.
- GitHub Fork - Convenient for custom development, but difficult to upgrade. Unless you are familiar with Jekyll and are determined to tweak or contribute to this project, this approach is not recommended.

우리는 나중에 블로그를 커스텀 할 수도 있어 Fork를 해서 작업을 해보겠다.

https://chirpy.cotes.page/posts/getting-started/ 참고

## ruby & Jekyll 설치하기

```shell
sudo apt-get install ruby-full build-essential zlib1g-dev
gem install jekyll bundler
```

## Chirpy fork 뜨기

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/fork-chirpy.png)
_https://github.com/cotes2020/jekyll-theme-chirpy/fork_

rename it to USERNAME.github.io

```shell
git clone https://github.com/vivace-log/vivace-log.github.io
```

## Chirpy의 구조

\_data, \_plugins ... etc 로 구성됨

## local에서 테스트해보기

```shell
bash tools/init
```

_[Node.js](https://nodejs.org/ko) 가 설치되어 있어야됨_

```shell
bundle
```

```shell
bundle exec jekyll s
```

_로컬에서 테스트해보기_

## 배포 하기

```shell
git push
```

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/first-deploy.png)
_아직 포스트가 없어 깨끗함_

## Jekyll Chirpy 설정하기

Chirpy의 기본적인 설정은 \_config에서 설정한다.

timezone, title, tagline, description, url, github, social 등을 설정하면 된다.

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

## 첫번째 글 작성하기

https://chirpy.cotes.page/posts/write-a-new-post/

글을 작성할때 파일을 \_posts에 yyyy-mm-dd-제목.md의 형식으로 만들어야 함

이때 제목은 영어로, 띄어쓰기는 - 로 작성할 것

```markdown
---
title: First Post
date: 2023-10-07 23:25:00 +0900
categories: [Test, First-Post]
tags: [test_tag, first-post]
---

First Post!
```

{: file="\_posts/2023-10-07-first-post.md" }

작성후 깃에 push 하면

```shell
git add .
git commit -m "Add comments"
git push
```

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/github-action.png)

Github Action에서 빌드를 하고 배포까지 해줍니다.

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/first-post-home.png)
![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/first-post.png)

이렇게 들어간 걸 볼 수 있다.

# 댓글 기능 추가하기

아마 \_config을 자세히 봤다면 comments 설정이 있었던 걸 볼 수 있었을 것 입니다.

Chirpy는 disqus, utterances, giscus를 기본적으로 제공하는데 이중 giscus를 이용해 댓글기능을 추가해 보겠습니다.

## Giscus 란?

Giscus는 깃허브의 Discussions 기능을 댓글처럼 사용하는 앱입니다.

## Giscus 설정하기

1. giscus 앱 설치하기

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus.app.png)
_https://giscus.app/ko_

에 접속해서 2. giscus 앱이 설치되어 있어야 합니다. 를 클릭하여

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-install-page.png)
_https://github.com/apps/giscus_

설치 페이지로 이동합니다.

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-install-target.png)

설치할 그룹을 정한뒤

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-install-repo.png)

github page를 호스팅할 repo를 지정합니다.

2. 레포에 Discussions 기능 활성화 하기

해당 레포의 설정에 들어가서 discussions를 활성화 하면 됨

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/set-discussions.png)

2개를 세팅한 다음에
![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-passed.png)

이렇게 모든 조건을 만족합니다가 뜨면 준비완료

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/giscus-script.png)

좀더 내려서 data-repo-id와 data-category-id 부분을 복사해서
\_config 파일에 아래처럼 설정하기

```yml
comments:
  active: giscus
  giscus:
    repo: vivace-log/vivace-log.github.io # <gh-username>/<repo>
    repo_id: R_kgDOKdKNnw # data-repo-id
    category: Announcements # category
    category_id: DIC_kwDOKdKNn84CZ8ZG # data-category-id
```

설정을 마치고 다시 올리기

```shell
git add .
git commit -m "Add comments"
git push
```

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/post-with-comments.png)
_성공적_

# 검색 엔진 최적화

통칭 SEO

우리가 블로그를 만들었으니 이제 유저들이 접근 하기 쉽게 만들고, 어떻게 유저들이 접근하는지 트래픽이 어떻게 되는지를 확인할 필요가 있다!

우리는 google search console에 우리 사이트를 등록하고, analytics를 붙여 유저들의 방문을 확인해 볼 것이다.

## SEO란?

우리는 보통 사이트를 방문할때 구글과 같은 검색엔진에 검색을 해서 접근하게 된다.
이때 검색엔진이 우리 사이트를 높은 랭킹을 줄 수 있도록 최적화 하는 걸 말한다.

## SEO는 어떻게 작동하는가?

## 구글 서치 콘솔 세팅

https://search.google.com/search-console/welcome?utm_source=about-page

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/google-search-console.png)

오른쪽 url 입력칸에 url을 입력하고

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/google-search-console-check.png)

HTML 태그로 소유권 확인을 클릭
메타 태그를 복사하여 content 부분만 아래 \_config 파일에 적기

```yml
google_site_verification: RgTWrBsyj_okl8Hts4kVawxuMI_pbqd6olqO8VPPl9o
```

깃에 푸시한뒤 확인을 누르면

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/google-search-console-check-pass.png)

이제 search console을 사용 가능

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/google-search-console-home.png)

## 구글 에널리틱스 설정

https://analytics.google.com/analytics/web/provision/#/provision/create

![fork-chirpy](/assets/img/create-a-new-blog-with-jekyll-chirpy/google-analytics-create.png)

계정을 세팅
