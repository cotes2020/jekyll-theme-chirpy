--- 
title: "깃허브 블로그 시작하기" 
description: 깃허브 블로그 시작하기
author: cylanokim 
date: 2025-09-16 17:00:00 +0800
categories: [Blog]
tags: [github, blog]
pin: true
math: true
mermaid: true
---
깃허브 블로그를 시작 합니다. 

아직 어떤 주제와 형식으로 시작해야할지 모르겠네요. 뭐 이쪽 분야가 다 그렇지만 계속 배우고, 구르면서 방향을 잡으려고 합니다. 가장 중요한 것은 **꾸준함**이겠지요. 하지만 이러한 꾸준함 이전에 블로그를 시작하기 조차 어려웠던 제 모습을 생각하며, 첫 번째 포스팅은 GitHub 블로그 개설하는 과정에서 부딧힌 문제 해결 방법에 대하여 글을 남기려합니다.

# 1. 왜 깃허브 블로그?

블로그를 시작하는 방법은 많습니다. 네이버, 티스토리, MEDIUM, 요즘은 Notion을 사용하는 경우도 많습니다. 다 장/단점이 있습니다만, 가장 큰 이유는 마크 다운 기반으로 자신만의 독특한 블로그를 만들 수 있다는 것이었습니다. 마크 다운으로 작성 시, 여러 경로를 통해 가져온 텍스타나 표, 링크 등을 쉽게 표시할 수 있었고, 무엇보다 이 기회에 자연스럽게 . 하지만 깃허브 블로그는 일단 시작하기에 일반 블로그 대비 상당한 난이도가 있습니다. 그럼에도 불구하고 먼저 블로그를 시작하신 선배님들을 샤라웃 하는 의미로, 참고한 사이트를 링크로 남겨둡니다.    

[ITability님의 있어보이는 건 따라해보기](https://tired-o.github.io/)

[Zedd0202님의 블로그](https://zeddios.tistory.com/)

<br>

---
# 2. 깃허브 블로그 시작하면서 직면한 문제들
## 2-1. jekyll 왜 설치 안되는거야?
깃허브 블로그에 이쁜 테마를 적용하려면 jekyll 아래와 같은 명령어로 설치해야합니다.
```console
gem install jekyll bundler
```
그런데 gem은 뭐지? 직감적으로 Python의 pip, Node.js의 mpm 비슷한 명령어일 것 같은데, 역시나 Ruby 언어에서 제공하는 패키지 관리 툴이었습니다. 즉 Ruby를 설치해야합니다. 그런데 윈도우에서 Ruby 설치하는게 쉽지가 않더라구요. 인터넷 검색과 GPT를 참고해가며 설치하려했지만 실패했었습니다. 정확한 원인을 알 수 없지만 환경 변수와 관련된 문제인 듯 한데, 환경 변수를 추가했음에도 GEM 명령어를 인식 못하였습니다.

결국 찾은 해답은, **Ubuntu**에서 설치하자는 것이었습니다. 정확히는 **WSL** 입력창을 통해 설치하는 것이었습니다.

✅ **단계별 설치 방법 (Ubuntu WSL 기준)**
1. 시스템 업데이트
```
sudo apt update
sudo apt upgrade -y
```  
2. Ruby 및 필수 패키지 설치
```
sudo apt install -y ruby-full build-essential zlib1g-dev
```
3. Jekyll & Bundler 설치 (드디어...)
```
gem install jekyll bundler
```
설치가 끝나고 버전 확인
```
jekyll -v
```
<br>
## 2-2. 테마를 적용했는데 적용이 안되는 문제
- **문제점**

    테마를 적용하고 저장소에 **Push**를 했는데 에러가 발생하였고, 저장소의 **Actions** 텝에서 왜 에러가 발생하였는지 확인해보았습니다.
    ![GitHub Develop Error](/assets/gitgub_build_error.png)


- **원인과 해결 방법**

    깃허브 테마는 크게 깃허브 페이지가 기본 제공하는 테마와 외부 Jekyll 테마가 있습니다. 이 경우 `_config.yml`에서 지정해주는 방식이 다른데, 위 에러는 제가 기본 테마에 remote_theme에서 제공하는 테마를 적용하여 발생한 문제였습니다. 이에 기본 테마인 jekyll-theme-modernist 적용하기 위하여 `_config.yml` 파일을 수정하였습니다. 

    ```
    (_config.yml)
    theme: 어디서 가져온 remote_theme (기존)
    theme: jekyll-theme-modernist
    ```

    아래는 GPT를 이용해 theme과 remote_theme을 비교한 내용입니다. 

    | 항목            | `theme`                                 | `remote_theme`                             |
    | ------------- | --------------------------------------- | ------------------------------------------ |
    | **설정 위치**     | `_config.yml`                           | `_config.yml`                              |
    | **사용법**       | `theme: jekyll-theme-modernist`         | `remote_theme: user/repo@version`          |
    | **설치 방법**     | 기본 제공하는 테마 중 하나 사용        | 외부 Jekyll 테마를 가져와서 사용           |
    | **지원 범위**     | 깃허브 페이지 공식 지원 테마만 가능               | 거의 모든 Jekyll 테마 사용 가능 |
    | **Gem 필요 여부** | Gem(`github-pages`)만 있으면 됨 | `jekyll-remote-theme` Gem 필요               |

<br>

## 2-3. 댓글 적용하기
깃허브 블로그에 댓글 기능을 추가하기 위하여 광고도 없고 깔끔하다는 [giscus](https://github.com/apps/giscus)를 사용하기로 하였습니다. 댓글을 작성하려면 깃허브 로그인을 해야합니다.

✅ **깃허브 블로그에 giscus 댓글 기능 추가하기**
### 1. 깃허브 블로그 저장소 > `Settings` > `Features` 항목에서 **Discussions** 체크하기
![discussions](/assets/discussions.png)

### 2. 저장소에 giscus 앱 설치
[https://github.com/apps/giscus](https://github.com/apps/giscus)에 접속 후 install!

### 3. giscuss 설정 
[https://giscus.app/ko](https://giscus.app/ko)에 접속하여 **설정** 항목에 블로그 정보를 입력합니다.

먼저 아래 항목에 본인의 깃 허브 저장소 주소를 입력합니다.
<p align="center">
  <img src="/assets/my_repo.PNG" alt="myrepo" width="400">
</p>

그리고 댓글 기능이 구현될 카테고리를 설정합니다. 아직 정확한 이유는 모르겠지만 **Announcements**를 선택 하라고 합니다.

### 4. 블로그 탬플릿에 script 태그 추가하기 
모든 입력이 마무리 되면 아래와 미슷한 script 태그가 만들어집니다. 이제 이 태그를 블로그 포스팅 템플릿에 추가하면 됩니다. 블로그 포스팅 템플릿은 블로그 > _layouts > post.html 파일에 있는데 적당한 위치에 붙여 넣으면 됩니다. 
<p align="center">
<img src="/assets/script_tag.PNG" alt="script" width="400">
</p>