---
title: Chirpy 테마 적용해서 Github 블로그 개발
author: lee
date: 2023-01-07-20:04:00 +0800
categories: [프로그래밍, 블로그]
tags: [블로그]
image:
  path: /assets/img/img4.png
---

<p data-ke-size="size16"><a href="https://tkdals1049.github.io/">tkals1049 | 지그의 게임 개발기 </a></p>


## 주의사항
절대로 자기 마음대로 진행하지 말고 
매뉴얼과 단계를 제대로 다 밟아 기초를 구현하고 
그 다음에 꾸미기 시작할 것

테마 개발자가 자기 방식으로 만든 것이기에 그 단계를 제대로 밟지 않는다면
제대로 구현되기는 커녕 기능이 일부 정지되어 오류가 잘 뜬다

## 준비물

<p data-ke-size="size16"><a href="https://https://jekyllrb-ko.github.io/">jekyll 설치 </a></p>
<p data-ke-size="size16"><a href="https://rubyinstaller.org/downloads/">루비 언어 설치 </a></p>
<p data-ke-size="size16"><a href="https://https://git-scm.com/">git bash 설치 </a></p>

## 준비물 설치

맥에는 Ruby가 설치되어 있어 따로 설치하지 않았다.

-gem 설치

gem install bundler
맥에는 System Ruby를 사용하고 있기 때문에 gem을 설치할 권한이 없어서 에러가 난다. rbenv를 이용해서 ruby 버전을 변경하여 문제를 해결한다. 👉 방법


-jekyll 설치

gem install jekyll
Jekyll 블로그가 github에서 렌더링되는데 필요한 의존성 패키지를 설치한다.

gem install github-pages

## Chirpy 테마 가져오기

Chirpy starter를 이용하는 방법과 fork로 가져오는 방법이 있다.

Chirpy starter : 업그레이드하기 쉬우며, 관련 없는 프로젝트 파일과 분리하여 작성에 집중할 수 있다.
Fork : 커스텀에 편리하나 업그레이드하기 어렵다. Jekyll에 익숙하지 않고 프로젝트에 기여할 의사가 없으면 권장되지 않는다.
커스텀을 시도해보기 위해서 fork로 진행하였다.

Fork Chirpy에서 리포지토리 이름을 <GH_USERNAME>.github.io 로 설정정하고 리포지토리를 만든다.

로컬에서 아래 명령을 통해 소스를 clone 한다.

git clone https://github.com/[username]/[username].github.io.git
프로젝트 루트 디렉토리로 들어가서 아래 명령어를 통해 chirpy를 초기화한다.

프로젝트 폴더에 들어가서 마우스 오른쪽 클릭으로 배쉬를 직접 키고 
bash tools/init.sh
그러면 다음과 같은 작업이 실행된다.

아래 파일과 디렉토리 삭제

.travis.yml
_posts 아래의 파일들
.github/workflows/pages-deploy.yml.hook에서 .hook 삭제하고 이를 제외한 .github내 다른 폴더와 파일들 삭제

Gemfile.lock을 .gitignore에서 삭제

변경을 저장하기 위해 자동적으로 commit 생성

dependencies 설치
첫 구동을 하기 전에 루트 디렉토리로 이동하여 아래 명령어를 통해 의존성을 설치한다.

bundle
로컬에서 실행해보기
원격으로 올리기 전에 먼저 보고 싶다면 아래 명령어를 실행하고 http://127.0.0.1:4000 으로 들어가본다.

bundle exec jekyll s

## 깃페이지 설정 

프로젝트 설정의 왼쪽 목록에 페이지 항목에 들어가 index를 다른 걸로 바꿔주고
바로 밑에 박스의 설정에 들어가 페이지를 활성화 시키고
https://tkdals1049.github.io에 접속한다.