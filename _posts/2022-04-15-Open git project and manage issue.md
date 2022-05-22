---
title: Git project 오픈 및 이슈 관리
author: Beanie
date: 2022-04-15 16:32:00 +0800
categories: [etc, git]
tags: [Git]
cover: assets/img/post_images/github_cover.jpeg
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/catty_cover2.png
---

처음에 Catty 서비스를 개발할 때 어짜피 1인 개발이라 따로 프로젝트 관리를 하지 않았다. 그렇지만 크롬익스텐션, 앱, 웹 3가지를 한꺼번에 관리해야하고 기능이 조금씩 복잡해지면서 체계적으로 프로젝트를 관리할 필요성을 느끼게 되었다. 예전에 프로젝트 관리용으로 `Jira`나 `Notion`을 써봤지만 아무래도 개발 프로젝트는 git repo에 프로젝트를 열어 관리하는 게 최고인 듯 하다.

## 이슈 라벨 꾸미기
&nbsp;

<div style="text-align: left">
  <img src="/assets/img/post_images/issue1.png" width="70%"/>
</div>
먼저 깃 이슈 관리에 사용될 라벨을 꾸며주었다. `🐞 BugFix`, `📃 Docs`, `✨ Feature`, `💪 Enhancement`, `🔨 Refactor` 등 기본적인 이슈 라벨과 추가로 몇번 코드 migrate 작업을 진행해 Migrate 태그도 추가해주었다.
또한, 사실 서비스의 많은 인상이 처음에 결정됨에도 불구하고 자꾸 개발하다보면 이런 부분을 놓치게 되어 `🥰 First impression`이라는 라벨을 따로 둬서 첫인상에 영향을 줄 수 있는 이슈를 별도로 관리하였다.

## 이슈 템플릿 추가하기
\
&nbsp;
또한, 이슈를 추가할 때 같은 형식으로 추가할 수 있도록 템플릿을 만들 수 있다. 이슈 템플릿 만드는 법은 다음의 블로그를 참고했다.
[https://shinsunyoung.tistory.com/35](https://shinsunyoung.tistory.com/35)

## Git 프로젝트와 이슈 연결하기
\
&nbsp;
마지막으로 Git 프로젝트를 생성하고 이슈와 연결할 수 있다. 간단하게 설정할 수 있는데, 프로젝트를 생성할 때 **Project template**을 `Automated kanban` 이나 `Automated kanban with review`를 사용하면 git이 알아서 다 연동해준다.
<div style="text-align: left">
  <img src="/assets/img/post_images/git issue1.png" width="100%"/>
</div>