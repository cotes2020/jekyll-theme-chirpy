---
title: (작성중) Capacitor와 Vue.js를 이용해서 cross-platform 모바일 앱 개발하기
author: Beanie
date: 2021-11-30 13:21:00 +0800
categories: [Projects]
tags: [Vue, Capacitor]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/vue_cover.png
---

이번 글에서는 21.10월부터 21.11월 2달 동안 외주 개발 프로젝트 개발한 하이브리드 앱에 Vue.js와 Capacitor를 적용해본 경험을 공유하려고 한다. 결론적으로 말하면 많이 불편했고 다시는 사용하지 않을 방법이다. 이후에 하이브리드 앱은 [리액트 웹뷰로 하이브리드 앱을 만들자](https://beanie00.github.io/posts/리액트 웹뷰에 네이티브 기능 연결해서 하이브리드 앱을 만들자) 포스팅에서 적은 것 처럼 웹뷰에 Capacitor를 붙이는 대신 네이티브 기능을 다이렉트로 붙이는 방향으로 작업하였다.

## 왜 이 프레임워크를 선택했나
---

우선 주어진 상황은 기존에 조금 구현된 Vue.js 코드를 참고할 수 있었고, 빠듯한 시간 내에 웹과 앱을 모두 구현해야 했다. 그래서 조금 있는 Vue.js 코드를 활용하기 위하여 Vue.js를 프론트엔드 프레임워크로 선택하였고, 모바일앱은 Vue.js로 모바일 웹앱을 구현한 뒤, Capacitor을 통해 cross-platform으로 개발하기로 결정하였다. (하지만 결국 조금있던 Vue.js 코드는 별로 도움이 되지 못하였고, 사실상 처음부터 짜는게 되긴 했다.)

## Capacitor란?
---

## Capacitor 세팅
---

## Capacitor의 플러그인들
---
이 프로젝트에서 모바일앱을 개발할 때 Status bar 디자인을 수정하고 사진, 영상을 촬영한 후 전송하는 등의 네이티브 기능 구현이 필요했다. 이렇게 네이티브 기능을 사용해야 할 때 Capacitor의 플러그인을 사용할 수 있다.

그렇지만 개발하다보니 꽤 불편했는데, 개인적으로 느낀 Capacitor의 단점은 다음과 같다.
* 네이티브 소스보다 느리고 무겁다.
* 프레임워크
* 제공되지 않는 플러그인이 많음
* 플러그인 버그

그래서 사실상 편하려고 쓴 프레임워크지만 기능 부족 / 버그 등의 이유로 네이티브 코드를 직접 건드릴 일이 왕왕 생겼다. 그러다보니 이럴바에는 복잡하게 이러지 말고 네이티브를 직접 건드리는 게 낫겠다는 생각이 들었다.

[https://medium.com/@lyslg1018/ionic-capacitor-ae9f7e691e70](https://medium.com/@lyslg1018/ionic-capacitor-ae9f7e691e70)