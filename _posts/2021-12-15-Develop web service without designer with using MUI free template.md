---
title: MUI 무료 template로 디자이너 없이 웹서비스 개발하기
author: Bean
date: 2021-12-15 13:21:00 +0800
categories: [Projects]
tags: []
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover:  assets/img/post_images/mui_cover.jpg
---

회사 개발 프로젝트로 빠르게 웹앱을 개발해야 할 일이 생겼다. 그렇지만 팀리소스가 부족해 디자이너가 없는 상황에서 웹서비스를 개발해야 했다.


개발하는 서비스는 `로우로깅` 서비스로, 체크리스트 형식으로 쉽고 빠르게 회사에 맞는 개인정보 처리방침을 생성해주고 이 내용을 기반으로 변호사 온라인 상담을 받을 수 있도록 지원해주는 서비스 이다. 서비스는 [https://www.lawlogging.kr/](https://www.lawlogging.kr/)에서 확인할 수 있다.

B2B 서비스이라 `심플` `깔끔`한 디자인의 서비스를 생각했다. 하지만 디자이너가 없는 상황에서 어떻게 깔끔한 디자인의 서비스를 개발할 수 있을 까 고민이 되었다. 직접 디자인까지 하면 너무 아마추어틱한 디자인의 서비스가 나올 거 같아 고민끝에 깔끔한 느낌을 담고 있는 무료 대시보드 템플릿을 찾아보기 시작했다.

&nbsp;
## MUI Minimal - Client & Admin Dashboard 템플릿
&nbsp;

여러개 다양한 템플릿을 서치하다가 MUI에서 제공하는 [Minimal - Client & Admin Dashboard](https://mui.com/store/#populars)가 괜찮아 보여 이 템플릿을 사용하기로 결정했다. 다양한 코드 예시가 있고 디자인도 깔끔해서 마음에 들었다. 이 템플릿은 칸반보드, 채팅, billing 등 광범위한 페이지와 컴포넌트로 구성되어 있는데 사실 이걸 다 사용하려면 최소 69달러로 라이센스를 사야한다. 로우로깅 서비스에는 이런 복잡한 기능까지 필요없어서 간소화된 [무료 데모](https://github.com/minimal-ui-kit/material-kit-react)를 이용하였다.

<div style="text-align: left">
  <img src="/assets/img/post_images/mui1.png" width="100%"/>
</div>

이 무료 디자인 데모는 다음과 같이 생겼다.

git에 들어가면 코드를 볼 수 있는데, 이 프로젝트 코드는 다음과 같이 구성되어 있다.

```
public
src
├── _mock (더미 데이터)
├── components (차트, popover 등 디자인 컴포넌트 정의)
├── hooks (커스텀 훅)
├── layouts
├── pages
├── sections
├── theme (색상 등 디자인 요소와 base가 되는 컴포넌트 정의)
└── utils
.eslintignore
.eslintrc
.prettierignore
.prettierrc
package.json
...
```

디자인을 참고하는 것 외에도 git 프로젝트 코드에 `prettier` `eslint` 세팅도 다 되어 있어 프로젝트 시작할 때 참고하면 좋을 것 같다. 또한, `theme` 폴더에서 color pallete와 버튼 등 자주 쓰는 디자인 컴포넌트들을 정의하고 반응형 컴포넌트에 대응할 수 있는 breakpoints 등을 정의하고 있다.

```javascript
const breakpoints = {
  values: {
    xs: 0,
    sm: 600,
    md: 900,
    lg: 1100,
    xl: 1536
  }
};

export default breakpoints;
```

MUI 디자인이 리액트 개발에 많이 쓰이기도 하고, 개발 폴더가 전체적으로 구성이 잘되어 있어 처음 리액트 개발을 한다면 시작 전에 구조를 한 번 뜯어보는 것을 추천하고 싶다.

또한 레이아웃도 별도 파일로 정의되어 있으며 drawer, chart, search bar, popover, scrollbar 등 많이 쓰는 컴포넌트들의 코드 예시도 포함되어 있다.

이렇게 디자인 요소가 컴포넌트별로 잘 구분되어 있어 개발 프로젝트에 적용하는 것도 간편했다. 색상코드를 제품 브랜딩에 맞춰서 바꾸고 컴포넌트 위치 바꾸는 것만으로도 쉽게 서비스 사이트를 개발할 수 있다!

&nbsp;
## 빠르게 완성!!

<div style="text-align: left">
  <img src="/assets/img/post_images/mui2.png" width="100%"/>
</div>

완성된 서비스 디자인은 다음과 같다. 다양한 컴포넌트들을 활용해서 디자인을 하니까 서비스 UIUX 디자인을 하는 데 하루이틀 정도면 충분했고 개발도 빠르게 진행할 수 있었다. 제대로 각잡고 서비스를 개발하려면 기획-UIUX 디자인-개발의 정석 프로세스를 따르는 게 좋겠지만 이렇게 적은 리소스로 빠르게 MVP를 개발하려고 한다면 시도해봄직한 방법인 듯 하다.
