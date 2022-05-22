---
title: 크롬 익스텐션 파일 크기 400mb에서 4mb로 줄이기(..)
author: Beanie
date: 2022-04-29 09:32:00 +0800
categories: [Web frontend, React]
tags: [Catty, Extension]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/catty_cover2.png
---

사실 이 이슈는 신경을 안써서 생겨난 해프닝이긴 한데 그래도 적잖이 당황했어서 정리해두기로 했다.

크롬 익스텐션은 배포 할 때 zip 형식으로 개발 파일을 올려야 한다.

<div style="text-align: left">
   <img src="/assets/img/post_images/extension_size1.png" width="100%"/>
</div>

그래서 별 생각없이 익스텐션을 개발하던 폴더 전체를 zip으로 압축해서 배포해왔다. 그런데 배포된 크롬 익스텐션을 다운받을 때 너무 오래 걸려서 확인해보니까 파일 크기가 200mb(!)로 확인되었다. 다른 익스텐션의 파일 크기는 거의 1~5mb로 매우 작은데 Catty만 유독 커서 확인해보니 Catty 개발 폴더가 400mb를 차지하고 있었다.

<div style="text-align: left">
   <img src="/assets/img/post_images/extension_size2.png" width="100%"/>
</div>

&nbsp;

Catty 디렉토리는 아래와 같이 구성되어 있는데, 더 세부적으로는 git 관련 내용이 저장되어 있는 .git 폴더가 200mb, npm 설치 파일이 들어있는 popup/node_modules 폴더가 200mb를 차지했다.
```
content_scripts
background.js
background.html
popup
   ├── node_modules
   ├── src
   └── package.json
manifest.json
package.json
.git
```

따로 익스텐션 배포할 때 이 파일들이 제외되지는 않는가보다. (너무 많은 걸 바랬나)

고로 이 폴더들을 제외하고 배포해야 하는데 .git 폴더도 없애면 git 설정을 다시해야하고, node_modules 폴더도 로컬에서 업데이트할 때 계속 필요해서 없애면 안되어서 고민하다가 그냥 이 두 폴더를 잠시 다른 곳에 옮긴 다음에 zip 압축을 하고, 그 후 다시 폴더를 가져오는 식으로 했다.

<div style="text-align: left" width="100%">
   <img src="/assets/img/post_images/extension_size3.png" width="100%"/>
</div>
&nbsp;
그 결과 새로 배포된 버전은 1mb 정도로 파일 크기가 줄어들었다!!

더 좋은 방법이 있을 거 같긴한데 아직은 잘 모르겠다. npm 모듈 중에 build하면서 바로 zip 파일을 생성해주는 것도 있는데 지금 디렉토리 구조가 popup 폴더 내에 또 package.json이 있고 여기서 node_modules가 생성되는 구조라 이 폴더에서 활용하기는 어려울 거 같다.(popup 내에서 build와 zip을 같이하면 popup 폴더 빌드 파일만 zip으로 압축된다.) 매번 배포할 때마다 폴더를 임시로 옮겼다가 돌려놓는게 많이 비효율적인 거 같긴해서 방법을 더 고민해봐야겠다.
