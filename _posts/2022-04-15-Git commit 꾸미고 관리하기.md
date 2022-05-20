---
title: Git commit 꾸미고 관리하기
author: Bean
date: 2022-04-15 19:19:00 +0800
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

Git 커밋 정리는 이전부터 해야지 마음먹었는데 계속 미루고 있었다. 계속 미루면 안될 거 같아 Git project 오픈하면서 같이 정리해보았다. 우선 커밋 메시지 템플릿은 검색하니까 많이 나왔다. 다 비슷한 내용인데 그 중 괜찮은 템플릿을 가져왔다.

&nbsp;

```
################
# <타입> : <제목> 의 형식으로 제목을 아래 공백줄에 작성
# 제목은 50자 이내 / 변경사항이 "무엇"인지 명확히 작성 / 끝에 마침표 금지
# 예) feat : 로그인 기능 추가

# 바로 아래 공백은 지우지 마세요 (제목과 본문의 분리를 위함)

################
# 본문(구체적인 내용)을 아랫줄에 작성
# 여러 줄의 메시지를 작성할 땐 "-"로 구분 (한 줄은 72자 이내)

################
# 꼬릿말(footer)을 아랫줄에 작성 (현재 커밋과 관련된 이슈 번호 추가 등)
# 예) close / fix / resolve #7

################
# feat : ✨ 새로운 기능 추가
# fix : 🐛 버그 수정
# docs : 📝 문서 수정
# test : ✅ 테스트 코드 추가
# refact : ♻️ 코드 리팩토링
# style : 💄 코드 의미에 영향을 주지 않는 변경사항
# chore : 🔧 빌드 부분 혹은 패키지 매니저 수정사항
################
```

이 템플릿을 프로젝트 폴더에 `.gitmessage.txt` 라는 이름의 파일로 추가하면 자동으로 이후에 `git commit`을 할 때마다 이 템플릿이 떠서 템플릿을 참고해서 커밋 메시지를 작성할 수 있다.

또한, 커밋을 이슈와 연결시킬 수 있는데 `#20`처럼 이슈 번호를 커밋에 남기면 아래와 같이 해당 이슈에 기록이 생성된다.
<div style="text-align: left">
  <img src="/assets/img/post_images/git commit2.png" width="100%"/>
</div>

&nbsp;

또 특히 `종료 키워드 + #이슈번호` 이렇게 커밋에 남기면 자동으로 이슈가 종료된다.
<div style="text-align: left">
  <img src="/assets/img/post_images/git commit1.png" width="100%"/>
</div>

&nbsp;

종료키워드는 다음과 같다.
* close
* closes
* closed
* fix
* fixes
* fixed
* resolve
* resolves
* resolved