---
title: "VRChat - 마왕총회"
description: "이세계아이돌 비챤님의 '마왕 토크쇼' 컨텐츠."
categories: [작업물, 버추얼]
tags: [작업물, VRChat, 유니티]
image: "/assets/img/post/works/mawang-assemble/250413-203153.png"
hidden: true

date: 2025-04-13. 00:00
# last_modified_at: 2025-04-16. 21:48 # Init
# last_modified_at: 2025-04-17. 21:59
last_modified_at: 2025-04-17. 22:40
---

이세계에 마왕들이 모였습니다 - 버튜버 마왕총회  
{% include embed/youtube.html id = "-yGT9IUrpHU" %}

## 머리말

---

이세계아이돌 비챤님의 '마왕 토크쇼' 컨텐츠입니다.  

25년 04월 13일 19시 방송이 진행됐습니다.  

망냥냥, 마왕이노리, 제갈금자, 마왕0216, 마왕루야, 웅터르님께서 참가자로 참여해주셨습니다.  

### 참여 / 담당

본 컨텐츠는 VRChat 게임 내에서 진행되었습니다.  

해당 컨텐츠에 사용된 VRChat World 내 기능 프로그래밍을 담당했습니다.  
또한 방송 진행을 위한 오퍼레이팅, 기능 조작을 담당했습니다.  

- 카메라 시스템
  - 에디터 타임에 카메라 구도를 미리 설정
  - `CinemachineVirtualCamera.Priority`를 조정하여 구도 전환

### 사용한 툴

- Unity 2022.3.22f1
- [U# (C# + VRChat SDK)](https://udonsharp.docs.vrchat.com/)
- [Woodon](https://github.com/wrchat/Woodon)
- TortoiseSVN

Discord를 통해 팀원/클라이언트와 소통했습니다.  

## 시작

---

25년 04월 03일에 프로젝트에 참여했습니다.  

## 과정

---

25년 04월 12일 작업을 진행했습니다.  

## 구현

---

1. 스크린에 주제 띄우기
2. 참가자 간의 인기투표

두 가지 기능이 필요하다는 사실만 전달 받고 작업을 진행했다.  

주제 띄우기는 간단히 문자열만 다르게 띄우면 됐고,  
인기투표의 경우 이미 예전에 만들어둔 `VoteManager` 기능이 있어서 그대로 활용했다.  

기능 구현보다는, UI 배치에 더 많은 시간이 소요됐다.  
UI에 사용된 글꼴은 [전주완판본 각체](https://noonnu.cc/font_page/625)이다.  

힉민님께서 아트 작업을 진행해주셨고, SVN으로 넘겨받아 기능을 씌웠다.  

## 후기

---

이번에도 오퍼레이팅/기능 조작 과정에서 실수가 있었다.  
방송으로 보다가 주제어 넘길 타이밍을 놓친다던지.  
한 번에 하나의 일만 할 것 !  

참가자 분들 전부 매력있고 재밌으셨다.  

특히 융터르님은 `나는모솔` 컨텐츠 때도 느꼈지만 정말 좋은 의미로 미친 사람인 것 같다.  
중간중간 튕기셔서 곤란하셨을텐데 정말 고생 많으셨다.  
마지막 나가실때 스태프분들 앞으로 한 번씩 가서 고개 끄덕끄덕하고 가셨다. (끄덕끄덕)  

망냥냥님의 개회사나 자기소개에 참고할 텍스트를 간단히 AI로 뽑았었는데,  
주어진 맥락에 맞게 생각보다 그럴싸하게 잘 나와서 '오..' 했다.  

## 기록

---

![250408-181444](/assets/img/post/works/mawang-assemble/250408-181444.png)
![250413-133559](/assets/img/post/works/mawang-assemble/250413-133559.png)
![250413-185903](/assets/img/post/works/mawang-assemble/250413-185903.png)
![250413-203105](/assets/img/post/works/mawang-assemble/250413-203105.png)
![250413-203123](/assets/img/post/works/mawang-assemble/250413-203123.png)
![250413-203128](/assets/img/post/works/mawang-assemble/250413-203128.png)
![250413-203142](/assets/img/post/works/mawang-assemble/250413-203142.png)
![250413-203153](/assets/img/post/works/mawang-assemble/250413-203153.png)
![250413-203709](/assets/img/post/works/mawang-assemble/250413-203709.png)
![250413-204234](/assets/img/post/works/mawang-assemble/250413-204234.png)

- [VRChat World](https://vrchat.com/home/world/wrld_8b9325e4-cc81-4f2c-b72c-8f105d67d43f/info)
  - Keyboard `~`, `1 ~ 6`, `F1 ~ F10` 까지 카메라가 등록돼있다.
  - PageDown으로 Staff UI를 활성화 할 수 있다.

- 주제
  - 개회사, 자기소개 타임
  - 진정한 마왕이란 무엇인가
  - 마왕이라 좋은 점
  - 마왕이라 나쁜 점
  - 마왕 이대로 괜찮은가
  - 마왕 중 서열 1위는?
  - 마왕 인기투표 (자투제외)
