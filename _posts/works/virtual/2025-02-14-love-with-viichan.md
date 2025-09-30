---
title: "VRChat - 챠니와 두근두근"
description: "이세계아이돌 비챤님의 '연애 시뮬레이션 토크' 컨텐츠."
categories: [작업물, 버추얼]
tags: [작업물, VRChat, 유니티]
image: "/assets/img/post/works/love-with-vii/love-with-vii-banner.png"
hidden: true

date: 2025-02-14. 00:00
last_modified_at: 2025-02-23. 00:44 # Init
---

_  
{% include embed/youtube.html id = "" %}

## 머리말

---

이세계아이돌 비챤님의 '연애 시뮬레이션 토크' 컨텐츠입니다.  

25년 02월 14일 방송이 진행됐습니다.  

비챤님을 제외한 이세계아이돌, 왁굳님께서 참가자로 참여해주셨습니다.  
비챤님께서 진행을 맡아주셨습니다.  

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

24년 12월 16일에 프로젝트에 참여했습니다.  

## 과정

---

### _

25년 01월 13일 1차 회의  
25년 02월 13일 리허설  

PM이셨던 힉민님, 문과공대생님  
영상 작업해주신 햐동님  
특히 고생 많으셨다  

영상 재생과 관련한 문제들  
영상이 맵에서 거꾸로 보인다거나  
힉민님 개인 나스에 올렸는데 링크 따는 법을 모른다거나  
<- 루나르님께 여쭤봐서 WebDAV 라는 걸 알아냄  
유튜브 일부공개 업로드로 바꾸고  
영상 캐싱이 안돼서 렉 걸리는 현상 발생 -> 미리 재생목록 만들어서 작업자들끼지 쭉 한 번 길뚫는 작업해두고  
영상 플레이어 딜레이 때문에 영상 중간부터 재생되는 현상 발생 -> 영상 앞뒤로 검은 빈 화면 추가  
각 영상을 그렇게 만드니 또 마가 너무 많이 뜸 -> t=5 링크 써서 해도 마가 뜸 -> 선택지 영상들 하나로 합치기  

Arizen님께서 SOOP 채팅 연동 투표 기능 개발  

트리거는 크게 적을 것이 없다  
물론 최신 Woodon 버전으로 컨버팅하면서 수정된 부분들은 많지만,  
전체적인 구조로 봤을 땐 크게 달라진 것이 없음  

## 구현

---

## 기록

---

![250214-230225](/assets/img/post/works/love-with-vii/250214-230225.png)
![250214-230740](/assets/img/post/works/love-with-vii/250214-230740.png)
![250214-230748](/assets/img/post/works/love-with-vii/250214-230748.png)
![250214-230756](/assets/img/post/works/love-with-vii/250214-230756.png)
![250214-230915](/assets/img/post/works/love-with-vii/250214-230915.png)
![250214-232046](/assets/img/post/works/love-with-vii/250214-232046.png)
![250214-232119](/assets/img/post/works/love-with-vii/250214-232119.png)
![250219-174646](/assets/img/post/works/love-with-vii/250219-174646.png)
![250219-174710](/assets/img/post/works/love-with-vii/250219-174710.png)
![250220-155902](/assets/img/post/works/love-with-vii/250220-155902.png)
![love-with-vii-banner](/assets/img/post/works/love-with-vii/love-with-vii-banner.png)
![love-with-vii-credit](/assets/img/post/works/love-with-vii/love-with-vii-credit.png)
