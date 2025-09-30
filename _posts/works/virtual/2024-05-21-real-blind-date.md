---
title: "인간 X 버추얼 소개팅 (리얼소개팅)"
# description: ""
categories: [작업물, 버추얼]
tags: [작업물, VRChat, 유니티]
image: "/assets/img/post/works/real-blind-date/240525-000000.png"
hidden: true

date: 2024-05-21. 00:00
# last_modified_at: 2024-11-09. 08:31 # Init
last_modified_at: 2025-03-13. 19:04
---

인간과 버추얼이 소개팅을 한다면? -인간X버추얼 소개팅  
{% include embed/youtube.html id = "dC1u2VNN7q8" %}

## 머리말

---
고세구님의 방송 컨텐츠로 진행된 '리얼 소개팅'.  
현실 모습의 남성과 가상 모습의 여성이 만나 소개팅을 하는 컨텐츠입니다.  

VRChat SDK를 이용한 Unity C# 프로그래밍을 담당하였습니다.  

### 참여 / 담당

작업한 기능은 다음과 같습니다.  

- 단순 Unity 애니메이션 재생
  - 무대 커튼
  - 투표소

- MeshRenderer의 Material 변경
  - 무대 양 옆 스크린에 남성 프로필 띄우기

- 투표 시스템
  - 첫 인상, 최종 선택 투표
  - 여성이 남성 선택지 팻말을 들어서 카메라에 표시
  - 스태프가 수동으로 투표현황 기록 (남성은 실제 캠 화면을 통해 투표를 진행하므로)
  - 투표 후 투표 결과 공개 (스크린 UI)

- CC (Cinemachine을 이용한 카메라 컨트롤)
  - 무대 카메라, 투표 카메라

### 사용한 툴

- Unity 2022.3.22f1
- [U# (C# + VRChat SDK)](https://udonsharp.docs.vrchat.com/)

## 기록

---

![240524-000000](/assets/img/post/works/real-blind-date/240524-000000.png)
![240525-000000](/assets/img/post/works/real-blind-date/240525-000000.png)
![250220-194001](/assets/img/post/works/real-blind-date/250220-194001.png)
![250220-194100](/assets/img/post/works/real-blind-date/250220-194100.png)
![250220-194119](/assets/img/post/works/real-blind-date/250220-194119.png)
![250220-194149](/assets/img/post/works/real-blind-date/250220-194149.png)

- 240521-123042: 작업

### 진행 순서

1. 소개
   - 무대 문 CC
   - 무대 문 애니메이션 (ON)
   - 프로필 공개
   - 무대 문 애니메이션 (OFF)

2. 투표
   - 투표소 애니메이션 (ON)
   - 투표소 안쪽 CC
   - 투표 진행 (픽업, 수동 기록)
   - 투표소 애니메이션 (OFF)
   - 무대 스크린 (ON)
   - 투표 결과 공개
   - 무대 스크린 (OFF)

출력할 투표 정보 설정 [남성: 여성] [첫 인상 득표수: 받은 하트 수]  
