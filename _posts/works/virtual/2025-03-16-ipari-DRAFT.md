---
title: "Team Ipari"
# description: ""
categories: [작업물, 버추얼]
tags: [작업물, 게임 개발, 유니티]
image: "/assets/img/background/20230112-151539.jpg"

date: 2025-03-16. 00:00
# last_modified_at: 2025-03-16. 00:42 # 1차 작업
# last_modified_at: 2025-03-17. 21:54 # 2차 작업
# last_modified_at: 2025-03-17. 23:57 # 회의: 개발 방향
# last_modified_at: 2025-03-20. 23:02 # 3차 작업
# last_modified_at: 2025-03-26. 18:46 # 4차 작업
# last_modified_at: 2025-03-29. 13:30 # 5차 작업
# last_modified_at: 2025-04-05. 16:11 # 6차 작업
# last_modified_at: 2025-04-10. 20:55 # 7차 작업
# last_modified_at: 2025-04-11. 21:22 # 7차 작업 마무리
# last_modified_at: 2025-04-17. 19:58 # 8차 작업
# last_modified_at: 2025-04-21. 18:42 # 작업: 유닛 별, 유닛 그룹 별 간격
last_modified_at: 2025-04-22. 17:46 # 작업: 이동 시 위치로
---

_  
{% include embed/youtube.html id = "" %}

## 머리말

---

### 참여 / 담당

서브 프로그래머로서 UI 개발을 담당했습니다.  

### 사용한 툴

- Unity 2022.3.8f1
- Github
- Google Presentation, Figma: 기획/ UI Flow 문서
- Google Drive: 리소스 공유

Discord를 통해 팀원/클라이언트와 소통하고 있습니다.  

## 시작

---

23년 12월 16일에 프로젝트에 참여했습니다.  

## 과정

---

## 구현

---

## 기록

---

- 2025-03-16. 00:00 ~ 00:44
  - UI branch 생성
  - Package 버전 업 / 정리
    - VSCode Package 추가
    - VisualScripting 제거 (이에 따른 일부 Script namespace 선언 제거)
    - 이 외 Package 버전 업
  - Pretendard Font Import
    - Font 확정 전까지 임시로 사용할 Font
  - 임시 배경화면 이미지 추가
  - HomeTown Scene, UI_HomeTown 추가
  - HomeTown 설정창 기반
- 2025-03-17. 19:00 ~ 21:50
  - NPC
    - UINPCPopup: NPC 말풍선
    - NPC Component
      - Talker
      - Interactive
        - StageEntrance
        - UIEntrance
  - 의상상점 (CostumeShop) 구현
    - 의상 Costume ScriptableObject 추가
  - 그 외
    - UnitManager.heroUnit을 Public Property로 변경
    - 게임 시작 시 설정창 비활성화
    - UI_Base의 Init이 Start가 아니라 Awake에서 호출되도록 수정
  - 22:30 회의 진행
- 2025-03-19.
  - 연공전 부활한 김에 연공전을 목표로 변경, 일정은 그대로
- 2025-03-20.
  - 전체적인 구조/흐름 구현
    - 일단 Hometown - Stage 간의 이동
- 2025-03-26.
  - 편성창
- 2025-03-29. 편성창, 업그레이드 상점
  - UIUpgradeDetail -> UIStatDetail
  - 편성창
    - Unit 사거리 기준 자동 우측 정렬, Null 포함
    - 편성 해제 UI Raycast 문제 해결
    - StatDetail
    - UI Develop
  - 업그레이드 상점
    - UI 기반 작성
    - 선택한 Goods Popup Animation, 초기 비활성화
    - Stat 증가량 게이지
  - Hero
    - Model LayerOrder 조정
    - 이동속도 조정 (5 -> 3)
  - UINPCPopup Pop, PopCoroutine 분리
  - Singleton, Resource Folder에서 Prefab 한 번 찾기
  - DataManager에 SO 저장
- 2025-04-05. 편성창 정보 Stage로 넘기기, 설정창
  - DataManager에 편성창 정보 저장
- 2025-04-10 ~ 04-11.
  - 자판기, 월드맵
- 2025-04-17
  - 전체적인 외관 기반 마무리 (클리어 UI +@)
  - 음악 에셋 적용
- 2025-04-18
  - 회의
    - priority: 전체 내용 들어간 빌드 공유
      - 정렬
      - *다수의 이파리가 겹칠 때 레이어 조건:
      - 캐릭터간 레이어가 겹칠 때가 있습니다. 제가 이파리들의 종류에 따라 레이어 순서가 정해놨지만 (사거리 짧음=더 앞), 같은 종류끼리는 생각하지 않았군요. 의 레이어 순서는 "더 앞(더 우측) = 앞 레이어" 입니다. 위치가 완전히 겹쳐질 경우, 겹쳐지기 전 앞 레이어에 있던 캐릭터가 계속 앞에 나오게끔 합니다.
      - HUD 재적용
      - 배경 적용
      - UI 남은 파트 (인트로, 스테이지 종료시 일어나는 일들, 설정 관련)
      - 새 브금 적용 (verse/chorus). 코러스는 피버때 출력됨
      - fever 입장음/퇴장음 -> 0으로.
      - 이파리 이동 랜덤 딜레이 -> 0으로.
- 2025-04-20
  - 빌드
- 2025-04-21
  - HomeTown BGM 적용
  - StageScene 통합
  - 유닛 별, 유닛 그룹 별 간격
- 2025-04-21
  - 유닛 이동 시 위치로

- Next

장애물, 필살기  
113 114 102 103  
113 스파인  
장애물 (공격 타겟)  

추가 인력  
