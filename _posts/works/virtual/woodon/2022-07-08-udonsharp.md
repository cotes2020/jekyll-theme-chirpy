---
title: "VRChat 월드 제작 (U# / UdonSharp)"
# description: ""
categories: [작업물, 버추얼]
tags: [작업물, VRChat, 유니티]
image: "/assets/img/background/20240827-140647.jpg"
hidden: true

# 🌔 VRChat 안개 (Fog)
# date: 2022-01-28. 09:48

# 🌔 VRChat 월드 에디터 테스트 시, ContextMenu Attribute
# date: 2022-06-28. 02:41

date: 2022-07-08. 14:31
# last_modified_at: 2023-10-10. 10:00
# last_modified_at: 2024-04-09. 13:44
# last_modified_at: 2024-08-19. 15:24
# last_modified_at: 2024-08-19. 16:54
last_modified_at: 2024-11-12. 11:44 # 라이브 스트리밍 ~
---

2024-04-09. 02:28: 글 계승  
`2022-01-28-USharp-Fog: 🌔 VRChat 안개 (Fog)`,  
`2022-06-28-USharp-ContextMenu: 🌔 VRChat 월드 에디터 테스트 시, ContextMenu Attribute`  

## 라이브 스트리밍 용 VRChat 컨텐츠 제작 시 신경 쓸 점

---

### VRChat

- 스트리머와 참가자의 PC/VR 플레이 유무
- 월드 인스턴스 최대 인원 제한

### 기획, 버그 가능성

- 수동 VS 자동화
  - 돌발 상황에 대비하여 수동으로 만들거나, 제한을 여유롭게 두기
  - '뭐 잘못 눌렀어요', 제한, 방지, 대책

- 테스트와 리허설 자주
  - 기능 자체의 버그와, 기획의 의도 확인

- 조작키
  - 예를 들어 스태프 전용 키를 자주 누르는 키로 설정하지 말기
  - 혹은 좀 더 복잡하게 만들던지

### 라이팅, 렌더링

- 아바타가 조명을 제대로 받는지
  - LocalPlayer, Player Layer만 잡는 RealTime 라이트
  - 라이트 프로브
- 리플렉션 프로브
- 포스트 프로세싱

### 영상, UI

- 채팅창 위치 고려를 고려하여 UI 제작
- (버추얼 한정) 화면에 띄워지는 아바타 위치를 고려하여 UI 제작

- 대기 화면 유무
- 로고를 UI로 띄울지 여부

- 글꼴
  - 글꼴 저작권
  - 한자 필요할 경우 한자 지원하는 글꼴로

### 비디오 플레이어

- 딜레이가 많이 길다
- 카메라로 비디오 플레이어 스크린을 찍으면, 가끔 프레임이 끊겨보인다 (검은 화면)
- 비디오 플레이어 소리 잘 나오는지
- URL 전부 제대로 입력했는지

### 사운드

- 플레이어 보이스 잘 들리는지 (기본 보이스 세팅, 증폭 필요한지)
- SFX, BGM, 비디오 플레이어 소리 잘 나오는지

### 카메라

- 포스트 프로세싱
  - [시네머신 버추얼 카메라](https://docs.unity3d.com/Packages/com.unity.cinemachine@2.10/manual/CinemachinePostProcessing.html)

- 들고다니는 유동 카메라
  - SmartSync ([LightSync](https://github.com/MMMaellon/LightSync))
  - 렌더 텍스쳐
  - 카메라를 들고 텔레포트를 하면, 싱크 오브젝트 특성 상 이동하는 위치 사이를 Lerp하게 보게 된다.

- 고정 카메라
  - 아바타가 해당 위치에서 의도한 대로 적절히 보이는지

### 그 외

- 프로젝트 후기는 바로바로 작성

## � 팁

---

### 오류 로그는 안뜨는데, 원하는 대로 작동안할 때

코드 잘못짜서 생긴 논리 오류를 제외하고,  

1. 호출하고자하는 CustomEvent가 Public 접근 제한자인지 확인한다
2. 똑같은 UdonBehaiviour가 여러 개 들어가있는지 확인한다 (프리팹에 Udon 추가하는 과정에서 주로 발생)

### Udon 싱크 크기

- <https://doc.photonengine.com/en-us/pun/current/reference/serialization-in-photon>
- 싱크 변수가 정말 많으면 싱크가 동작하지 않음.

### UI 인터렉션 가능하게 하는 조건 3가지

1. 오브젝트 Layer Default
2. VRC UI Sharp 컴포넌트
3. Box Collider

### VRChat World에서 VideoPlayer로 데이터 불러오기

- [링크1](https://feralresearch.org/lab/api-calls-from-inside-vrc/)
- [링크2](https://ask.vrchat.com/t/http-requests/1803)
- [링크3](https://github.com/Roliga/udon-video-decoder)
- [링크4](https://gitlab.com/anfaux/pixel-proxy/-/blob/main/server-node/modules/encode.js)
- [링크5](https://vrchat.com/home/launch?worldId=wrld_7508e408-ba6a-4478-b772-6af430c89286&instanceId=51500~private(usr_74fd4823-008f-4434-969c-c892e7c143e2)~region(eu)~nonce(031b2879-124f-4943-b075-2700f61ee200))

### Fog

`Fog` 안켜둔 채로 빌드하면, 런타임에서 Fog를 켜도 적용이 안됌.  
켜둔 채로 빌드하고, 월드 들어오자마자 꺼주기.  

220128 기준.  

### ContextMenu

`ContextMenu` 사용 시,  
오브젝트를 껐다키는 등의 단순 명령들은 잘 실행되지만,  
변수 값을 변경하는 등의 명령은 제대로 실행되지 않음.  

```cs
int temp = 0;

void Update() { Debug.Log(temp); }

[ContextMenu("Add")]
void Add() { temp++; Debug.Log(temp); }
```

예를 들어, 위 같은 코드에서 `Add`를  
`SendCustomEvent`로 호출하면 `Add`에서 `1`, `Update`에서 `1`이 찍히는데,  
`ContextMenu`로 호출하면 `Add`에서 `1`, `Update`에서 `0`이 찍힌다.  

테스트 시 주의  

220628 기준.  

### 오버레이 UI

Overlay 캔버스로 만들어도 되지만, VRChat에서 메뉴(R키)를 조작할 때 Overlay 캔버스가 전부 꺼지는 문제가 종종 있음  
-> 대신 DepthOnly 카메라로 WorldSpace 캔버스를 찍어서 위에 띄우기  

### ?

`SyncMode(None)`인 `오브젝트 토글 우동 A`로  
`우동 B가 포함된 오브젝트`를 자식으로 가지는 부모 오브젝트 C를 활성화 시킬 때,  

`우동 B`가 `SyncMode(Continue or Manual)`이면 `우동 B` 동작안함  
`우동 B`가 `SyncMode(None)`이면 `우동 B` 동작함  

다시말해,  
`A`가 `SyncMode(Continue or Manual)`이면 `B`의 `SyncMode`가 뭐든간에 `B`가 정상적으로 동작  
`A`가 `SyncMode(None)`일 때, `B`의 `SyncMove(Continue or Manual)`라면 `B`가 동작안함  

### 메모

- [VRC, 영화 자막](https://x.com/vr_hai/status/1495774702521958407?s=20)
- Light Probes Volme Settings
- Light Probe Group
- Occlusion Area
- Editor Only 태그 활용
