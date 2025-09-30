---
title: "승리의 여신: 니케, 2주년 업데이트 광고"
description: "'승리의 여신: 니케, 2주년 업데이트 광고' 컨텐츠."
categories: [작업물, 버추얼]
tags: [작업물, VRChat, 유니티]
image: "/assets/img/post/works/nikke-2nd-anniversary/nikke-2nd-anniversary-ppt-title.png"
hidden: true

# date: 2024-10-19. 09:55
date: 2024-11-04. 00:0
last_modified_at: 2024-11-18. 06:24 # Init
---

역대급 바디수트 신의상 출시...?!ㄷㄷ  
{% include embed/youtube.html id = "E9C4HeGa2bQ" %}

(진짜 변태같다)  
{% include embed/youtube.html id = "HczXoXZ0PMs" %}

초특급 섹시 등장  
{% include embed/youtube.html id = "8R0I0Xa5Gbw" %}

니케 2주년 이벤트! (with. 엔젤님, 이슬 성우님)  
{% include embed/youtube.html id = "rwP5rQXirUA" %}

니케 2주년을 맞이하여 라푼젤 성우님이 오셨습니다!  
{% include embed/youtube.html id = "VUseKBVODg0" %}

## 머리말

---

24년 11월 04일에 진행된 '승리의 여신: 니케, 2주년 업데이트 광고' 컨텐츠입니다.  

당시 여러 인플루언서 광고가 진행되었으며,  
이세계아이돌(주르르, 고세구, 비챤), 뢴트게늄, 엔젤(김수현 아나운서)이 참여한 본 컨텐츠는 '패러블 엔터테인먼트'에서 주관하였습니다.

니케 앨리스 성우님께서 인터뷰 게스트로 참여하셨습니다.  

### 참여 / 담당

본 컨텐츠는 VRChat 게임 내에서 진행되었습니다.  

해당 컨텐츠에 사용된 VRChat World 내 기능 프로그래밍을 담당했습니다.  
또한 방송 진행을 위한 오퍼레이팅, 기능 조작을 담당했습니다.  

- `Unity Overlay Canvas`를 이용해 컨텐츠 내 PPT 및 영상 화면을 풀 스크린으로 출력하는 기능
- 카메라 시스템
  - 에디터 타임에 카메라 구도를 미리 설정
  - `CinemachineVirtualCamera.Priority`를 조정하여 구도 전환
- '위험한 초대' 미니 게임 기능
  - `Unity Animator`/`VRChat Station`을 이용한 플레이어 애니메이션
- '두뇌 풀 가동' 미니 게임 기능

### 사용한 툴

- Unity 2022.3.22f1
- [U# (C# + VRChat SDK)](https://udonsharp.docs.vrchat.com/)
- [Woodon](https://github.com/wrchat/Woodon)

Unity Package를 통해 팀원과 파일을 공유했습니다. (파일 공유 수가 적었습니다.)  
Discord를 통해 팀원/클라이언트와 소통했습니다.  

## 과정

---

24년 10월 04일, 본 광고 프로젝트를 주관한 '패러블 엔터테인먼트'로부터 프로젝트 제안을 받았습니다.  

24년 10월 14일, 프로젝트 팀에 합류하였으며,  
24년 10월 25일 - 11월 04일, 약 2주 간 작업했습니다.  

### 위험한 인터뷰

레퍼런스: 우왁굳 - '위험한 초대라는 예능 기억나십니까..?'  
{% include embed/youtube.html id = "GYPX6mj0CMA" %}

![위험한 인터뷰](/assets/img/post/works/nikke-2nd-anniversary/241118-081759.png)
![위험한 인터뷰](/assets/img/post/works/nikke-2nd-anniversary/241021-145341.png)

왁굳님의 '위험한 초대' 컨텐츠를 레퍼런스로 삼아 제작했습니다.  
구현 관점에서 요약하면 1. `VRCStation`으로 플레이어 위치를 고정시키고, 2. `VRCStation` 오브젝트를 애니메이션으로 날려버리면 되는 미니 게임입니다.  

게임 **Among Us**에서 임포스터가 우주선 밖으로 방출되는 상황이 있는데 (OO님은 임포스터가 아니었습니다), 여기에서 아이디어를 얻어 트리거가 된 플레이어가 우주선 밖으로 방출되는 애니메이션을 만들었습니다. 플레이어가 허우적거리는 애니메이션을 입혔으면 더 재밌는 장면이 나왔을 것 같은데, 마땅한 애니메이션을 찾지 못해 다소 빳빳하고 어색하게 날라가게 된 점이 조금 아쉽습니다.  

`Animation Event`를 통해 애니메이션이 처음 재생될 때 SFX를 재생하고, 플레이어가 어느 정도 날라갔을 때 `Camera.Priotiry`를 조정하여 플레이어가 날아가는 모습을 볼 수 있도록 시점을 바꿨습니다.  

게임 특성상, 플레이어가 날라가는 상황이 많다는 점을 고려하여, 카메라 시점은 날라가는 플레이어 본인만 바뀌도록(로컬) 설정했습니다.  

### 두뇌 풀 가동

레퍼런스: 우왁굳 - '전설의 그 예능... 재현했습니다.'  
{% include embed/youtube.html id = "T3Vh8hWXwTU" %}

![두뇌 풀 가동](/assets/img/post/works/nikke-2nd-anniversary/241118-081507.png)
![두뇌 풀 가동](/assets/img/post/works/nikke-2nd-anniversary/241030-155453.png)

'위험한 인터뷰'와 마찬가지로 왁굳님의 '두뇌 풀 가동' 컨텐츠를 레퍼런스로 삼아 제작했습니다.  
구현 관점에서 요약하면 1. 문제를 띄우고, 2. 애니메이션으로 벽을 이동시키면 되는 미니 게임입니다.  

`Animation Event`를 통해 애니메이션이 처음 재생될 때 SFX를 재생하고, 벽이 끝에 다다랐을 때 `Camera.Priotiry`를 조정하여 플레이어가 떨어지는 모습을 볼 수 있도록 시점을 바꿨습니다.  

게임에 참가한 플레이어가 정답을 맞추지 못해 벽에 밀려나고, 우주 밖으로 떨어지고 있는 상황에만 시점을 바꿔주고 싶었습니다. 스테이지 밖에 콜라이더를 하나 두고, 플레이어 위치를 `bounds.Contains(playerPos)`로 특정하여, 플레이어가 콜라이더 안에 있는 경우(떨어지고 있는 경우)에만 카메라의 부모 오브젝트를 활성화하여 시점을 바꿔주었습니다.  

### 진행

사소한 이슈가 조금 있었습니다.  

- **VRChat**과 **니케**가 같이 켜지지 않음
  - 뢴트님께서 방송 전에 미리 확인해주신 부분
  - VRChat을 끄고 니케 플레이를 진행해주셨습니다.
- 이슈
  - 엔젤님 VRChat 계정에서 `VR챗 옵션 -> 편의성 및 안전 -> 신뢰할수없는 URL 허용 체크`가 안되어 있어서 중간에 리조인이 필요했다.
    - 엔젤님 시점은 방송되지 않았기 때문에, 방송 진행 상 문제는 없었습니다.
  - 아바타 키
    - 플레이어들간의 키 밸런스가 맞지 않았던 문제. 컨텐츠 진행 인게임에서 조정해주셨습니다.
  - 소리
    - 기본 플레이어 보이스 범위를 키웠지만, 실제 방송 때 작게 들렸다.
    - 방송 송출 소리도 조금 작았다.

## 후기

---

'이 컨텐츠 구성이 방송에서 재밌게 나올 수 있을까' 하는 걱정이 조금 있었는데, 엔젤님과 이세돌/뢴트게늄님께서 역시 재밌게 상황을 잘 살려주셔서 생각보다 재밌게 진행이 되었던 것 같습니다.  

## 기록

---

![241021-145341](/assets/img/post/works/nikke-2nd-anniversary/241021-145341.png)
![241021-145415](/assets/img/post/works/nikke-2nd-anniversary/241021-145415.png)
![241021-145428](/assets/img/post/works/nikke-2nd-anniversary/241021-145428.png)
![241021-145433](/assets/img/post/works/nikke-2nd-anniversary/241021-145433.png)
![241021-145440](/assets/img/post/works/nikke-2nd-anniversary/241021-145440.png)
![241021-145445](/assets/img/post/works/nikke-2nd-anniversary/241021-145445.png)
![241021-155535](/assets/img/post/works/nikke-2nd-anniversary/241021-155535.png)
![241029-141808](/assets/img/post/works/nikke-2nd-anniversary/241029-141808.png)
![241030-143600](/assets/img/post/works/nikke-2nd-anniversary/241030-143600.png)
![241030-155408](/assets/img/post/works/nikke-2nd-anniversary/241030-155408.png)
![241030-155453](/assets/img/post/works/nikke-2nd-anniversary/241030-155453.png)
![241031-115959](/assets/img/post/works/nikke-2nd-anniversary/241031-115959.png)
![241031-120431](/assets/img/post/works/nikke-2nd-anniversary/241031-120431.png)
![241103-200625](/assets/img/post/works/nikke-2nd-anniversary/241103-200625.png)
![241104-000000](/assets/img/post/works/nikke-2nd-anniversary/241104-000000.gif)
![241104-184032](/assets/img/post/works/nikke-2nd-anniversary/241104-184032.png)
![241104-184708](/assets/img/post/works/nikke-2nd-anniversary/241104-184708.png)
![241104-184744](/assets/img/post/works/nikke-2nd-anniversary/241104-184744.png)
![241104-184800](/assets/img/post/works/nikke-2nd-anniversary/241104-184800.png)
![241104-185846](/assets/img/post/works/nikke-2nd-anniversary/241104-185846.png)
![241104-190216](/assets/img/post/works/nikke-2nd-anniversary/241104-190216.png)
![241104-190238](/assets/img/post/works/nikke-2nd-anniversary/241104-190238.png)
![241104-190438](/assets/img/post/works/nikke-2nd-anniversary/241104-190438.png)
![241104-190451](/assets/img/post/works/nikke-2nd-anniversary/241104-190451.png)
![241104-190849](/assets/img/post/works/nikke-2nd-anniversary/241104-190849.png)
![241104-190916](/assets/img/post/works/nikke-2nd-anniversary/241104-190916.png)
![241104-190933](/assets/img/post/works/nikke-2nd-anniversary/241104-190933.png)
![241104-190935](/assets/img/post/works/nikke-2nd-anniversary/241104-190935.png)
![241104-190940](/assets/img/post/works/nikke-2nd-anniversary/241104-190940.png)
![241104-190954](/assets/img/post/works/nikke-2nd-anniversary/241104-190954.png)
![241104-191002](/assets/img/post/works/nikke-2nd-anniversary/241104-191002.png)
![241104-191021](/assets/img/post/works/nikke-2nd-anniversary/241104-191021.png)
![241104-191038](/assets/img/post/works/nikke-2nd-anniversary/241104-191038.png)
![241104-191047](/assets/img/post/works/nikke-2nd-anniversary/241104-191047.png)
![241104-191101](/assets/img/post/works/nikke-2nd-anniversary/241104-191101.png)
![241104-191106](/assets/img/post/works/nikke-2nd-anniversary/241104-191106.png)
![241104-191136](/assets/img/post/works/nikke-2nd-anniversary/241104-191136.png)
![241104-191227](/assets/img/post/works/nikke-2nd-anniversary/241104-191227.png)
![241104-200712](/assets/img/post/works/nikke-2nd-anniversary/241104-200712.png)
![241104-200722](/assets/img/post/works/nikke-2nd-anniversary/241104-200722.png)
![241104-200740](/assets/img/post/works/nikke-2nd-anniversary/241104-200740.png)
![241104-200749](/assets/img/post/works/nikke-2nd-anniversary/241104-200749.png)
![241104-202336](/assets/img/post/works/nikke-2nd-anniversary/241104-202336.png)
![241104-202658](/assets/img/post/works/nikke-2nd-anniversary/241104-202658.png)
![241104-210151](/assets/img/post/works/nikke-2nd-anniversary/241104-210151.png)
![241104-210305](/assets/img/post/works/nikke-2nd-anniversary/241104-210305.png)
![241104-210648](/assets/img/post/works/nikke-2nd-anniversary/241104-210648.png)
![241104-210839](/assets/img/post/works/nikke-2nd-anniversary/241104-210839.png)
![241104-211050](/assets/img/post/works/nikke-2nd-anniversary/241104-211050.png)
![241118-062254](/assets/img/post/works/nikke-2nd-anniversary/241118-062254.png)
![241118-081507](/assets/img/post/works/nikke-2nd-anniversary/241118-081507.png)
![241118-081759](/assets/img/post/works/nikke-2nd-anniversary/241118-081759.png)
![241119-140500](/assets/img/post/works/nikke-2nd-anniversary/241119-140500.png)
![nikke-2nd-anniversary-game2-logo](/assets/img/post/works/nikke-2nd-anniversary/nikke-2nd-anniversary-game2-logo.png)
![nikke-2nd-anniversary-ppt-title](/assets/img/post/works/nikke-2nd-anniversary/nikke-2nd-anniversary-ppt-title.png)

- 일정
  - 1104 18:00 방송 대기: VRChat 엔젤님 영접
  - 1104 19:00 방송 진행: Discord 스태프 모여서 오퍼레이팅/방송 진행
- Link
  - PV: `https://youtu.be/zyB4V_xhSps?si=Pko64FAYMXUmxKJm`
  - 뢴트게늄님 SOOP 주소: `https://play.afreecatv.com/jey422/278399548`
- 그 외
  - [엘리스디지털배움체](https://noonnu.cc/font_page/671)
  - 컨텐츠 플로우
    1. 가챠 컨텐츠
    2. 스토리 컨텐츠
    3. 위험한 인터뷰
    4. 두뇌 풀 가동

| 참가자   | 금지어      | 퀴즈 정답 유무 | 최종 등수 |
| -------- | ----------- | -------------- | --------- |
| 주르르   | 표현        | OOOOOX         | 1         |
| 고세구   | 니케        | OXOXOX         | 3         |
| 비챤     | 음... 그... | OXOOOO         | 2         |
| 뢴트게늄 | 라푼젤      | _              | _         |
