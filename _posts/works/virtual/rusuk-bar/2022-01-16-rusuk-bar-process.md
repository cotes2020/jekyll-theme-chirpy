---
title: "RusukBar | 루석바 - 제작 과정"
description: "왁타버스 고정멤버 해루석님을 위한 VRChat 월드 제작 프로젝트."
categories: [작업물, 버추얼]
tags: [작업물, VRChat, 유니티, 루석바]
image: "/assets/img/post/works/rusuk-bar/rusuk-bar-banner.png"
hidden: true

date: 2022-01-16. 00:00
# last_modified_at: 2024-11-09. 08:07 # Init
last_modified_at: 2024-11-11. 02:55 # Init
---

{% include embed/youtube.html id = "I5bkJ4S9qic" %}

## 머리말

---

'루석바'를 만들기까지의 과정.  
내가 VRChat과 3D CG를 처음 입문을 하고 공부해나가는 과정이기도 하다.  

## 시작

---

때는 2022년 1월. 당시 '우왁굳 2021 연말 공모전'에 왁굳님을 위한 팬 게임 '왁그리드'를 만들어 출품한 이후, 여러 복잡 미묘한 감정을 느끼던 때가 있었다. 실패도 성공도 아닌 애매한 결과, 지난 몇 달간 열심히 달려온 이후 찾아온 번아웃. 그와 동시에 찾아오는 다른 팬 게임들에 대한 열등감. 무엇인지는 모르겠지만 - 일단 한 단계 나아가야 할 것만 같은 불안감. 그때 눈에 들어왔던 것이 VRChat이었다. '그래, 2D 게임 한 번 만들어봤으니, 이번엔 3D도 한 번 공부 해봐야지.' 했던 생각을 시작으로.. 시작했다.  

## 과정

---

### VRChat, 3D CG 입문

무작정 입문했다.  

당시 VRChat SDK 버전이 2에서 3로 막 넘어가던 시점이었다. SDK 2는 CyanTrigger 라는 패키지를 이용하여 VRChat World 내에 기능을 구현할 수 있었고, SDK 3부터는 노드 그래프인 Udon Graph 혹은 U# (C# + VRChat SDK)으로 월드 내 기능을 유저가 직접 구현할 수 있었다. 아무래도 복잡한 기능을 만들기 위해서는 SDK 3를 사용해야 했고, 때문에 SDK 2 대신 SDK 3를 바로 공부하기로 했다. 하지만 앞서 말했듯, 이제 막 버전이 넘어가던 시점이라 SDK 3의 자료가 많이 없었다. 그래서 상대적으로 자료가 많던 SDK 2 자료로 전체적인 틀을 이해하고, 이를 사용할 때는 직접 SDK 3에서 어떻게 적용되는지 찾아가며 공부했다. 이때 `Korea Udon Community`라는 디스코드 서버가 큰 도움이 되었다. 'rage147'님을 비롯한 여러 제작자들을 만나고 교류할 수 있었다.  

3D CG에 대한 이해가 전무했다. 전문적인 3D Tool? 그런 건 건들 용기조차 나지 않았다. 나는 VRChat World 제작 강의 영상에 자주 보이던 Unity 내장 패키지 ProBuilder를 이용하여 건물을 만들기 시작했다. 네모네모 사각사각한 벽을 만들고, ProBuilder가 알아서 펴주는 UV에, 텍스쳐 딸랑 한 장 들어간 머티리얼을 얹는 것이 내가 할 수 있는 전부였다. 복잡한 모델이나 텍스쳐가 필요한 것들은.. Unity 에셋 스토어(당시 '절대강좌 유니티5' 책에서 처음 딱 한 번 들어가보고 다신 들어갈 일이 없었는데)에서 괜찮은 Free 3D 모델들을 찾아 맵에 넣었다.  

- 당시 썼던 글들
  - [왁물원 (VRC 맵 제작 입문 기록)](https://cafe.naver.com/steamindiegame/4310077)
  - [왁물원 (Udon 질문 ) 스카이박스 Material 배열로 변경하는 그래프 오류)](https://cafe.naver.com/steamindiegame/4321339)
  - [왁물원 (Udon 신사)](https://cafe.naver.com/steamindiegame/4328558)
  - [왁물원 (OnPlayerCollision 이벤트)](https://cafe.naver.com/steamindiegame/4333352)
  - [왁물원 (질문 ) Material 텍스쳐 방향?)](https://cafe.naver.com/steamindiegame/4340459)

이제보니 게시글 댓글에 지금은 익숙한 닉네임들이 많이 보인다. 비슷한 시기에 맵 제작을 시작하신 분들이 많았다.  

### VRChat 맵 제작 스터디

사실 이때 쯤 `왁타버스 맵 제작 스터디` 2기 모집이 진행되고 있었다. VRChat과 3D CG 공부를 시작함과 동시에, `왁타버스 맵 제작 스터디` 2기에도 지원했다.  

![Wakta_VRChat_Study_Mail_0](/assets/img/post/works/rusuk-bar/Study/Wakta_VRChat_Study_Mail-0.png)

지금보면 참 부끄러운 지원서. 다른 건 그렇다치고, '최소한의 미적 감각은 있다고 생각합니다' 의 근거로 제시한 것이 '학교 예술 관련 과목 최고 성적' 이라는 것이 참 부끄럽다. 당시 대학교 1학년이었기 때문에, 결국 초중고 때 수업을 들었던 '미술' 과목 성적이 좋다는 것을 근거로 '난 미적 감각이 있어' 라고 주장한 것이다. 그와중에 보험 들어두겠다고 '최소한의' 를 앞에 붙인 걸 보니, 참 예나 지금이나 사람이 한결 같은 것 같다.  

어쨌든, 스터디에 들어가고 싶었던 마음은 진심이었다. 왁물원 카페 'VRChat 맵 제작소' 게시판에 '여기에 글 자주 쓰면 멘토분들께서 좋게 봐주시지 않을까?' 하는 의도를 가지고 게시글을 여럿 올렸었다. 전략이라면 전략.  

![Wakta_VRChat_Study_Mail-1.](/assets/img/post/works/rusuk-bar/Study/Wakta_VRChat_Study_Mail-1.png)

그리고 그 전략이 통했는지, 얼마 뒤 1차 서류 전형 합격 메일을 받았다.  

![220112-210207](/assets/img/post/works/rusuk-bar/Study/220112-210207.png)
![220112-215253](/assets/img/post/works/rusuk-bar/Study/220112-215253.png)

![Wakta_VRChat_Study_Mail_2](/assets/img/post/works/rusuk-bar/Study/Wakta_VRChat_Study_Mail-2.png)

이후 2022년 1월 12일, VRChat에서 2차 면접 전형을 보았고, 여기서도 합격을 받아 최종 합격이 되었다. 그렇게 '왁타버스 맵 제작 스터디' 2기에 멘티로 참여하게 되었다.  

면접을 회상해보자면 그렇다. 일단, 이 면접은 나에게 있어서 인생 두 번째 대면(?) 면접이었다. 대학 입시 때 미쳤다고 수시 6개 쓸 수 있는 걸 딸랑 2개만 썼었는데, 그것도 상향 교과 하나, 내 성적에 맞는 곳 종합 하나를 넣었었다. 그래서 난 입시 때 면접을 딱 한 번 봤다. 어쨌거나 이 TMI 에서 내가 하고 싶은 말은, 당시 내 인생에 있어서 제대로 된 면접 경험이 없었다는 것이다. 그래서 이 두 번째 면접을 많이 떨면서 진행했었다. 면접이 진행되는 과정(VRChat 화면)을 영상으로 녹화해뒀는데, 다시 보면서 내가 말 하는 걸 들어보면 정말 굳어있는 것이 느껴진다. 면접 성격도 그렇고, 당시 면접관 (김치만두번영택사스가님, 클로버님, 공속팬치님) 분들도 그렇고, 그렇게 막 딱딱한 분위기에서 진행되는 면접은 아니였는데, 나 혼자 쫄아서 굳어있었다.  

그래도 어쨌거나 과거 경험도 있고, 할 말은 어떻게 다 하기는 해서, 합격이라는 좋은 결과를 받은 것 같다. 스터디에서 밝히길, 가장 중요한 심사 기준은 "왁굳형 및 왁타버스 관련 VR챗 컨텐츠 조공 의도성 및 열정" 이었는데, 이 부분도 잘 풀어낸 것 같다.  

`Korea Udon Community` 디스코드 서버와 더불어, `왁타버스 맵 제작 스터디` 디스코드 서버에서도 많은 도움을 받으며 VRChat과 3D CG를 공부했다.  

### 어떤 맵을 만들까

며칠 공부하고 나니, 이제 제대로 된 맵 하나를 만들어 VRChat에 업로드 해보고 싶다는 생각을 했다. '어떤 맵을 만들어볼까?' 고민하다가, 고멤분들의 분위기에 맞는 월드를 만들어보자는 생각이 들었다. 나중엔 이런 월드를 하나로 합쳐서, 왁타버스 도시를 만들면 좋겠다는 생각도 했다. 그렇다, 원래는 모든 고멤분들의 월드를 만들려고 했다. 꿈도 참 크지. 어쨌든, 당시 내가 고멤분들 중에서 특히 좋아했던 해루석님을 위한 맵을 가장 먼저 만들어보기로 했다.  

이덕수 할아바이 - Memories of the last night (마지막 밤의 추억)  
{% include embed/youtube.html id = "4G06wJv76Go" %}

2321년 12월 31일에 업로드 된 고멤 놀이터 영상. 나는 당시 이 영상을 인상 깊게 봤었다. 재즈, 바, 노래와 MV의 분위기가 아주 마음에 들었다. 영상을 내 유튜브 음악 플레이리스트에 넣고 계속 돌려봤었다. 이 MV 맨 처음에 루석님이 바텐더로 잠깐 나오는데, 여기서 영감을 받아 '루석님의 바 건물을 만들자!' 라는 생각을 했다.  

핵심 주제가 정해진 후, 왁물원 'VRChat 맵 제작소' 게시판과, 여러 VRChat 맵을 구경하며 레퍼런스를 찾아다니기 시작했다.  

### 레퍼런스 및 구현

#### [비밀다방](https://vrchat.com/home/world/wrld_998d476a-78e1-4dd1-a4cd-79c98f5bc9cb)

왁물원 'VRChat 맵 제작소' 게시판에 공속팬치님께서 올리신 [비밀다방](https://cafe.naver.com/steamindiegame/4004362)이라는 VRChat 맵 제작 후기를 보았다. '비밀다방'은 21년 말 진행된 상황극 콘테스트 컨텐츠에 공개되었던 VRChat 맵으로, '비밀소녀님의 카페' 컨셉을 기반으로 만들어진 맵이다.  

여기서 핵심 아이디어를 얻었다. 지금도 그렇지만 당시에도 해루석님과 비밀소녀님은 자주 커플로 묶였고, 루석님의 건물을 만들 때 이를 기믹으로 쓰고 싶었고, 마침 있었던 '비밀다방'을 모티브로 루석님의 건물을 만들면 좋겠다고 생각했다. 무작정 따라 만들기보다는, '비밀다방'와 구조와 요소를 참고하고, 일부 요소를 반전되게 만들어보자는 생각이었다. 결과적으로 전체적인 건물 디자인은 유사하게 가져가되 (정면 외관 및 내부 구조), 입구을 바라보는 방향을 기준으로 좌우 반전을 시키고 (곡선 벽), 시간대를 낮과 밤으로 반전되게 만들었다.  

![비밀다방 & 루석바]()  

맵 이름도 '비밀(소녀) + 다방'의 조합처럼 '(해)루석 + 바'로 지었다. 이때부터 '루석바'라는 이름이 생겼다.  

#### [로망스 바](https://vrchat.com/home/world/wrld_d6e42474-9c72-4ef3-be92-848b0e9c5726)

### 제작

- 요소들
  - 그림 액자
  - 루서키다스
  - 무대
  - 철봉

- [VRC 맵 제작 기록 - 루석 바](https://cafe.naver.com/steamindiegame/4369899)
- [VR챗 루석바, 퍼블릭으로 올렸습니다!](https://cafe.naver.com/steamindiegame/4546786)

### 확장 공사

## 업데이트

---

- [Screen Space - Overlay 캔버스 UI를 클릭으로 상호작용하는 방법](https://cafe.naver.com/steamindiegame/4641015)
- [루석바 설정창 UI 업데이트](https://cafe.naver.com/steamindiegame/4863990)
- [개같이 뜬금 루석바 확장 공사 업데이트](https://cafe.naver.com/steamindiegame/5140269)
- [[비밀다방 X 루석바] Mood Indigo 업데이트 기념 콜라보 이벤트](https://cafe.naver.com/steamindiegame/5562196)
- [Mood Indigo 업데이트 기념 비밀다방 x 루석바 콜라보 이벤트 [루비를 찾아주세요!]](https://cafe.naver.com/steamindiegame/5562204)
- [포토샵/디자인/기획 하시는 분을 찾습니다](https://cafe.naver.com/steamindiegame/6355112)
- [○○시 시계 (Clwak)](https://cafe.naver.com/steamindiegame/6772551)
- [김피탕 토크쇼를 루석바 모든 맵에서 중계하세요](https://cafe.naver.com/steamindiegame/6847750)
- [@@ 김피탕 토크쇼 버튼을 눌러도 안보일 때 @@](https://cafe.naver.com/steamindiegame/6848008)

- 요소들

## 기록

---

### 메모

- <https://youtu.be/9S5WFb2uOsU?si=KnzmOWhCSrzsO2KS>
