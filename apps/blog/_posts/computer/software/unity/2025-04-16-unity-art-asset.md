---
title: "Unity Art Asset"
# description: ""
categories: [컴퓨터, 소프트웨어]
tags: [유니티]
image: "/assets/img/background/20240827-140647.jpg"

date: 2025-04-16. 21:15 # Init
# last_modified_at: 2025-04-16. 21:15
---

## 머리말

---

## 아트 관련 Package

---

UNITE Seoul 2025  
`Keijiro Takahashi` 님의 `성공적인 Unity 아트 프로젝트를 위한 6가지 강력한 패키지` 세션 기반.  

### VFX Graph

Unity 내장 Package  
GPU를 이용해서 움직이는 파티클을 그래프를 써서 제작 가능  

선능 좋고, 쓰기 쉬운 GUI 등  

최대 장점으로는 다듬어진 기술 설계  
파티클 쓴 사람 알 수 있겠지만, 용도 다양하고 성능 좋고 그런 게 많이 없다  
반면 VFX Graph 기술적으로 우수해서 계속 쓰게 된다.  

특히 아트 분야에서는 활용이 두드러진다  
Unity 보다도 VFX 어플리케이션을 쓴다는 느낌  
그만큼 중요하게 느껴진다  

### VFX Graph Assets Package

그래서 [VFX Graph Assets Package](https://github.com/keijiro/VfxGraphAssets)  

표준 기능에는 부족한 부분이 있다  
그런건 커스터마이징  

여러 프로젝트에서 공유해서 쓸 수 있는 공통 부분이 포함됨.  

#### TextureLess Particle Shader

Parameter 만으로 Particle 용 텍스쳐를 만들어내는.  
단순한 파티클을 만들기 위해 텍스쳐를 만드는 것은 비효율.  

#### Depth of Field - Particle Shader

보케/블러를 만드는 셰이더  
이를 통해 깊이감 있는 이펙트를 셰이더로  
\+ 블러  

#### Exponential Random

난수/ 지수함수적인 편향을 주는 함수  

예를 들어 화면에 0~100 크기의 원을 랜덤하게 그린다고 했을 때,  
일반 랜덤은 큰 파티클이 눈에 띄게 많이 보이게 된다  
작은 건 그 그림자에 가려져서 잘 안보인다  

이런 문제를 해결하기 위해 이 함수 (Exponential Random)가 쓰인다  
함수에 편향을 줘서 작은 파티클도 잘 보이게  
인상이 평군화되어 있다  
큰 파티클이 비교적 듬성듬성 있는  

Particle Lifetime에도 적용할 수 있다  
임펙트를 주고 싶은 경우, Particle이 계속 남아있는 경우 좀 그렇다  
여운을 남기는 것은 이 노드를 쓰는 편이  

#### SetRandom Angle, SetRandom Angular, Velocity

단순히 각도를 랜덤화하면  
(구체) 아래와 가운데에 편향성이 생긴다  
이걸 이용하면 평균화된다 (편향없이 분포가 된다)  

마찬가지로 랜덤 벡터도 랜덤을 쓰면 편향이 생기는데  
RandomVector/Sphere가 있다  

#### 그 외

그 외에도 여러 노드  

- Tween
- Axis Rotation
- Divergence-FreeNoiseField
- Low Frequency Field
- ...

### VJUITK

[VJUITK](https://github.com/keijiro/VJUITK)  
VJ UI for UI Toolkit  

간단하게  
BUtton, Knob, Toggle에 대한 컨트롤  

Dense Layout Support  
몇 가지 중요한 컨셉을 위해 설계된 것  

밀도가 높은 레이아웃에  
UX를 희생하더라도 어쨌든 많은 파라메터를 조작해야  
컴팩트하게 줄 세우기 좋은 디자인을 만들 필요가 있다  

슬라이더 여러 개 보다는 Knob 쓰는 것이 좋겠다  

이런 컨트롤은 멀티 조작이 가능해야  
한 쪽은 부드럽게, 한 쪽을 격렬하게 조작하고 싶은 경우  
2개 이상의 컨트롤을 동시에 컨트롤 할 수 있어야  
이것은 UI Toolkit에서? 이미 지원되고 있어서 쉽게 구현할 있었다  

Enhanced Visibility  
시인성이 좋은가 <- 중요  

UI랑 Preview가 서로 겹쳐서 잘 보이도록 시인성이 좋아야  
Preview랑 UI를 분리해도 되는데,  
눈 왔다갔다 하는것이 불편  
그래서 요즘은 가능하면 둘을 겹쳐서 보기  
보면서 동시에 파라메터 조정 가능  

[VJ UI for Unity GUI (uGUI)](https://github.com/keijiro/VJUI)  
같은 기능을 UGUI에도 만들었는데 지금은 (더) 지원 안함 (UI Toolkit 써서)  
줄 세우기 같은 UI는 Toolkit이 좋다  

### Minis

[Minis](https://github.com/keijiro/Minis)  
MIDI Input for new Input System  

VJ같이 리얼타임 감각적인 파라메터가 필요한 경우  
MIDI 컨트롤러  
이런 외부 하드웨어  

InputSystem을 MIDI에 대응시키는  

지원하는 타입

- Midi Notes
  - 128Notes
  - Velocity 건반 눌렀을 때의 강도 (pressure)
- MiDI Control Change (CC)
  - 128 control (쓰로틀, 보통 LR 좌우만 있는데, 128개면 대단하죠 - 주요 이유)
  - 7 bit value 127

MIDI: 건반을 눌렀을 때 정보 (어떤 트리거적인 이벤트)  
CC는 값의 변화 노드/페이더로 입력하게 된다  

이것을 nodeGraph로 잘 쓸 수 있게 변환  

C#에서 어케 쓸까?  

PlayInput 같은 걸 쓸 수 있는데,  
이런건 인게임을 위해 만들어진 것이라  
아트 작업에는 쓰기 어려울 수 있다  

대신 InputAction을 쓸 수 있다  
이는 범용적  
입력 검출 가능, 물론 MIDI 컨트롤러에 대한 것도 수용 가능  

실제 사용방법  

```cs
public InputAction ~;
// 바인딩 인터페이스 띄우기 가능  
// 이 드롭다운에서 AddBinding
// 더블 클릭하면 Dialog가 나오고 경로를 클릭하고, Listen을 누르면 입력 대기 상태
// 이 상태에서 미디 컨트롤러 누르면 자동으로 설정된

// 자동 검출 뿐만 아니라
// T를 눌러서 Path를 이용해 모시꺵할 수 있다

// 문법
// #(Display)/Name -> 복수 여러 개 동시에 설정 가능

// (*Channel)/note060
// (*Channel)/note* -> 와읻드 컨트롤롤

testAction.ReadValue<float>();

testAction.Enable();
testAction.performed += _ => ~vfx.Play()

// 미디
// 키보드
// 혹은 둘 다 쓰는 등
// 유연하게 대응 가능
```

악기 본래의 사용 방법도 쓸 수 있는  
악기의 연주를 리얼타임 비주얼 활용 가능  

MIDI는 악기를 위한 프로토콜  
이것이 본래의 사용 방법일 것  

음악/비주얼  
완전히 싱크되는 것은 보통 어려운데, 잘 할 수 있다  

### LASP

[LASP](https://github.com/keijiro/Lasp)  
Low-Latency Audio Signal Processing Plugin  

직접 음악(음형/파형)을 이용해서 비주얼라이제이션  

저시현 오디오 입력/간단한 파형 분석을 지원하는  

유니티 내장 오디오 시스템 X
libsoundio 라는 외부 라이브러리 사용  
실시간성 높은 오디오 처리  
멀티 패널 처리도 가능 (이것은 예술 분야에서는 중요한 기능)  

Audio Level Tracker Component  
입력을 얻을 수 있는?  

프로퍼티 바인더를 이용해섶 값을 얻을 수 있다  

Frequency filters  
고음/저음 필터링도 가능  

Spectrum Analyzer FFT Component  
자세히 주파수 성부 분석  

#### LASP VFX  

[LASP VFX](https://github.com/keijiro/LaspVfx)  

샘플 전혀 C# 사용하지 않음  
LASP와 VFX만 써서  

### KlakNDI

[KlakNDI](https://github.com/keijiro/KlakNDI)  
NDI Video Connection Plugin  

랜 상으로 비디오 송신  
NDI를 유니티 내에서..?  

(번역 제대로 못 들음)  

어떤 것이 가능한가  

Camera -> HDMI Capture -> COmputer - HDMI - Computer  
일반적으로는 캡처보드 등을 써야하는데  
이런게 많아지면 복잡하고, 문제 가능성도 높아짐  

그림으로는 일자이지만, 현장은 더 복잡하게 함  
케이블 길이 제한도 있고 자유롭지 않다  

이를 한 개의 같은 것에 연결하는 것만으로 간단하게 만들 수 있다 (Like 중앙집중처리방식)  
어쨌든 같은 것에 연결되어 있으면 통신 가능하기 때문에  
쉽게 추가도 가능하다  

이런 것은 길이 자유가 있고, Wifi도 사용 가능하다.  
(큰 공연장은 어렵지만, 작은 폐쇠된 곳이라면)  

iphone Video를 유니티에 (같은 랜)  
Ndi Receiver 컴포넌트를 추가하기만 하면 되는  
NDI Name 에서 선택하기만 하면 된다  

이렇게 간편하게 연결할 수 있다  

유사한 프로토콜이 있긴한데, NDI 프로토콜 장점  
설정이 간단하다  
같은 랜 이기만 하면  그 자체로 연결 가능하다  

에코 시스템 (개발 환경/생태계)  
NDI 지원하는 것이 많다  

스마트폰에서 쓸 수 있는 NDI 앱이 있다?  
= 쉽게 카메라를 늘릴 수 있다?  

개인적으로 중요한  
PowerFrame 메타 데이터  

이미지 뿐만아니라 metaData를 포함하여 보낼 수 있고, 자유럽게 정의 가능 (사용자가)  
XML 형식이라면 뭐든지 가능하다 라는 느낌  

위치 회전 추적 데이터를 같이 보낸다던지  
화면 UI 입력 정보를 같이 보낸다던지  

싱크 깨질 염려가 없다  
엄격하게 동기화 가능  

어떤 의미인가?  

트래킹 같은 확장 데이터를 보내기 위해서는  
독집된 별개의 스트림을 만들어야 한다  
2개 있다는 것  
이것은 엄격하게 싱크로 되어 있어야  
차이가 있다면 어색해보인다  

타임 코드를 만들거나, 버퍼를 만들거나  
그런 복잡한 방법을 고려해야 하는데  

이미지에 메타데이터를 하나로 모이게 되면  
동기화가 어긋날 모시꺵이 없다  

AR 관련 플젝트에서 많이 쓴다  

Color  
Depth  
Human Stencil 이미지  

메타 데이터에는 Camera Position + Rotation + UI Input Data  
이런 것들을 Base64 Encoding 해서 XML로 보내고  

이런 걸 이용해서 볼류메트릭 비디오를  

동기화가 어긋날 일도 없고, 스트림 1개, 연결도 간편하다 (랜)  

### Procedural Motion Package

[Procedural Motion](https://github.com/keijiro/ProceduralMotion)  

가장 간단하지만  
가장 많은 곳에서 쓰이는  

절차적인 움직임 컴포넌트  

- Brownian
  - 가장 많이 쓰이느
  - Noise, 무작위 적인 움직임
  - 주파수, 옥타브 수를 조정해서 세밀하게, 마구잡이로 움직이는 다양한 움직임 구현
  - 카메라 랜덤 움직임 (대표적인 사용 사례)
  - 손으로 들고 있는 것처럼 강한 움직임
  - 혹은 크레인 등을 이용한 부드러운 (약하게)

- Natural Animation
  - 짧을 애니메이션을 쓰면 부자연스러운
  - 의미는 없지만, 부드럽게 -> 자연적인 것 등에서 쓰기 적절하게 간편하다
  - 나비 움직임
  - 풍향계 움직임을 이용한

- 좀 더 복잡한
  - Human Idle
  - 정지한 인간의 자연스러운 움직임
  - 무중력 상태 등에서
  - 간단한 구조이지만, 설득력 있는 움직임을 보인다다

cyclic  
Linear  
Random

- Smooth Motion
  - Exponential
  - Spring
  - Damped Spring
  - @ Ease 같은?

#### Procedural Motion Tack Package

Timeline으로 쓸 수 있도록  
되감기 / 반복 / Fade(Cross) 가능  

(세션 시간 부족으로 4가지 소개 하고 생략)  

## 메모

---

### 참고
