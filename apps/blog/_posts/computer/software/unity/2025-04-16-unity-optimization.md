---
title: "Unity Optimization"
# description: ""
categories: [컴퓨터, 소프트웨어]
tags: [유니티]
image: "/assets/img/background/20240827-140647.jpg"
hidden: true

date: 2025-04-16. 20:43 # Init
# last_modified_at: 2025-04-16. 22:12 # 메모: Read/Write, MipMap
last_modified_at: 2025-05-28. 21:30 # +메모:
---

## 머리말

---

## A

---

공통적 최적화 이슈  
모바일 플랫폼 대상  
경험적인 부분들이 많이 포함됨  

### 높은 메모리 점유 요소

- Shader Variants
- Keyword를 추가하면 2의 지수 형태로 variant가 증가
  - Keyword가 10가 만 있어도 1024
  - Build in Shader나 상용 Asset 수정 시 keyword 증가 주의
- Dynamic SHader Loading
  - Chunck 단위 Variants 압축/해제
    - PlayerSetting / Shader Variant Loading Settings
    - 유니티 블로그를 통해서도 소개됨 (2021)
- Stripping 자동 제거
  - Project Settings/Grapjjics
- " 직접 제거
  - Play log에서 variant 정보 추출
  - GraphicsStateCollection.BeginTrace (Unity6)
- AssetBundle 관련련
- PersistentManager.Remapper
  - Instance ID와 File GUID, Local ID mapping 정보
    - 늘어나면 즐어들지 않기 때문에 최대 asset 개수 최소화
- SerialzedFile
  - Assetbundle metadata
- 장면에 필요한 asset만 포함된 bundle 구성 (사실상 불가능)
- 동기 Load된 bundle 개수 제한
- Instance 직접 관리로 bundle 해제 (개발 비용 상승)
  - AssetBundle.UNload(False)
    - 이때 로드된 인스턴스들도 같이 내려가서
  - Addressable - Custom assetbundle provider
    - 커스텀 필요

```cs
// AssetBundleProvider.cs

// 14번줄
~ m_AssetBUndle.UnloadAsync(true); // 수정?
```

### _

IL2CPP Metadata
일부 고정크기, 일부는 실행 중 크기 증가

- 전체 규모 화이
  - Platform seting ScriptBdebbuing 설정 (Debugging 정보 부하)
- Native Profiler
  - `il2cpp:: filtering`

- 관련 성정
  - Managed Code Stripping - Medium
    - Medinum이랑 다른거 성능 20%차이이
    - 코드 삭제 되는 경우
    - namespace 설정 가능
  - IL2CPP Code Generation - Faster (smaller) builds
  - Package Manager - 불필요한 package, plugin 제거
    - Build 후 다음 경로의 dll크기로 module 별 크기 유추
    - Library/Bee/artifact/(platform)

### 단편화 문제

- 잦은 작은 할당에 의한 2가지 이슈
  - Memory hole에 의한 amanaged heap 증가
  - 작은 할당에 의해 거의 빈 bucket의 해제 ㅂㄹ가
- 일반적인 가이드 (난이도 상)
  - 수명이 긴 데이터를 미리 할당하고 최대한 유지
  - ~
  - ~

고민해볼 만한 내용  

- 할당 규모와 빈도가 높은 주요 구간
  - Text data parsing - parsing 별 string 할당 규모 확인
  - 명시적 장면 전환 - 장면간 빈 장면 추가로 heap 확장 억제
    - 이전 씬, 다음 씬 메모리 모두 올라가는 경우가

- GarbageCollector..GCMode를 사용한 해제 시점 관리
  - (사례 부족으로 비추)
  - 동시 해제를 통한 단편화 감소 효과 ㄱ댜
  - 해제 시점의 CPU 부하 주의

### CPU Stuttering

주의점  - Profiler Marker 수집 부하  

- Call count에 비례한 오차 증가
  - 의심스러운 경우 native profiler의 결과와 꼭 비교
- GC Allocation의 call stack 확인
  - Deep Profiler대신 call stack 옵션 사용

Shader.CreateGPUProgram  
셰이더 로드  

Shader warmup

- OpenGGL, DX11
  - Shader.WarmupAlShaders
  - ShaderVariantCollection.WarmUp
- +
- off ScreenRendering
  - 가장 안전?

Instantiate

- 최적화
  - 오브젝트 풀 사용
  - Serialized field 구모에 비례한 CPU 부하
  - 생성시점분리 예) 캐릭터 렌더러와 정보
  - Object. Instantiate Async
    - Produce & Copy 비동기 (Deserialied 크기) -> Job으로 돌리고, AWake 비동기

GC.Collect

- GC allocation 제거로 GC.Collect 지연
  - 특히 Update, 반복문 비명시적 할당 (object.name 이런 복사본 반환 멤버 주의)
- Incremental GC
- 가장 확실한 방법 GCMode 활용에 대한의견
  - 메모리 사용량 증가에 대해 관리가 가능한 사왕인지 검토

### 주요 CPU 부하 요인

주요 CPU 부하 요인

REndering

- Draw 고급 기능 활용
  - **SBP Batcher, Instancing
  - Indirect draw, Batch Render Group API
  - GPI Resident Drawer (MeshRenderer, Foward*, Unity 7)
- Cull
  - 원거리 초소형 물체 및 사전 제거
    - Camera Cull
    - Light
    - ~
  - Culling Group (장면 구조에 따라 유효, 그림자 문제)

UGUI

핵심 최적화 및 고려사항

- 동적 요소에 의한 Batc 비용 최소화
  - 동적.정적 요소 canvas 분리 (Animator, Scroll , ...)
- Layout system 부하
  - 계층 구조 최소화 및 비활성화
- REctMask2D cull query 부하
  - 대상 요소 규모에 비례함 (특히 Text 포함 시 급격한 증가 - 글자 단위로 Culling 하기 때문에)

Animator
~ 사진 참고  
Job을 통해서 돌아가는데  
Event 등을 쓰면?? MainThread에서 돌아가도록 되어 있음음

Skinning
~ 사진 참고
X  

기타  
불필요한 카메라  
 RenderObjects에 Override ?
 커스텀 렌더 피처?
 사용에 따라서는 ..?

동일한 REdnerer를 지정하는 경우도 잇는데  
서브 카메라가 꼭 필요한 경우가 아니라면 ,최적화된 별개의 Renderer를 만드는 방식으로..

물리  
Fixed UPdate

RayCase  
D
rawMesh는 직적 Culcing해야하는 부분이 고려  

### GPU 부하

PixelOVerDraw  
주로 복잡한 UI, Particle에서 겹쳐져 있는 부분  
파티클, 주로 Billboard를 쓰는데, 완전 투명한 영역이 있음  
타이트한 지오메트리를 쓰는 것을 추천  

훔질이 허용하는 선에서는 병합해서,  

UI, 복잡도가 높기 때문에  
꼭 가려지는 1,2개 정도 있는데  
꼭 꺼주시고  

Alpha 0 일때 아예 꺼주는 것이 있는데  
최신와서는 활성화되어 있는지 여부를 확인해야야

Shader  
SHader가 가지는 Pixel Coverage  
Native Profiler를 쓰면 , 셰이더, 스크린 기준으로 픽셀? 이 얼마나 쓰이고 있는 지 확인 가능  
직접 Shader 최적화 한다고 했을때, 정적 정보를 수집하시고, 사용되는 인스트럭션의 정밀도?? 등이 포함되는데, 높은 정밀도의 인스트력션인 경우 정밀도를 낮추는 경우 효과?? 위치나 시간과 과련된 인스트력션인 겨우는 주의를 해야 ㅎ마.

- 동적 분기 제거?
  - 동기성을 위해 제거할 필요가 있다?

픽셀보다 작은 삼각형의 경우 3개의 버텍스? 낭비가  
Micro triangle  

측정이 쉽지 않다  
모바일..?  
HDRP만 제공하느 ㄴ중  

NativePRofiler 쓸 수 있지만 ,쉽지 않다  

설명하기 어려운 GPU 부하는 이것을 의심해볼 수 있다.  

## B

---

5가지 내용  

### 성능타겟러정

가장 중요한 것은  Mindsetㅇ르  성능 주도적으로  
성능에 대해서 이야기 할 때는 

타겟 프레임 타임  
이것을 밀리타임 단위로 생각해야 함  
특정 프레임을 렌더링/마무리 CPU에서 마무리ㅏ는데 얼마나 많은 시간이 ㅓㄹ리는 지 중요하니까요  

주의할 부분  
특정 FPS 예로 30으로 하면 33, 60이면 16ms, 90ms 11ms 초가 됩니다  
그래서 밀리초 단위로 계산을 해야  

일관성이 있어야?  
FPS가 울렁이면 디터, 물리, 애니메션 등이 튄다  

성능 버퍼을 만들어야 한다  

그 가운데 65%만 써야 한다  
배터리를 아낄 수 있고, 성능이 약간 저하될 때 버퍼역할을 한다  
발열, 등  

숨을 돌릴 수 있는 공간을 둬야 한다  
공간을 둬서 쿨다운 할 수 이쎅  

버킷을 전부 쓰면 어렵다  

33ms 21ms초  
60FPX 16 10ms 초 등  

다양한 부분에서 프로파일링  
여러 종류의 기기 (저가 고가 OS ㅔ조자 드라이버 등)

일관성있는 프로파일링을 재현가능한 환경에서  
언제나 이씬을 자동으로 재생하면서  혹은 렌더링함녀서  
일반적인 고객처럼 일관성 있게 재현 가능해야  
일관성 확보가 중요하다  

Profile early, oftne, and on the target device  

프리프로덕션, 등등 성능 검증 등  
모발일 - 모든 디바이스가 성능이 다르고 뭔가 고민거리를 주고  
에디터 재현 불간으 

- 렌더 타겟 아키텍켜

Dedicated Graphics Memory  
OldSchool  
노트북, 컴 등 개별적인  처리  

이게 Old School 인이유  
이제 대부분은 Unified  
CPU GPU 공유한 RAM을 쓰고 있다는 뜻  

아닌경우가 이것이 정말 리소스를 많이 쓰고 있다  

PC, Labtop은 전기 연결하면 되는데  
모바일은 그렇지 않죠  

그래서 (타일 메모리)를 쓰는 것이 중요  
렌더링이 화면 한 번ㅇ ㅔ그리지 않고, 타일 상으로 된다  ?

좀 더 효율적으로  

실제 렌더링 워크 플로우  

씬이 렌더링 되면 씬이 프로세싱을 핞다  
버테긋가  어떤 숫ㄴ서로 렌더링 되어야 하는지  

카메라 안보이는 오브젝트를 제거한다던지 Culling  등을  

모든 리소스가 Ram쪽으로  

이것을 GPU가 읽어낸다  Struct Resoruces  

타일 메모리를 쓰게 되고  

Vertex, Fragment SHader를 쓰고  
전부 기억하지는 않아도 되고  

하나하닁 프레임에 
계속 카피하고 읽어내고 바녹되기 때문에  

나눠서 효율적으로 하는 것이 중요  

너무 많이 생성되지 않도록 유의해야 한다 (데이터 카피, 읽는 과정)

- 프로파일 툴

Project Auditor  
새로운 기능  
WIP  
코드 이슈가 뭐가 있는지, 그 강도는 어떤지 등
정적인 분석  
런타임이 안리 ㅏ최적화 첫걸음으로 ㅗ게된ㄴ 

Profiler  

Frame Debugger 
유니티 파이프라인에서 뭐가 어떻게 벌어지고 있느지 등  
자세하진 않지만, 어떻게 되고 있거나  

Shader DrawCall Batching 등을 디버깅 간으  

Frame Debugger 좀 더  

어떻게 씬이 만들어져 있는지 확인 가능  
Bath, Batch Break가 어디서 어떻게 되느지 안ㄹ 수 있고  
그것을 알아내는 것을 시작으로 ~  

역에는 GPU Profierl  
추천되는 것은 (유니티에서도 쓰이는)
RenderDOc AOD
Metal GPI Debugger IOS  

RenderDOc 예시  

TimeLine 등이 있고  (이벤트들)  
좌긏ㄱ 상단에는 Event 브라우저가 있다  
어느정도 느리게 진행이 되는ㄷ, 상대적으로 각각의 콜들이 그래픽스 콜에서 얼마나 차지하는지 (쌍대적으로 (절대 X))  

그리고 오른쪽은 텍스쳐 뷰어  
실제로 어떤 Output이 나오는지  
느린데 뭘 그리고 있는건지?  

등을 알려주는  

구체적으로 하면 끝이 없고  
핵심은 시작을 잡을 수 있다

- 성능저하 그래픽 실수들

가장 중요한 것은 우리 모두가 한다는 것  

Mesh Mistiakes  

복잡한 지오 메트리대쉬  

많은 삼각형  
삼각형의 형태도 중요하다  
모바일은 타일 기준으로 화면을 쪼개는데, 이 삼각형이 타일에 딱 맞아 떨어지면 좋을 수록 좋다  
납작한 게 좋다?  
완벽하면 GPU가 더 효율적으ㅗㄹ 가능  

마이크로 삼각형  
엄청나게 많은 삼각형 (하나의 타일에 엄청 많음)  

LOD를 사용하지 안흔 것  
타일 렌더링을 십분 화룡하려명 LOD 

카메라 사용 숫자 줄여야한다  
2개 이상은 하나로 줄일수있냐 언제나 물어본디 - 엄청 복잡하긴 하죠
CPU 오버헤드가 엄청 걸린다

컬링 레이어를 쓰는 것을 추천  
Read/Write enabled 화성화 하지 않기?  
-> 원래 텍스쳐 데이터는 GPU에서 읽을 수 있도록 GPU 메모리에 상주, 유저 스크립트에서 Texture.GetPixel/SetPixel() 하기위해 CPU에도 메모리 올려놓을지 여부 옵션 (기본 비활성화)
-> Mesh도

텍스쳐 실수들  
Max Size를 사용하지 않는 것  

- 내부 지식? 교훈?

## 메모

---

MipMap  
2의 배수로 작아지는 (다 합쳐서 원본의 33% 정도?)  
카메라 거리마다 다르게  

- VS Doc 느린 열거형 .toString 성능
  - 성능향상을 위해 toString 캐싱.
- Unity how 게임 개발자를 위한 성능 프로파일링

### 참고

- UNTIE SEOUL 2025
  - '류태규': 'Unity 프로젝트 개발시 반드시 체크해야 할 최적화 관련 기능 공유'
  - 'Elbert Perez': '부드러운 모바일 게임: 최고의 성능을 위한 그래픽 최적화'

Canvas 직접 움직이는 것은 괜찮지만  
Canvas 내 Element를 움직이는 것은 Rebuild  

RenderTexture 설정되어 있으면, 카메라 꺼져있어도 렌더링?  

Disclaimer  

CameraStacking URP  
