---
title: "셰이더 종류"
# description: ""
categories: [컴퓨터, 그래픽]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-11-13. 06:42 # Init
# last_modified_at: 2025-05-28. 05:57 # +Surface Shader
# last_modified_at: 2025-05-28. 20:38 # +Shader Graph, +정리: from CG
# last_modified_at: 2025-06-10. 23:11 # ~정리
last_modified_at: 2025-06-13. 00:29 # +공부 방향성
---

## 언어

---

셰이더 프로그래밍 언어.  
하나만 잘 배워두면 나머지는 쉽게 터득할 수 있다. (like 프로그래밍 언어)  

### Cg

CG (C for Graphics) 엔비디아가 마이크로소프트와 협력하여 만든 언어.  
Cg Old built-in Only  

### HLSL

HLSL (High Level Shading Language) 가장 유명하고 보편적으로 넓게 쓰임.  
new SRP  
Cg랑 유사함. NVIDIA x Microsoft  

### GLSL

GLSL (OpenGL Shading Language) OpenGL에서 사용하는 언어.  
Unity에서는 안 씀. 쓸 수는 있는데 안 씀. Cg/HLSL은 OpenGL/webGl 쓰는 플랫폼 빌드시 GLSL로 변환됨  

## Unity 언어

---

Unity는 CG 언어를 사용, URP부터는 HLSL 사용. (언리얼도 HLSL).  
Unity는 추가적으로 ShaderLab 언어를 제작하고 이를 지원.  

### Shader Lab

(고정 파이프라인 셰이더: fixed function shader)

예쩐 방식. 권장 x, 이미 만들어진 틀. 호환성은 좋음  

- 매우 가볍고, 하드웨어 호환성이 좋지만, 기능이 상당히 부족 > 고급효과 X
- 자체 문법 > 다른 셰이더 문법과 거의 호환 X + 거의 지원 중단
- ㄴ ☆ 다양한 환경에 맞는 셰이더로 자동으로 분기되어 만들어줌
- ㄴ 모바일, PC, 기타 콘솔 기기, 라이트 맵 유무, 조명이 픽셀 라이팅일 때, 버텍스 라이팅일 때 등 모든 경우의 수마다 다른 셰이더를 만들어야 하지만, 서피스 셰이더는 이런 경우의 수를 자동으로 제작
- ㄴ 때문에 편리 > 불필요한 셰이더 양은 늘어남 > 최적화 X

호환성은 가장 높지만, 그만큼 할 수 있는 게 제한적

### Surface Shader

Shader Lab 스크립트 + Cg 셰이더 코드.  
(서피스 셰이더)

Built-In only. 아티스트 단계  

- ShaderLab 스크립트와 함께 일부분은 CG 셰이더 코드를 사용
- 기본적인 조명 코드와 버텍스 셰이더의 복잡한 부분은 스크립트를 이용하여 자동으로 처리
- 픽셀 셰이더 부분만 간편하게 작성할 수도 있음 > 편함
- 비주얼 셰이더 에디터와도 상당히 비슷한 개념 > 쉽게 공부하고 응용하기에도 좋음
- 단, 최적화에는 다소 무리, 일정 수준 이상의 고급 기법 X

가장 쉽고 멀티 플랫폼에서 잘 대응되는 셰이더, 프로그래머가 아니더라도 배우기 쉬운 개념, 아티스트 레벨에서 배우는

- 이걸 배워두면 Vertex & Fragment Shader 도 이해할 수 있고, 랜더 몽키로도 갈 수 있고, 노드로도 갈 수 있다

스탠다스 서피스 셰이더는 유니티 5부터 기본으로 적용된 셰이더 형태  
하지만 물리 기반 셰이더 라이트이기 때문에 모바일과 같은 저사양 기기에서 구동하기에는 다소 무거운 것이 사실

_

설정 부분
전처리라고 할 수도 있고 스니핏(snippet)이라고 부르기도 합니다.
 이 부분은 말 그대로 셰이더의 조명계산 설정이나, 기타 세부적이 분기를 정해주는 부분입니다.

 ★ Albedo는 빛을 받는다는 의미가 아닙니다. Diffuse와 동일하게 생각하면 곤란합니다.
 ★ 3ds Max에서 셀프 일루미네이션 (Self-illumination) 이라고 부르는 '자기 발광' 기능이 바로 Emission입니다
 ★ o.Albedo와 o.Emission의 값은 최종적으로 서로 더해집니다. 즉, 둘을 모두 쓰면 필연적으로 밝아집니다.
 Albedo > 조명 연산을 추가로 받게 되고, Emission 조명 연산을 받지 않아서
'조명과 상관없는 순수한 색상' 만이 출력 > 순수한 결과물을 보고 싶을 때는 Emission을 즐겨서 사용
 float4.rgb => rgb만 쓰겠다, 자동 형변환 느낌인듯?
 float.rgb
float.grb
float.bgr
float.rrr
 1 = 1,1,1 / 0.5 = 0.5, 0.5, 0.5
 o.Albedo = test.b 라고 쓰면 0을 입력한 것과 같기 때문에 검정색이 출력
 ㄴ 이렇게 변수의 부분값을 자유자재로 바꾸는 것은 스위즐링 (Swizzling) 이라고 합니다
 예전에는 오류가 생기면 무조건 마젠타 색을 출력했지만,
유니티 5.5부터 계산할 수 있을 때 까지 계산하고 오류 메세지 출력
심각한 오류라면 여전히 마젠타 컬러를 보여줌
 float x
x.r 가능

float2 xx
float3(0, xx.rg)
float3(0, xx.gg)
 인터페이스 변수를 CG코드 내에서 사용하려면 똑같은 이름의 변수를 안에서 선언해주면 됨
 밝기를 조절하는 방법
-1 ~ 1 Range로 받아오고
o.Albedo = rgb~ + 변수 값
 struct 안에 아무것도 들어있지 않으면 안됨
 float4 color: OLOR
이 내용은 버텍스 컬러를 받아오는 마법의 주문, 현재로서는 어디에도 쓰진 않습니다. 단지 에러를 피하고자 쓴 것이지요.
 o.Alpha = 1 로 불투명함을 표시, 하지만 '아직은' 0.5 나 0 을 넣는다고 반투명해지거나 투명해지지 않습니다.
 LOD > 이 셰이더의 환경 설정에 따른 옵션 값에 관련된 내용

<https://celestialbody.tistory.com/5>

텍스쳐는 UV 좌표와 함께 계산되어야 float4로 출력될 수 있기 때문에, 아직 UV와 계산되지 않은 텍스쳐는 색상 (float4)으로 나타낼 수 없다. 그래서 이때까지는 sampler라고 부른다.  

#### 4

물리 기반 셰이더에서 Metalic이 0이면 스페큘러 컬러가 흑백(흰 빛을 비추었을 때)이 되며, Metalic이 1이면 스페큘러 컬러가 Albedo에 넣은 색이 됩니다. 금속은 고유의 스페큘러 컬러를 가지고 있기 때문입니다. 가급적 Metalic에서 0과1 외의 값은 쓰지 않는 것이 정확한 물리 기반 셰이더를 다루는 방법입니다.  

Smoothness는 재질이 매끄러운지 거친지를 결정하는 부분.  
0이면 완벽히 거칠어서 난반사만 일어나고, 1이면 완벽히 매끄러워서 정반사만 일어나게 됩니다.  

난반사가 일어나는 이유는 '거친 표면' 때문이며, 특히 겉으로는 부드러워 보이더라도 미세한 표면 단계에서의 거칠기 때문에 (거의 분자 단위로) 피부나 천의 경우에는 스페큘러가 적게 나타나는 것입니다.  

즉, 재질이 매끈하면 난반사(Diffuse) 연산보다 정반사(Specular) 연산이 늘어나며, 이것을 조절할 수 있는게 Smoothness 인자 입니다.  

벤타 블랙 같은 특수한 물질을 제외하고는 대부분 어느 정도의 스페큘러를 가지고 있다.  
우리가 질감을 느끼게 되는 요소는 거의 스페큘러, 그래서 스페큘러가 없으면 거의 질감을 느끼기가 힘든 것.  

유니티에서 Standard Shader라는 '물리 기반 렌더링' 의 기본 개념은 '에너지 보존 법칙'으로 '나가는 빛의 양은 들어온 빛의 양을 넘을 수 없다'라고 간단하게 설명할 수 있습니다.  

[](https://docs.unity3d.com/Manual/StandardShaderMaterialCharts.html)  

일반 텍스쳐를 노말맵으로 만들 수 있음.  
Create from Grayscale 옵션을 체크하고, Bumpniss나 Filtering을 적절히 조절  

NormalMap은 일반적인 텍스쳐가 아니다.  
NormalMap은 일반적인 게임용 텍스쳐 포맷인 DXT1 혹은 DXT5가 아니라 DXTnm이라는 파일 포맷이다.  
(플랫폼에 따라 다를 수 있다. 예를 들어 안드로이드에서는 ETC 파일 포맷을 쓰기도 한다. 여기서는 PC 기준으로 설명하겠다.)  

이 파일 포맷은 일반적인 텍스쳐의 압축에 의한 NormalMap 품질의 저하를 막기 위해 만든 AG 파일 포맷이다.  
이 포맷은 NormalMap의 R과 G의 퀄리티를 최대한 보전하여 A와 G에 넣어 저장한다. (B는 가지고 있지 않다)  
이렇게 보전된 R과 G는 NormalMap의 X와 Y로 계산되며, Z는 삼각함수를 이용하여 수학적으로 추출이 된다.  

그러므로 이 텍스쳐를 이용해서 NormalMap으로 온전하게 생성해내려면 앞에 설멍혀나 공식이 적용되어 있는 함수를 이용하면 간편한다.  
`float3 y = UnpackNormal ( float4 x );`  
`o.Normal = UnpackNormal ( tex2D (_BumpMap, IN.uv_bumpMap) );`  

or  

`fixed3 n = UnpackNormal(tex2D(_BumpMap, IN.uv_bumpMap));`  
`o.Normal = float3(n.x * 2, n.y * 2, n.z);`  
강도 조절 (Z축은 거의 영향을 끼치지 않기 때문에 연산해줄 필요가 없다, 물론 1이상을 곱한 후 내부적으로 normalize를 한다면 약하게 만들어 줄 수 있겠습니다만 이 단계에서는 적절치 않으므로 설명을 생략하겠습니다.)  

Occlusion (Ambient Occlusion)은 구석진 부분의 추가적인 음영을 표현하는 기능  
일반적으로 환경광 (Ambient Color)로 가득 차 있느 세상에서 그림자가 드리워진 부분도 사방에서 오는 환경광 정도를 받고 있는 것이 일반적입니다.  
하지만 매우 구석져 있거나 복잡한 물체들로 가려져서 환경광도 닿지 못하는 부분은 더욱 더 어두워지는데 이 부분을 Ambient Occlustion (환경 차폐)라고 부릅니다.  

특이하게도 Occlusion 맵은 독립된 UV를 받으면 에러가 납니다. 반드시 _MainTex같은 UV를 사용해야 정상적으로 작동.  
o.Occlusion이 float을 받기 때문에, RGB를 모두 사용하는 텍스쳐 한 장을 단지 Occlusion 기능을 위해 추가하는 것은 낭비.  
_MainTex의 알파 채널을 Occlusion으로 사용한다던지  

지금까지 유니티에 내장된 PBS 셰이더인 Standard Shader의 Metail Pass의 기본 조작접에 대해 알오보았다.  

[https://unity.com/kr/releases/editor/archive](https://unity.com/kr/releases/editor/archive)  
다운로드 아카이브에서 내장 셰이더를 다운 받을 수 있다.  

### Vertex & Fragment Shader

자주쓰이는 건 HLsl x vertex&fragment 조합  

- ShaderLab 스크립트와 함께 CG 셰이더 코드를 사용, 좀 더 본격적인 셰이더 작성 방법
- 자동적으로 처리되는 부분이 별로 없어서 제대로 된 CG 셰이더 방식으로 버텍스의 좌표변환부터 제대로 처리해야 작동
- 배우기는 조금 어렵지만, 완전히 수동으로 제어 가능 > 최적화 + 고급 기법 O

Surface Shader의 상위 버전, CG를 더 디테일하게 다룸, Surface Shader가 오토 모드라면. Vertex & Fragment Shader는 수동이라는 느낌.  

### Compute Shader

gpu 연산. 관련 지식 필요  

### Post Processing

## HLSL Doc

---

fixed 4, half 4, tex2D( )~, _Time.x y z w  

### 함수

- frac(float), fraction 일부 -> 소수점만. 10.1 -> 0.1, 217.89 - > 0.89.
- length(vector2) -> 길이 (피타고라스 삼각 정리)
  - dist = length(w) // 원점 ~ 거리
- smoothStep
  - ring = smoothStep(radius + width, radius, dist) - smoothStep(radius, radius - width, dist)

## Shader Graph

---

Unity shader-graph  

- Graph Target
  - Build-In 있긴 함
  - 근데 소개는 대부분 SRP
  - 대부분의 피처는 SRP 위주일듯
- Target
  - CanvasShaderGraph

### Sub Graph

여러 노드를 하나의 노드로 합치기.  

### Amplify Shader Editor

## 공부 방향성

---

### 코드 vs 그래프

- 코드
  - 자유도, 고점 높음 (복잡한 효과, 최적화, 커스텀 연산 가능)
  - 오픈소스, AI 도움 받을 수 있음
  - 버전 관리에 유리
  - 러닝 커브: 높음, 셰이더 개념 뿐만아니라 셰이더 코드 문법도 알아야 함.
  - 실수 시 디버깅 어려움
  - 보통 빌트인이 많고, 최신 SRP 코드는 많이 없음
  - 그래픽 디자이너가 다루기 어려움. 에디터를 만들면 되긴하는데, 작업이 필요함.
- 그래프
  - 오픈소스, AI 도움 받기 어려움
  - 자료, 에셋은 그래프가 더 많음
  - 버전 관리 어려움
  - 러닝 커브: 툴은 상대적으로 쉬움, 셰이더 개념은 똑같기에 어려움
  - 시각적, 직관적. 결과를 중간에도 바로 확인 가능. 디버깅 쉬움. 유지보수/인수인계도 쉬움. 코드 설명 대신 중간 결과 보면서 이해하면 되니까. (근데 유지보수/인수인계 하는 일 많이 없을 듯)
  - SRP 쓰는 것이 적합.
  - 그래픽 디자이너가 다룰 수도 있음
  - 노드 -> 이미 구현된 것이 많음. 가져다 쓰면 됨. 빠름.
  - 최적화 어려울 수 있음.

보통 혼용 한다고는 함.  
애초에 서로 같은 셰이더를 어떻게 만드냐의 차이. 서로 트레이드 오프가 있고, 둘 중 하나만 써야하는 것도 아님. 서로 잘만하면 컨버팅도 가능.  

그래프에 Custom Function 노드를 통해 코드를 일부 삽입 가능 (연결 가능)  

에셋 같은 건 그래프가 많아서, 유지보수 하려면 그래프도 알아야 할 듯.  
복잡성, 최적화 고점을 보려면 코드가 맞는 것 같고.  

코드로 이론 깊이 공부하고, 그래프로 예제 넓게 공부하는 것이 좋을 듯.  
중요도를 따진다면 코드인듯.  

공통 기능: '코드: `.hlsl`, `.cginc`', '셰이더: 서브 그래프'  

> Shader Graph와 코드 셰이더 모두 익히되, 코드를 더 깊게 공부하는 것이 장기적으로 유리
> 빠른 프로토타입 -> 그래프, 최적화/특수효과 -> 코드

### 빌트인 vs URP

빌트인 셰이더가 URP에서 호환이 안됨  
코드 구조도 다르고, 지원하는 기능도 다르고  

결국 URP 쓸텐데 시작을 URP로 하는 것이 좋을 듯  
컨버팅 하면서 두 번 작업하느니  

> URP 환경에 익숙해지는 것이 앞으로의 Unity 셰이더 개발에 가장 효율적

## 메모

---

### 키워드

- shader keyword
