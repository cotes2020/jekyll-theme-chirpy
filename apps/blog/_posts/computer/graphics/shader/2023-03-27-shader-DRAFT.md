---
title: "셰이더 Shader"
# description: ""
categories: [컴퓨터, 그래픽]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-03-27. 13:27
# last_modified_at: 2023-03-27. 14:10
# last_modified_at: 2023-10-26. 13:09
# last_modified_at: 2024-07-26. 13:29
# last_modified_at: 2024-07-30. 22:20
# last_modified_at: 2024-09-02. 11:15
# last_modified_at: 2024-09-27. 22:44
# last_modified_at: 2024-10-16. 08:14 # 메모
# last_modified_at: 2024-11-13. 07:19 # -Cg
# last_modified_at: 2025-03-14. 23:32 # 메모
# last_modified_at: 2025-05-28. 05:53 # +메모 from career-learning
# last_modified_at: 2025-05-28. 20:09 # +메모 from CG
# last_modified_at: 2025-06-08. 20:18 # +메모
# last_modified_at: 2025-06-09. 21:00 # ~메모
last_modified_at: 2025-06-10. 23:10 # ~정리, +유니티 셰이더: 코드
---

## 셰이더

---

[유니티 셰이더&렌더링 에센스 E01 셰이더는 무엇인가?](https://youtu.be/4iSJW7YGrjY)  

- 목적: 화면에 색을 칠하는(shading) 프로그램
- 동작: 렌더링 파이프라인의 일부를 유연하게 변경하는 프로그램
- 기술: 3D 컴퓨터 그래픽에서 최종적으로 화면에 출력하는 픽셀 색을 결정하는 함수
- 감성: 그래픽 데이터의 음영과 색상을 계산하여 다양한 재질을 표현하는 방법
- 직역: 음영, not 그림자 but 그늘에 가까운
  - 빛이 비치는 쪽은 밝게, 반대 방향은 어둡게 표현하는 그것.
  - 왜? 평면에 입체감 표현, 사람의 눈은 음영을 통해 공간감을 느낄 수 있음
- 좀 더 그래픽 아티스트 친화적으로 말하면, '머티리얼' material. 언리얼 엔진:
- 단지 기계적 영역의 설명뿐 아니라 감성적 영역까지 아우르는 멋진 공예품에 가깝다.
  - 자동차
    - '이동을 위해 제작된, 금속과 다양한 재질들로 제작된 내연기관과 그 부속물' 이라고 말할 수도 있겠지만
    - 다른 영역에서 바라면 자동차는 품위를 높여주거나 자신의 개성을 나타내며 유행을 선도하는 예술품으로 느껴지기도
- 자신만의 예술적인 개성이 넘치는 드로잉 스타일도 가능하게 되는 것. 마치 자기만의 색연필이나 물감을 만드는 것처럼.

## 유니티 셰이더: 코드

---

```shader
// 2023 URP, https://youtu.be/EHLMkNBUIdM
Shader "karmoDDrine/HelloShader"
{
    // 머티리얼 프로퍼티 (외부에서 셰이더로 값을 전달할 수 있는 필드)
    // 머티리얼 프로퍼티 값은 셰이버 내부의 변수로 전달된다

    Properties
    {
        // _는 권장사항
        // 이름 ("인스펙터 이름", 타입) = 기본 값
        // Int, Float, Range (단일 숫자)
        // Color, Vector (4차원 벡터)
        // 2D, Cube, 3D (세 가지 텍스처 유형)
        // 텍스처 기본값은 약간 더 복잡하며 문자열 뒤에 괄호로 묶인 한 쌍으로 지정됨.
        // 여기서 문자열은 비어 있거나 "흰색", "검은색", "회색" 또는 "Bump".로 되어있음
        // 괄호의 목적은 원래 일부 텍스처 속성을 지정하는 것.

        _BaseColor ("Color", Color) = (1, 1, 1, 1)
        _Scale ("Scale", Float) = 1
    }

    // SubShader: 실제 구현
    // 여러 개의 SubShader 넣을 수 있는데, 위에서부터 최초로 동작하는 SubShader가 사용됨
    // 하드웨어 따라서 지원 가능한 셰이더 다른 경우
    // 어떤 것도 지원되지 않으면 Fallback

    SubShader
    {
        // Tags: 어떻게 동작해야 할지, 해당 세이더를 어떻게 사용해야 할지에 대한 추가 정보
        // "Key" = "Value"
        // 셰이더 내부 동작을 다루기 위한 것도 있는데, 유니티 에디터에서 어떻게 셰이더를 다룰지에 대한 것도
        // 렌더링 엔진간의 브릿지
        Tags { "RenderPipeline" = "UniversalRenderPipeline" "PreviewType" = "Sphere" }

        // RenderSetup

        // 게임 오브젝트 그리기 한 번에 대응 (한 번의 완전한 렌더링 프로세스 정의)
        // Pass: 여러 개 넣을 수 있는데, 하나의 오브젝트를 여러 번 덧대어 그리는 것
        // 보통 낭비/성능 저하,
        // 카툰 렌더링 등에서 외곽선을 구현하는 방법 등에는 사용 (모델 확대하여 단색, 원래 크기 + 원래 색)
        Pass
        {
            HLSLPROGRAM

            // HLSL 셰이더 라이브러리를 가져와야 함.
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // 컴파일러 등에 옵션을 알려주기 위한
            // 우리가 작성할 vertex/fragment 함수의 이름을 결정 (자유롭게 지어도 됨)
            #pragma vertex vert
            #pragma fragment frag

            // 유니폼 변수 (vert, frag 모두 사용 가능한 변수)
            // 프로퍼티 변수랑 이름이 같다면 프로퍼티 값이 여기에 들어옴
            half4 _BaseColor;
            half _Scale;
            // fixed: 더 작음. fixed로 충분한 것 -> Color, Vector
            // half: float 절반 크기. 범위, 정밀도가 필요한 것 -> 
            // Nvidia 쪽 GPU의 경우에는 half로 할 때 연산수가 2배로 빨라집니다.
            // 나머지 float

            // vert랑 frag에 어떤 모양의 구조체를 넣을지
            
            // 버텍스 함수에게 전달할 입력
            struct VertexInput // 이름 자유
            {
                // 3D 모델의 각 정점의 위치를 가진다
                // 시멘틱: 변수에 대한 추가 정보를 제공하는 문법/키워드/마커
                // POSITION - 벝텍스 함수의 입력으로 사용할 오브젝트 정점을 표시
                // 그래픽스 드라이버가 오브젝트 위치를 전달
                float3 objectSpacePosition : POSITION ;
            };

            // 프로그먼트 함수에게 전달할 입력
            // 동시에,
            // 버텍스 함수의 출력이기도 함
            struct FragmentInput
            {
                // 화면상의 위치
                // 왜 WHY XY가 아니라 XYZW
                // x,y - 화면 위치, z - 깊이, w - 동차 좌표계에서 쓰던 값이 남은 것 (여기서 쓰지는 않음)
                // SV_POSITION - 프래그먼트 함수의 입력으로 사용할 화면상의 정점을 표시하는데 사용
                float4 screenPosition : SV_POSITION ;
            }

            FragmentInput vert(VertexInput input)
            {
                half3 objectSpacePosition = input.objectSpacePosition;
                objectSpacePosition *= _Scale;

                // 오브젝트 공간의 정점을 월드 공간으로 변환
                // Core.hlsl에 있는 함수
                half3 worldPosition = TransformObjectToWorld(ObjectSpacePosition);
                // 월드 공간에 있는 정점을 뷰 공간으로 변환
                half3 viewPosition = TransformWorldToView(worldPosition);
                // 뷰 공간에 있는 정점을 클립 공간(동차 클립 좌표계 Homogeneous Clip Space)
                half4 clipPosition = TransformWViewToHClip(viewPosition);

                FragmentInput output;
                // vert -> screenPosition - HClip 위치 클립 공간 위치
                // --> 래스터라이저를 거치면서 --> frag : screenPosition 화면상의 좌표 위치로 변환 된 것을 받게된다.
                output.screenPosition = clipPosition;
                return output;
            }

            // 렌더 타겟 시멘틱
            // 여러 카메라가 여러 개의 화면을 동시에 그릴 수 있는데, 여러 화면을 그리고 있다면 이 쉐이더 어떤 화면을 기준으로 색을 그릴지, 각각의 그리기 대상이 되는 화면을 렌더 타겟
            // 별 다른 설정을 안하면 가장 첫 번째 렌더 타겟을 대상으로 함
            // SV_Target, SV_Target0 모두 첫 번째 렌더 타겟을 대상으로 함
            half4 frag(FragmentInput input) : SV_Target
            {
                // input.screenPosition 받긴 하는데
                // 단색으로 표현하려 한다면
                return _BaseColor;
            }

            ENDHLSL
        }
    }

    Fallback
    {
        // 모든 서브 셰이더가 이 그래픽 카드에서 실행되지 않을 경우, 최하 수준으로 실행되는 셰이더
        // 폴백은 그림자 투사에도 영향을 끼치고, 기본적으로 유니티 범용 패스가 포함되어 있기 떄문에, 직접 모든 패스를 구현할 필요가 없음.
    }
}
```

```shader
// Built-in
Shader "Custom/Simple Surface Shader"
{
    SubSHader
    {
        Tags { "RenderType" = "Opaque" }
        
        CGPROGAM
        
        #pragma surface surf Lambert
        
        struct Input
        {
            float4 color : COLOR;
        };
        
        void surf (Input IN, inout SurfaceOutput o)
        {
            o.Albedo = 1;
        }
        
        ENDCG
    }
    Fallback "Diffuse"
}
  
Shader "Custom/Simple VertexFragment Shader"
{
    SubShader
    {
        Pass
        {
            CGPROGRAM
            
            #pragma vertex vert
            #pragma fragment frag
            
            float4 vert (float4 v : POSITION) : SV_POSITION
            {
                return mul(UNITY_MATRIX_MVP,v);
            }

            fixed4 frag() : SV_Target
            {
                return fixed4(1.0,0.0,0.0,1.0);
            }
            
            ENDCG
        }
    }
}
```

- HLSL,GLSL도 아닌 셰이더 랩(ShaderLab)이라는 고급 렌더링 추상화 레이어
- 머티리얼 표시하는 모든 것을 정의
  - 고정 파이프라인의 텍스처 좌표 생성을 제어해야 하는 경우,
  - 버텍스 셰이더에서 해당 텍스처 좌표를 계산하는 코드를 작성해야함
- 셰이더 랩은 이런 일련의 랜더링 상태 설정 명령을 제공하여, 블렌딩/뎁스 여부등의 다양한 상태를 설정할 수 있음
  - Cull 컬링 모드 설정
  - ZTest 심도 테스트 설정 시 사용되는 기능(그림자 그릴때 씀 그에 대해서 내가 쓴 글 링크)
  - ZWrite 딥 라이팅 켜기/끄기
  - Blend 블렝딩 모드 켜기 및 설정
- 위의 랜더링 상태가 서브셰이더 블록에 설정되면 나머지 모든 패스에 일괄적용됨. 이를 원하지 않을 경우, 패스 시멘틱 블록에서 별도로 설정가능함
- 서브셰이더는 기본적으로 태그는
  - 유니티가 지원하는 서브셰이더 태그 유형
    - Queue -랜더링 순서 제어, 객체가 속하는 렌더링 대기열 지정
    - RenderType - 셰이더 종류 분류
    - DisableBatching - 일부 서브 셰이더에서 버텍스 애니메이션을 위해 공간 좌표 쓸때, 유니티의 일괄 처리기능 여부를 키고 끌수있음.
    - ForceNoShadowCasting - 객체가 그림자를 투사할지
    - IgnoreProjector -서브 셰이더를 사용하는 객체의 반투명 여부
    - CanUseSpriteAtlas -서브 셰이더가 스프라이트에 사용되는 경우 False
    - PreviewType - 패널에서 미리보기
  - 패스 태그 유형
    - LightMode Unity의 랜더링 파이프에서 이 패스 역할 정의
    - RequireOptions 특정 조건 충족시 패스가 랜더링되도록 지정
- 셰이더 유형
  - 기본적으로 표면 셰이더(SurfaceShader)는 서브 셰이더의 CGPROGRAM과 ENDCG사이에 정의됨
  - 그 이유는 표면 셰이더를 사용할 때, 개발자가 사용할 패스 수, 각 패스가 랜더링 되는 방식에 대해서 생각할 필요가 없고, 이 작업을 유니티 엔진 차원에서 해줌
  - CGPROGRAM과 ENDCG사이의 코드는 CG/HLSL을 사용하여 작성되는데, Cg/HLSL 언어는 셰이더 언어와 중복되서 쓸 수가 있음
  - 버텍스/ 프래그먼트 셰이더 또한 CGPROGRAM과 ENDCG사이에 정의되야함
  - 표면 셰이더와 차이점은 SubShader가 아닌 Pass 블록에서 작성된다는거
- 따라서 유니티 셰이더는 실제 셰이더가 아님.
- 전통적인 셰이더에서는 입력 출력을 하기 위해서 출력 위치 대응을 해야했지만, 유니티 셰이더에서는 특정 블록만 활성화하면됨.
- 유니티 셰이더는 모델에 제공되는 버텍스 위치, 텍스쳐 좌표, 노멀에 직접 접근할 수 있으므로 개발자는 셰이더에 데이터를 전달하기위한 추가 코딩이 필요 없음

## 유니티 셰이더: 그래프

---

## 유니티 셰이더: 유형

---

- StandardSurfaceShader: 기본 조명 모델을 포함하는 표면 셰이더 템플릿
- UnlitShader: 조명 없이, 그러나 FOG를 포함하는 기본 버텍스, 프래그먼트 셰이더
- ImageEffectShader: 화면 후처리 효과
- ComputeShader: 특수 셰이더, GPU 병렬성을 이용한 일반 렌더 파이프라인과 관련 없는 계산 수행
- RayTracingShader: 실시간 레이트레이싱

## 특징

---

- 단점
  - 기능마다 셰이더 만들어야 함. 컴포넌트처럼 조립할 수 없음.
  - 로그 못 찍음
  - 값(인스턴스) 차이를 주려면 각 머티리얼 파일을 만들거나, 스크립트에서 수정하거나 (디버깅 어려움). 컴포넌트처럼 씬 오브젝트 단위로 구분하고 저장할 수 없음.
  - cpu와 다르게 if 문에서 쓰지 않을 내용도 모두 계산
    - 이를 해결하기 위해 shader variant
    - 빌드 용령, 속도 늘어남. 메모리도 영향 줌.
- 장점
  - 에디터 타임 동작. 보면서 수정 가능. 비교적 컴파일 빠름.
- 특이
  - 플레이 모드 수정 저장됨.
  - 중간 과정 디버깅 어려움. -> 셰이더 그래프는 가능.

## 수학적 모양

---

## 메모

---

- 셰이더는 사용 중인 RP 버전에 따라 전처리 지시어의 사용 방법이나 라이트 루프 처리 방식이 기존과 다를 수 있습니다. 가장 확실한 방법은 사용 중인 RP의 ShaderLibrary/RealtimeLights.hlsl 파일을 확인하여 추가 조명 계산 및 처리 방식을 살펴보는 게 좋습니다.

- ShaderGraph Tutorial and ShaderCode
- Shader Compile
- UIObject -> Canvas가 주심. Shader에서 움직이기 어려움
- UIEffectV5
- UIOutline
- 횡스크롤 강 셰이더 만들기
- [\[최적화\] Shader Variants와 효율적인 사용](https://asatala.tistory.com/171)
- Unity Manual - Rendering 추가 리소스 및 예제
  - 20가지 고급 2D 셰이더 효과
- 유니티 공식 UI 셰이더
  - UGUI 캔버스 셰이더
  - 2023 업?
  - 트랜지션?
- Shade Graph in Built-in Pipeline. Unity 2021.2
  - Youtube
- UnitySHadersIntroPart2: HLSL/CG EdgeDistortion~
- Image.material.setColor('_Color', ~);
  - 이때, shared material x, 임의로 만들어 넣어줘야 함
  - \_Color에 1 넘게 넣은 수 있음 (\_Emission, PP Bloom)
- unity, 3dMax uv 좌표계 좌우 하상
- RGB, XYZ, UVW
- 디더링
- 그래픽스
- 디더링
- 텍스처 압축 포맷
- 소프트마스크
- 스텐실
- 알파테스트, 알파블렌딩
- 스크린 스페이스 셰이더
  - sssShader
- Scene Depth: 카메라부터 연산을 시작하는 점까지의 깊이?
  - Transparent는 통과하는 듯..?
- [The Unity Shaders Bible](https://learn.jettelly.com/unity-shader-bible/#buy-now)
- [Graph 그리기, Position에 따른 Color](https://catlikecoding.com/unity/tutorials/basics/building-a-graph/)
- [Surface Shader (눈)](https://blog.naver.com/PostView.naver?blogId=plasticbag0&logNo=221439156276&parentCategoryNo=&categoryNo=45&viewDate=&isShowPopularPosts=false&from=postView)
- [Shadertoy](https://www.shadertoy.com/)
- [가짜 투명도, 디더링](https://gall.dcinside.com/mgallery/board/view/?id=game_dev&no=117790&page=1)
- [알베도, 이미션, 디퓨즈](https://m.blog.naver.com/sorang226/222940558803)
- [2D→3D](https://x.com/asidys230/status/1635799802100482049?s=20)
- [Post-processing outlines](https://x.com/TheMirzaBeig/status/1658643110409261056?s=20)
- [light scrolling](https://x.com/cmzw_/status/1655536784485527552?s=20)
- [Tiling Vs Hex Tiling](https://x.com/_kzr/status/1621052638723993600?s=20)
- [일반적인 2Pass 방식일 때 노말 방향 보정 공부](https://x.com/longlong_stone/status/1664844118491553793)
- [darkcatgame](https://darkcatgame.tistory.com/79)
- [펭귄 게임 개발일지 - 툰 셰이더](https://gall.dcinside.com/mgallery/board/view?id=game_dev&no=126408)
- 쉐이더 자동 컨버전
- 포워드 디퍼드 전환
- 키바님
  - [숫자 타일](https://x.com/kjh030529/status/1754052982621274570)
  - [버튼](https://x.com/kjh030529/status/1757252520051888242)
  - [홀로그램 셰이더](https://x.com/kjh030529/status/1631561982842396677?s=20)
- 경섭님
  - [1](https://x.com/ryurud_n5/status/1822665541909434376)
  - [2](https://x.com/ryurud_n5/status/1820451843941745102)
    - 명조 스타일 만들기 #1 명조는 Grass가 카메라와 가까워지면 아래로 숨겨짐 Dithering을 활용하는 경우는 봤어도 이렇게 아래로 숨는 경우는 처음 봤음 원래 툰게임들은 이렇게 구현하나? 비슷하게 따라해본 결과 Distance Field 맵과 Pixel Depth를 활용하는 방향이 쉽겠다 싶었는데, 뭔가 명조는 더 딱딱한걸 보면 단순 메쉬의 월드 포지션을 조절하는건가 싶다
  - [3](https://x.com/ryurud_n5/status/1756354222159994889)
  - [4](https://x.com/ryurud_n5/status/1747572879464833498)
  - [5](https://x.com/ryurud_n5/status/1746504485915246713)
  - [6](https://x.com/ryurud_n5/status/1845474453024895215)
  - [7](https://x.com/ryurud_n5/status/1831341519913255228)
- 하람쥐님
- [CatLikeCoding](https://catlikecoding.com/)
- 무슨 말인지 이해하고 싶다
  - 이펙터는 웬만한 셰이더를 반투명으로만 사용한다. Translucent 아니면 Additive 즉 최적화를 크게 신경쓰지 않는. 반면 모델러들은 양면 렌더도 제대로 못 쓰게 한다 최적화에서 가장 크게 잡아먹는 Shadowdepth도 이펙터들은 크게 신경쓰지 않는다
  - 모델러들은 라이트 베이킹부터 VRAM 때문에 라이트맵 해상도부터 줄여나가고 Batching과 Instancing도 해야하는 작업을 가진다. 이쁜 gi를 쓰고싶어도 forwardrender 쓰는 모바일에선 라이트베이킹만 써야하고. 하다못해 그것도 안써서 페이크 라이팅 쓰는곳도 많음. 특히 커스텀 셰이딩 모델은 대부분 캐릭터에서 나오지 이펙트에서 나오지 않는다.
- Graphic material materialForRendering baseMaterial sharedMaterial?
- material -> 사설 가능, 렌더링 사용 여부 X, 수정 가능 여부  O
- base -> 원본, X, X
- materialForRendering -> 렌더링에 쓰는, O, X
- 렌더링 영향 주려면
- Material mat = new Mat(renderMat)
- mat.SetColor (~)
- tmp.fontMaterial = mat
- TMP fontMaterial fontSharedMaterial

### 역사

```yml
# CG
date: 2024-11-13. 06:35 # Init
last_modified_at: 2025-03-24. 00:03 # 메모
```

2025-05-28. 글 병합,  
`2024-11-13-computer-graphics-DRAFT: CG`.
