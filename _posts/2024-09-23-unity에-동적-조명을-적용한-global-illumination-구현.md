---
title: 'Unity에 동적 조명을 적용한 Global Illumination 구현'
description: 동적 조명의 영향을 받는 글로벌 일루미네이션을 연구했던 과정을 설명해보고자 합니다.
author: ounols
date: 2024-09-23 16:31:00 +0800
categories: [Dev]
tags: [Coding, Unity, dev, graphics, made]
pin: true
math: true
mermaid: true
redirect_from:
  - unity에-동적-조명을-적용한-global-illumination-구현
image:
  path: /media/unity에-동적-조명을-적용한-global-illumination-구현/image%2011.png
---

안녕하세요! 오랜만에 블로그 글을 작성해봅니다.

이번엔 회사 내부 프로젝트로 사용하기 위한
**동적 조명의 영향을 받는 글로벌 일루미네이션**을 연구했던 과정을 설명해보고자 합니다.

> 어디까지나 회사 내부 프로젝트에 적용되는 내용이라<br/>
> 상세한 구현은 보여드릴 수 없는 점 양해 바랍니다.
>
> 해당 포스트는 GI를 간단한 기존 유니티 기능을 통해 연구했던 내용만 간략하게 포함됩니다!
{: .prompt-info }

## 구현 결과

{% include embed/video.html src="/media/unity에-동적-조명을-적용한-global-illumination-구현/1.mp4" title="GI를 위한 커스텀 G-Buffer(512x512)가 각 섹션 별로 3장 적용되어 있습니다." %}

먼저 영상을 확인하면서 ‘동적 조명을 적용한 GI’를 어떻게 구현이 되었는지를 확인해봅시다!

위 영상에서 보여주는 해당 구현의 특징은 아래와 같습니다.

- 저 영상에서 GI을 제외한 라이팅은 **Directional Light 하나가 전부**입니다.
- Directional Light가 향하는 방향으로 **GI Diffuse가 동적**으로 변하고 있습니다.
- Directional Light가 향하는 방향으로 **GI Reflection 역시 동적**으로 변하고 있습니다.
- 중간에 위치한 작은 구가 섹션을 이동할 때 마다 각 섹션에 맞는 GI 값을 적용받고 있습니다.
- 유니티 URP에서 커스텀 된 G-Buffer가 구워진 상태로 돌아가기 때문에 **GI 연산이 타 GI 기술에 비해 상대적으로 매우 가볍습니다.**

그럼 이야기를 계속 진행해보겠습니다!

## 어쩌다 이걸 유니티에…?

이 질문이 가장 많았습니다. 왜 이런 기술을 유니티에 적용하게 되었는지…<br/>
하지만 답변은 간단했습니다. 회사 프로젝트에 필요했기 때문이죠…ㅋㅋ

아무튼, 회사 프로젝트엔 시간에 따른 **낮과 밤의 실시간 라이팅 변화**가 필요했고<br/>
로우 폴리곤 형태의 그래픽에 사실적인 라이팅을 곁들이는 기술이 필요했습니다.

꽤 까다롭고 처음 연구를 해보는 그런 내용이였지만 이런 기술은 뭔가 꽤 익숙한 느낌이 들었습니다.

{% include embed/youtube.html id='00QugD5u1CU?start=1631' title='이 영상의 일부분에서 해당 설명이 나옵니다.' %}

![image.png](/media/unity에-동적-조명을-적용한-global-illumination-구현/image.png)

바로 유나이트 서울 2020에서 원신의 컨퍼런스 내용 중<br/>
**작은 크기의 G-Buffer를 사용하여 기존 GI를 리라이팅** 후 적용해서 넣는 방식이였습니다.

이미 상용 게임에서도 구현을 했고, 컨퍼런스에도 친절하게 설명을 했기에 이걸 도전해보자고 마음먹고 구현을 해봤습니다.

## 어떻게 구현할지 구상하기

이 기술은 제한된 환경(고정된 배경 오브젝트)에서 작동하지만, 상대적으로 자유로운 GI를 사용할 수 있으면서도 **가벼운 성능으로 구현되는 것이 가장 큰 장점**입니다. 이러한 이점을 살리는 핵심 요소가 바로 **G-Buffer**입니다. (미호요는 이미지 압축에도 공을 들였지만, 이 게시글에서는 구현 결과 확인에 초점을 맞추어 이미지 압축은 다루지 않았습니다.)

G-Buffer는 렌더링을 조금 공부해본 분들이라면 디퍼드 렌더링에 사용되는 "그것"이라고 아실 겁니다. 하지만 G-Buffer는 전통적인 디퍼드 렌더링을 넘어 다양한 용도로 활용할 수 있는 만능의 기술입니다. 이 만능의 기술이 여기서도 빛을 발하네요! 벌써부터 구현할 생각에 설렙니다.

자, 그럼 필요한 준비물을 정리해볼까요?

- **GI 정보를 저장할 G-Buffer**
    - albedo 텍스처
    - normal 텍스처
- **쉐이더**
    - G-Buffer 굽기용 (사전에 구워두며, 런타임에는 사용하지 않습니다)
    - G-Buffer 라이팅 계산용 (Probe에서 계산)
    - 계산된 GI 최종 렌더링용 (GI가 적용되는 모든 렌더링 객체에서 계산)
- **스크립트**
    - G-Buffer 굽기용 (사전에 구워두며, Editor에서 작동)
    - G-Buffer 라이팅 계산용 (Probe에서 계산)
    - 계산된 GI 최종 렌더링용 (주변 Probe를 파악하여 최종적으로 GI를 계산)

## G-Buffer를 어떻게 사용할까?

먼저 G-Buffer라고 부르는 이유를 살펴보겠습니다.

G-Buffer는 특정 프레임 버퍼를 통해 렌더링에 필요한 데이터를 미리 저장하고, 이후 해당 버퍼를 사용해 계산을 수행하는 개념입니다. 이러한 방식이 일반적인 지오메트리 버퍼의 사용과 유사하여 G-Buffer라고 부릅니다.

![왼쪽 : albedo 텍스쳐, 오른쪽 : world normal 텍스쳐](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%201.png)
_왼쪽 : albedo 텍스쳐, 오른쪽 : world normal 텍스쳐_

위 이미지를 보면 albedo 값과 world normal 값이 저장되어 있습니다. 이 두 가지 정보만으로도 간단한 디퓨즈 라이팅을 동적으로 구현할 수 있습니다.

그러나 재질별 PBR 라이팅은 데이터 부족으로 어렵고, 깊이 값을 이용한 레이마칭으로 그림자를 구현하면 성능 부하가 큽니다.

원신에서도 이러한 세부적인 요소들을 포기하고 구현한 것으로 보입니다.

이제 G-Buffer의 개요 설명을 마치고 실제 구현으로 넘어가겠습니다!

## G-Buffer 구현

우리가 구현하고자 하는 GI의 핵심은 주변 환경의 형태는 유지하면서 동적 라이팅에 따라 색상이 변화하는 것입니다. 이를 위해 G-Buffer를 런타임에서 계산할 필요 없이 Unity Editor에서 미리 구워두도록 설계했습니다.

여기서 중요한 과제가 있습니다. 바로 G-Buffer를 스카이박스처럼 360도로 렌더링해야 한다는 점입니다. 이를 해결하기 위해 두 가지 방법을 고려했습니다

- 6면의 FOV를 90도로 설정하여 정사각형으로 촬영한 후, 6개의 텍스처로 저장한다.
- Equirectangular 맵으로 저장한다.

저는 두 번째 방법인 Equirectangular 맵 저장을 선택했습니다.
그 이유는 다음과 같습니다

- 텍스처 6개보다 1개를 유니폼으로 전달하는 것이 더 효율적이다.
- 런타임 중 큐브맵에서 Equirectangular 맵으로의 변환은 시간이 걸리지만, 사전에 구워두면 이 문제를 피할 수 있다.

자, 이제 실제 구현을 시작해보겠습니다!

```jsx
// Albedo 텍스쳐 렌더링을 위한 Pass
Pass {
    Name "Albedo"
    
    [...] // 상세 내용 생략
    
    // -------------------------------------
    // Includes
    #include "Packages/com.unity.render-pipelines.universal/Shaders/UnlitInput.hlsl"
    #include "Packages/com.unity.render-pipelines.universal/Shaders/UnlitForwardPass.hlsl"
    ENDHLSL
}

// Normal 텍스쳐 렌더링을 위한 Pass
Pass {
    Name "Normal"
    
    [...] // 상세 내용 생략
    
    // -------------------------------------
    // Includes
    #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
    #include "DeferredReflectionNormal.hlsl"
    ENDHLSL
}
```
{: file='Unity shader'}

먼저 유니티 쉐이더 코드입니다.

albedo는 기존 unlit 쉐이더에서 재활용해도 문제가 없기 때문에 unlit 그대로 사용했습니다.<br/>
하지만 normal은 **모든 방향에 대한 노멀 텍스쳐**이기 때문에 구현해야할 파트가 어느정도 있었기 때문에 따로 hlsl을 작성했습니다.

```hlsl
[...] // 상세 내용 생략

// 기존 월드 노멀로 바꿔주는 코드
float3 normalWS = TransformTangentToWorld(
                      normalTS,
                      half3x3(input.tangentWS.xyz
                      bitangent.xyz, input.normalWS.xyz)
                  );
													
normalWS = PackingNormal(NormalizeNormalPerPixel(normalWS));

// 노멀값을 패킹하는 함수
half3 PackingNormal(const half3 n) {
    return n * 0.5 + 0.5;
}
```

여기서 PackingNormal을 통해 음수로 된 값 까지 모두 양수로 패킹하고 넘기는 모습입니다.

![image.png](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%202.png)

렌더링 준비는 끝났으니 유니티 에디터를 통해 렌더링 할 위치값을 잡고 구워줍니다.

![image.png](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%203.png)

그럼 위와 같은 텍스쳐를 구울 수 있게 됩니다.

## G-Buffer를 이용한 환경광 Relighting

![image.png](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%204.png)

이제 G-Buffer를 구했으니, 이를 하나로 합쳐 환경광에 맞게 리라이팅해야 합니다. 제 경우, G-Buffer를 구웠던 위치에 "Deferred Reflection Probe"라는 스크립트를 통해 독자적인 Probe 개념을 만들었습니다.

그 다음, Custom Render Texture를 사용하여 최종 환경맵을 렌더링합니다.

![image.png](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%205.png)

최종 환경맵 생성을 위해서는 기존에 만든 G-Buffer를 가져와 합치는 쉐이더가 필요합니다.<br/>
여기서 저희는 Unity 기능을 사용하기 때문에, 상단에 노멀맵을 사용하지만 노멀맵 설정이 되지 않은 텍스처를 사용한다는 경고 메시지가 표시됩니다.

저희의 노멀맵은 -1에서 1까지의 값을 패킹한 데이터이므로, 노멀맵으로 따로 지정하지 않고 일반적인 텍스쳐 형태로 직접 진행해야 합니다.

이제 쉐이더 구현으로 넘어가 볼까요?

```hlsl
[...] // 기존 PBR 픽셀 쉐이더가 진행 됨

half strength = 1.5; // 라이팅 가중치
half4 color = half4(0.); // 최종 렌더링 값

Light mainLight = GetMainLight();
float3 normal = tex2D(_NormalMap, input.localTexcoord) * 2. - 1.;
float3 n = normalize(normal); // 노멀값 추출 (월드 좌표의 노멀값이기 때문에 그대로 사용)
float diffuse = dot(n, mainLight.direction) * 0.5 + 0.5;
diffuse = ease_in_quad(diffuse); // Diffuse 계산
half3 albedo = brdfData.diffuse;
half3 bakedGI = SampleSH(n); // 기존 URP에서 사용하는 환경광 데이터

[...] // 기존 데이터에서 추가적인 보정을 진행

// 최종 환경광 값을 적용
color.rgb = (diffuse * mainLight.color * strength + bakedGI) * albedo;

return color;
```

버텍스 쉐이더는 평범한 quad 렌더링이기 때문에 넘어가고, 핵심은 픽셀 쉐이더 입니다.

기존 URP에 적용해야하기 때문에 URP에서 계산할 때 사용되는 데이터들을 들고와줍니다.<br/>
그리고 위와 같은 코드로 계산을 진행합니다.

노멀값은 언패킹만 해주면 월드 좌표의 노멀값이 나오기 때문에 언패킹 이외에 별 다른 조치가 필요하지 않습니다.

이 후 간단하게 MainLight 데이터를 가지고 diffuse 값을 램버트로 추론하면 됩니다.

뭔가 엄청난게 들어있지도 않고 평범한 라이팅 계산식만 있습니다.<br/>
그렇기에 이해 난이도도 높지 않고, 꽤 가볍지 않을까 생각이 듭니다.

{% include embed/video.html src="/media/unity에-동적-조명을-적용한-global-illumination-구현/2.mp4" title="실시간 라이팅의 변화에 따라 렌더 텍스쳐의 결과값도 달라지는 모습을 확인할 수 있다" %}


이렇게 쉐이더도 완성하게 되면 위와 같이 메인 라이팅에 맞게 환경광도 변화하는 모습을 확인할 수 있습니다. (오른쪽 텍스쳐 프리뷰 참고)

## Probe를 통해 오브젝트 렌더링

Probe도 완성했고, 렌더 텍스쳐에 렌더링 되는 모습까지 확인했으니 이제 본격적으로 렌더링을 해보겠습니다.

기존 URP의 Lit 쉐이더에서 조금만 추가할 예정입니다.<br/>
ForwardLit 파트의 픽셀 쉐이더 파트를 알아보도록 하겠습니다.

먼저, 쉐이더에 필요한 함수부터 짚어보겠습니다.

```hlsl
// 기존 노멀값을 equirectangular 좌표계로 변환하는 함수입니다
float2 NormalToUV(float3 normal) {
		// 유니티식 노멀값을 equirectangular 좌표계에 연동하기 위해 일부 데이터를 스왑
    normal = float3(normal.z, normal.y, -normal.x);
    // equirectangular 좌표로 변환
    float lon = atan2(normal.z, normal.x);
    float lat = acos(normal.y / length(normal));
    
    // equirectangular 좌표를 UV로 전환
    float u = (lon + 3.14159265359) / (2.0 * 3.14159265359);
    float v = 1. - lat / 3.14159265359; // v는 거꾸로 있어야 함
    
    return float2(u, v);
}

// 계산된 환경광을 들고오는 함수입니다.
half4 GetEnvColor(const half2 uv, const float lod) {
    return SAMPLE_TEXTURE2D_LOD(_Reflection, sampler_Reflection, uv, lod).rgba;
}
```

지금까지 렌더링을 2D 텍스쳐 형태의 Equirectangular 맵으로 진행했기 때문에, 노멀값을 해당 Equirectangular 좌표계로 변환하는 함수가 필수적입니다.

[텍스쳐 6개를 skybox처럼 계산하는 것보다 훨씬 가볍죠.](/posts/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/#4-레거시-렌더링에-적용하기)<br/>
그래서 제 엔진에서 만들었던 SDFDDGI의 프로브를 Equirectangular 맵으로 전환할지 고민이 많이 됩니다… ㅎㅎ

그리고 환경맵을 LOD로 받아옵니다. 여기서 LOD를 사용하는 이유가 뭘까요?<br/>
바로 roughness를 표현하기 위해서입니다!

roughness (유니티에선 비슷한 개념으로 smoothness)를 표현할 때 간단하게 LOD로 블러 효과를 줍니다. 이렇게 하면 블러 효과를 직접 적용하지 않고도 흐린 이미지를 얻을 수 있습니다. 이 LOD 레벨을 자유롭게 가져오기 위해 `SAMPLE_TEXTURE2D_LOD`라는 매크로 함수를 사용합니다.

자, 이제 본격적으로 픽셀 쉐이더를 구성해볼까요?

```hlsl
half2 diffuse_uv = NormalToUV(normalWS);
half3 r = float3(normalWS.x, normalWS.y, normalWS.z);
r = reflect(-inputData.viewDirectionWS, r);
half2 reflect_uv = NormalToUV(r);
```

해당 오브젝트의 월드 노멀값을 가지고 오브젝트에 비칠 diffuse와 reflection을 계산하기 위한 각각의 uv값을 받아옵니다.

여기서 diffuse는 난반사, reflection은 정반사이기 때문에 각 노멀값을 이용한 계산식은 서로 다른 점 유의하면서 uv값을 가지고 왔습니다.

```hlsl
...

half4 drDiffuse = GetEnvColor(diffuse_uv, 6);
half4 drReflection = GetEnvColor(reflect_uv, (1. - surfaceData.smoothness) * 7);
half NdotL = saturate(pow(dot(normalWS, -inputData.viewDirectionWS) * 0.5 + 0.5, 0.5 - surfaceData.metallic));
drReflection.rgb = pow(drReflection.rgb, 1) * (NdotL);
...
```

이렇게 기존 환경광을 가져올 수 있습니다. 하지만 리플렉션의 경우, 유니티의 PBR 계산을 수정하기에는 너무 많은 부분을 건드려야 합니다. 또한, URP 버전에 따라 이 수준의 함수들이 작동하지 않거나 수정될 가능성이 매우 높습니다. 그래서 안타깝게도 직접 계산하여 적용하는 방식으로 진행했습니다.

결과적으로 완전한 물리 기반의 환경광은 아니지만, 간단히 구현한 Rim Light까지 적용했습니다.

이렇게 기존 Lit 쉐이더에 코드를 조금 더 추가하고 색상을 보정하면 구현이 완료됩니다!

{% include embed/video.html src="/media/unity에-동적-조명을-적용한-global-illumination-구현/3.mp4" %}

그렇게 동적 라이팅이 적용되는 실시간 GI는 위와 같이 작동하게 됩니다!

> 참고로 제작 예제로 사용한 배경은 (회사 프로젝트는 공개할 수 없기 때문에) 제가 영상 CG제작을 위해 만들고 배치했던 레벨입니다.<br/>
> [관련 영상은 해당 링크에서 확인하실 수 있습니다! 많관부!](https://youtu.be/oSIvKaemkfw)
{: .prompt-tip }


|                          동적 조명 GI가 적용된 화면                           |                      기존 URP의 Lit 쉐이더만 적용된 화면                      |
| :---------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
| ![](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%206.png) | ![](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%207.png) |


이 GI는 오브젝트 옆에 벽이 있다면 확실한 효과가 나타납니다.

## 성능 테스트?

사실 제가 구현한 내용과 가장 비슷한 요소는 Reflection Probe 일 것 같습니다.<br/>
하지만 구현 방식이 전혀 다르기도 하고 목적도 다르기 때문에 Reflection Probe와 비교하는건 옳지 않은 것 같은 느낌입니다.

![image.png](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%208.png)

위 이미지는 Reflection Probe 하나를 업데이트할 때의 데스크톱 프로파일링 결과입니다. Reflection Probe는 주변 환경맵을 업데이트할 때 6면을 촬영하여 렌더링하는 로직을 사용합니다. 이 과정은 전반적인 렌더링을 포함하기 때문에 상당한 성능을 소모합니다.

따라서 Reflection Probe는 동적 조명에 따라 실시간으로 변화하는 라이팅의 목적에 부합하지 않습니다. 기존 기능과의 단순 비교보다는 다양한 디바이스에서의 성능 확인이 더 적절하다고 판단했습니다.

이러한 고민 끝에 웹 데모를 제작하게 되었습니다!

## 웹 데모 체험하기

![왼쪽은 mac os, 오른쪽은 ios에서 돌아가는 모습](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%209.png)
_왼쪽은 mac os, 오른쪽은 ios에서 돌아가는 모습_

> [Unity WebGL Player DeferredReflectionDemo](https://ounols.github.io/Deferred-Reflection-Demo)
{: .prompt-info }

성능을 직접 확인해보는 것이 가장 좋기 때문에, WebGL이 구동되는 모든 플랫폼에서 테스트할 수 있도록 웹 데모를 준비했습니다.

다만 급하게 준비하는 바람에 무료 에셋의 최적화가 미흡하여 배칭이 많이 발생합니다. 또한 WebGL의 고질적인 문제로 인해 중간에 스로틀링이 걸릴 수 있는 점 양해 부탁드립니다.

하지만 GI 계산은 Probe마다 순차적으로 V-sync가 적용되어 렌더링되기 때문에, GI Probe 계산이 스로틀링에 큰 영향을 받지 않는다는 점을 참고해 주시면 감사하겠습니다!

![GZY8i25asAA6QZf.png](/media/unity에-동적-조명을-적용한-global-illumination-구현/GZY8i25asAA6QZf.png)

위의 데모에는 총 4개의 Probe가 있습니다. 앞서 설명했듯이 V-sync가 적용된 상태이기 때문에, 10개 이상의 프로브를 거의 동시에 렌더링하는 것이 가능합니다.

이는 가벼운 G-Buffer를 사용하여 간단한 라이팅 계산만으로 처리되기 때문입니다.

![image.png](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%2010.png)

해가 가장 밝고 선명할 정오에 가까운 라이팅 표현에선 바닥으로부터 올라오는 빛의 세기가 가장 강하기 때문에 다른 오브젝트에도 간접광이 적용되어 주변 색상에 더 어울리는 라이팅 계산을 하게 됩니다.

위 사진처럼 굳이 리플렉션의 영향이 적고 디퓨즈 라이팅이 많은 부분을 담당하더라도 전반적인 환경의 색감을 잡아줍니다.

## 정리

|                               프로브 별 GI 데모                                |                              PBR 재질의 변화 데모                              |                               시연용 WebGL 데모                                |
| :----------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :----------------------------------------------------------------------------: |
| ![](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%2011.png) | ![](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%2012.png) | ![](/media/unity에-동적-조명을-적용한-global-illumination-구현/image%2013.png) |

Unity에서 동적 조명을 적용한 Global Illumination(GI) 구현에 대해 3가지 정도의 데모를 통해 살펴보았습니다. 주요 내용을 정리하면 다음과 같습니다:

- 기존 Unity의 Directional Light 하나만으로 동적 GI 효과를 구현했습니다.
- 실시간으로 변화하는 라이팅을 위해 Reflection Probe 대신 커스텀 솔루션을 개발했습니다.
- 쉐이더 코드를 수정하여 diffuse와 reflection을 계산하고 적용했습니다.
- 성능 테스트를 위해 WebGL 데모를 제작하여 다양한 플랫폼에서의 동작을 확인할 수 있게 했습니다.

이 구현 방식은 완전한 물리 기반의 환경광은 아니지만, 효율적이고 실용적인 방법으로 동적 GI 효과를 달성했습니다. 특히 오브젝트 주변의 환경에 따라 자연스러운 간접광 효과를 제공하여 원신과 비슷한 GI 효과를 적용할 수 있으니 즐겁네요!

엠바고로 인해 보여드릴 수 있는 내용은 여기까지만 설명을 드리지만, 여러분의 추가적인 의견과 피드백은 언제나 환영합니다!

긴 글 읽어주셔서 감사합니다!
