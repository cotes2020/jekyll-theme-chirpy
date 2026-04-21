---
title: "Normal, NormalMap"
# description: ""
categories: [컴퓨터, 그래픽]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-04-01. 00:00 # ?
# last_modified_at: 2024-11-13. 07:02 # Init
---

## 머리말

---

Normal과 NormalMap에 대해.  

## Normal, NormalMap

---

### Normal

법선 벡터.  
표면/Face의 방향/각도/기울기를 나타내는 벡터.  

### NormalMap

Normal을 텍스쳐로 만든 것.  

### Use

굴곡을 만들어주는, 특정 부준을 그림자지게 만들어주는, 그래서 퀄리티가 높아보이게 만들어주는 텍스쳐.  
실제 메쉬 폴리곤 디테일이 없는 부분을 디테일이 있는 것처럼 보이게하는 맵  

일반적으로는, 로우폴리메시에서 하이폴리메시의 디테일을 살려주는 역할  

### Why

<https://docs.unity3d.com/kr/2020.3/Manual/StandardShaderMaterialParameterNormalMap.html>  

오브젝트의 디테일. 나사, 조금씩 파인 곳 같은.  
이런 것들은 메쉬로 표현된다.  

디테일을 메쉬로 표현한다는 것은, 큰 메쉬에서 미세한 폴리곤을 써야 한다는 것.  
미세한 폴리곤의 수가 많아진다면, 최적화 측면에서, 관리 측면에서 복잡해진다.  

미세한 폴리곤 대신, 디테일을 표현하는 대체 방법으로 NormalMap을 사용한다.  

(하이폴리의) 노말 정보를 텍스쳐 이미지 형태로 저장하여 로우폴리에 적용시키는 것이다.  

하이폴리 → 폴리곤이 많음 → 노말이 많음  
로우폴리 → 폴리곤이 적음 → 노말이 적음  

## NormalMap은 무슨 색일까 (WIP)

---

### 궁금증

일반적으로 Unity Editor의 Project 창에서 NormalMap은 파랗게 보인다.  
그런데, NormalMap을 Albedo에 넣은 Material을 Mesh에 적용시키면, 이상하게도 빨갛게 나온다  

Project 창에서 파랗게 보이던 NormalMap이 왜 빨간색으로 나오는지 궁금했다.  
왜 이렇게 나오는지, 그래서 NormalMap은 어떤 색인 것인지에 대해 알아보았다.  

### 1. NormalMap은 빨간색이 맞다

결론적으로, NormalMap은 빨간색이 맞다.  

이 이유를 알기 위해서는 NormalMap의 원리와 압축방식과 대해 알아야 한다.  

R U  
G V  
B Normal  

<https://www.reddit.com/r/Unity3D/comments/vcty4f/pink_normals/>  
Unity Normal맵 압축에 DXTnm을 사용하고 노멀을 RGBA = (1.0, y, y, x)로 저장  

<https://steamdb.info/patchnotes/5003146/>
<https://docs.unity3d.com/kr/2020.3/Manual/SL-BuiltinIncludes.html>  

<https://80.lv/articles/overwatch-technical-overview/>  
오버워치는 실제로 노란 노말맵을 쓴다  
BC5 같은 압축 방법을 써서, 추후에 언팩(압축해제)할때 블루 채널을 만들어낸다  
이는 성능이 뛰어나고, 적은 아티팩트  

일반 노멀 맵을 얻으려면 알파 채널을 녹색 채널로 이동하고 파란색 채널을 순수한 흰색

<https://catlikecoding.com/unity/tutorials/rendering/part-6/>

<https://docs.unity3d.com/kr/2020.3/Manual/SL-BuiltinIncludes.html>
유니티경로/Data/CGIncludes/UnityCG.cginc  

UnpackScaleNormal

```c

inline fixed3 UnpackNormalDXT5nm (fixed4 packednormal)
{
    fixed3 normal;
    normal.xy = packednormal.wy * 2 - 1;
    normal.z = sqrt(1 - saturate(dot(normal.xy, normal.xy)));
    return normal;
}

// Unpack normal as DXT5nm (1, y, 1, x) or BC5 (x, y, 0, 1)
// Note neutral texture like "bump" is (0, 0, 1, 1) to work with both plain RGB normal and DXT5nm/BC5
fixed3 UnpackNormalmapRGorAG(fixed4 packednormal)
{
    // This do the trick
   packednormal.x *= packednormal.w;

    fixed3 normal;
    normal.xy = packednormal.xy * 2 - 1;
    normal.z = sqrt(1 - saturate(dot(normal.xy, normal.xy)));
    return normal;
}

inline fixed3 UnpackNormal(fixed4 packednormal)
{
#if defined(UNITY_NO_DXT5nm)
    return packednormal.xyz * 2 - 1; //DXY가 아니면 그냥범위만 조정해서  
#else
    return UnpackNormalmapRGorAG(packednormal);
#endif
}

```

<https://community.gamedev.tv/t/baking-normal-map-on-orc-horn-turning-out-yellow/182487>  
파랑 노말맵에 중간중간 노란 노말맵  
면이 뒤집혀져 있다던지?  

- Normalmap을 쓰기 위해 프로퍼티에 _BumpMap("Normalmap" ,2D) = "bump"{} 를 선언
- Normalmap은 게임용 텍스쳐 포맷인 DXT1, DXT5가 아니라 DXTnm의 포맷
- DXTnm은 일반적인 텍스처 압축으로 인한 Normalmap 품질저하를 막기위한 AG파일 포맷
- 이 텍스쳐를 이용하여 Normalmap을 생성해내려면 함수를 사용해야 함

이 파일 포맷은 일반적인 텍스쳐의 압축에 의한 NormalMap 품질의 저하를 막기 위해 만든 AG 파일 포맷. 이 포맷은 NormalMap의 R과 G의 퀄리티를 최대한 보전하여 A와 G에 넣어 저장함. (B는 가지고 있지 않음) 이렇게 보전된 R과 G는 NormalMap의 X와 Y로 계산되며, Z는 삼각함수를 이용하여 수학적으로 추출됨

그러므로 이 텍스쳐를 이용해서 NormalMap으로 온전하게 생성해내려면 앞에 설명한 공식이 적용되어 있는 함수를 이용하면 간편함.

DXT5nm에서의 노멀 처리.

위는 DXTnm나 BC5의 노멀 텍스쳐의 RGB 색상에 1개의 채널을 더해 RGBA로 바꾸고 소스 R은 G로, 소스G는 A로 변경해 R과 B는 공백으로 둔다.(이렇게 됨으로 아티팩트를 줄일수 있음) 따라서 packednormal.xy는 packednormal.ag와 동일하다.

이는 DXT5  R:5 G:6 B:5 A:8로 변환(RGB16bit를 24bit RGBA로 변환)할때 G와A채널 값이 가장 크기 때문

<https://docs.unrealengine.com/5.1/ko/vector-operation-material-expressions-in-unreal-engine/>  
<https://forum.unity.com/threads/modifying-standard-shader-to-work-with-dxt-normal-textures.774353/>  
<https://forum.unity.com/threads/runtime-generated-bump-maps-are-not-marked-as-normal-maps.413778/>  
5.5보다 새로운 Unity 버전을 사용하는 경우 빨간색과 파란색 채널이 검은 색이 아닌 흰색이어야합니다. 그러나 더 이상 작동하기 위해 노멀 맵을 스위즐링할 필요도 없습니다.  

전체 "노멀 맵" 지정은 게임과 렌더링 시스템에 개념이 없는 순전히 에디터 텍스처 임포터이며, 특별한 방식으로 포맷된 텍스처2D 애셋일 뿐입니다.

경고가 귀찮고 커스텀 셰이더를 사용하는 경우 `_BumpMap` 또는 `_NormalMap`라는 텍스처 속성 또는 [Normal] 재질 속성 서랍을 사용하지 마십시오.  

2017.2 ~ 2018.4 사이에  
이전에는 float4(1, y, 1, x)로 표시되었지만 이제는 float4(x,y,?,1)입니다.  

2017.1에 BC5 노멀맵에 대한 지원 추가  
선택적으로 RGBA 1y1x (실제로는 스위즐 DXT5인 "DXTnm" 또는 스위즐 RGBA32인 "선형 Nm 32비트"로 표시됨) 및 RG xy01 팩 노멀(BC5는 항상 0.0B 및 1.0A를 반환하는 5채널 RG 전용 형식임)을 모두 사용할 수 있습니다. 6.2018 및 이전 버전의 경우 실제로 yyyx로 압축되었지만 GA 채널 만 사용되었습니다.

4.5에서는 둘 다 계속 지원됩니다. HDRP의 기본값은 BC5 노멀 맵입니다. LWRP 다른 개정판의 기본 형식에 BC1 및 이전 "DXTnm"을 모두 사용하는 것을 보았지만 현재 기본값을 살펴 않았습니다. 내장 렌더링 경로 인 AFAIK는 여전히 DXTnm을 사용하지만 1y01x 및 xy<>이 모두 지원됩니다.

RGBA(107, 153, 250, 255)는 Unity 2017, 1부터 RGBA(153, 107, 255.153)가 되어야 합니다. 실제로 B 값은 다소 임의적이며 0, 255, G 또는 텍스처에 압축하려는 다른 임의의 데이터가 될 수 있지만 일부 추가 압축 아티팩트가 필요합니다. 그 이유는 지난 게시물에서 볼 수 있듯이 Unity가 R 채널과 A 채널을 함께 곱하여 인코딩 된 X를 얻었 기 때문에 Unity 2017.1 이상은 RG 및 AG 일반 인코딩을 모두 지원합니다.

이 값에 대해 조금 더 다루고 싶었습니다. BC (일명 DXTC) 압축 형식이 작동하는 방식, 빨간색 및 파란색 채널은 각각 5 비트를 사용하고 녹색 채널은 블록 색상 팔레트 당 인코딩에 6 비트를 사용합니다. 즉, 녹색 값이 빨간색 또는 파란색보다 정밀도가 약간 더 높기 때문에 Y가 녹색 채널에 남아 있습니다. Unity는 실제로 "DXTn"을 (1.0, y, y, x)로 저장하지만 G와 B의 정밀도 차이는 셰이더로 샘플링할 때 완벽하게 동일하지 않을 수 있음을 의미합니다. 솔직히 왜 파란색 채널에서 Y를 복제하기로 선택했는지 모르겠지만 그레이 스케일 값으로 저장했기 때문에 3 개 채널 모두 Y를 유지하고 빨간색을 1.0으로 변경하는 것이 기존 코드에 대한 가장 작은 변경 사항이었습니다. 데이터가 RGB 채널에 복제 될 때 일부 인코더가 약간 더 잘 수행 될 수 있지만 형식이 그렇게 더 잘 압축 될 기술적 인 이유는 없습니다.

<https://someiyoshino.info/entry/20171205/1512486725>

DXTC, DXT5, BCn BC3  
Direct X Texture Compression  
Block Compression  

DXT5 = BC3  

<https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=hram01&logNo=221489477514>  

## 메모

---

- [참고](https://forum.unity.com/threads/normal-map-looks-red-in-shader-graph.1071962/)  
- [참고](https://github.com/Perfare/AssetStudio/issues/529)  
- [참고](https://www.reddit.com/r/unity/comments/txdew7/why_does_the_texture_turn_red_or_purple_when_i/)  
- [참고](https://www.reddit.com/r/Unity3D/comments/19u6ea/normal_maps_in_unity_why_do_some_normal_maps_get/)  
- [참고](https://raypop.tistory.com/71)  
- [참고](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=gardiss&logNo=221099030447)  
- [참고](https://raypop.tistory.com/53)  
- [참고](https://docs.unity3d.com/2023.2/Documentation/Manual/StandardShaderMaterialParameterNormalMap.html)  
- [참고](https://forums.unrealengine.com/t/is-bc5-higher-quality-than-dxt5-nm-compression/279848/4)  
- [참고](https://forum.unity.com/threads/what-the-texture-type-normal-does.428180/)  
- [참고](https://www.reddit.com/r/blenderhelp/comments/11b0lx2/convert_red_normal_map/)  
- [참고](https://forum.unity.com/threads/normal-map-reversed-at-import.720629/)  
- [참고](https://forum.unity.com/threads/normal-map-reversed-at-import.720629/)  
- [참고](https://forum.unity.com/threads/texture-type-set-as-normal-map-shows-up-as-red-in-secondary-textures-and-transparent.1284611/)  
- [참고](https://mgun.tistory.com/1892)  
- [참고](https://velog.io/@kimpro/Normal-Map%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4)  
- [참고](https://darkcatgame.tistory.com/84)  
- [참고](https://forums.unrealengine.com/t/pink-normal-maps/39707/12)  
- [참고](https://www.reddit.com/r/Unity3D/comments/vcty4f/pink_normals/)  
- [참고](https://forum.unity.com/threads/runtime-generated-bump-maps-are-not-marked-as-normal-maps.413778/#post-4935776)  
- [참고](https://forum.unity.com/threads/runtime-generated-bump-maps-are-not-marked-as-normal-maps.413778/#post-4935776)  
- [참고](https://forums.unrealengine.com/t/is-bc5-higher-quality-than-dxt5-nm-compression/279848)  
- [참고](https://someiyoshino.info/entry/20171205/1512486725)  
- [참고](https://forum.unity.com/threads/runtime-generated-bump-maps-are-not-marked-as-normal-maps.413778/)  
- [참고](https://exien.tistory.com/48)  
- [참고](https://namu.wiki/w/%ED%85%8D%EC%8A%A4%EC%B2%98%20%EC%95%95%EC%B6%95%20%ED%8F%AC%EB%A7%B7)  
- [참고](https://namu.wiki/w/S3%20Texture%20Compression)  
- [참고](https://koreascience.kr/article/JAKO200606141801680.pdf)  
- [참고](https://www.google.com/search?q=dxt5+4bit&ei=wF4oZKD9CuaQ2roPi96j4Ag&ved=0ahUKEwig3Z_-j4n-AhVmiFYBHQvvCIwQ4dUDCA8&uact=5&oq=dxt5+4bit&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIFCCEQoAEyBQghEKABOgoIABBHENYEELADOgYIABAIEB46BQgAEIAEOgQIABAeOgcIABCKBRBDSgQIQRgAUKsLWJ8iYNEjaARwAXgAgAGGAYgBiAuSAQQwLjEymAEAoAEByAEBwAEB&sclient=gws-wiz-serp)  
- [참고](https://docs.unity3d.com/kr/2020.3/Manual/StandardShaderMaterialParameterNormalMap.html)  
- [참고](https://www.youtube.com/watch?v=G531ABhyxfE)  
- [참고](https://gammabeta.tistory.com/3531)  
- [참고](https://www.google.com/search?q=dxt5+bit&ei=pWIoZMjOOZyM2roPvv6q0Ak&ved=0ahUKEwiItYLak4n-AhUchlYBHT6_CpoQ4dUDCA8&uact=5&oq=dxt5+bit&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIGCAAQCBAeOggIABCiBBCwAzoLCAAQiQUQogQQsAM6BQgAEIAEOgQIABAeOgUIIRCgAUoECEEYAVDxCVi0DGDhDmgCcAB4AIABiQGIAfMCkgEDMC4zmAEAoAEByAEFwAEB&sclient=gws-wiz-serp)  
- [참고](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=hram01&logNo=221489477514)  
