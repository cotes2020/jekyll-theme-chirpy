---
title: '자체 엔진에 Global Illumination을 적용하기 위한 삽질기 2'
description: 2달 동안 SDF를 이용한 글로벌 일루미네이션을 구현하고자 목표를 잡고 구현을 진행했었고, 아직 온전하진 않지만 SDF로 생성한 리플렉션 맵을 적용하는 것 까진 성공적으로 구현하게 되었습니다
author: ounols
date: 2023-06-07 01:07:00 +0800
categories: [Dev, 자체 게임 엔진 프로젝트]
tags: [Coding, cpp, dev, made]
pin: true
math: true
mermaid: true
redirect_from:
  - 자체-엔진에-global-illumination을-적용하기-위한-삽질기-2
  - unity--global-illumination
image:
  path: /media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/화면_캡처_2023-06-05_173803.webp
---

{% include embed/video.html src="/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/C_2023-06-05_23-35-49.mp4" title="레이마칭을 이용한 SDF로 실시간 렌더링한 리플렉션 맵을 적용한 모습" %}

낮에는 회사일하고, 밤에는 렌더링 리서칭을 계속 하다보니 꽤 바쁘게 지냈었습니다..ㅎㅎ

참고로 여기서 렌더링 리서칭은 바로 <br/>
레이 마칭을 통한 글로벌 일루미네이션을 적용하는 내용이였습니다!

제가 이전에 [기존 레거시 렌더링을 통해 무작정 글로벌 일루미네이션을 위한 프로브 맵을 만드는 짓](https://velog.io/@ounols/%EC%9E%90%EC%B2%B4-%EC%97%94%EC%A7%84%EC%97%90-Global-Illumination-%EB%A5%BC-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0-%EC%9C%84%ED%95%9C-%EC%82%BD%EC%A7%88%EA%B8%B0-1)을 했었습니다.

그리고 대실패로 돌아갔었지만 레이 마칭으로는 충분히 할만해 보인다고 말하며 글을 마쳤습니다.

그리고 최근 2달 동안 SDF를 이용한 글로벌 일루미네이션을 구현하고자 목표를 잡고 구현을 진행했었고, 아직 온전하진 않지만 SDF로 생성한 리플렉션 맵을 적용하는 것 까진 성공적으로 구현하게 되었습니다!🥳🥳

여기서부터 아래는 제가 구현하면서 알게 된 정보나 개발 히스토리를 정리했습니다.<br/>
먼저 개발 단계는 다음과 같습니다.

1. Mesh에서 SDF Volume Texture로 변환하기
2. SDF Volume Texture를 레이마칭으로 표현하기
3. SDF Reflection Map 생성하기
4. 레거시 렌더링에 적용하기

절차를 확인하기 전에 먼저 어떤 렌더링 플로우인지 확인해보겠습니다.

## 렌더링 플로우

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled.png)

전체적인 렌더링은 위 이미지로 간단하게 설명할 수 있습니다.

SDF Render Group에서는 볼륨 텍스쳐를 각 프로브 객체만큼 렌더링하여 환경 텍스쳐로 만듭니다.<br/>
그 이후 최종 렌더링에서 해당 환경 텍스쳐를 사용하는 방식입니다.

## 1. Mesh에서 SDF Volume Texture로 변환하기

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%201.png){: width="300" .w-40 .left}

<br/><br/>
매쉬에서 SDF의 볼륨 텍스쳐로 변환하기까지 여러 방법들이 있습니다. 
저는 여기서 쉽고 빠르게 할 수 있는 방법을 사용했죠!

바로 x, y, z축으로 일정 간격으로 렌더링해서 저장하는 방식을 사용하게 되었습니다!

![주전자의 초기 SDF 볼륨 텍스쳐](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%202.png)
_주전자의 초기 SDF 볼륨 텍스쳐_

![유니티 텍스쳐 프리뷰로 본 볼륨 텍스쳐](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%203.png)
_유니티 텍스쳐 프리뷰로 본 볼륨 텍스쳐_

저는 위 이미지와 같이 구간별로 자른다음 이걸 SDF 볼륨 텍스쳐로 사용해보기로 했습니다.<br/>
여기서 녹색은 x축, 시안색은 y축, 노란색은 z축으로 확인하실 수 있습니다.

그런데 몇가지 문제가 생겼었습니다… 문제점은 다음과 같았습니다.

### z축 종횡비가 안맞는 문제

이 부분은 저도 잘 모르겠지만… z축 종횡비가 옆으로 2배 늘어난 형태로 렌더링 되는 현상이 있었습니다.

그렇기에 z축에 해당하는 부분만 0.5배로 적용하긴 했지만… 전체적으로 4배 작게 나오고, 공백도 많은 편이기 때문에 해당 부분은 나중에 어느정도 개선을 거칠 예정입니다.

### OpenGL에서 다르게 적용되는 데이터 구성

![OpenGL에서 바로 사용 가능하도록 데이터 구성을 바꾼 SDF 볼륨 텍스쳐](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%204.png)
_OpenGL에서 바로 사용 가능하도록 데이터 구성을 바꾼 SDF 볼륨 텍스쳐_

OpenGL에서는 GL_TEXTURE_3D를 받아올 때 해당하는 index는 다음과 같은 형태로 받아오게 됩니다.

```cpp
int index = (z * size * size) + (y * size) + (x);
```

이에 따라 사실 상 CT촬영같던 이전 볼륨 텍스쳐와는 달리 세로 방향으로 쭉 늘어난 형태의 특이한 형태로 만들어졌음을 알 수 있습니다.

## 2. SDF Volume Texture를 레이마칭으로 표현하기

![제가 구현했던 SDF 볼륨 텍스쳐를 레이마칭을 통해 데이터를 받아오는 과정을 설명하는 그림입니다.](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%205.png)
_제가 구현했던 SDF 볼륨 텍스쳐를 레이마칭을 통해 데이터를 받아오는 과정을 설명하는 그림입니다._

레이마칭을 이용하여 해당 SDF의 값을 가져오는 단계입니다. 이전 볼륨 텍스쳐 생성은 런타임 중에 만들어지지 않아도 무방했지만, 여기서부터는 런타임 중에 실시간으로 작동하게 됩니다.

위 그림에서 보시다시피 크게 2단계로 나눠 볼륨 텍스쳐의 정보를 불러오고 있습니다.

1. 레이의 AABB Box 형식의 Intersection 판단
2. 볼륨 텍스쳐 내부에서의 레이마칭

### 1단계 : 레이의 AABB Box 형식의 Intersection 판단

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%206.png)

첫번째는 간단하게 AABB로 구성된 Box의 Intersection을 구하는 단계입니다.

여기서는 굳이 레이마칭을 사용하지 않고 Box의 교차점을 구하여 박스의 범위값인 near와 far만 얻어옵니다.<br/>
딱히 정확한 거리값은 필요없기도 하고, 런타임 중 쓸데없는 부가 연산을 하는건 나중에 문제가 발생할 수 있기 때문에 위와 같은 구현 방식을 선택하게 되었습니다.

코드는 다음과 같습니다.

```glsl
vec2 RayAABBIntersection(vec3 ro, vec3 rd) {

    vec3 aabbmin = vec3(-AABB_SIZE/2.) * 0.5;
    vec3 aabbmax =  vec3(AABB_SIZE/2.) * 0.5;

    vec3 invR = vec3(1.0) / rd;

    vec3 tbbmin = invR * (aabbmin - ro);
    vec3 tbbmax = invR * (aabbmax - ro);

    vec3 tmin = min(tbbmin, tbbmax);
    vec3 tmax = max(tbbmin, tbbmax);

    float tnear = max(max(tmin.x, tmin.y), tmin.z);
    float tfar  = min(min(tmax.x, tmax.y), tmax.z);

    return tfar > tnear ? vec2(tnear, tfar) : vec2(-999.);
}

.../*중락*/...

vec3 renderTexture(vec3 origin, vec3 direction) {
    vec2 isct = RayAABBIntersection(origin, direction);

    if (isct.x <= -999.) {
        return vec3(0.);
    }

.../*중락*/...
```
{: .nolineno }

### 2단계 : 볼륨 텍스쳐 내부에서의 레이마칭

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%207.png)

Box의 범위가 구해지면 이제 본격적으로 레이를 일정 스탭에 맞게 전진하도록 합니다.

전진하면서 해당 부분에 3D 텍스쳐의 alpha값이 일정 수준 이상일 때 면으로 판단을 하게 됩니다. 여기서 near에서 얼마나 레이가 진행했는지 길이값을 담고 이를 통해 SDF 값을 받아옵니다.

저는 여기서 SDF(A)값과 함께 해당 위치의 RGB값을 받아오도록 합니다.<br/>
아래는 관련 코드입니다.

```glsl
    float D = abs(isct.y - isct.x);

    vec3 wp = origin + direction * isct.x;
    vec3 vol_size = vec3(AABB_SIZE);
    vec3 tp = wp + (vol_size * 0.5);
    float steps = D / 512.f; // steps를 잘게 자르면 자를 수록 표현 정확도가 높아집니다.

    // D만큼의 거리를 지나갑니다
    for (float t = 0.0; t < D; t += steps) {
        // 레이의 방향을 향해 레이를 전진시킵니다.
        vec3 currentPos = tp + direction * t;

        // 밀도값과 함께 RGB값을 가져옵니다.
        vec4 density = texture(u_sdf_tex, currentPos / vol_size);

        // 추가적인 데이터 포팅을 진행할 수 있지만 지금은 RGB값으로만 적용해봅니다.
        vec4 src = vec4(density);

        if (density.a > 0.5) {
						// restore color
            return (src.rgb + src.a * src.rgb * 1.5) * (1. - (t / D) * 2.);
        }
    }
```

이와 같은 로직을 통해 아래와 같이 볼륨 렌더링을 가시화한 렌더링으로 확인하실 수 있습니다.

{% include embed/video.html src="/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/C_2023-06-06 22-56-36.mp4" title="왼쪽은 기존 레거시 렌더링, 오른쪽은 SDF 볼륨 텍스쳐(256^3)로 그려진 레이마칭 렌더링입니다." %}


이처럼 레이마칭 렌더링이 가능하도록 준비가 되었다면 다음단계로 넘어가게 됩니다.

## 3. SDF Reflection Map 생성하기

![언리얼 루멘 시스템을 소개하며 공개한 사진 중 하나](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled.jpeg)
_언리얼 루멘 시스템을 소개하며 공개한 사진 중 하나_

언리얼의 ‘루멘’ 시스템에선 다이나믹 글로벌 일루미네이션을 위해 여러 빛 데이터를 모아놓은 루멘 맵이라는게 존재한다고 합니다.<br/>
이 루멘 맵에는 레이 트레이싱을 통해 계산된 조도 맵, 쉐도우 맵, 리플렉션 맵 등 다양한 데이터를 가지고 있다고 합니다.

이 중 가장 정확도에 민감하고, 특수한 텍스쳐의 형식으로 저장을 필요로하는 리플렉션 맵을 먼저 구현해보고자 했습니다.

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%208.png){: .w-50 .left}

<br/>
사실 구글에 쉽게 검색할 수 있는 SDF를 이용한 GI기술은 대부분 옛날의 다이나믹 디퓨즈만을 지원하는 논문과 포스트글입니다.

왼쪽은 쉽게 접할 수 있는 논문 중 하나인 SDFDDGI(Signed Distance Fields Dynamic Diffuse Global Illumination)입니다.

이 논문을 간단하게 요약해보자면 당시 RTXGI의 기법에서 영감을 받아 SDF로 소프트웨어 렌더링을 통해 다이나믹 디퓨즈 렌더링을 하는 내용입니다.

<br/><br/>

저 역시 다이나믹 디퓨즈 GI만 지원한다면 SDF의 모델 역시 정확도를 요구하지 않기 때문에 <br/>
논문에서 말한 것처럼 그냥 간단하게 표현할 수 있는 SDF 프리미티브들을 직접 배치해서 표현하고, 빛샘현상과 같은 문제를 좀 더 유심히 살펴봤을겁니다.

하지만 이왕 여기까지 온거, “언리얼도 구현했는데 나도 얼추 만들면 할 수 있다(?)”라는 무모한 생각 하나로 리플렉션 맵을 도전하게 되었습니다.

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%209.png){: .w-50 .right}

<br/>
말이 길어졌네요! 이제 구현단계로 들어가보겠습니다.

이번엔 전체 화면에서 좀 특이하게 레이를 쏘게 됩니다. 왜냐하면 리플렉션 맵은 각 프로브에 해당하는 요소에 6면이 들어있는 큐브맵 형태로 렌더링을 해야하기 때문이죠.

그래서 오른쪽과 같은 형태는 무난하게 레이마칭으로 렌더링할 때의 모습이라면, 이번엔 각 가로 세로를 일정 크기에 맞게 나눠서 UV값을 재할당 시켜줍니다.

UV를 재구성하는 코드는 다음과 같이 작성됩니다.

<br/>

```glsl
float index_pos_y = u_node_size.x * u_node_size.y * u_node_size.z;
vec2 new_uv = vec2(1. - fract(v_textureCoordOut.x / (1./6.)),
                       fract(v_textureCoordOut.y / (1. / index_pos_y)));

...

vec2 p = vec2(2. * (new_uv - 0.5));
```

일단 당장의 리플렉션 맵 구성은 가로로 6개의 큐브맵을 일자 형태로 배치를 하고, 세로로는 각 프로브에 맞는 리플렉션 맵을 구성할 예정입니다. 이에 따라 `index_pos_y` 에 총 프로브의 개수를 지정하고 그에 맞게 UV를 재구성합니다. 그리고 포지션값인 `p` 를 만들어줍니다.

아 참! ray origin값은 프로브 위치값으로 설정하면 그만이지만… 바라보는 방향에 따라 perspective view 90도로 설정을 해줘야 합니다!

따라서 다음과 같이 추가적인 코드도 작성해줍니다.

```glsl
const float c_pv_d = 1.0;

const mat3 c_pv_m0 = mat3(
vec3(0., 0., 1.),
vec3(0., -1., 0.),
vec3(1., 0., 0.)
);
const mat3 c_pv_m1 = mat3(
vec3(0., 0., -1.),
vec3(0., -1., 0.),
vec3(-1., 0., 0.)
);

...중략...

const mat3 c_pv_m5 = mat3(
vec3(1., 0., 0.),
vec3(0., -1., 0.),
vec3(0., 0., -1.)
);
```

위 코드는 perspective를 구성할 때 필요한 값인 $d$값과, $LookAt$의 행렬값을 6방면에 맞게 넣어놓고 `const`로 선언을 합니다.

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%2010.png)

여기서 $d$값은 위 이미지처럼 보입니다. 여기서 만약 $\theta$값이 $90^\text{o}$라면 어떻게 될까요? 그렇다면 $\frac{\theta}{2}$값은 $45^\text{o}$가 되면서 이등변 직각 삼각형, 즉 $d$값이 1이 되는 상황이 됩니다.

> 관련된 상세한 수학적 분석은 [‘이득우의 게임수학’](https://www.google.com/search?client=firefox-b-d&q=%EC%9D%B4%EB%93%9D%EC%9A%B0%EC%9D%98+%EA%B2%8C%EC%9E%84%EC%88%98%ED%95%99)과 [제 블로그 게시글에서 확인하실 수 있습니다!](https://velog.io/@ounols/%EA%B2%8C%EC%9E%84-%EC%88%98%ED%95%99-%EC%9B%90%EA%B7%BC-%ED%88%AC%EC%98%81)
{: .prompt-info }

무료 홍보는 여기까지 하고 계속 진행해볼까요?ㅎㅎ

어쨌든 ray direction은 재구성된 uv값에서 z축은 perspective의 $d$값으로 구성하면 내가 원하는 프로젝션으로부터 나온 방향이 됩니다.

코드는 다음과 같이 나타납니다.

```glsl
// camera
vec3 ro = vec3(0, 0, 0) - pos * u_node_space + u_node_size * u_node_space * 0.5;

// Setting View Matrix
mat3 viewMat = mat3(0.);
{
    int i = int(mod(node_index, 6.));
    if(i == 0) viewMat = c_pv_m0;
    if(i == 1) viewMat = c_pv_m1;
    if(i == 2) viewMat = c_pv_m2;
    if(i == 3) viewMat = c_pv_m3;
    if(i == 4) viewMat = c_pv_m4;
    if(i == 5) viewMat = c_pv_m5;
}

// ray direction
vec3 rd = viewMat * normalize(vec3(p.xy, c_pv_d));
```

렌더링은 이렇게 하면 모든 준비가 끝났습니다!

이제 엔진 상에서 이 SDF Map을 렌더링하려면 어떻게 하냐가 중요한데요. 이는 제가 [예전에 작성했던 블로그 글](https://velog.io/@ounols/%EB%A0%8C%EB%8D%94%EB%A7%81-%EB%AC%B4%EC%9E%91%EC%A0%95-%EA%B0%9D%EC%B2%B4%ED%99%94-%ED%95%98%EA%B8%B0)의 `RenderGroup` 클래스를 이용하여 구현했습니다.

> [`RenderGroup`으로 구현된 레이마칭 렌더러의 코드는 해당 깃허브 링크에서 확인하실 수 있습니다!](https://github.com/ounols/CSEngine/blob/6809cc70883b33be37abc0cef0fc9f0bd3f487e5/src/Manager/Render/SdfRenderGroup.cpp)
{: .prompt-tip }

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%2011.png)

이렇게 렌더링하면 다음과 같은 리플렉션 맵이 실시간으로 그려지게 됩니다.

## 4. 레거시 렌더링에 적용하기

레거시 렌더링에 적용하기 위해선 다음과 같은 조건으로 적용해야합니다.

1. 2D 텍스쳐를 큐브맵 텍스쳐처럼 렌더링 해야함
2. 프로브 위치값에 맞게 큐브맵 텍스쳐를 선택해야함
3. 각 프로브마다 존재하는 큐브맵 텍스쳐를 자연스럽게 처리해야함

먼저 2D 텍스쳐를 큐브맵 텍스쳐로 적용할 수 있는 방법에 대해 정말 막막하고 시간도 부족했지만, 다행스럽게도 [스택 오버플로우에서 설명과 함께 코드가 있었습니다.](https://stackoverflow.com/questions/53115467/how-to-implement-texturecube-using-6-sampler2d)

설명 되어있는 수식은 아래와 같습니다.

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%2012.png)

여기서 $s_c, t_c,m_a$는 아래의 표에 해당하는 값으로 치환이 가능합니다.

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%2013.png)

이 내용을 토대로 코드를 작성하면 다음과 같은 함수를 작성할 수 있습니다.

```glsl
void cubemap(vec3 r, out float texId, out vec2 st) {
	vec3 uvw;
	vec3 absr = abs(r);
	if (absr.x > absr.y && absr.x > absr.z) {
		// x major
		float negx = step(r.x, 0.0);
		uvw = vec3(r.zy, absr.x) * vec3(mix(-1.0, 1.0, negx), -1, 1);
		texId = negx;
	} else if (absr.y > absr.z) {
		// y major
		float negy = step(r.y, 0.0);
		uvw = vec3(r.xz, absr.y) * vec3(1.0, mix(1.0, -1.0, negy), 1.0);
		texId = 2.0 + negy;
	} else {
		// z major
		float negz = step(r.z, 0.0);
		uvw = vec3(r.xy, absr.z) * vec3(mix(1.0, -1.0, negz), -1, 1);
		texId = 4.0 + negz;
	}
	st = vec2(uvw.xy / uvw.z + 1.) * .5;
}

...

cubemap(direction, texId, st);
vec4 color = vec4(0);
for (int i = 0; i < 6; ++i) {
	vec4 side = texture(u_sampler_sdf[i], st);
	float select = step(float(i) - 0.5, texId) * step(texId, float(i) + .5);
	color = mix(color, side, select);
}
return color;
```

이제 이 코드에서 6개의 텍스쳐를 가로로 받아와 적용하고, 프로브 위치에 따른 인덱스값도 추가를 한다면 아래와 같이 코드를 작성할 수 있습니다.

```glsl
vec4 texCubemap(vec3 uvw, vec3 pos) {
	float texId;
	vec2 st;
	vec3 ipos = floor(pos);
	float index_pos_y = u_node_size.x * u_node_size.y * u_node_size.z;
	float index = (u_node_size.z - ipos.z) * u_node_size.y * u_node_size.x
							+ (u_node_size.y - ipos.y) * u_node_size.x
							+ (u_node_size.x - ipos.x);
	index = max(min(index, index_pos_y), 0.);

	cubemap(uvw, texId, st);
	st = vec2(st.x / 6., st.y / index_pos_y);
	vec4 color = vec4(0);
	for (int i = 0; i < 6; ++i) {
		vec4 side = texture(u_sampler_sdf, st + vec2(float(i) * (1./6.), 
												index * (1./index_pos_y)));
		float select = step(float(i) - 0.5, texId) *
		step(texId, float(i) + .5);
		color = mix(color, side, select);
	}
	return color;
}
```

이렇게 2D 텍스쳐를 큐브맵 텍스쳐로 사용하는 코드를 완성했습니다!<br/>
한번 돌려볼까요?

{% include embed/video.html src="/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/C_2023-06-07_00-39-52.mp4" %}

하지만 위 영상처럼 위치값에 따라 온전한 프로브 간의 전환이 이루어지지 않고, 매우 부자연스러운 전환이 이루어집니다.

이를 해결하기 위해 아래의 코드처럼 삼중 선형 보간을 적용합니다.

```glsl
vec4 texCubemapSmooth(vec3 uvw, vec3 pos) {
	vec3 spos = fract(pos) * 0.5;
	vec3 spos_abs = abs(spos);
	vec3 direction = normalize(uvw);

	vec3 offset_x = vec3((spos.x < 0 ? -1. : 1.), 0., 0.);
	vec3 offset_y = vec3(0., (spos.y < 0 ? -1. : 1.), 0.);
	vec3 offset_z = vec3(0., 0., (spos.z < 0 ? -1. : 1.));

	vec4 c000 = texCubemap(direction, pos);
	vec4 c100 = texCubemap(direction, pos + offset_x);
	vec4 c010 = texCubemap(direction, pos + offset_y);
	vec4 c110 = texCubemap(direction, pos + offset_x + offset_y);
	vec4 c001 = texCubemap(direction, pos + offset_z);
	vec4 c101 = texCubemap(direction, pos + offset_x + offset_z);
	vec4 c011 = texCubemap(direction, pos + offset_y + offset_z);
	vec4 c111 = texCubemap(direction, pos + offset_x + offset_y + offset_z);

	vec4 color = mix(
		mix(
		mix(c000, c100, spos_abs.x * 2.),
		mix(c010, c110, spos_abs.x * 2.),
		spos_abs.y * 2.
		),
		mix(
		mix(c001, c101, spos_abs.x * 2.),
		mix(c011, c111, spos_abs.x * 2.),
		spos_abs.y * 2.
		),
		spos_abs.z * 2.
	);
	return color;
}
```

어쩔 수 없이 3차원 공간에 프로브를 놓다보니 삼중 선형 보간을 사용하게 되었습니다.<br/>
하지만 부드러워졌죠?

{% include embed/video.html src="/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/C_2023-06-07_23-14-21.mp4" %}

그리고 위 영상과 같이 프로브가 위치한 구간은 모두 실시간으로 리플렉션이 되는 모습을 확인하실 수 있습니다!

## 마무리

![Untitled](/media/자체-엔진에-global-illumination을-적용하기-위한-삽질기-2/Untitled%2014.png)

사실 이게 끝이 아닙니다! 아직 남은 일들이 정말 많습니다.

전체적인 최적화라던가… 다른 오브젝트 볼륨 텍스쳐 합치는 작업이라던가… 맵을 밉맵으로 적용하기 위한 작업이라던가… 근경 및 원경 처리에 대한 최적화라던가… 등등….

그래도 다이나믹 글로벌 일루미네이션을 구현하는건 제 꿈의 목표 중 하나기도 했고<br/>
전혀 하지도 못하고 생각으로만 설계하고 그러하던 것이 이렇게 구현되니 기분이 매우매우 좋습니다ㅎㅎㅎ

뭐, 아무튼! 앞으로 지금까지 구현한 내용을 가지고 더 기능을 추가하거나 최적화를 한다면 이 시리즈도 진행할 수 있을 것 같습니다!

뭔가 조금이라도 성장한 느낌이 드니 뿌듯하기도 하네요!

정말 긴 글 읽어주셔서 감사합니다! 다음에 다시 뵙도록 할게요!
