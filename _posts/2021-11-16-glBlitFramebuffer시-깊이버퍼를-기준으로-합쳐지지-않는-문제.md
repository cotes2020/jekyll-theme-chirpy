---
title: glBlitFramebuffer시 깊이버퍼를 기준으로 합쳐지지 않는 문제
description: 
author: ounols
date: '2021-11-16 00:00:00'
categories: []
tags: []
pin: false
math: false
mermaid: false
image:
  path: /media/2021-11-16-glBlitFramebuffer시-깊이버퍼를-기준으로-합쳐지지-않는-문제/image.png
---

![](/media/2021-11-16-glBlitFramebuffer시-깊이버퍼를-기준으로-합쳐지지-않는-문제/image.png)

```cpp
    /** ======================
     *  3. Blit the depth buffer
     */
    gbuffer.AttachGeometryFrameBuffer(GL_READ_FRAMEBUFFER);
    if(frameBuffer == nullptr) {
        m_mainBuffer->AttachFrameBuffer(GL_DRAW_FRAMEBUFFER);
    }
    else {
        frameBuffer->AttachFrameBuffer(GL_DRAW_FRAMEBUFFER);
    }

    glBlitFramebuffer(0, 0, *m_width, *m_height, 0, 0, bufferWidth, bufferHeight, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
```
디퍼드 렌더링을 구현하면서 포워드 렌더링과 깊이버퍼를 기준으로 메인 프레임버퍼에 합치는 작업을 진행하였고, 윈도우 플랫폼에선 보시다시피 정상적으로 둘 다 정상적으로 렌더링 되는 모습을 보실 수 있습니다.

그런데 관련해서 문제는 리눅스에서 나타나고 말았습니다.

| ![](https://images.velog.io/images/ounols/post/d5b56a59-d016-45a0-87a6-7a58f19f6c92/image.png) | ![](/media/2021-11-16-glBlitFramebuffer시-깊이버퍼를-기준으로-합쳐지지-않는-문제/image.png) |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 리눅스에선 포워드가 렌더링 되지 않는 모습                                                      | 리눅스에서 `glDepthFunc(GL_ALWAYS)` 옵션을 킨 모습                                          |

리눅스에선 포워드가 렌더링 되지 않았습니다.
그런데 또 이상한 점은 `glDepthFunc(GL_ALWAYS)` 옵션을 활성화하면 깊이값과 상관없이 모두 렌더링을 한다는 의미가 되어 멀쩡하게 합쳐져 렌더링되는 모습을 볼 수 있습니다.

여기까지 보면 깊이버퍼가 서로 같은 포맷이 아닌게 원인인 느낌이 납니다.
하지만 아직 깊이버퍼만의 문제가 아닐 수 있어 다른 부분도 체크를 해봤습니다.

### 문제 1. `glBlitFramebuffer`함수를 잘못 사용한 게 아닌가?

처음엔 이쪽 문제가 유력하다고 생각했습니다.
왜냐하면 얼마 전까지만 해도 깊이값을 통한 프레임 버퍼 합치기 작업은 멀쩡하게 돌아갔기 때문입니다. 게다가 코드 리펙토링 하기 전엔 잘 작동했기 때문에 이 부분에 문제가 있다고 확신했습니다.

그런데 `glGetError`에도 아무런 로그가 남지 않았고, `glCheckFramebufferStatus(GL_FRAMEBUFFER)`를 통해 버퍼 생성 중에 문제가 있는지 알아봤지만 전혀 없었습니다. 코드를 아무리 다시 봐도 문제가 전혀 없었습니다.

게다가 아래의 코드가 잘 작동함을 확인하자 이 쪽 문제가 아님을 깨달았습니다.
```cpp
m_mainBuffer->AttachFrameBuffer(GL_READ_FRAMEBUFFER);
glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
glBlitFramebuffer(0, 0, *m_width, *m_height, 0, 0, bufferWidth, bufferHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);
```
위 코드는 엔진의 메인 프레임버퍼를 OpenGL 메인 프레임버퍼로 컬러값을 그대로 옮기는 작업입니다. 포스트 프로세싱(구현 예정)도 모두 포함된 최종적인 렌더링이기 때문에 깊이값과 상관없이 무작정 컬러 버퍼를 옮기는 작업을 합니다.

이 부분은 윈도우, 리눅스, 안드로이드 모두 잘 작동하였습니다. 따라서 해당 부분은 문제가 없는 것으로 볼 수 있었습니다.

### 문제2. 깊이버퍼 포맷이 맞지 않아서 생기는 문제인가?
그 다음으로 예상해본 문제는 깊이버퍼의 포맷이 맞지 않는 문제라고 생각했습니다.
사실 엔진 메인 프레임버퍼는 OpenGL에서의 메인 프레임버퍼를 그대로 사용하고 있었기 때문에 메인 프레임버퍼의 깊이값을 설정하기 애매하므로 이쪽에서 문제가 발생했다고 생각했습니다.

그래서 따로 `m_mainBuffer`라는 객체를 선언하고 여기서 모두 렌더링을 진행한 다음 최종적으로 OpenGL의 메인 프레임버퍼에 넣으려고 했습니다.

그러나 결과는 그대로였습니다. 분명 `m_mainBuffer`의 깊이버퍼와 디퍼드를 렌더링한 G-Buffer의 깊이버퍼와 같은 포맷임에도 불구하고 나오지 않게 되었습니다.

사실 이 부분은 어느정도 예상했습니다. 앞서 말했듯이 이전에 잘 작동했기 때문입니다.

### 문제3. OpenGL의 버그?
제가 주요 타겟 플랫폼으로 설정한 윈도우, 우분투, 안드로이드는 각자 다른 OpenGL의 프로파일을 담고 있습니다.
* Windows : OpenGL Core
* Ubuntu(Linux) : OpenGL Compatibility
* Android : OpenGL ES

이로 인해 각자 환경이 조금씩 달라 골때리는(?)일도 많지만 OpenGL의 모든 프로파일에서 정상적으로 돌아가는지 확인할 수 있어서 이대로 환경을 유지하고 있습니다.

근데 얼마전에 우분투에서 코드 그대로 진행하던게 작동 안하는 일이 발생하였습니다...
마치 Core에서 작동하던 코드들이 ES형식의 코드로 바꿔줘야 잘 돌아가는 것처럼 말이죠...

여기서 작동하지 않게 된 코드 중 하나가 바로 깊이버퍼를 기준으로 프레임버퍼를 합치는 작업입니다.
특히 이게 좀 악질적인 문제인게 ES환경에선 G버퍼가 너무 무거워 온전히 렌더링되지 않지만 깊이버퍼는 잘 작동하는 것을 확인할 수 있었습니다.

다시 말하면 Core, ES 환경에선 잘 돌아가는데 Compatibility 환경에선 안돌아가는 상황이 되었습니다...
</br>

사실 이정도 문제면 OpenGL 버그를 의심해볼만해서 관련해서 찾다가 아래의 글을 발견하게 됩니다.
[https://community.khronos.org/t/framebuffer-not-bliting-depth-buffer/66280](https://community.khronos.org/t/framebuffer-not-bliting-depth-buffer/66280){:target="_blank"}

이 글 역시 저와 같은 문제를 겪고 있지만 저와 다른 환경의 문제임을 알 수 있습니다.
저 글의 환경은 GPU가 AMD사의 라데온 그래픽 카드를 사용하고 있고, 저는 Nvidia사의 RTX 그래픽카드를 사용하고 있다는 차이점이 있습니다.

저 글은 AMD 그래픽카드의 버그라고 하며 깊이버퍼만 쓰지말고 깊이버퍼+스텐실버퍼를 사용하면 정상적으로 렌더링이 된다고 설명하고 있습니다.
그런데 저의 경우엔 AMD 그래픽카드가 아니라 좀 애매하네요... 가상 환경으로 우분투를 돌려도 같은 문제가 발생하는지도 잘 모르겠습니다ㅠㅜ
</br>
### ~~결론. 더 큰 문제가 있다...~~
~~문제를 해결하기 위해 몇가지 환경 세팅을 적용하고 있는데 여기서 이상한 환경의 설정이 보였습니다...~~

~~먼저 윈도우는 4.6.0 Core 였고, 리눅스는 3.3 Compatiblity였습니다...
게다가 리눅스에서 3.3 Core로 돌리면 아무것도 렌더링되지 않았습니다..
이는 아마 3.3 호환성 프로파일은 코어에서 삭제된 함수도 지원하는 형태이기 때문에 간당간당하게 렌더링이 되었던걸로 보입니다.~~

~~ES 3.0도 4.3 버전을 기반으로 만들었으니 이제서야 왜 이런지 알 것 같습니다...~~
~~어쨌든 당장은 오픈지엘의 3.3 이상으로 올라가는 기술이 딱히 없기 때문에 관련해서 렌더링 문제점을 찾는게 가장 급선무인 것 같습니다...ㅠ~~

~~미래의 나 화이팅...~~

---------------

=====2022.01.19 수정사항=====

### 킹능성1. 깊이 버퍼 형식이 진짜 서로 다르다!

[https://stackoverflow.com/questions/9914046/opengl-how-to-use-depthbuffer-from-framebuffer-as-usual-depth-buffer](https://stackoverflow.com/questions/9914046/opengl-how-to-use-depthbuffer-from-framebuffer-as-usual-depth-buffer){:target="_blank"}
2022년이 된 지금까지도 해결을 못하고 있었는데 우연하게 구글링 해보다가 위 글이 나왔습니다.

요약해보자면 기본적으로 제공되는 깊이버퍼를 가지고 `glBlitFramebuffer`를 진행하면 엔비디가 윈도우에선 정상적으로 작동하지만 다른 그래픽카드나 os 환경에선 다른 형식의 깊이값으로 인해 **합치려고 하는 프레임버퍼들의 깊이버퍼 형식이 다를 수 있다**는 내용입니다.

위의 문제3과 비슷한 원인입니다.
근데 이번엔 좀 더 확실하게 안전한 방법으로 깊이를 적용할 수 있는 방안이 나와있습니다!

바로 직접 깊이 텍스쳐를 가져와서 `gl_FragDepth`로 뿌려주는 방식입니다.

이게 작동할 가능성이 높은 가장 큰 이유가 **그림자는 모든 플랫폼에서 구현되었기 때문**입니다!
그림자가 깊이버퍼를 텍스쳐로 받아와서 직접 쉐이더에서 계산하는 방식인데 여기서 깊이 텍스쳐는 정상적으로 표현이 되었습니다.

오오....드디어 이 문제를 해결할 수 있게 되는 것인가.....흑흑....제발 해결됐으면 좋겠습니다...흑ㅎ그....

=====2022.02.02 수정사항=====

## 진짜 최종 결론!

음... 일단 깊이 버퍼 형식이 서로 다른게 아니였습니다... 프레임버퍼의 정보를 확인할 수 있는 정보 다 써봤는데 서로 같은 형식이였기 때문에 킹능성1의 문제와는 조금 먼 거 같네요..


사실 왜 이런지 아직도 모르겠는데 일단 `glBlitFramebuffer`를 안쓰는 방향으로 진행했습니다.
그래서 직접 합치는 작업 자체를 쉐이더를 통해 또 다른 프레임버퍼에서 이루어집니다..

```cpp
void SFrameBuffer::BlitFrameBuffer(const SFrameBuffer& dst, BlitType type) {
    if (m_mainColorBuffer == nullptr || m_depthBuffer == nullptr) {
        Exterminate();
        GenerateFramebuffer(PLANE, m_width, m_height);
        GenerateTexturebuffer(RENDER, GL_RGB);
        GenerateTexturebuffer(DEPTH, GL_DEPTH_COMPONENT);
        RasterizeFramebuffer();
    }

    const SFrameBuffer* a;
    const SFrameBuffer* b;
    if (type == REVERSE) {
        a = this;
        b = &dst;
    }
    else {
        a = &dst;
        b = this;
    }
    const auto& aColorTexture = a->m_mainColorBuffer->texture;
    const auto& bColorTexture = b->m_mainColorBuffer->texture;
    const auto& aDepthTexture = a->m_depthBuffer->texture;
    const auto& bDepthTexture = b->m_depthBuffer->texture;

    [...]

    AttachFrameBuffer();
    glViewport(0, 0, m_width, m_height);
    glUseProgram(m_blitObject.handle->Program);
    aColorTexture->Bind(m_blitObject.aColor, 0);
    bColorTexture->Bind(m_blitObject.bColor, 1);
    aDepthTexture->Bind(m_blitObject.aDepth, 2);
    bDepthTexture->Bind(m_blitObject.bDepth, 3);

    ShaderUtil::BindAttributeToPlane();
}
```

```c
/// Blit 쉐이더 코드
precision highp float;

//Uniforms
//[a.color]//
uniform sampler2D u_a_color;
//[b.color]//
uniform sampler2D u_b_color;
//[a.depth]//
uniform sampler2D u_a_depth;
//[b.depth]//
uniform sampler2D u_b_depth;
//Varying
in mediump vec2 v_textureCoordOut;
out vec4 FragColor;

void main(void) {

	float a_depth = texture(u_a_depth, v_textureCoordOut).r;
	float b_depth = texture(u_b_depth, v_textureCoordOut).r;

	vec4 color = vec4(1.0);
	float depth = 0.f;

	if(a_depth < b_depth) {
		color = vec4(texture(u_a_color, v_textureCoordOut).rgb, 1.0);
		depth = a_depth;
	}
	else {
		color = vec4(texture(u_b_color, v_textureCoordOut).rgb, 1.0);
		depth = b_depth;
	}

	FragColor = color;
	gl_FragDepth = depth;
}
```

제가 짠 blit 코드를 보면 진짜 너무 리소스를 FLEX 해버리기도 하고 코드도 너무 더럽습니다.... 
근데 이거말고 더 빠르게 합칠 수 있는 방안을 더 이상 생각하지 못하겠더군요...ㅠㅜㅜㅜㅜ

그래도 윈도우랑 리눅스 그리고 4.3 core와 3.3 Compatiblity에서 돌아가는 방법이다 보니...
혹시 이 방법 말고 '어? 이렇게 하는게 더 나은데?' 하는 부분이 있으면 제발 알려주시면 감사하겠습니다...ㅠㅜ



> 📣 프로젝트 Git 주소 : [https://github.com/ounols/CSEngine](https://github.com/ounols/CSEngine){:target="_blank"}
