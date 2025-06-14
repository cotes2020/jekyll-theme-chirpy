---
title: OpenGL Framebuffer의 객체화
description: 오픈지엘에 있는 프레임버퍼라는 요소를 어떻게 객체화를 진행할지 고민이 많았었습니다.
author: ounols
date: '2021-06-22 20:00:00 +0800'
categories:
- Dev
tags:
- Coding
- dev
pin: false
math: true
mermaid: true
image:
  path: "/media/2021-06-22-OpenGL-Framebuffer의-객체화/maxresdefault.jpg"
---

오픈지엘에 있는 프레임버퍼라는 요소를 어떻게 객체화를 진행할지 고민이 많았었습니다.

사실 렌더링 프로그래밍하면서 프레임버퍼는<br>
튜토리얼에 가끔 보이는 프레임 버퍼 생성 부분이 전부라<br>
그동안 객체화를 하지 않았던 것이 아닌가 생각이 드네요ㅎㅎ;;

하지만 게임제작을 해보면 프레임 버퍼의 개념이 정말 많이 쓰이기 때문에
객체화는 해야합니다!

<br/>

그렇게 객체화를 진행하게 되었습니다.

일단 프레임버퍼의 기본개념은 아래의 링크로 대체하도록 할게요
* [LearnOpenGL Framebuffer 튜토리얼 번역본](https://heinleinsgame.tistory.com/28)
* [LearnOpenGL Render to texture 튜토리얼 원본](http://www.opengl-tutorial.org/kr/intermediate-tutorials/tutorial-14-render-to-texture/)

-------------
## 1. 어떻게 객체화를 진행할 것인가? ##

![](/media/2021-06-22-OpenGL-Framebuffer의-객체화/maxresdefault.jpg) 저는 Unity의 렌더 텍스쳐의 개념을 참고하여 제작하기로 했습니다.<br>
렌더 텍스쳐도 프레임 버퍼 그 자체이기 때문이죠!

그리고 다음과 같은 규칙을 정하고 코딩하기로 했습니다.
* 프레임 버퍼(렌더 텍스쳐)는 **리소스의 개념**이다.
* **카메라**가 프레임 버퍼를 가지고 있다.
* 렌더링 순서는 `광원 그림자 깊이버퍼 → 서브 프레임버퍼 → 메인 프레임버퍼`다.

</br>

이와 같이 규칙을 정한 이유는 단순합니다.<br>
유니티를 참고하면서 게임 엔진을 제작하다보면 유니티로 객체화 개념을 잡기가 정말 편합니다.

'으악 난 유니티가 싫어!'하면서 유니티랑 다르게 뭔가 만들려고 해도
결국 유니티처럼 설계하는게 가장 편하더군요ㅋㅋㅋㅋ

</br>

그렇게 첫번째 두번째 규칙을 정하게 되었고
마지막 렌더링 순서는 정말 고민이 많았습니다.

안그래도 렌더큐랑 쉐이더에 따라 렌더링 순서를 정했기 때문에
여기서 또 큰 의미로 순서를 정해야 하나 싶었기 때문이죠

그러다 기껏 생각한게 위와 같은 방식입니다.<br>
관련해서 피드백을 받을 곳이 딱히 없어서 급하게 정해본 순서라 안타깝긴 합니다..ㅠㅜ
횩시 이 게시글에 피드백 해주신다면 정말 감사하겠습니다ㅎㅎ

어쨌든 이렇게 진행하였습니다ㅎㅎ

## 2. 설계에 맞게 코드 작성 ##

사실 코드 작성은 튜토리얼에 있는 코드를 클래스에 맞게 작성한게 전부라
전체적인 코드를 원하신다면 제 [Git](https://github.com/ounols/CSEngine/commit/7b76fcce56adbb9db56d57e31269d4248ae687f6)에서 확인해보시면 될 것 같습니다ㅎㅎ

### framebuffer 클래스 ###

```cpp
class SFrameBuffer : public STexture {
    public:
        enum BufferType {
            RENDER = 0, DEPTH = 1, STENCIL = 2,
        };
    public:
        SFrameBuffer();
        ~SFrameBuffer();

        void InitFrameBuffer(BufferType type, int width, int height);
        void AttachFrameBuffer(int index = 0, int level = 0) const;
        void DetachFrameBuffer() const;
};
```
일단 프레임버퍼는 텍스쳐에 상속되어 작동하도록 설계하였습니다.

프레임버퍼와 텍스쳐는 각자 다른 개념이긴 하지만...<br>
메인 프레임버퍼가 아니면 텍스쳐에 그리는게 전부라 그냥 텍스쳐에 상속시켜
특수한 텍스쳐 클래스가 되었습니다!

어쩌다보니 유니티와 비슷하네요! 세상에!

### CameraComponent 클래스 ###

```cpp
class SFrameBuffer;

class CameraComponent : public SComponent, public CameraBase {
public:
    
    ...
    
    SFrameBuffer* GetFrameBuffer() const override;
    void SetFrameBuffer(SFrameBuffer* frameBuffer);
    
private:
    SFrameBuffer* m_frameBuffer = nullptr;
    ...
    
};
```
음.. 카메라 컴포넌트는 기존 컴포넌트에서 방금 만들었던 프레임버퍼 클래스를 넣은게 끝입니다.<br>
그래도 나름 중요한 포인트라서 짧지만 넣었습니다 헤헷

### RenderMgr::Render() 함수 ###
```cpp

void RenderMgr::Render() const {
    // Render Order : Depth Buffers -> Render Buffers -> Main Render Buffer

    // 1. Render depth buffer for shadows.
    const auto& lightObjects = lightMgr->GetAll();
    const auto& shadowObjects = lightMgr->GetShadowObject();
    const auto& shadowEnvironment = m_environmentMgr->GetShadowEnvironment();
    for (const auto& light : lightObjects) {
        if(light->m_disableShadow) continue;
        RenderShadowInstance(*light, *shadowEnvironment, shadowObjects);
    }
    lightMgr->RefreshShadowCount();

    const auto& cameraObjects = cameraMgr->GetAll();
    const auto& mainCamera = cameraMgr->GetCurrentCamera();

    // 2. Render active sub cameras.
    for (const auto& camera : cameraObjects) {
        if(!camera->GetIsEnable() || camera == mainCamera || camera->GetFrameBuffer() == nullptr) continue;
        RenderInstance(*camera);
    }

    if(mainCamera == nullptr) return;
    // 3. Main Render Buffer
    RenderInstance(*mainCamera);

}
```

여기는 제가 말했던 렌더링 순서에 맞게 일단 깡코드로 때려박았습니다.<br>
아직까진 이게 최선인거 같네요..ㅎㅎ;;

물론 디퍼드 렌더링을 구현할 때 되면 갈아엎을 확률 100% 겠지만<br>
일단 지금은 이렇게 짜놓고 나중에 제가 알아서 수정할 것 같습니다!<br>
화이팅 미래의 나!

다시 본론으로 돌아와서
저 함수에서 작성하기 가장 까다로웠던 얘가 바로 광원 그림자 버퍼였습니다.<br>
까다로웠던 이유는 아래와 같았슴니다.
* 그림자 렌더링 여부를 RenderComponent에서 정하기 때문에
기존 렌더큐랑 다른 방식으로 코드를 작성해야 그나마 효율이 생깁니다.
* 그림자는 카메라가 아닌 **광원을 기준으로** 합니다.
* 그림자의 인식 범위가 따로 존재합니다.
* 그냥 따로 렌더링되는게 마음에 들지 않아여..흑흑

그래서 결국...
`CameraBase`라는 인터페이스를 만들어 카메라, 광원에 모두 상속시켰고,<br>
렌더링은 기본 렌더링과 그림자 렌더링으로 나누게 되었습니다.

## 3. 구현 끝! ##

영 찝찝하지만 나름 기본적인 구현은 끝났습니다.<br>
프레임 버퍼를 객체화 함으로써 렌더링 속 렌더링이 가능해졌고
리플렉션이나 그림자, 깊이 관련 렌더링도 더욱 편리해졌습니다!

이제 OpenVR을 통해 vr렌더링도 가능해지겠군요!

문제가 생긴다면.. 미래의 제가 여기에 새로운 링크를 통해 해결책을 제시해 줄 것입니다!<br>
고마워요 미래의 나!


> 📣 관련 프로젝트 Git 주소 : https://github.com/ounols/CSEngine
