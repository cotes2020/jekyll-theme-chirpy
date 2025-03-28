---
title: OpenGL을 iOS에서 무작정 돌려보기
description: 드디어 제 오랜 염원이였던 ios에서 자체엔진 돌리기에 성공을 했습니다!
author: ounols
date: 2023-02-12 23:12:00 +0800
categories: [Dev, 자체 게임 엔진 프로젝트]
tags: [Coding, OpenGL, cpp, dev]
pin: false
math: true
mermaid: true
redirect_from:
  - opengl을-ios에서-무작정-돌려보기
image:
  path: /media/opengl을-ios에서-무작정-돌려보기/IMG_9B7D936C02CE-1.jpeg
---

![Untitled](/media/opengl을-ios에서-무작정-돌려보기/Untitled.png)

안녕하세요! 드디어 제 오랜 염원이였던 ios에서 자체엔진 돌리기에 성공을 했습니다!<br/>
와!!! 👏👏👏👏👏👏👏👏

그래서 이번 포스팅은 제가 개발하면서 했던 삽질과 일대기를 간략하게 작성해볼 예정입니다!

## 1. 무엇이 두려운가?

사실 iOS를 마냥 포팅해보자! 하는 마인드는 별로 없었습니다.<br/>
왜냐하면 제가 애플 생태계를 진짜 아는게 거의 없기도 하고 개발환경도 잘 모르기 때문이였습니다.

그러다가 최근에 아이폰을 갈아타면서 맥북과 아이폰을 가진 오우너가 되었습니다.<br/>
그리고 문득 두려운 무언가가 다가오고 있었습니다… 바로 iOS에다가 자체엔진 포팅하기…

이렇게 저는 대부분 환경이나 지식이 전무해서 iOS로의 포팅이 두려울 수 밖에 없었습니다.

## 2. 풀어야 할 문제들

이런 두려움을 극복할 수 있는 문제들은 다음과 같았습니다.

1. **iOS의 OpenGL 지원 문제**
2. **기존 코드의 적합성 문제**
3. **Xcode에 대해 아무것도 모르는 문제**
4. **Objective-C 언어를 모르는 문제**


하나씩 플어가보도록 하겠습니다!

## 3. iOS의 OpenGL 지원 문제

![Untitled](/media/opengl을-ios에서-무작정-돌려보기/Untitled%201.png)

iOS는 12버전 이후로 OpenGL의 지원을 중단하였습니다. 그렇게 그래픽스 라이브러리는 Metal로 갈아타는 시점에서 OpenGL로 구현된 제 자체엔진이 잘 돌아갈지 걱정이 좀 있었습니다.

하지만 지원이 중단되었다는게 어느정도 버전에서는 지원되었기 때문에 돌아가기는 한다는 뜻이기도 하고, 저는 OpenGL ES 3.0 이상이기만 하면 됩니다.

다행스럽게도 OpenGL ES 3.0은 지원하는 터라 문제없이 돌아갈 것이라는 이론은 세워졌습니다!

## 4. 기존 코드의 적합성 문제

![Untitled](/media/opengl을-ios에서-무작정-돌려보기/Untitled%202.png)

지금까지 새로운 플랫폼으로 갈 때 마다 항상 기존 코드에서 특정 부분이 지원하지 않는 형태가 존재했었습니다.
이번엔 Xcode라는 처음 접해보는 IDE이기 때문에 관련 문제에 대해서 고민을 했었고, 실제로 위와 같은 에러가 나타났습니다.

![Untitled](/media/opengl을-ios에서-무작정-돌려보기/Untitled%203.png)

다행스럽게도 기존 cpp컴파일러 기본값이 c++14 이상으로 지원하지 않는 문제였기 때문에 <br/>
다음과 같이 설정화면에서 직접 설정해주니 큰 에러없이 잘 컴파일되었습니다!

## 5. Xcode와 Objective-C를 잘 모르는 문제

![Untitled](/media/opengl을-ios에서-무작정-돌려보기/Untitled%204.png)

Xcode와 Objective-C는 둘 다 유명하긴 하지만… 써볼 일이 없었습니다..ㅋㅋ<br/>
그래서인지 이 두가지를 다루려고 할 때가 가장 무서웠습니다.

그래도 구글링을 통해 내가 원하는건 다 얻을 수 있는 방대한 정보들 덕분에 생각보다 쉽게 진행할 수 있었던 것 같습니다. 그래도 비주얼 스튜디오처럼 Xcode는 얘만의 불친절한 무언가가 있더군요..ㅋㅋㅋㅋ

일단 제가 느꼈던 Xcode의 불편한 점은 다음과 같았습니다.

- 리소스 하나씩 설정에서 넣어줘야 함. 폴더 넣으면 인식 못함
- 소스코드 하나씩 설정에서 넣어줘야 함. 폴더 넣거나 다른 확장명을 넣으면 에러남
- 처음 맛보는 단축키들. 이클립스 처음 써보는 느낌

뭐 당연한 불편한 점들도 있겠지만 역시 cmake의 편안한 방식에서 벗어나지 못했나봅니다ㅋㅋㅋ
<br/><br/>

이제 오브젝티브c에 대해서도 이야기 해볼게요.<br/>
이번 언어는 따로 배우려고 하거나 딱히 검색을 하진 않았습니다. 다음과 같은 이유가 있었기 때문이죠!

- mm 확장명 설정 후, 오브젝티브C와 cpp의 문법을 혼용하여 사용해도 문제가 없었음
- chatGPT가 알아서 내가 원하는 언어로 포팅해주거나 여러 함수들을 알려줌

![Untitled](/media/opengl을-ios에서-무작정-돌려보기/Untitled%205.png)

네, 여기서 chatGPT가 가장 큰 역할을 해냈습니다ㅋㅋㅋ <br/>
제가 모르는 부분은 얘한테 다 물어봤는데 다 알아서 코드까지 다 짜고 설명도 해주더군요ㅋㅋㅋ

싸랑해요 오픈AI!

---

자, 이렇게 문제되는 점들은 다 해결하고 빌드까지 모두 성공했습니다!<br/>
그럼 이론 상 정상적으로 떠야합니다! 과연 어떻게 떴을까요?

## 6. 메인 프레임버퍼는 항상 0일까?

![Untitled](/media/opengl을-ios에서-무작정-돌려보기/Untitled%206.png)

바로 저렇게 떴습니다! 하얀색 아니면 검은색이였어요…ㅋㅋㅋ 역시 바로 될리가 없죠!<br/>
일단 콘솔 로그에 뜨는 내용들을 봤을 떄 렌더링 로직은 문제가 없는 것 같았습니다.

문제가 없었기에 어느 부분에서 안보이는지 삽질을 좀 많이 했습니다…<br/>
다른 플랫폼들에서 잘 돌아가는걸 얘 혼자 못돌아간다니… 좀 많이 슬프죠…ㅠ

그렇게 소스코드를 이리저리 보다가 문득 다음과 같은 코드를 보게 됩니다.

```objectivec
- (void)setupLayer
{
    _eaglLayer = (CAEAGLLayer*) self.layer;
    _eaglLayer.opaque = YES;
    _eaglLayer.drawableProperties = [NSDictionary dictionaryWithObjectsAndKeys:
                                    [NSNumber numberWithBool:NO], kEAGLDrawablePropertyRetainedBacking, kEAGLColorFormatRGBA8, kEAGLDrawablePropertyColorFormat, nil];
}
```

이 코드는 현재 종속된 UIView 객체에서 layer를 들고와 이 레이어에 렌더링 속성값을 부여합니다.<br/>
그런 뒤 이 레이어로 context를 생성한 후 프레임 버퍼를 만듭니다…

…뭔가 이상하지 않나요? 메인 프레임버퍼라면 그냥 그대로 아이디값 0으로 넣어서 그리면 될텐데<br/>
UIView 객체의 레이어를 통해 프레임버퍼를 생성한다니…

이렇게 되면 실질적인 메인 프레임버퍼는 0이 아닌 저기서 생성한 프레임버퍼가 될 것이다는 겁니다!<br/>
이 때 가장 먼저 생성하는 프레임버퍼는 저 레이어 객체에서 만들기 때문에 항상 아이디값이 1이였습니다.

따라서 저는 메인 프레임버퍼에 렌더링 하는 코드를 다음과 같은 코드로 수정했습니다.

```cpp
		GetMainBuffer()->AttachFrameBuffer(GL_READ_FRAMEBUFFER);
#ifdef IOS
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 1);
#else
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
#endif
    glBlitFramebuffer(0, 0, *m_width, *m_height, 0, 0, *m_width, *m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
```

위 코드는 엔진 상의 메인 버퍼를 실제 메인 버퍼에 그려주는 코드인데<br/>
iOS만 레이어 객체에 렌더링 하기 때문에 렌더링 객체에서 생성한 프레임버퍼 아이디인 1을 넣어주었습니다.

## 7. 드디어 렌더링 🎉

![아이폰14 프로에서 돌아가는 CSEngine의 Animation 씬](/media/opengl을-ios에서-무작정-돌려보기/IMG_9B7D936C02CE-1.jpeg)
_아이폰14 프로에서 돌아가는 CSEngine의 Animation 씬_

마참내! iOS 16버전의 아이폰에서도 제 자체엔진이 잘 돌아가게 되었습니다!<br/>
정말…힘든 여정이였지만 이렇게 잘 돌아가는 모습이 너무 행복하네요!

드디어 제가 목표로 잡던 모든 플랫폼에서 제 자체 게임엔진이 돌아가게 되었습니다….!<br/>
해당 게임엔진은 깃허브에 있으니 아래의 링크를 참고해주세요ㅎㅎ

긴 글 읽어주셔서 감사합니다!

> 자체 게임엔진 Github 주소 : [https://github.com/ounols/CSEngine](https://github.com/ounols/CSEngine)
{: .prompt-tip }
