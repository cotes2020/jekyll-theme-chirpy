---
title: OpenGL Framebuffer to Texture 사용시 팁
description: 프레임버퍼와 텍스쳐의 관계 관련하여 오늘 얻은 지식을 작성하고자 합니다.
author: ounols
date: '2021-06-25 00:00:00'
categories:
- Dev
- 자체 게임 엔진 프로젝트
tags:
- Coding
- dev
- OpenGL
- cpp
- graphics
pin: false
math: false
mermaid: false
image:
  path: "/media/2021-06-25-OpenGL-Framebuffer-to-Texture-사용시-팁/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-06-25%20032437.png"
---

프레임버퍼와 텍스쳐는 땔래야 땔 수 없는 관계죠!
그래서 프레임버퍼와 텍스쳐의 관계 관련하여 오늘 얻은 지식을 작성하고자 합니다.

## 1. 렌더링 되지 않는 문제 발생! ##

|                                                                                                                                                                          |                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![](/media/2021-06-25-OpenGL-Framebuffer-to-Texture-사용시-팁/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-06-25%20032121.png)_프레임버퍼가 정상적으로 렌더링 된 모습_ | ![](/media/2021-06-25-OpenGL-Framebuffer-to-Texture-사용시-팁/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-06-25%20032437.png)_OpenGL ES 3.0에서 정상적으로 나타나지 않은 모습_ |



둘의 차이가 보이시나요?<br>
정상적인 모습은
오른쪽 원반형태의 **Render To Texture**와 잘 보이지 않지만 존재하는 **그림자 깊이 버퍼**가 렌더링되어 확인이 가능한 상태입니다.

그러나 오른쪽의 ES환경에선 아예 프레임버퍼가 모두 렌더링되지 않았습니다.<br>
같은 코드에 이런 문제가 생겼습니다ㅠㅜ
<br>
## 2. 문제 분석 ##

디버그를 해서 프레임버퍼가 문제가 있는지<br>
`glReadPixels`함수를 사용하여 프레임버퍼를 확인했는데 멀쩡하게 나왔습니다.

그렇다면 텍스쳐에 문제가 있다는 것이 분명하겠군요!<br>
그렇게 하나씩 삽질하면 알아낸 원인은 다름아닌 생성 단계부터였습니다!

아래의 `glTexImage2D`함수를 사용하는 코드를 봐주세요

## 3. 텍스쳐 코드 수정 ##

```cpp
// 수정 전
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, data); 
```
흠... 이렇게만 보면 뭐가 문제인지 알 수가 없습니다.🤔<br>
사실 OpenGL Core단에서는 무리없이 잘 돌아가거든요!<br>
게다가 LearnOpenGL에서도 이렇게 작성되어있기 때문에 의심없이 작성했었습니다.

그러나 ES의 까다로운 환경에선 다음과 같이 코드를 수정해야합니다.
```cpp
// 수정 후
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, data);
```
인터널 포맷 파라미터가 수정되었습니다...!<br>
이렇게 수정하면 마침내 ES 환경에서도 잘 돌아갑니다!

음... 그럼 깊이버퍼는 어떻게 되어야할까요?
```cpp
// 수정 전 깊이버퍼
glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, data);
// 수정 후 깊이버퍼
glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, data);
```
깊이버퍼 역시 이렇게 세부적인 포맷 설정이 이루어지면 ES 환경에서 잘 돌아갑니다.


## 4. 음, 포맷이 좀 복잡한데요.. ##

네 맞습니다ㅠㅜ ES환경으로 인해 세부적인 포맷형식을 지정해야 멀쩡하게 돌아가다니...<br>
뭐 그게 ES특이니 어쩔 수 없네요ㅠㅜ

그래도 상황에 따른 포맷 설정은 아래의 링크에 있답니다!
> [glTexImage2D - OpenGL ES 3 Reference Pages](https://www.khronos.org/registry/OpenGL-Refpages/es3.0/html/glTexImage2D.xhtml)

![](/media/2021-06-25-OpenGL-Framebuffer-to-Texture-사용시-팁/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-06-25%20035716.png)
위 그림과 같이 잘 적혀있으니 원하는 것에 맞게 쓰시면 될 것 같습니다.


## 5. 요약 ##
프레임버퍼와 텍스쳐를 병행하며 사용할 때 ES환경에서 문제가 생긴다면<br>
프레임버퍼나 텍스쳐를 생성할 때 포맷을 잘 확인해서 적용해봅시다!
</br>
혹시 관련한 소스코드 및 프레임버퍼 객체화 관련 게시글이 궁금하다면<br>
아래의 링크를 참고해주세요!

> 📑 프레임버퍼 객체화 관련 글 : [https://velog.io/@ounols/Framebuffer의-객체화](https://velog.io/@ounols/Framebuffer%EC%9D%98-%EA%B0%9D%EC%B2%B4%ED%99%94) 
📣 관련 프로젝트 Git 주소 : https://github.com/ounols/CSEngine
