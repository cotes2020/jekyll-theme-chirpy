---
title: 'C++ 코드를 웹으로 포팅하는 Emscripten 알아보기 (feat. WebGL)'
description: 내가 작성한 C++ 코드를 손쉽게 웹으로 포팅할 수 있다...?
author: ounols
date: 2023-01-17 20:49:00 +0800
categories: [Dev, 자체 게임 엔진 프로젝트]
tags: [Coding, Emscripten, OpenGL, cpp, dev]
pin: false
math: true
mermaid: true
redirect_from:
  - c-코드를-웹으로-포팅하는-emscripten-알아보기-feat-webgl
image:
  path: /media/Cpp-코드를-웹으로-포팅하는-Emscripten-알아보기/Untitled 1.png
---

![Untitled](/media/Cpp-코드를-웹으로-포팅하는-Emscripten-알아보기/Untitled.png)

내가 작성한 C++ 코드를 손쉽게 웹으로 포팅할 수 있다...?<br/>
라는 말은 거짓말 같아 보이는데 여기 진짜 쉽게 할 수 있는 엄청난 프로젝트가 있습니다!<br/>
바로 Emscripten이라고 하는 프로젝트입니다!

emscripten은 LLVM 기반으로 돌아갑니다.<br/>
여기서 LLVM은 이미 안드로이드 ndk를 다루신 분들이라면 익숙하실텐데요.<br/>
더욱 다양한 아키텍쳐 환경 속에서 쉽게 포팅 가능하게 해주는 툴체인 정도로 알아두시면 좋을 것 같습니다!

어쨌든 여기서는 OpenGL로 돌아가는 C++기반 게임엔진 프로젝트를 웹으로 포팅해보는 과정을 최대한 담아보고자 합니다.<br/>
(사실 한번 포팅 성공하고 나중에 다시 만지려니 과정을 까먹어버려서 따로 작성하려고 합니다ㅋㅋ)

### 프로젝트 환경

- 개발 환경 운영체제 : Ubuntu 20.04
- 개발 IDE : CLion
- 빌드 가능한 컴파일러 : gcc, **clang**, msvc
- 지원 가능한 OpenGL 버전 : **OpenGL ES 3.0 이상** 또는 OpenGL 4.3 이상

프로젝트 환경은 위와 같습니다. 아무래도 제가 개발 중인 자체 게임엔진은 크로스 플랫폼이 가능하게 설계를 하였기 때문에 바로 웹으로 포팅 가능하겠다고 판단하고 진행하였습니다.

만약 윈도우 msvc에서만 빌드를 해보셨으면 **최소한 clang 컴파일러로 성공적으로 컴파일이 가능한지 알아보시고 진행하시는걸 추천드립니다!**<br/>
왜냐하면 clang이 llvm 환경에서 기본적으로 깔고 가는 컴파일러이기 때문에 emscripten에서도 clang을 기본 컴파일러로 채택하였습니다!


> OpenGL을 WebGL로 포팅하기 위해선 최소한 OpenGL ES 2.0 환경에서 돌아가는지 확인해주셔야 합니다.
> 이름은 WebGL이지만 사실 상 오픈지엘의 ES 환경으로 돌아가기 때문이죠!
{: .prompt-tip }



> **해당 글은 리눅스 환경으로 진행합니다!**
> 
> 윈도우로도 충분히 진행할 수 있지만 리눅스가 윈도우보다 개발하기 더 수월한 환경이라 추가적으로 해야하는 작업이 상대적으로 적습니다.
> 게다가 CI/CD 작업도 도커를 돌리는 경우가 많은데 이 역시 리눅스 기반으로 짜여져있다보니 이런 부분에서는 큰 메리트가 있다고 생각을 합니다!
{: .prompt-info }

그럼 이제 본격적으로 포팅해보도록 하겠습니다!

## 1. emsdk 설치

설치 방법은 아래의 링크에 친절하게 설명이 적혀있습니다.<br/>
비록 영어이지만 영알못도 충분히 따라 진행할 수 있기 때문에 그대로 진행해주시면 됩니다.

[https://emscripten.org/docs/getting_started/downloads.html](https://emscripten.org/docs/getting_started/downloads.html)

간단하게 설명하자면 깃허브에 있는 emsdk 레포지토리를 그대로 내려받아 설치를 하는 형태입니다.

```bash
# Fetch the latest version of the emsdk (not needed the first time you clone)
git pull

# Download and install the latest SDK tools.
./emsdk install latest

# Make the "latest" SDK "active" for the current user. (writes .emscripten file)
./emsdk activate latest

# Activate PATH and other environment variables in the current terminal
source ./emsdk_env.sh
```

## 2. 전용 CMakeLists.txt 생성

일반적인 cpp 프로젝트라면 기존 CMakeLists.txt를 만들어서 빌드를 하셨을겁니다.<br/>
이번 emscripten를 위한 CMakeLists.txt는 조금 다른 점이 있더군요!

아래에서 간단하게 알아보도록 하겠습니다.

```cmake
cmake_minimum_required(VERSION 3.4.1)

PROJECT(CSEngineWeb)

# 만약 Shared 라이브러리가 존재하면 해당 프로퍼티 설정을 넣어주셔야 합니다!
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

SET(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_BUILD_TYPE Debug)

# 기존 /usr 내에 있는 라이브러리는 링크하지 않습니다!
# 이미 emsdk에 내장되어있기 때문에 링크가 되어있다면 해제해줍니다.
#link_directories(/usr/lib)
#link_directories(/usr/lib32)
#link_directories(/usr/local/lib)

# 만약 WebGL을 사용한다면 아래처럼 설정해주시면 됩니다.
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

...

# 제 프로젝트에 들어가는 Shared 라이브러리 입니다.
add_library( SquirrelLib
            SHARED
            ${SQUIRREL_SRC})

add_executable(CSEngineWeb CSEngine_Web_glfw.cpp ${CSENGINE_SRC})

# 기존 /usr 내에 있는 include 디렉토리도 선언하지 않습니다!
# 이미 emsdk에 내장되어있기 때문에 선언되어있다면 해제해줍니다.
#include_directories(/usr/include)
#include_directories(/usr/local/include)

include_directories(../../External/Squirrel/include)

target_link_libraries(CSEngineWeb glfw)
target_link_libraries(CSEngineWeb SquirrelLib)

# html 형식으로 빌드하기 위해선 아래와 같이 옵션을 추가로 설정해줘야 합니다.
# 해당 파트가 가장 중요하다고 볼 수 있겠습니다!
set_target_properties(CSEngineWeb
        PROPERTIES SUFFIX ".html"
        LINK_FLAGS " --bind -s USE_GLFW=3 -s ASSERTIONS=1 -s WASM=1 -O3 -std=c++14 -s ALLOW_MEMORY_GROWTH=1 -s EXCEPTION_CATCHING_ALLOWED=[..] -s USE_WEBGL2=1 -g4 --preload-file Assets.zip")
```

제가 실제로 사용하고 있는 내용이라 몇가지 쓸데없는 부분들도 존재합니다.<br/>
하지만 여기서 가장 중요한건 맨 아래에 있는 `set_target_properties` 부분이라고 볼 수 있겠습니다.

### LINK_FLAGS

여기서 사용하는 옵션에 대한 설명은 아래의 링크에 있습니다!
[https://emscripten.org/docs/tools_reference/emcc.html](https://emscripten.org/docs/tools_reference/emcc.html)

이번엔 여기서 중요하다고 생각되는 부분에 대해 제가 다시 설명을 작성해보고자 합니다.

- `USE_GLFW=3` : GLFW3을 사용한다고 설정합니다. 코드에서 GLFW를 사용하는지 확인하고 작성해주시면 됩니다.
- `ASSERTIONS=1` : 메모리 할당 오류에 대한 검사를 활성화 합니다. 기본적으론 1로 지정되어있으나 최적화 단계가 `O1`이상이 되면 해제되기 때문에 따로 넣어주었습니다.
- `O3` : 코드의 최적화 단계를 뜻합니다. 기본적으론 `O0`부터 `O3`까지 진행을 하게 되는데 디버그 목적이라면 낮은 단계를 사용하는걸 추천드립니다. **참고로 OpenGL에서의 Warning은 `O3`에서도 상세하게 나타나게 됩니다!**
- `ALLOW_MEMORY_GROWTH=1` : [프로그램이 요구하는 메모리 양이 변경되는 상황에 대해 허용해줍니다.](https://emscripten.org/docs/optimizing/Optimizing-Code.html#memory-growth) 해당 옵션 없이 빌드하면 프로그램이 런타임 도중 메모리를 사용할 수 있는 최대치를 넘으면 뻗어버립니다.
- **`EXCEPTION_CATCHING_ALLOWED=[..]`** : 이게 참.. 중요한 요소인 것 같습니다. 왜냐하면 `try catch`문이 정상적으로 작동하지 않기 때문입니다!
[원래 doc의 내용대로라면 정상적으로 예외처리가 되어야 하는데](https://emscripten.org/docs/porting/exceptions.html) 해당 구문이 저에겐 먹히지 않네요... 이것저것 다 달고 구글링을 열심히 했는데도 마땅한 방안을 못 찾았습니다.
게다가 emscripten에서도 exception 처리는 생각보다 무겁기 때문에 정상적으로 처리하기 힘들다고 적혀있습니다. 다시 정리해보면 **예외처리문인`try catch`문은 사용을 피하는게 정신건강에 이롭습니다!**
- `USE_WEBGL2=1` : WebGL 2.0을 활성화 여부입니다.
참고로 ES 2.0은 WebGL 1.0, ES 3.0은 WebGL 2.0으로 돌아갑니다! 저같은 경우엔 OpenGL ES 3.0이 최소사양이기 때문에 WebGL 2.0을 활성화하였습니다.
- `-preload-file` : 기존의 파일들을 불러와서 사용하고 싶다면 저 문구를 통해 어떤 파일들을 사용할지 설정해줍니다. 여기서 설정된 파일들은 .data 확장명을 가진 zip 형식 파일로 들어가게 됩니다.

## 3. CLion의 Toolchains 설정

CLion없이 바로 emsdk를 통해 cmake처럼 빌드하면 끝이긴 합니다!<br/>
하지만 제가 CLion을 사용하고 있기 때문에 여기서 툴체인을 설정을 해줄 필요가 있었습니다.

그래서 따로 CLion에서 Toolchain을 설정하는 과정도 넣게 되었습니다.

### Toolchains

![Untitled](/media/Cpp-코드를-웹으로-포팅하는-Emscripten-알아보기/Untitled%201.png)

툴체인 설정은 위와 같이 진행하였습니다. 위 이미지를 정리하면 다음과 같습니다.

- CMake	: 설정 그대로
- Make	: 설정 그대로
- C Compiler : `emsdk/upstream/emscripten/emcc`를 설정
- C++ Compiler : `emsdk/upstream/emscripten/em++`를 설정
- Debugger : 설정 그대로

물론 이대로 빌드를 실행해보면 cmake는 emcc랑 em++가 뭔지 몰라하면서 실패하게 됩니다.<br/>
따라서 아래와 같이 또 따로 설정을 해줄 필요가 있습니다.

### CMake

![Untitled](/media/Cpp-코드를-웹으로-포팅하는-Emscripten-알아보기/Untitled%202.png)

여기서 빌드를 위한 CMake 설정 프로파일을 따로 생성하게 됩니다.<br/>
Toolchain은 위에서 만들었던 프로파일로 설정하고<br/>
CMake options는 아래와 같이 설정해줍니다.

```cmake
-DCMAKE_TOOLCHAIN_FILE=[emsdk가 설치되어있는 경로]/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake
```

여기까지 설정이 완료되면 CLion에서도 빌드할 수 있는 환경이 조성됩니다!

## 4. 빌드 된 html 확인하기

![Untitled](/media/Cpp-코드를-웹으로-포팅하는-Emscripten-알아보기/Untitled%203.png)

정상적으로 빌드가 되었다면 위와 같은 파일들이 생성되어있을겁니다. 그럼 바로 html 파일을 클릭해서 확인해볼까요?

![Untitled](/media/Cpp-코드를-웹으로-포팅하는-Emscripten-알아보기/Untitled%204.png)

앗! CORS 문제로 데이터를 불러올 수 없다고 하네요...<br/>
여기서 CORS 관련 문제를 만나보신 분들은 얼마나 끔찍한 문제인지 알고 계실겁니다...

하지만 저희는 복잡한 문제가 아니고 http형식이 아니기 때문에 불러오지 못한다는 뜻이니 아래처럼 간단하게 호스팅 되도록 하겠습니다.

### 파이썬을 이용해 로컬 http 서버 실행하기

놀랍게도 파이썬이 기본 내장인 우분투에선 저 파일들이 있는 위치에서 터미널을 켜고 아래와 같이 간단한 한 줄을 작성해주시면 간단하게 http 서버를 실행할 수 있습니다!

```bash
python -m http.server
```

작동하지 않는다면 `python3`로 대체해서 넣어보시면 됩니다!

## 5. 알아두면 좋은 내용들

### 전처리기 매크로

모든 플랫폼 환경에는 각 플랫폼마다 선언된 매크로가 있습니다.
여기 emscripten도 마찬가지로 존재합니다!

```cpp
#ifdef __EMSCRIPTEN__
    static_cast<SceneMgr*>(m_sceneMgr)->SetScene(new WebDemoScene());
#endif
```

### 예외처리구문 대용?

예외처리구문인 `try catch`를 사용 못하는건 생각보다 큰 단점으로 다가옵니다.<br/>
저도 이런 문제가 있을 줄 모른채 유용하게 썼기 때문입니다.

덕분에 예외처리구문에 대한 예외처리(...)를 하는 덕분에 코드가 좀 더러워졌습니다..ㅠㅜ<br/>
미리 이 상황을 파악하고 코드를 작성하는게 가장 베스트이지만<br/>
저는 아래와 같이 해결을 하였습니다.

1. 예외처리를 `try` 대신 `bool` 형태로 반환하는 예외 확인 함수를 따로 작성한다.
2. `goto`문을 사용한다(...)

저도 `goto`문을 사용할 줄 몰랐는데 좀 복잡하게 적용된 코드는 필요하더군요...흑흑...<br/>
`goto`문을 안쓰자니 같은 코드를 똑같이 작성해야하고...

여러분들은 이런 불상사가 일어나지 않길 빕니다..

## 6. 마무리

![Untitled](/media/Cpp-코드를-웹으로-포팅하는-Emscripten-알아보기/Untitled%205.png)

문제없이 잘 작동한다면 정상적으로 포팅 된 모습을 확인하실 수 있습니다!

인터넷에 기존 OpenGL과 emscripten을 같이 사용하여 웹으로 포팅하는 예제가 생각보다 한정적이여서 몇가지 내용은 제가 삽질을 하며 알아냈습니다...하핫...

제가 해놓은 삽질이 다른 분들과 미래의 나에게 도움이 되었으면 합니다!

그리고 저 프로젝트가 궁금하시다면 아래의 링크를 통해 확인해보셔도 좋습니다!<br/>
[https://github.com/ounols/CSEngine](https://github.com/ounols/CSEngine)
