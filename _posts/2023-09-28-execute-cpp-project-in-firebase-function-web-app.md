---
layout: post
title: Execute cpp Project in Firebase Function Web App
date: 2023-09-28 11:32 +0900
category: [Framework]
tag: [Python, Firebase, cmake, pybind11, wheel]
---

github의 어떤 cpp프로젝트를 보고 이를 활용하여 만들고 싶은 웹 사이트가 생겼다. cpp와 웹은 좀 거리가 있어보이는 조합이다... 이때 시행착오 과정이 꽤 길었어서 마주친 문제점과 해결책을 정리해보았다.

### 웹 사이트에서 서버 없이는 실행파일을 실행할 수 없다

다룰 수 있는 웹 사이트 제작 프레임워크는 flutter밖에 없었기 때문에 일단은 dart에서 cpp 빌드 파일을 실행할 수 있는 방법을 모색해보았는데 웹의 경우엔 서버 없이는 불가능하다는 것 같다. react를 사용하더라도 기술적인 문제로 불가능한 것 같다. 따라서 firebase 서버를 사용해보기로 하였다.

### Firebase Functions을 Python 서버로 사용

firebase가 제공하는 여러 기능 중 functions는 html 통신을 통해 서버와의 정보를 주고받을 수 있도록 설계되어있다. 서버의 종류는 typescript, python 이 두 가지가 있는데 typescript에는 문외한이므로 python으로 사용하기로 하였다.

>### Python 서버 환경과 Local 환경은 다르다
>firebase는 `firebase emulators:start` 명령어를 통해 firebase functions 기능을 로컬에서 테스트할 수 있는데, 이때 잘 작동하더라도 `firebase deploy`로 배포했을 경우에는 작동하지 않는 경우가 많았다. 앞으로 설명할 다양한 문제 상황도 여기에 해당되는 경우가 많다.

>### on_request함수는 CORS 오류가 발생한다
>이유는 모르겠지만 on_request 함수 대신 on_call 함수를 사용해야 CORS error가 발생하지 않는다. 자세한 예제는 [여기](https://github.com/seokjin1013/flutter-web-firebase-func-python-example) 참고. 이때 Local 테스트 환경이 아닌, 배포할 경우에는 lib/main.dart의 `FirebaseFunctions.instance.useFunctionsEmulator` 부분을 지워줘야 한다. [여기](https://github.com/seokjin1013/flutter-web-firebase-func-python-example/blob/master/lib/main.dart#L13C35-L13C35) 참고. on_call 함수를 사용하면 반환하는 값이 on_request와 달리 문자열이어야 한다는 점 주의하자. 그리고 이 문자열은 json serializable 해야 한다.

### subprocess.run을 사용하면 오류가 발생한다

파이썬에서는 `subprocess.run`함수를 통해 명령어를 실행할 수 있는데, 이를 이용해서 실행하려고 하는 cpp 프로젝트를 빌드하여 실행파일로 만들어놓고 실행시키는 방법을 생각해볼 수 있다. 하지만 이렇게 할 경우 아래 오류가 나온다.

>`INTERNAL | Internal server error. Typically a server bug.`

그것도 Local에서 테스트했을 때는 오류가 나지 않지만 배포했을 때만 오류가 난다. 그래서 cpp프로젝트를 단순히 실행파일로 빌드한 것이 아닌, 파이썬 확장 모듈의 형태로 만들고 파이썬 환경을 꾸려줘야 한다.

### shard object의 형태로 import하면 오류가 발생한다

cpp프로젝트를 pybind11을 통해 파이썬 라이브러리로 만들고 cmake를 통해 빌드를 성공시키면 .so파일(shared object)을 얻을 수 있는데, python 소스코드와 같은 위치에 놓고 import하여 사용할 수 있다. 하지만 import하는 코드를 작성하고 `firebase deploy`를 했을 때 배포가 안되는 오류가 발생한다.

그러면서 <https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start> 이 링크를 통해 오류를 해결하라는 로그가 나오지만 여기에 해당되는 경우가 없는 것 같다. Local으로 테스트한 경우 오류가 발생하지 않고, 64비트 linux에서 컴파일했기 때문이다. 따라서 pip install으로 설치가능한 형태로 만들어줘야 한다. 하지만 .so파일은 바로 설치할 수 없고 .whl파일으로 빌드해야 한다.

>배포 실패 오류를 몇 번 겪다보면 이상하게도, 문제가 되었던 import부분을 제거하고 배포했을 때 CORS 오류가 발생한다. 나의 경우에는 배포 실패 오류를 겪었던 함수 A가 계속 CORS오류가 발생하여, 그대로 복사하고 이름만 바꾼 함수 B를 만들어 새로 배포했는데 B는 잘 작동하는 반면 A는 그대로 오류가 발생하는 이상한 현상이 발생했다. 이 경우 firebase console에 들어가 함수를 모두 삭제하고 다시 배포하니 해결되었다.

>`<라이브러리 이름>.cpython-311-x86_64-linux-gnu.so`와 같은 이름은 환경에 따라서 다르게 나올 수 있다. 나는 python 구현이 cpython이고, 3.11버전이면서 Windows WSL Ubuntu환경에 Linux버전의 파이썬을 설치했기 때문에 위와 같은 이름이 나온 것 같다.


### 해결 방법

내가 찾은 유일한 방법은 cpp프로젝트를 wheel으로 빌드하고 가상환경에 설치하여 환경을 만들어주는 것이었다. wheel으로 프로젝트를 빌드하려면 `pip install build`를 통해 build를 설치하고 `setup.py`를 작성한 뒤 `python -m build --wheel`을 실행해야 한다.

setup.py는 cmake 프로젝트라면 <https://github.com/Klebert-Engineering/python-cmake-wheel> 이 문서를 활용하여 `CMakeLists.txt`를 약간 수정하는 것으로 `cmake --build .`를 했을 때 자동으로 setup.py가 생성되도록 할 수 있다.

생성된 .whl파일은 dist 폴더 안에 있다. 이걸 firebase functions 폴더에 옮기고 requirements.txt에 상대경로로 .whl파일 위치를 적어주면 Local에서도 잘 작동하고 배포도 잘 되며 실행도 잘 된다.