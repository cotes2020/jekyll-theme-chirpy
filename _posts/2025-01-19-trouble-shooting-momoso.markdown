---
title: "[Troubleshooting] docker 빌드 시 권한 문제로 install이 안되는 문제"
author: kwon
date: 2025-01-19T23:00:00 +0900
categories: [toubleshooting]
tags: [docker, npm]
math: true
mermaid: false
---

# 🚫 현상

`docker-compose build`로 빌드를 진행할 때 아래와 같은 문제가 발생

```bash
[+] Building 14.0s (14/19)                             docker:desktop-linux
...
 => ERROR [frontend internal] load build context                      12.7s 
 ...
 => CANCELED [api 4/6] RUN pip install --no-cache-dir -r requirement  12.5s
------
 > [frontend internal] load build context:
------
failed to solve: archive/tar: unknown file mode ?rwxr-xr-x
```
---

# 💡원인

```bash
failed to solve: archive/tar: unknown file mode ?rwxr-xr-x
```

- 위 메세지에서 권한 관련 문제라는 것을 추측하였고, `frontend internal` 에서 발생한 문제이므로 frontend contianer의 문제라고 생각하여 이에 대해 비슷한 문제가 있는 경우를 탐색
- 자료가 많이 있지는 않았지만 `.dockerignore` 에서 `node_modules`를 추가하고 정상적으로 빌드가 가능했다는 사례에 착안하여 `node_modules` 내부의 파일을 함께 이미지로 build 하려고 했을 때 권한 문제가 있다고 판단.
---

# 🛠 해결책

- 첫 시도는 `docker-compose build --no-cache` 명령어를 통해 기존의 cache를 모두 날리고 `.dockerignore`에 `node_modules`를 추가한 뒤 다시 빌드를 진행
    - 하지만 똑같은 문제가 발생하였음
- 두 번째 시도로 `node_modules`와 `package-lock.json`를 삭제하고 처음 시도했던 방식으로 빌드를 진행
    - 이 경우 정상적으로 빌드가 진행되었음
---

# 🤔 회고

- 새로운 npm 환경을 가지고 왔을 때는 `node_modules`와 `package-lock.json` 를 삭제하고 다시 설치해보는 시도를 해봐야겠다.
- 다른 곳에서 가지고 온 module 파일과 같은 환경 파일의 권한이 잘못 설정되어 있을 수 있다.

---
# 📚 Reference

- [https://github.com/docker/for-win/issues/14083](https://github.com/docker/for-win/issues/14083)