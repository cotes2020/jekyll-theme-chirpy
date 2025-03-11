---
title: "[Troubleshooting] docker에서 volume을 연결해도 파일이 보이지 않는 문제"
author: kwon
date: 2025-02-16T23:00:00 +0900
categories: [toubleshooting]
tags: [docker]
math: true
mermaid: false
---

# 🚫 현상

- `npm run build`로 build 파일을 만들고 image를 만들었음에도 불구하고 mount한 local에 build 파일들이 나타나지 않았음
---


# 💡원인

1. 기존에 이미지에 있는 파일들은 자동 반영 안됨
    - 기존에 이미지에 있었던 build 파일들은 mount되어 있더라고 자동으로 반영되지 않는다.
2. mount를 할 경우 local(source)의 상태가 덮어 씌워짐
    - 이미지를 build 하는 도중에 확인한 log에서는 분명 node build 파일이 정상적으로 생성되었었다.
    - compose를 통해 mount를 하고 실행했을 경우 build 파일이 모두 사라진 것을 확인할 수 있었는데 이건 local의 상태(비어있는 directory)가 덮어 씌워졌기 때문
---


# 🛠 해결책

- `Dockerfile`에서 build 파일들을 복사하는 과정을 추가하였다.
    
    ```bash
    ...
    CMD ["cp", "-r", "/app/dist", "/app/frontend_build"]
    ```
---


# 🤔 회고

- volume mount를 할 때 local의 상태가 덮어 쓰기 된다는 것을 명심하자
---


# 📚 Reference