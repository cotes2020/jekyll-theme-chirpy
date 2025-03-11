---
title: "[Troubleshooting] gitlab-runner 상에서 docker 빌드가 안되는 문제"
author: kwon
date: 2025-01-20T23:00:00 +0900
categories: [toubleshooting]
tags: [docker, gitlab-ci-cd]
math: true
mermaid: false
---

# 🚫 현상

- `gitlab-runner` 이미지로 docker 빌드를 하려 할 때 아래와 같은 문제 발생

    ```bash
    Status: Downloaded newer image for gitlab/gitlab-runner:alpine
    docker: Error response from daemon: manifest has incorrect mediatype: application/vnd.oci.image.index.v1+json.
    See 'docker run --help'.
    ```
---

# 💡원인

- image와 현재 환경이 맞지 않는 것으로 판단
---

# 🛠 해결책

- 버전을 v14으로 낮춰서 빌드를 진행하였고, 성공함
---

# 🤔 회고

- 자료들을 찾아본 결과 mediatype문제는 거의 최신 버전을 제대로 지원하지 못해 생기는 문제인 것 같다. 버전을 바꿔서 시도해보자.

---
# 📚 Reference

- [https://github.com/docker/for-win/issues/14083](https://stackoverflow.com/questions/59092140/docker-push-fails-manifest-invalid)