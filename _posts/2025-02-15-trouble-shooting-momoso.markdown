---
title: "[Troubleshooting] gitlab ci 상에서 permission denied가 발생하는 문제"
author: kwon
date: 2025-02-15T23:00:00 +0900
categories: [toubleshooting]
tags: [gitlab-ci-cd]
math: true
mermaid: false
---

# 🚫 현상

```bash
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock:
Post "": dial unix /var/run/docker.sock: connect: permission denied
```
---


# 💡원인

- `.gitlab-ci.yml`을 다음과 같이 작성했다
    
    ```yaml
    ...
    build_backend:
      tags:
        - backend-runner
      script:
        - cd Backend
        - docker build -t $IMAGE_BACKEND:$TAG -f Dockerfile.dev .
        - docker push $IMAGE_BACKEND:$TAG
      only:
        - develop
        - master
    
    build_frontend:
      tags:
        - frontend-runner
      script:
        - cd Frontend
        - docker build -t $IMAGE_FRONTEND:$TAG -f Dockerfile.dev .
        - docker push $IMAGE_FRONTEND:$TAG
      only:
        - develop
        - master
    ...
    ```
    
    - 이런 식으로 작성할 경우 문법 상으로는 문제가 없지만 실제 commit 후 pipeline이 작동할 때 runner 때문에 문제가 발생한다.
        - 두 job은 runner를 구분하지 않았기 때문에 자연스럽게 한 runner를 통해 build를 진행하려고 하는데 이 때 먼저 runner를 선점하지 못한 job은 docker daemon에 대한 권한을 얻을 수 없게 된다.
        - 즉, docker를 사용하는 job이 같은 stage에서 작동한다면 runner를 분리해줘야 한다는 것
---


# 🛠 해결책

```yaml
...
build_backend:
  stage: build
  tags:
    - backend-runner
  script:
    - cd Backend
    - docker build -t $IMAGE_BACKEND:$TAG -f Dockerfile.dev .
    - docker push $IMAGE_BACKEND:$TAG
  only:
    - develop
    - master

build_frontend:
  stage: build
  tags:
    - frontend-runner
  script:
    - cd Frontend
    - docker build -t $IMAGE_FRONTEND:$TAG -f Dockerfile.dev .
    - docker push $IMAGE_FRONTEND:$TAG
  only:
    - develop
    - master
...
```

- 이처럼 `tags`를 활용하여 해당 tag에 맞는 runner를 골라 사용할 수 있다.
    - 지정한 tag를 가지는 runner는 생성해줘야 한다.
---


# 🤔 회고

- 😡진짜 화가 잔뜩 났던 오류였고, 해결까지 정말 오래 걸렸다. 애먼 `.gitlab-ci.yml`만 계속해서 바꿔보고, dind도 적용해봤지만 오류는 계속 발생했다.
- 그런 도중 backend만 성공하고 frontend는 빌드를 실패하는 것을 보고 docker를 중복으로 사용하려는 문제인가 하고 생각하게 되었다.
- 진짜 오류는 겉만 보고는 모른다…
---


# 📚 Reference