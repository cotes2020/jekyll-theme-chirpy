---
title: "[Troubleshooting] gitlab-runner에서 panic: EOF [recovered]가 발생하는 문제"
author: kwon
date: 2025-01-21T23:00:00 +0900
categories: [toubleshooting]
tags: [gitlab-ci-cd, docker]
math: true
mermaid: false
---

# 🚫 현상

- gitlab-ic의 컨테이너 내부에서 `gitlab-runner register`를 실행했을 때 다음과 같은 오류 발생

    ```bash
    gitlab_runner   | Runtime platform                                    arch=amd64 os=linux pid=7 revision=782c6ecb version=16.9.1
    gitlab_runner   | Running in system-mode.                            
    gitlab_runner   |                                                                                                                                   
    gitlab_runner   | Enter the GitLab instance URL (for example, https://gitlab.com/):                                                                 
    gitlab_runner   | [https://lab.ssafy.com]: panic: EOF [recovered]                                                                                   
    gitlab_runner   |       panic: EOF   
    ```
---

# 💡원인

- GPT의 답변

    이 문제는 GitLab Runner 컨테이너가 실행되는 동안 사용자 입력을 요구하지만, 컨테이너의 표준 입력(stdin)과 터미널 연결(tty)이 활성화되지 않아 발생합니다. 이를 해결하기 위해 `stdin_open: true`와 `tty: true`를 설정하면, 컨테이너가 인터랙티브 모드에서 실행되어 사용자 입력을 처리할 수 있게 됩니다.

    ### 문제의 원인

    1. **컨테이너의 입력 요구**:
        - GitLab Runner는 실행 시 `gitlab-runner register` 명령어를 통해 GitLab 서버 URL 및 토큰과 같은 설정 값을 입력받아야 합니다.
        - 기본적으로 Docker 컨테이너는 입력을 처리할 수 있는 터미널 환경을 제공하지 않으므로, 입력이 차단되거나 EOF(End of File) 상태로 간주되어 프로그램이 비정상 종료됩니다.
    2. **`panic: EOF`의 의미**:
        - 프로그램이 사용자 입력을 기다릴 때, EOF가 발생하면 입력 스트림이 끝났다는 신호를 받습니다.
        - GitLab Runner는 입력이 종료된 상황을 처리하지 못해 `panic: EOF` 오류를 발생시킵니다.

    ---

    ### 해결 방법: `stdin_open: true`와 `tty: true`

    1. **`stdin_open: true`**:
        - Docker Compose에서 `stdin_open: true`를 설정하면 컨테이너의 표준 입력(stdin)을 열어줍니다.
        - 이는 Docker의 `i` 옵션과 동일하며, 컨테이너가 표준 입력을 지속적으로 대기하도록 만듭니다.
    2. **`tty: true`**:
        - `tty: true`를 설정하면 컨테이너에 가상 터미널(tty)을 할당합니다.
        - 이는 Docker의 `t` 옵션과 동일하며, 인터랙티브 세션에서 터미널처럼 동작하게 만듭니다.
    3. **결합된 효과**:
        - 이 두 옵션을 함께 사용하면 컨테이너가 입력을 받을 수 있는 터미널 환경을 갖추게 되어, GitLab Runner가 사용자 입력을 정상적으로 처리할 수 있습니다.
        - 결과적으로 `panic: EOF` 오류가 발생하지 않고 필요한 설정 과정을 완료할 수 있습니다.

    ---

    ### 실행 원리

    아래와 같이 Docker Compose 파일에 설정하면:

    ```yaml
    services:
    gitlab_runner:
        image: gitlab/gitlab-runner:latest
        stdin_open: true
        tty: true

    ```

    이 설정은 컨테이너가 시작될 때:

    1. 표준 입력이 열려 있어 사용자 입력을 받을 준비가 됨.
    2. 터미널 환경이 활성화되어 입력과 출력이 자연스럽게 연결됨.

    ---

    ### 대안 방법

    만약 자동화된 방식으로 설정을 완료하고 싶다면, 다음 방법을 사용할 수 있습니다:

    1. **사전 등록 설정**:
        - 필요한 환경 변수를 `.gitlab-runner` 디렉토리나 환경 변수로 전달하여 수동 입력을 피할 수 있습니다.
        - 예:
            
            ```yaml
            environment:
            - RUNNER_NAME=my-runner
            - REGISTRATION_TOKEN=your-token
            - CI_SERVER_URL=https://gitlab.example.com/
            
            ```
            
    2. **스크립트 사용**:
        - `gitlab-runner register` 명령어를 스크립트로 작성하여 입력을 자동화할 수 있습니다.

    이 방법을 사용하면 `stdin_open`과 `tty` 설정 없이도 자동으로 등록을 완료할 수 있습니다.
---
# 🛠 해결책

- docker-compose를 다음과 같이 수정하여 해결
    
    ```yaml
    # docker-compose
    ...
      gitlab-runner:
        image: gitlab/gitlab-runner:v16.9.1
        stdin_open: true # -i 옵션과 동일
        tty: true        # -t 옵션과 동일
        container_name: gitlab_runner
        restart: unless-stopped
    ...
    ```
    
    이 경우 사용자의 입력을 받아야 하는 상황에서 입력을 정상적으로 받지 못해 `panic: EOF` 오류가 발생한 것.
    
    `docker run`의 `-it` 옵션을 통해 표준 입력과 가상 터미널을 할당하는 것으로 오류를 피할 수 있지만 이 경우 입력을 받아야 하기 때문에 명령이 다음 상황에서 더 진행되지 않음
    
    ```bash
    2025-01-21 10:46:08 Runtime platform                                    arch=amd64 os=linux pid=7 revision=782c6ecb version=16.9.1
    2025-01-21 10:46:08 Running in system-mode.                            
    2025-01-21 10:46:08                                                    
    2025-01-21 10:46:08 Enter the GitLab instance URL (for example, https://gitlab.com/):
    ```
    
    바로 입력을 반영할 수 있는 방법이 필요
    
- 입력을 받지 않을 것이므로 `-it` 옵션을 지우고 `entrypoint.sh`를 다음과 같이 구성하여 환경 변수를 통해 실행되도록 함
    
    ```bash
    #!/bin/bash
    set -e
    
    if [ ! -f /etc/gitlab-runner/config.toml ]; then
      echo "Registering GitLab Runner..."
      gitlab-runner register \
        --non-interactive \
        --url "$CI_SERVER_URL" \
        --registration-token "$REGISTRATION_TOKEN" \
        --executor "$RUNNER_EXECUTOR" \
        --docker-image "alpine:latest" \
        --description "$RUNNER_NAME" \
        --tag-list "docker,fastapi" \
        --run-untagged="true" \
        --locked="false" \
        --access-level="not_protected"
    fi
    
    echo "Starting GitLab Runner..."
    gitlab-runner run --working-directory=/etc/gitlab-runner
    
    ```
    
    docker-compose는 다음과 같이 수정하여 entrypoint가 실행되도록 한다.
    
    ```yaml
    ...
      gitlab-runner:
        image: gitlab/gitlab-runner:v16.9.1
        container_name: gitlab_runner
        restart: unless-stopped
        volumes:
          - ./gitlab-runner/config:/etc/gitlab-runner
          - /var/run/docker.sock:/var/run/docker.sock
        environment:
          - RUNNER_NAME=my-runner
          - TZ=Asia/Seoul
          - CI_SERVER_URL=${CI_SERVER_URL}
          - REGISTRATION_TOKEN=${REGISTRATION_TOKEN}
          - RUNNER_EXECUTOR=${RUNNER_EXECUTOR}
        entrypoint: ["/bin/bash", "/entrypoint.sh"]
    ...
    ```

---

# 🤔 회고

- 만약 `panic: EOF` 오류가 발생한다면 `-it` 옵션으로 입력을 받도록 할 수 있다.
- 하지만 이번에 발생한 본질적인 문제는 환경변수가 제대로 들어가지 않아 발생한 것.
    - 표면적인 메세지에 집중하지 말고 더 깊이 있는 본질적인 문제를 아는 것이 중요.
- 동작해야 하는 명령이 많은 경우 `entrypoint.sh` 를 작성하여 관리하는 것이 관리 측면이나 디버깅 측면에서 좋을 것 같음

---
# 📚 Reference

- [https://gitlab.com/gitlab-org/gitlab-runner/-/issues/29078](https://gitlab.com/gitlab-org/gitlab-runner/-/issues/29078)