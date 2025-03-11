---
title: "[Troubleshooting] gitlab-runner가 연결하려 할 때 403 error"
author: kwon
date: 2025-01-22T23:00:00 +0900
categories: [toubleshooting]
tags: [gitlab-ci-cd, docker]
math: true
mermaid: false
---

# 🚫 현상

### 명령어

```bash
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
```

### 오류

```bash
025-01-21 16:46:56 ERROR: Checking for jobs... forbidden               runner=t3_TgbhXF status=POST https://lab.ssafy.com/api/v4/jobs/request: 403 Forbidden
2025-01-21 16:46:59 ERROR: Checking for jobs... forbidden               runner=t3_TgbhXF status=POST https://lab.ssafy.com/api/v4/jobs/request: 403 Forbidden
2025-01-21 16:47:02 ERROR: Checking for jobs... forbidden               runner=t3_TgbhXF status=POST https://lab.ssafy.com/api/v4/jobs/request: 403 Forbidden
2025-01-21 16:47:02 ERROR: Runner "https://lab.ssafy.comt3_TgbhXF" is unhealthy and will be disabled for 1h0m0s seconds!  unhealthy_requests=3 unhealthy_requests_limit=3
```
---


# 💡원인

- ~~매개변수의 역할을 제대로 확인하지 않아 발생한 문제~~
    - ~~`--token`을 사용하면 이미 발급된 인증 토큰으로 바로 GitLab 서버와 통신하기 때문에 **403 Forbidden** 오류가 발생하지 않음.~~
    - ~~`--registration-token`을 사용하려면 등록 후 인증 토큰이 올바르게 저장되었는지 확인해야 합니다.~~
- ~~gitlab에서 발급한 토큰을 사용해 통신을 하기 위해서는 `--token` 을 사용해야 함~~
- 기존에 등록했던 runner들이 남아 있고, 함께 run 되려 해서 생긴 문제인듯
    
    ```bash
    / # gitlab-runner list
    Runtime platform                                    arch=amd64 os=linux pid=52 revision=66a723c3 version=17.5.0
    Listing configured runners                          ConfigFile=/etc/gitlab-runner/config.toml
    my-runner                                           Executor=docker Token=glrt-t3_TgbhXFrg957wAC1GotYH URL=https://lab.ssafy.com
    my-runner                                           Executor=shell Token=glrt-t3_iLs69W3NdhwWXSAPzd-j URL=https://lab.ssafy.com/
    my-runner                                           Executor=shell Token=glrt-t3_iLs69W3NdhwWXSAPzd-j URL=https://lab.ssafy.com
    ...
    my-runner                                           Executor=shell Token=glrt-t3_AnQNmob79scgmUco33qb URL=https://lab.ssafy.com
    my-runner                                           Executor=docker Token=glrt-t3_T4J5VMyULHjzsqBAXww5 URL=https://lab.ssafy.com/
    my-runner                                           Executor=docker Token=glrt-t3_Nb98WV4oxRjNVcMxx9_J URL=https://lab.ssafy.com
    ```
---
# 🛠 해결책

- ~~명령어 변경~~
    
    ```bash
    gitlab-runner register \
      --url "$CI_SERVER_URL"\
      --token "$REGISTRATION_TOKEN"\
      --listen-address ":9252" \
      --executor "docker"\
      --docker-image "alpine:latest"
    ```
    
- `config.toml`파일을 제거하고 다시 등록하니 403 발생 안됨

---

# 🤔 회고

## 1. **`--registration-token`**

### 역할

- *Runner를 등록(Register)**할 때 사용되는 **일회성 토큰**입니다.
- GitLab에서 새 Runner를 프로젝트, 그룹, 또는 인스턴스에 연결하기 위해 사용합니다.
- 이 토큰은 Runner를 등록한 뒤에는 더 이상 사용되지 않습니다.

### 발급 위치

- **프로젝트**나 **그룹**의 **Settings > CI/CD > Runners** 섹션에서 확인 가능합니다.
- 예: `Settings > CI/CD > Specific Runners`에서 `registration-token`으로 표시됩니다.

### 주요 특징

- **일회성 토큰**: Runner 등록에만 사용되며, 이후 Runner가 작업 요청을 할 때는 사용되지 않습니다.
- Runner를 등록한 후 GitLab이 **`token`(인증 토큰)**을 발급합니다.

---

## 2. **`--token`**

### 역할

- Runner가 GitLab 서버와 **지속적으로 통신**하기 위한 **인증 토큰**입니다.
- 작업(Job)을 요청하거나 실행 결과를 보고할 때 사용됩니다.
- 등록 과정에서 자동으로 발급되며, Runner의 `config.toml` 파일에 저장됩니다.

### 저장 위치

- `config.toml` 파일에 저장됩니다:
    
    ```toml
    [[runners]]
      name = "my-runner"
      url = "https://gitlab.example.com/"
      token = "authentication-token"  # 인증 토큰
      executor = "docker"
    ```
    

### 주요 특징

- **지속적으로 사용**: Runner가 GitLab 서버와 통신하는 모든 과정에서 사용됩니다.
- 만약 잘못된 값이 설정되면 **403 Forbidden** 오류가 발생합니다.
- GitLab UI에서는 보이지 않습니다. 대신 Runner가 등록된 후 자동으로 발급됩니다.

---

## 차이점 요약

| 구분 | `--registration-token` | `--token` |
| --- | --- | --- |
| **역할** | Runner를 등록하기 위한 토큰 | Runner와 GitLab 서버 간 인증용 |
| **발급 위치** | 프로젝트/그룹의 Runner 설정 | 등록 시 자동 생성 |
| **사용 시점** | Runner 등록 시 한 번 사용 | 등록 후 지속적으로 사용 |
| **가시성** | GitLab UI에서 확인 가능 | `config.toml`에 저장됨 |
| **만료 여부** | 필요시 GitLab UI에서 재발급 가능 | 등록 시 자동으로 새로 생성됨 |

---

## 실질적인 사용 예시

### 1. Runner를 등록할 때:

```toml
gitlab-runner register \
  --url "https://gitlab.example.com/" \
  --registration-token "registration-token" \
  --executor "docker" \
  --description "my-runner"
```

- 이 명령어로 GitLab 서버에 Runner를 등록합니다.
- 등록 과정에서 GitLab이 인증 토큰(`-token`)을 생성하여 `config.toml`에 저장합니다.

---

### 2. Runner 실행:

```bash
gitlab-runner run
```

- 이 명령어는 `config.toml`에 저장된 인증 토큰(`-token`)을 사용해 GitLab 서버와 통신합니다.
- `-registration-token`은 이 단계에서 더 이상 사용되지 않습니다.

---

## 결론

- *`-registration-token`*은 Runner를 GitLab 서버에 **등록**하기 위한 초기 설정 토큰입니다.
- *`-token`*은 등록 후 **GitLab과의 지속적인 인증**을 위해 사용됩니다.
- `403 Forbidden` 문제가 발생한다면, 대개 `config.toml`에 저장된 **`token`** 값이 잘못되었거나 유효하지 않기 때문입니다.
---
# 📚 Reference