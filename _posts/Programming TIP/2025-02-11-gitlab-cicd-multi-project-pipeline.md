---
title: "GitLab Multi-Project Pipelines로 CI/CD 최적화하기"
categories: [Programming TIP]
tags:
  [
    GitLab,
    CI/CD,
    MultiProjectPipelines,
    DevOps,
    PipelineOptimization,
    Automation
  ]
---

혹시 Gitlab의 CI/CD 파이프라인을 구성하시면서, **A 프로젝트에서 CI/CD 작업이 완료되면 B 프로젝트의 CI/CD 작업을 자동으로 실행**하고 싶은 경험이 있으셨나요?

저도 이번에 PF팀의 메인 프로젝트의 파이프라인의 속도가 너무 오래 걸려 이를 개선하기 위한 방법으로 프로젝트 분리를 진행했는데요, 이 과정에서 Gitlab의 Multi-Project Pipeline 작업을 적용해보았습니다!

## 📌 Multi-Project Pipelines란?

GitLab의 Multi-Project Pipelines는 **여러 개의 프로젝트 간에 파이프라인을 연결하고 실행**할 수 있도록 해주는 기능입니다. 이 기능을 활용하면 **서로 다른 GitLab 프로젝트의 빌드 및 배포 프로세스를 효율적으로 연계**할 수 있습니다.

> 참고 : [📚 Gitlab의 Multi-Project Pipelines에 대한 공식 doc](https://docs.gitlab.com/ee/ci/pipelines/downstream_pipelines.html#multi-project-pipelines)

### 🔹 Multi-Project Pipelines의 주요 특징

1. **프로젝트 간 파이프라인 트리거**
   - 한 프로젝트의 파이프라인이 다른 프로젝트의 파이프라인을 실행할 수 있음
   - 예: `A` 프로젝트의 배포가 끝나면 `B` 프로젝트의 테스트 및 배포 자동 실행
2. **Downstream Pipeline**
   - 특정 프로젝트의 파이프라인 실행 후 다른 프로젝트의 파이프라인을 자동으로 실행하는 방식
   - `trigger:` 키워드를 활용하여 구현 가능
3. **독립적 관리 가능**
   - 각 프로젝트의 CI/CD를 개별적으로 운영하면서도 상호 연계 가능
   - 불필요한 빌드 중복을 방지하여 CI/CD 성능 최적화
4. **Cross-Project Dependency 해결**
   - 프로젝트 간 종속성이 있는 경우, 필요한 프로젝트를 먼저 빌드하고 후속 프로젝트의 배포 진행 가능

## 🛠 Multi-Project Pipeline 설정 방법

GitLab CI/CD 설정 파일(`.gitlab-ci.yml`)에서 Multi-Project Pipelines를 적용하려면 `trigger:` 키워드를 사용하여 다른 프로젝트의 파이프라인을 트리거할 수 있습니다.

```
stages:
  - build
  - test
  - deploy

build_project:
  stage: build
  script:
    - echo "Building project..."

trigger_downstream:
  stage: deploy
  trigger:
    project: group-name/downstream-project
    branch: main
    strategy: depend
```

### 🎯 주요 설정

- **`project:` : 트리거할 프로젝트 경로 (예: `group-name/downstream-project`)**
  - ❌이 때 반드시 프로젝트 경로의 full path를 기재해주셔야 합니다! 상대 경로 작성 시 제대로 수행이 되지 않아요.
- **`branch:` : 트리거 대상 프로젝트의 실행할 브랜치 지정**
- **`strategy: depend` : Upstream이 실패하면 Downstream 실행하지 않음 (의존적 관계 설정)**

## 🏗 Multi-Project Pipelines 적용 사례

### 기존 문제

- `SpringBoot` 프로젝트의 CI/CD에서 **Spring Boot 소스코드 수정 후 배포 시, 매번 매뉴얼 리소스를 불필요하게 빌드**
- 매뉴얼 리소스에 변경이 없음에도 **소스코드 배포 과정에서 매뉴얼 빌드로 인해 불필요한 시간 소요**
- 결과적으로 **소스코드 배포 속도가 저하됨 (기존 SpringBoot 배포 소요 시간: 20분, 그 중 매뉴얼 빌드 시간 약 10-15분)**

### 해결 방법

**1.** **`매뉴얼`을 별도의 GitLab 프로젝트로 분리**

1. `main` 브랜치 배포 시, Docusaurus 빌드 실행
2. 빌드 결과물을 Package Registry에 저장
3. ✅`SrpingBoot` 프로젝트의 `master` 브랜치 pipeline 트리거 => `Multi-Project Pipeline 적용!`

> 매뉴얼 프로젝트 Gitlab CI/CD Script

```yml
stages:
  - build
  - deploy
  - trigger

variables:
  BUILD_DIR: "build"

# Build stage
build-docusaurus:
  image: node:22-alpine
  stage: build
  tags:
    - funky-docker
  cache:
    key: "${CI_COMMIT_REF_SLUG}"
    paths:
      - node_modules/
  script:
    - echo "Installing dependencies..."
    - npm install
    - echo "Building the project..."
    - export NODE_OPTIONS=--openssl-legacy-provider
    - npm run build
  artifacts:
    paths:
      - ${BUILD_DIR}
    expire_in: 1 hour
  only:
    - main

# publish build output to GitLab Package Registry
deploy-docusaurus:
  stage: deploy
  image: curlimages/curl:latest
  tags:
    - funky-docker
  script:
    - tar -czf manual-build.tar.gz -C ${BUILD_DIR} .
    - echo "Contents of ${BUILD_DIR}:"
    - |
      curl --header "JOB-TOKEN: ${CI_JOB_TOKEN}" \
           --upload-file manual-build.tar.gz \
           "GITLAB_API_HTTP_URL/${CI_PROJECT_ID}/packages/generic/manual/${CI_COMMIT_REF_NAME}/manual-build.tar.gz"
  only:
    - main

# Trigger PF deploy pipeline
trigger_pf_deploy_job:
  stage: trigger
  trigger:
    project: springboot
    branch: master
    strategy: depend
```

**2.** **`SpringBoot` 프로젝트 파이프라인 수정**

- 기존 매뉴얼 빌드 작업 제거
- 대신, 매뉴얼 빌드 결과물을 `매뉴얼` 프로젝트의 Package Registry에서 획득

### 파이프라인 진행 과정

![Image]({{"/assets/img/posts/2025-02-11-gitlab-cicd-multi-project-pipeline/1739253745981.png" | relative_url }})

- `PF 매뉴얼` 프로젝트의 main 브랜치로 배포 시, `build` -> `deploy` -> `trigger` 순으로 진행
- `trigger` job으로 인해 **Downstream(PF 프로젝트의 master 브랜치)의 파이프라인 실행 요청**
- `PF` 프로젝트의 `deploy:pf:dev` job을 포함한 **master 브랜치 파이프라인 실행**

### 효과

1. 🚀 **소스코드 배포 속도 2.5배 향상**

   - 기존 배포 시간: **16분** → 개선 후: **6분**
   - **배포 시간 약 10분 단축**

2. 🔄 **매뉴얼과 소스코드의 독립성 확보 → 유지보수 용이**

   - 소스코드 변경과 무관하게 매뉴얼 빌드 가능
   - 불필요한 매뉴얼 빌드 제거로 CI/CD 효율성 향상

3. 🔧 **기존 매뉴얼 배포 방식 유지**

   - 매뉴얼 수정 시 **Trigger Job을 통해 서버에 배포하는 기존 구조 유지 가능**
   - 추가적인 설정 변경 없이 기존 배포 방식과의 연속성 보장

4. ✅ **소스코드 배포 시, 매뉴얼 빌드 작업 제거 → 배포 프로세스 최적화**

   - Spring Boot 프로젝트 배포 시, **매뉴얼 빌드 없이 즉시 배포 가능**
   - **배포 속도 개선 및 불필요한 리소스 낭비 방지**

---

GitLab Multi-Project Pipelines를 활용하면 **배포 시간 단축, 독립적 유지보수, 효율적 CI/CD 운영**이 가능하며, 실제로 PF 매뉴얼 프로젝트에서도 이점을 확인할 수 있었습니다.

프로젝트 규모가 커지고 배포 시간이 중요해지는 환경에서 Multi-Project Pipelines는 강력한 도구가 될 수 있습니다.
