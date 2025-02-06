---
title: "[Trouble Shooting] GitLab CI/CD에서 Group Token 권한 문제!"
categories: [Programming TIP]
tags: [gitlabCICD, GroupToken, 권한문제]
---

개발을 하다 보면 같은 그룹 내에서 CICD를 통해 레포지토리를 클론 받고, 커밋하고, 푸시해야 하는 경우가 있습니다. 이를 위해 GitLab CI/CD에서 Group Token을 활용하려 했으나, 예상치 못한 권한 문제를 겪었습니다. 이 문제를 해결한 과정을 공유합니다.

## 작업 개요

- CI/CD 파이프라인을 통해 같은 그룹 내에 있는 repository를 clone 후 commit, push하는 작업

## 수행한 작업

1. **Group Token 생성**

- Role: `developer`
- Scope: `api` , `read_repository` , `write_repository`
  > 배포 대상 레파지토리에 protected_branch로 배포하지 않을 목적이기에 `developer`로 Role을 정했습니다.

2. **CI/CD 환경변수 등록**

- 생성한 `group_token` 을 배포 진행 프로젝트의 GitLab CI/CD 변수로 등록

3. **CI/CD 스크립트에서 환경 변수 활용**

   **CICD script**

   ```yml
     git clone https://oauth2:${GROUP_ACCESS_TOKEN}@git.bwg.co.kr/gitlab/product/pf/pf-api-docs.git # <- 여기서 오류 발생
     cd pf-api-docs
     git add .
     git commit -m "[Automation Commit] Deployed Javadoc to Markdown conversion for product-api"
     git push https://oauth2:${GROUP_ACCESS_TOKEN}@git.bwg.co.kr/gitlab/product/pf/pf-api-docs.git
     echo "Pushed changes to pf-api-docs repository successfully."
   ```

## 문제 상황

CI/CD에서 `group_token` 을 통해 git clone을 사용하려고 했으나, **권한이 없다는 오류 발생**

저는 흔히 발생하는 아래의 상황을 체크했음에도 문제가 없었습니다.

- CICD 변수명 혹은 변수값을 잘못 작성하였나? -> 아니다!
- `group_token`에 권한이 부족한가? -> 아니다! read_repo, write_repo 모두 있다! 그리고 developer 권한은 pull&commit&push 모두 가능하다!

그래서 일단 혹시 모르니 echo를 통해 등록된 토큰 환경 변수를 출력해보았습니다.

### 원인 분석

- CI/CD 실행 로그에서 `group_token` 이 정상적으로 출력되지 않음 😑(echo 출력값이 없음)
- 찾아보니 GitLab CI/CD의 환경 변수 설정에서, **기본적으로 변수가 Protected Branch에서만 사용 가능하도록 설정**되어 있었음
- 하지만 나는 CI/CD 테스트를 위해 별도의 브랜치에서 작업을 진행했음
- 즉, **Protected Branch가 아닌 브랜치에서는 token을 읽어올 수 없었음**

## 해결 방법

![Image]({{"/assets/img/posts/2025-02-05-21-14-26-i14pmg3.png" | relative_url }})

- GitLab CI/CD 변수 설정에서 `protected_branch` 제한을 제거
- 이후 다시 실행하니 정상적으로 token을 활용할 수 있었음

## 정리

GitLab CI/CD에서 **Protected Branch에서만 접근 가능**하도록 변수가 설정되어 있을 수 있습니다. 따라서, CI/CD에서 특정 브랜치에서 실행할 때 token이 정상적으로 읽히지 않는다면, 환경 변수 설정을 확인하고 **Protected Branch 제한을 해제하는 것**이 해결책이 될 수 있습니다.

이와 같은 CI/CD 관련 이슈를 해결할 때는 다음과 같은 점을 고려해야 합니다.

1. **CI/CD 환경 변수 설정을 확인**
2. **로그를 출력하여 변수 값이 정상적으로 읽히는지 확인**
3. **GitLab의 기본 보안 정책을 숙지하고 필요한 경우 설정 변경**

비슷한 문제를 겪고 있는 분들에게 도움이 되길 바랍니다!
