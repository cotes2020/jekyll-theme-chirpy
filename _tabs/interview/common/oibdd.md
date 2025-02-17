# 아키텍쳐



OiBDD는 Client(local server)와 Server(E2C Instance)로 구분된다.

#### OiBDD Client

- **Front소스** : 빌드 후 MainUI로 띄운다. (Flask의 내장 web서버 사용)
- **Feature/StepDefs File I/O:** 로컬 레파지토리의 cucumber project의 feature, step defintion 파일을 읽고 쓴다.
- **LLM API Proxy:** LLM 연동 API 호출의 proxy 역할을 한다. (OIBDD Server API 사용)
- **Cucumber Project**
  - Gitlab Cucumber project를 로컬 레포지토리에 클론받은 경로 내 프로젝트이다. (경로: `.env`의 `bdd_features_dir` 에 할당)
  - 유저가 보고있는 MainUI에서 Feature/Step Defs 파일은 로컬 Cucumber Project 내 파일이다.
  - Feature/ Step Defs 생성 시, 서버(엔진)으로부터 파일을 받아 로컬에 파일을 생성한다.
  - 서버와 로컬의 파일 존재여부가 다를 경우 MainUI화면에서 [아이콘](https://pages.bwg.co.kr/tools/OiBDD/user_manual/#31-%EA%B8%B0%EB%8A%A5%EC%9A%94%EA%B5%AC%EC%82%AC%ED%95%AD-%EB%AA%A9%EB%A1%9D-%EC%A1%B0%ED%9A%8C)을 통해 표현해준다.

#### OiBDD Server

OiBDD의 엔진역할이다.

- **Admin UI:** Prompt 관리를 위한 Admin 페이지(nginx로 띄움, 비밀번호 존재, Prompt는 소스 내 json으로 관리)
- **LLM 연동:** GPT와 연동된 LLM
  - feature 파일 생성
  - step defintion 생성
- **Embedding**
  - **RAG 유사도검색**을 위해 Feature와 Step Defition 파일의 필요한 부분을 각 각 임베딩하여 저장한다.
  - **스케쥴 작업**-현재 1분 간격으로 임베딩을 진행한다. (간격 설정: `appconfig.py`의 `embedding_scheduler_interval` 변수)
    - 임베딩 대상 조건: DB의 Feature 테이블의 feature/step_defs status 컬럼이 `CREATED`, `MODIFIED`, `DELETED`, `FR_DELETED `인 경우
    - 임베딩 대상 파일 경로: Feature/StepDefs Repository에 있는 파일들을 대상 (`.env`의 `bdd_features_dir` 에서 설정한 경로)
  - 임베딩 완료 후, 각 파일들이 임베딩 완료되었음을 DB에 상태 업데이트한다. ([상태관리 wiki](./09_상태관리.md))
- **Resource 관리**
  - **SRS, Example Mapping, Feature 상태**(연동 상태, 생성상태, 임베딩 상태 등), 파일 연동 로그 관리 등
  - **파일변경 감지**: SRS를 Pages와 연동 시, 파일변경 감지하여 DB 내용을 업데이트
- **Gitlab Project 연동**
  - Cucumber Project 연동 - 필수 / SRS Pages 연동 - 선택 (연동 안할 경우 환경변수 `BDD_SRS_DIR` = `None`으로 설정 )
  - `pull_repo.sh` 파일에 각 gitlab repository 및 token 할당
  - crontab에서 `pull_repo.sh` 파일 호출하여 설정한 간격에 따라 변경사항 존재 시 pull 받아서 각각의 Server의 repository path로 전송
  - server의 repository path는 환경변수로 설정(`BDD_FEATURES_DIR`, `BDD_SRS_DIR`)
- **파일 변경 감지**
  - watchdog을 통해 Feature/StepDefs Repository의 파일을 감지하며, 변경사항(추가, 삭제, 변경) 시, DB에 상태 업데이트
    - Feature 상태 변경 시, feature_status 컬럼 / Step Defs 상태 변경 시, step_defs_status 컬럼 상태 변경
  - (SRS pages 연동 시) watchdog을 통해 SRS Repository의 파일을 감지하며, 변경사항(추가, 삭제, 변경) 시, DB에 상태 업데이트
    - SRS 변경 시, DB SRS 테이블에 변경 내용 적용

---

### LLM을 통한 생성 - 상세 프로세스

#### Feature 생성

1단계~3단계로 이루어진 과정을 통해 한번 생성

- 입력: SRS ID, Example Mapping ID
- 출력: Feature 내용(client로 생성 내용 전달)
- **1단계: 초안 생성**

  - DB의 example mapping 기반 초안 생성
- **2단계: 유사한 step 통합**

  - (기존에 이미 임베딩된 데이터) Step Definitions 파일에서 keyword + 스텝명으로 임베딩 청크 구성
  - 초안으로 생성된 Feature 파일에서 Step 목록 색출
  - Step 목록 하나하나씩 돌며 임베딩 벡터에서 유사도 검색을 통해 기존 존재하는 Step 목록과 유사한 목록 발견 시 응답
  - Feature파일에 유사한 step 목록 주석으로 제시
- **3단계: Before Action 존재 시, Given 스텝 생성**

  - (기존에 이미 임베딩된 데이터) Feature 파일의 example name, feature명, 태그, 위치 등으로 임베딩 청크 구성
  - example mapping에 before action 존재 시, before action과 vectordb의 feature 파일청크 유사도 검색을 통해 before action 대상이 되는 feature example 탐색
  - 존재 시, 조회된 feature 파일의 example의 step을 feature 파일의 Given step으로 추가

#### Step Defintion 생성

생성 타입에 따라 각 각 생성

- **초안 생성: Step Definition Template을 통해 초안 작성**

  - 입력: 생성된 Feature 내용(local로 부터 파일내용 전송받음), Feature 실행 로그(어떤 스텝 만들어야 하는지 정보 획득)
  - 출력: Step Definition 내용(client로 생성 내용 전달)
- **상세 작성: Step Definition에 상세 수행동작(연동 API, 입력, 결과값 등)을 주석으로 입력하여 완성된 코드 생성**

  - 입력: 상세 수행동작 주석이 포함된 Step Defintion 내용
  - 출력: Step Definition 내용(client로 생성 내용 전달)
- 

# 배포 프로세스

`oibdd-server`의 배포 프로세스는 **GitLab CI/CD, Docker, Docker Compose, Nginx, PostgreSQL**을 활용하여 이루어집니다. 배포의 핵심 흐름은 **CI/CD 파이프라인을 통한 자동화**, **Docker 기반의 컨테이너화된 애플리케이션 관리**, **서버 내 `update_server.sh` 스크립트를 통한 배포 갱신**으로 구성됩니다.

------

## **1. 배포 주요 구성 요소**

배포 프로세스는 다음과 같은 핵심 컴포넌트들로 이루어져 있습니다.

- **GitLab CI/CD (`.gitlab-ci.yml`)**:
  - GitLab에서 **자동으로 빌드 및 배포**를 수행하는 CI/CD 파이프라인을 정의합니다.
  - `build-ui-job` → `deploy-job` → `release` 순서로 진행됩니다.
- **Dockerfile**:
  - FastAPI 기반의 애플리케이션을 실행할 수 있도록 Docker 이미지를 빌드하는 설정입니다.
  - Python 3.12 및 Node.js 환경을 포함하여 실행을 준비합니다.
- **서버 배포 스크립트 (`update_server.sh`)**:
  - **최신 소스 코드 pull**
  - **Docker 이미지 빌드**
  - **Nginx 기반 Admin UI 배포**
  - **FastAPI 컨테이너 실행**
  - `docker-compose`를 이용하여 서비스를 실행 및 관리합니다.
- **Docker Compose (`docker-compose.yml`)**:
  - FastAPI 서버, Nginx, PostgreSQL을 포함한 서비스들의 실행 및 네트워크 설정을 정의합니다.

------

## **2. 전체적인 배포 흐름**

### **📌 1) CI/CD를 통한 자동화 빌드 & 배포**

`oibdd-server`의 GitLab CI/CD는 `.gitlab-ci.yml`을 통해 배포를 자동화합니다.

1. **UI 빌드 (`build-ui-job`)**
   - `dev` 브랜치에서만 실행 (`npm install && npm run build`).
   - `front/build/` 폴더에 빌드된 파일 저장.
2. **서버 배포 (`deploy-job`)**
   - `"deployserver"`가 포함된 커밋 메시지가 있는 경우 **자동 배포**.
   - EC2 서버로 빌드된 UI 파일을 업로드.
   - `update_server.sh`를 실행하여 서버를 갱신.
3. **릴리즈 (`release`)**
   - 태그가 붙은 커밋에서 실행.
   - Dockerfile 및 의존성 파일을 포함하여 **배포 가능한 패키지**를 생성.

------

### **📌 2) 서버에서 애플리케이션 실행 (`update_server.sh`)**

서버 측에서는 `update_server.sh`가 실행되어 애플리케이션을 최신 상태로 업데이트합니다.

1. Git 저장소에서 최신 코드 `pull`
2. Docker 이미지 빌드 (`finlabbwg/oibdd-server:dev`)
3. Nginx를 통해 **Admin UI 배포**
4. `docker-compose up -d fastapi`를 실행하여 FastAPI 애플리케이션 재실행

> 만약 GitLab CI/CD 없이 수동으로 배포해야 할 경우, `pull_repo.sh` 스크립트를 실행하여 최신 코드를 가져올 수도 있습니다.

------

### **📌 3) Docker Compose를 통한 컨테이너 관리**

`docker-compose.yml`을 사용하여 FastAPI 애플리케이션과 관련 서비스를 관리합니다.

- **FastAPI 컨테이너** (`finlabbwg/oibdd-server:dev`)
  - FastAPI 애플리케이션을 실행.
  - PostgreSQL과 연결하여 데이터베이스를 사용.
  - 환경 변수 설정을 통해 OpenAI API 및 파일 디렉토리 설정.
- **Nginx 컨테이너**
  - `nginx.conf` 설정을 사용하여 Admin UI 정적 파일을 서빙.
  - `htpasswd`를 사용하여 인증을 추가 가능.
- **PostgreSQL 컨테이너**
  - FastAPI 애플리케이션에서 사용할 DB 환경을 설정.

------

## **3. 최종 정리**

**배포 프로세스는 크게 3단계로 나눌 수 있습니다.**

### **🚀 1) 코드 푸시 & CI/CD 실행**

- GitLab CI/CD (`.gitlab-ci.yml`)가 실행되어 코드가 자동으로 빌드되고 서버로 배포됨.
- `build-ui-job` → `deploy-job` → `release` 순으로 실행.

### **🔄 2) 서버에서 업데이트 & 실행 (`update_server.sh`)**

- 최신 코드 `pull`
- `Dockerfile`을 기반으로 새로운 컨테이너 이미지 빌드
- Nginx를 통해 정적 UI 업데이트
- `docker-compose up -d`를 사용하여 FastAPI 컨테이너 실행

### **🛠️ 3) 컨테이너 실행 (`docker-compose.yml`)**

- `fastapi`, `nginx`, `postgres`를 포함한 컨테이너 환경을 정의하고 실행.
- 각 서비스가 올바르게 동작하도록 네트워크 및 볼륨을 설정.

------

## **4. 결론**

- **GitLab CI/CD를 이용한 자동 배포 프로세스**를 구축하여 `dev` 브랜치에서 코드 푸시 시 자동으로 배포가 이루어짐.
- **서버에서 `update_server.sh` 실행**을 통해 최신 소스를 받아 컨테이너를 재실행.
- **Docker & Docker Compose**를 사용하여 애플리케이션을 컨테이너 기반으로 실행 및 관리.
- **Nginx를 통해 Admin UI 배포** 및 **PostgreSQL을 활용한 데이터 관리**.

------

### **💡 배포 프로세스를 더욱 개선하려면?**

✅ **CI/CD에서 `docker-compose` 사용하여 배포 자동화**
현재 `update_server.sh`에서 `docker-compose up -d fastapi`를 실행하지만, CI/CD에서 직접 실행하도록 개선 가능.

✅ **롤백 기능 추가**
배포 실패 시 자동으로 이전 버전으로 롤백할 수 있도록 `docker-compose`와 `git checkout`을 활용한 스크립트 추가.

✅ **캐시 최적화**
Docker 이미지 빌드시 캐싱을 활용하여 `pip install`과 `npm install`의 속도를 최적화할 수 있음.

------

### **📌 요약**

1. CI/CD를 활용하여 자동 배포
   - GitLab `.gitlab-ci.yml`에서 빌드, 배포, 릴리즈를 정의.
2. 서버 배포 스크립트 실행 (`update_server.sh`)
   - 최신 코드 `pull` → Docker 이미지 빌드 → `docker-compose` 실행.
3. Docker Compose 기반 컨테이너 실행
   - FastAPI 서버, Nginx, PostgreSQL을 하나의 환경에서 실행.

이와 같은 구조를 통해 **자동화된 배포와 안정적인 운영**이 가능해집니다. 🚀