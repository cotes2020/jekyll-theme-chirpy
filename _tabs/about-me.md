---
# the default layout is 'page'
icon: fas fa-solid fa-user
order: 4
---

<!-- <div style="display: flex; align-items: center;"> -->
<div style="display: flex; align-items: flex-start; gap: 20px;">
  <img src="/assets/img/posts/2024-12-12-15-32-24-i32pmg4.png" alt="profile-img" style="border-radius: 10rem; width: 300px; height: auto; margin-right: 20px; margin-bottom: 20px;">
  <div>
    <h3>AI 어플리케이션 개발자, 정예울입니다.</h3>
    <p>/////가안만 작성하고, 마지막에 업무 경력 및 내 활동을 보고 다시 일관되는 점을 찾아 작성하자///</p>
    <p>새로운 것을 시도하고 경험을 통해 배웁니다.</p>
    <p>배운 것을 실생활에서의 아이디어와 결합하여 일상의 새로움을 창조합니다.
</p>
  </div>
</div>

|E-mail|papooo.dev@gmail.com|
|Github| [github.com/papooo-dev](https://github.com/papooo-dev)|
|Blog|`now`: [https://papooo-dev.github.io/](https://papooo-dev.github.io/)<br/> `previous`: [https://creamilk88.tistory.com/](https://creamilk88.tistory.com/)|

## Skills

- AI : Langchain
- 언어 : Java, Python, javascript
- 프레임워크: Spring Boot, FastAPI, Flask, Node.js, React
- 도구 및 플랫폼: Docker, AWS, Git
- DB: PostgreSQL, Oracle, MySQL

## Work Activities

#### 2021.03 ~ 현재 | Bankware Global

`사내 툴 개발`, `코어뱅킹 개발(고객)`, `테스트 자동화 프로그램 운영`

- AI를 활용한 어플리케이션 개발을 진행하였습니다. AI를 활용하여 BDD 개발 방식에 자동화를 도입하는 툴과 대량 건수의 테스트 데이터를 AI를 통해 생성하는 프로그램을 개발하였습니다.
- AI 프레임워크인 LangChain 사내 강의를 직접 진행하였습니다. 10회 차로 구성되어 기본 이론부터 실습까지 진행하였으며, 이는 사내 AI를 활용한 제품 개발에도 영향을 주었습니다.
- 대외 연계 시뮬레이터 솔루션인 BXI Simulator의 버전2 개발을 진행했습니다. REST API도 시뮬레이터를 통한 거래 테스트를 진행할 수 있도록 하였습니다.
- LINE Bank Japan 의 코어뱅킹 시스템 개발을 진행했습니다. 비대면 고객가입 및 고객관리를 개발하였으며, 테스트 자동화 프로그램을 개발&운영 하였습니다.

## 프로젝트

#### `OiBDD` : AI를 활용한 BDD 개발 보조 프로그램

- 기간: 2024/08 - 2024/12 (4M)
- 프로그램 개요: BDD의 디스커버리 워크숍 과정을 AI로 자동화하여 Example Mapping부터 Feature 및 Step Definition 코드 생성까지 지원하는 AI 기반 BDD 보조 프로그램
- 기술 스택: `LangChain`, `Python`, `Flask`, `FastAPI`, `PostgreSQL`, `React`
- 주요 성과

  - API 코어 서버와 클라이언트 서버를 분리하여 클라이언트의 다양화(웹, 플러그인)가 가능하도록 확장성을 증가하고, 코어 기능(프롬프트, 토큰)을 보호할 수 있었습니다.
  - AI 응답의 퀄리티를 높이기 위해, Memory, RAG 기술을 적용하였습니다.
  - FastAPI의 Background Task 기능을 적용하여, LLM 응답 대기 중 실시간으로 진행 상태 조회할 수 있도록 하였습니다.
  - 서버에 스케쥴을 통해 파일 변경 시 변경 파일에 대한 자동 임베딩 시스템을 개발하였습니다.
  - AWS에 E2C 인스턴스를 구성하여, Docker 기반의 Gitlab CICD 배포 시스템을 구축하였습니다.

- 개선사항: 프로그램 개요는 간단히, 주요기능에 대해서 상세하게 쓰기 / 주요 성과에서는 기능의 얘기는 뺴기

#### Test Data Generator : AI를 활용하여 대량의 다양한 형식에 최적화된 테스트 데이터를 생성하는 프로그램

- 기간: 2024/07 - 2024/08 (1.5M)
- 프로그램 개요: SQL DDL, XML 파일을 통해 데이터 구조와 타입, 관계 정보를 인식하여 사용자가 자연어로 작성한 데이터 생성 규칙을 통해 mock 데이터를 생성한다. 이후 사용자가 원하는 형태(csv, json, sql, DB에 직접 insert)로 데이터를 출력한다.
- 기술 스택: `LangChain`, `Python`, `Flask`, `PostgreSQL`, `React`
- 주요 성과

  - 데이터 생성 건마다 AI를 호출하는 것이 아닌 데이터 생성 규칙을 통한 무한의 데이터를 생성할 수 있는 python 코드를 AI를 통해 생성합니다. 그렇기 때문에 AI 사용 비용이 최초 1번만 소요되고, AI보다 프로그램 수행 속도가 훨씬 빠르기 때문에 매우 빠르게 대량의 데이터를 생성할 수 있습니다.
  - 테이블 1건(컬럼 수:22개)에 대해 1,000,000건의 mock data를 72.38초만에 생성하였습니다.

- 개선 사항: smartbear의 컨셉과 비슷하기 때문에 용어는 거기서 따와도 좋을 듯

#### 사내 LangChain 강의 진행

- 기간: 2024/06 - 2024/07 (1.5M, 10회)
- 강의 내용: LangChain 기초+응용 스터디 자료 준비 및 진행
- 주요 성과
  - 다른 사람들이 AI를 우리 회사 사업에 적용할 수 있는 아이디어를 낼 수 있었음
  - 실제 개발이 되었음 - blue hound, oibdd

#### CS Portal : MA(유지보수 계약) 고객사를 위한 이슈 트래킹 웹 사이트

- 기간: 2024/01 - 2024/04 (4M)
- 프로그램 개요:
- 기술 스택:
- 주요 성과

- 리액티브 프로그래밍
- Java17 + springboot3
- 멀티 모듈
- gitlab service desk 연동
- 리액트 (MUI)

#### Simulator Version 2 : 대외 연계 시뮬레이터 솔루션 버전 업 진행

- 기간: 2023/08 - 2023/12 (5M)
- 프로그램 개요:
- 기술 스택:
- 주요 성과

- 어려웠던 기존 소스 파악하는 능력 배움
- TCP 거래에 대해 알게됨
- REST API 거래 시뮬레이터 개발하며, REST API HTTP, HTTPS 통신에 대한 공부 할 수 있었음

#### LINE Bank Japan 코어뱅킹 개발

- 기간: 2021/04 - 2021/04 (13M)
- 프로그램 개요:
- 기술 스택:
- 주요 성과

- 고객팀 업무 개발
- 자동화 통합 테스트 진행
