---
title: "AI Engineer가 되기 위한 Roadmap"
categories: [Medium]
tags: [AI, AI Roadmap]
---

## AI Engineer가 되기 위한 Roadmap

> Skills, learning resources, and project ideas to become an AI Engineer in 2024

[medium 원문](https://medium.com/@dswharshit/roadmap-to-become-an-ai-engineer-roadmap-6d9558d970cf)

### 전제 조건

1. python/js 프로그래밍에 대한 중간정도의 이해
2. Flask/Rails/Node.js를 통해 2-3개 정도의 적당한 복잡한 수준(블로그앱)의 코딩 개발 경험
3. documentation 을 편안히 읽을 수 있는 능력
4. IDE(vs code)를 편안히 사용
5. 깃헙, 깃 사용 <- 하지만 공부 하면서 자연스레 익힐 수 있음

## Roadmap

![Image]({{"/assets/img/posts/2024-12-21-17-06-11-i6pmg6.png" | relative_url }})

### **Beginner**

> LLM API를 활용한 기본 앱 개발 <- 앱에 적합한 프롬프트 엔지니어링을 고려하며 해보세요!

- LLM의 기본 이해 - ChatGPT이 어떻게 높은 수준으로 작동하는지 알기
- 개발자를 위한 Prompt Engineering 배우기 - LLM의 답변을 향상시키는 프롬프트 작성법
- API에서 데이터(특히 JSON)가 이용되는 방법 배우기
- closed LLM과 open-source LLM의 함수 호출, 프롬프트 전달, 응답 파싱 방법 배우기
- 대화에서 context space를 관리하는 방법 배우기
- 작업을 생성 / 자동화 방법 배우기 - 랭체인 사용
- Gradio나 Streamlit을 이용해서 간단한 POC/demo app 만들어보기
- 접근가능하도록 app을 배포하기 - HuggingFace Space혹은 Streamlit Cloud를 통한 기본 배포 이용
- 멀티 모달 생성 - HuggingFace `transformer` library를 통해 code, 이미지, 음성을 이용

### **Intermediate**

> - RAG를 이용해서 컨텍스트 인식을 더 잘하는 앱을 개발
> - vector DB에 대해 배우고, 어떻게 동작하는지에 대해 배워보세요
> - LLM을 이용한 agent와 tool을 개발하는 것에 대해 배워보세요

- vector 임베딩과 DB에 대해 이해하기
- 나의 app에 vector db 적용하는 법 배우기
- RAG 생성해보기
- 고급 RAG 파이프라인 개발하기 - 여러 데이터 소스를 거친 후 응답을 제공할 수 있도록 하위 질문 쿼리 엔진 구축
- Agent 구축 - 반복적 워크플로우를 통한 큰 작업 수행
- Multi Agent 어플리케이션 구축 - 단일 Agent보다 서로 더 좋은 솔루션을 제공할 수 있는 협력 Agent 구축
- Multi Agent를 통한 자동화 - Autogen, CrewAI
- RAG 평가 - RAGAs framework
- DB관리 , retrieval, 완성된 app 배포, versioning, logging, 모델 행동 monitoring

### **Advanced**

> - app개발 마스터 후, 배포/최적화/운영에 대해 배워보세요 => LLMOps
> - 사전 학습된 모델을 파인튜닝을 해서 다운스트림 애플리케이션에 효율적이고 저렴한 비용하고 적용하는 방법을 배워보세요

- domain-specific knowledge를 위한 파인튜닝 - 의료연구, 금융연구, 법률분석과 같은 맞춤형 대응(tailored reponses)
- 모델 미세 조정을 위한 dataset 및 엔지니어(ETL 파이프라인) 파이프라인 큐레이션
- 모델 성능에 대한 평가 및 벤치마킹
- LLMOps - 모델 레지스트리, observablity, 자동화 테스트를 통해 완전한 e2e 파이프라인 구축
- 멀티모달 어플리케이션 구축 - 텍스트와 이미지에 대해 하이브리드 시맨틱 검색
- 다른 개발자가 사용 가능한 SDK 패키지 및 맞춤형 솔루션 구축
- AI app 보안 - 프롬프트 해킹과 같은 기술을 통해 취약점과 잠재적 위험을 확인하여 방어조치를 취함

![Image]({{"/assets/img/posts/2024-12-21-17-07-00-i7pmg6.png" | relative_url }})

### [Learning resources, references and projects](https://github.com/dswh/ai-engineer-roadmap) 📚

위 깃헙에 다양한 리소스와 각 컨셉별 학습을 위한 좋은 프로젝트들이 있으니 참고하세요.

---

저는 Beginner와 Intermediate 사이 어디쯤엔가 있는 것 같군요..
제공된 github 자료들을 통해 하나하나 도장깨기 해가며 공부하면 많이 배울 수 있을 것 같아요!
