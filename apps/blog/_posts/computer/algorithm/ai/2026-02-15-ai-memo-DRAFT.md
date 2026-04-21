---
title: "AI"
# description: ""
categories: [컴퓨터, 알고리즘, AI]
tags: [AI]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2026-02-15. 15:42 # Init
# last_modified_at: 2026-02-15. 15:42
---

## AI Memo

- AWS Bedrock
  - 분산된 여러 가지 회사의 모델들을 한 곳에서 사용할 수 있게 해주는 서비스
- Anthropic
  - AI 모델을 개발하는 회사
  - Claude OPUS, Claude Sonnet
  - OpenAI가 Microsoft 투자 받고 영리 BM 만든 것을 계기로, 몇몇 직원이 나와서 만든 회사. Google의 지원을 받으며 OpenAI & Microsoft v.s. Anthropic & Google 구도가.
- Cursor
  - Max Mode
  - VSCode 기반
- .md
  - <https://github.com/forrestchang/andrej-karpathy-skills/blob/main/CLAUDE.md>
  - <https://news.hada.io/topic?id=26655>
- OpenClaw
- 학습
  - [Anthropic 실험](https://news.hada.io/topic?id=26364)
  - 뇌 토큰
  - 에이전트를 ‘지능’이 아닌 ‘성실함’을 자동화하는 존재로 보는 게 맞음

- 똑같은 음식을 만들어서 배달하는 음식점인데, 요즘 자동화기계를 통해서 수익성을 올리는 음식점들이 많거든요. 근데 그런곳이라고 해서 특별히 가격이 싼것도 아니고, 자동화로 만드는 업체다라고 표기를 해야하거나, 요구하는걸 본적이 없는데, 왜 게임에만 이런식으로 반응하게 되는걸까? 라는 것에 대해서 좀 고민하고 있습니다. 올바른 비유인지는 모르겠지만 조리 과정이 프로그래밍이고 음식 재료가 리소스라고 생각합니다. 그래서 조리 과정 자체가 ai로 대체되는건 괜찮지만 음식 재료가 배양육이나 대체 가공품으로 섭취되는 것에 대한 거부감이지 않을까라는 생각입니다. ai태그는 없어도 되지않을까요. ai태그가 없을때 부작용을 생각하면, 이겜 플레이중에 ai티가 나서 구져가 아니라, 나 이겜 재밋고 아트와 스토리에 감동도 느꼈지만 알고보니 에이아이라 나를 기만한 느낌이 들어 인거같은데.. 이건 사실 '팡션 쓰지마세요'랑 같은 감성으로 저에겐 좀 다가오는거 같아요
- cot
- 활용
  - 영상을 NotebookLM으로 슬라이드 화
  - 더 깊은 내용이 궁금하면 보고서화
  - 영상 예시가 필요하면 그때 유튜브를 보기
  - 제미나이 CS 퀴즈 (면접 질문 같은 것)

명령을 잘 내려ㅎ야 할 듯  
명령 조각을 만들어서 블럭 조립하듯  
기억 조가  

### Cursor Q&A 요약 (2026년 3월 기준)

- **Pro 사용 방법**
  - Chat: `Ctrl+L` / 인라인 편집: `Ctrl+K` / Composer: `Ctrl+I` / Tab 자동완성: `Tab`
  - 사용량: Settings → General → Account (또는 [usage dashboard](https://cursor.com/dashboard?tab=usage))

- **Fast Premium vs Slow Premium**
  - Fast: 우선 처리·즉시 응답, 월 한도(예: Pro 500회). 특정 모델 선택 또는 Premium 선택 시.
  - Slow: Fast 한도 소진 후 같은 프리미엄 모델 사용 시. 무제한이지만 피크 시 대기열(5~60초~수 분).

- **사용량 풀 (현재 정책)**
  - **API 풀**: 특정 모델 또는 Premium 선택 시 차감. Pro는 월 $20 포함.
  - **Auto + Composer 풀**: Auto 또는 Composer 1.5 선택 시 차감. 별도 풀, "넉넉한 포함량".
  - Auto 사용 시 **500회가 아닌** Auto+Composer 풀에서 차감됨.

- **무제한·대기열 없는 모델**
  - 예전에 알려진 cursor-small, cursor-fast, gpt-4o-mini 등은 **2026년 3월 기준 Models UI/공식 목록에 없음**. (레거시/내부 이름 또는 정책 변경.)
  - Auto 선택 시 Cursor가 내부적으로 저렴·빠른 모델을 골라 씀. 직접 선택 메뉴에는 없음.

- **모델 옆 뇌 아이콘**
  - **Thinking / Reasoning** 지원 모델. 추론 단계(chain-of-thought) 후 응답. 복잡한 추론에 유리, 비용·토큰 더 소모 가능.

- **효율적 사용**
  - 계획 먼저(Plan 모드), 컨텍스트는 필요한 만큼만, 프롬프트 구조화(맥락·의도·제약·형식), 간단한 작업은 Auto·무제한 계열 위주로.

### _

- IDE에서 점차 CLI로
  - 직접 코드를 작성하거나 수정하기 않고, AI가 작업한 내용을 컨펌하는 의사결정이 주가 됨. last_modified_at: 2026-04-06. 02:38
