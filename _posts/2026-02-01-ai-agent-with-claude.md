---
title: Claude API로 AI 에이전트 만들기
date: 2026-02-01 10:00:00 +0900
categories: [AI, 에이전트]
tags: [ai, claude, llm, python, 자동화]
---

## AI 에이전트란?

단순히 질문에 답하는 것을 넘어, **도구를 사용**하고 **목표를 향해 반복적으로 행동**하는 AI 시스템입니다.

## Claude API 기본 사용

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "파이썬으로 피보나치 수열을 구현해줘"}
    ]
)

print(message.content[0].text)
```

## Tool Use (함수 호출)

```python
tools = [
    {
        "name": "get_weather",
        "description": "특정 도시의 현재 날씨를 가져옵니다",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "도시명"}
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "서울 날씨 알려줘"}]
)
```

## 에이전트 루프 구현

```python
def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return response.content[0].text

        if response.stop_reason == "tool_use":
            # 도구 실행 후 결과를 메시지에 추가
            tool_results = execute_tools(response.content)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
```

## 활용 사례

- 뉴스 자동 수집 및 요약
- 주식 데이터 분석 에이전트
- 코드 리뷰 자동화
- 문서 작성 보조
