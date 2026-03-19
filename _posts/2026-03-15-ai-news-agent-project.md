---
title: "프로젝트: AI 뉴스 에이전트 개발기"
date: 2026-03-15 10:00:00 +0900
categories: [프로젝트, AI]
tags: [ai, python, 자동화, claude, 프로젝트]
---

## 프로젝트 개요

매일 수십 개의 뉴스 중에서 투자/기술 관련 핵심 뉴스를 자동으로 수집하고 요약해주는 AI 에이전트를 만들었습니다.

## 핵심 기능

- 다양한 뉴스 소스 크롤링 (RSS, API)
- Claude AI로 뉴스 요약 및 중요도 분류
- 카테고리별 필터링 (방산, 반도체, AI, 금융)
- 매일 오전 7시 자동 발송 (이메일/슬랙)

## 아키텍처

```
뉴스 수집 (크롤러)
    ↓
데이터 전처리 (중복 제거, 필터링)
    ↓
Claude API 요약 및 분류
    ↓
PostgreSQL 저장
    ↓
알림 발송 (이메일/슬랙)
```

## 뉴스 수집 코드

```python
import feedparser
import httpx
from datetime import datetime

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.investing.com/rss/news.rss",
]

async def fetch_rss_news() -> list[dict]:
    news_list = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            news_list.append({
                "title": entry.title,
                "summary": entry.get("summary", ""),
                "link": entry.link,
                "published": entry.get("published", ""),
            })
    return news_list
```

## Claude로 뉴스 요약

```python
import anthropic

client = anthropic.Anthropic()

async def summarize_news(news_list: list[dict]) -> str:
    news_text = "\n\n".join([
        f"제목: {n['title']}\n내용: {n['summary']}"
        for n in news_list
    ])

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""다음 뉴스들을 분석하고 투자자 관점에서 중요한 순서로 요약해주세요.
각 뉴스에 대해 중요도(상/중/하)와 핵심 포인트를 정리해주세요.

{news_text}"""
        }]
    )
    return response.content[0].text
```

## 배운 점

1. RSS 피드마다 형식이 달라 전처리가 까다로움
2. Claude의 요약 품질이 뉴스 도메인에서 매우 뛰어남
3. 중복 뉴스 제거에 제목 유사도 체크(TF-IDF)가 효과적
4. 스케줄링은 `APScheduler`보다 `Celery + Redis` 조합이 안정적

## 앞으로 개선할 것

- [ ] 종목별 관련 뉴스 추적
- [ ] 감성 분석 추가
- [ ] 웹 대시보드 구축
