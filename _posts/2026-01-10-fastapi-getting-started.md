---
title: FastAPI 시작하기 — 기초부터 배포까지
date: 2026-01-10 10:00:00 +0900
categories: [백엔드, FastAPI]
tags: [python, fastapi, rest-api, backend]
---

## FastAPI란?

FastAPI는 Python 기반의 현대적인 웹 프레임워크로, 빠른 속도와 자동 문서화가 특징입니다.

## 설치

```bash
pip install fastapi uvicorn
```

## 기본 예제

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

## 실행

```bash
uvicorn main:app --reload
```

`http://localhost:8000/docs` 에서 자동 생성된 Swagger UI를 확인할 수 있습니다.

## Pydantic으로 데이터 검증

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
def create_item(item: Item):
    return item
```

FastAPI는 Pydantic을 통해 입력 데이터를 자동으로 검증하고 타입 변환합니다.

## 다음 단계

- 데이터베이스 연동 (SQLAlchemy / asyncpg)
- 인증/인가 (JWT)
- Docker로 배포
