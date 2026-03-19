---
title: FastAPI 보안 — OWASP Top 10 대응 방법
date: 2026-02-20 10:00:00 +0900
categories: [보안, FastAPI]
tags: [security, owasp, fastapi, python, 취약점]
---

## 1. SQL 인젝션 방지

**절대 f-string으로 쿼리 만들지 마세요.**

```python
# 위험
query = f"SELECT * FROM users WHERE name = '{user_input}'"

# 안전 (파라미터 바인딩)
result = await db.fetch("SELECT * FROM users WHERE name = $1", user_input)
```

## 2. 깨진 인증 (Broken Authentication)

```python
# 비밀번호 해싱 필수
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```

## 3. 민감 데이터 노출 방지

```python
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    # password 필드는 응답 스키마에 포함하지 않음

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    ...
```

## 4. 보안 헤더 추가

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com"])

# CORS 제한
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # "*" 사용 금지
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 5. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/login")
@limiter.limit("5/minute")
async def login(request: Request):
    ...
```

## 환경변수 관리

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    secret_key: str
    database_url: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
```

`.env` 파일은 반드시 `.gitignore`에 추가하세요.
