---
title: FastAPI JWT 인증 구현하기
date: 2026-01-20 10:00:00 +0900
categories: [백엔드, FastAPI]
tags: [python, fastapi, jwt, 인증, security]
---

## JWT 인증 흐름

1. 클라이언트가 ID/PW로 로그인 요청
2. 서버가 JWT 토큰 발급
3. 클라이언트는 이후 요청에 토큰 첨부
4. 서버가 토큰 검증 후 응답

## 설치

```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

## 토큰 생성

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
```

## 보안 엔드포인트

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401)
    return username

@app.get("/users/me")
async def read_users_me(current_user = Depends(get_current_user)):
    return {"user": current_user}
```

## 주의사항

- `SECRET_KEY`는 환경변수로 관리
- 토큰 만료 시간을 짧게 유지 (30분 권장)
- Refresh Token 패턴 병행 사용 권장
