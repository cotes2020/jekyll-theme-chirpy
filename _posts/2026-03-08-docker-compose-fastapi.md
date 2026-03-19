---
title: Docker Compose로 FastAPI + PostgreSQL 배포하기
date: 2026-03-08 10:00:00 +0900
categories: [백엔드, 배포]
tags: [docker, fastapi, postgresql, devops, 배포]
---

## 프로젝트 구조

```
project/
├── app/
│   ├── main.py
│   ├── models.py
│   └── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## docker-compose.yml

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydb
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 5s
      timeout: 5s
      retries: 5

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certbot/conf:/etc/letsencrypt
    depends_on:
      - api

volumes:
  postgres_data:
```

## 실행

```bash
# 빌드 및 시작
docker-compose up -d --build

# 로그 확인
docker-compose logs -f api

# 데이터베이스 접속
docker-compose exec db psql -U user -d mydb

# 중지
docker-compose down
```

## .env 파일 관리

```bash
# .env
DATABASE_URL=postgresql://user:strongpassword@db:5432/mydb
SECRET_KEY=your-very-secret-key-here
DEBUG=false
```

```yaml
# docker-compose.yml에서 env_file 사용
services:
  api:
    env_file:
      - .env
```

## GitHub Actions 자동 배포

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd /app
            git pull
            docker-compose up -d --build
```
