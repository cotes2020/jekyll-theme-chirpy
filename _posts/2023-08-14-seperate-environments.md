---
title: 개발, 프로덕션, 테스트 환경 분리 대작전(Package.json 과 dotenv)
date: 2023-08-14
categories: [troubleshooting]
tags: [.env, node.js, nest.js, package.json, dotenv]
---

## 🤔 Problem

로컬에서 개발을 하면서 Dev 용 데이터베이스와 Test 용 데이터베이스를 분리해서 진행하고 싶었다.

그래서 도커로 각각 따로따로 데이터베이스를 띄워두고 사용했는데 제법 편리했다. dotenv 와 package.json 으로 명령어를 분리하니 어렵지 않게 구현이 가능했다.

그러던 중 배포서버에서 Oauth 로그인을 진행하려면 별도의 환경변수와 redirect URL을 기록해야 했기 때문에 각각의 환경에서 사용할 .env 파일과 npm 명령어를 나누어 작성하는 테크닉을 사용하기로 했다.

## 🌱 Solution

<iframe width="560" height="315" src="https://www.youtube.com/embed/GHTA143_b-s" title="NestJs Course for Beginners - Create a REST API" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

1.  Dotenv 패키지를 설치한다.

    `npm install dotenv --save`

    <br/>

2.  각각의 환경에 맞는 환경변수 파일을 작성한다.

    ```zsh
      .env.development
      .env.test
      .env.production
    ```

    <br/>

3.  package.json 의 Script 부분에 해당 .env 에 맞는 파일을 사용하게끔 명령어를 작성한다.

    ```zsh
    $ dotenv -e ./.env.development -- <<명령어>>
    ```

    <details markdown="block"><summary>package.json 코드</summary>

    ```py
    "scripts": {

      # 개발환경 --------------------------------------------------------------------------

      # 개발 환경용 데이터베이스 컨테이너 제거
      "db:dev:rm": "docker compose rm dev-db -s -f -v",

      # 개발 환경용 데이터베이스 컨테이너 실행
      "db:dev:up": "docker compose up dev-db -d",

      # 상기 명령어 조합 : 개발 환경용 데이터베이스 컨테이너 제거 후 재시작 및 Prisma 마이그레이션 실행
      "db:dev:restart": "npm run db:dev:rm && npm run db:dev:up && sleep 1 && npm run prisma:dev:deploy",

      # Prisma 스튜디오 실행 (개발 환경)
      "prisma:dev:studio": "dotenv -e ./.env.development -- npx prisma studio",

      # 개발 환경용 Prisma 마이그레이션 실행
      "prisma:dev:deploy": "dotenv -e ./.env.development -- prisma migrate dev",

      # 개발 환경 설정을 적용한 개발 모드로 Nest.js 앱 실행
      "dev": "dotenv -e .env.development -- nest start --watch",


      # 테스트 환경 --------------------------------------------------------------------------

      # 테스트 환경용 Prisma 마이그레이션 실행
      "prisma:test:deploy": "dotenv -e ./.env.test -- prisma migrate deploy",

      # 테스트 환경용 데이터베이스 컨테이너 제거
      "db:test:rm": "docker compose rm test-db -s -f -v",

      # 테스트 환경용 데이터베이스 컨테이너 실행
      "db:test:up": "docker compose up test-db -d",

      # 테스트 환경용 데이터베이스 컨테이너 제거 후 재시작 및 Prisma 마이그레이션 실행
      "db:test:restart": "npm run db:test:rm && npm run db:test:up && sleep 1 && npm run prisma:test:deploy",

      # 엔드투엔드 테스트 전에 테스트 환경용 데이터베이스 컨테이너 재시작
      "pretest:e2e": "npm run db:test:restart",

      # 엔드투엔드 테스트 실행
      "test:e2e": "dotenv -e ./.env.test -- jest --watch --no-cache --config ./test/jest-e2e.json",

      # Prisma 모델 코드 생성
      "prisma:generate": "npx prisma generate",


      # 프로덕션 환경 --------------------------------------------------------------------------

      # 프로덕션 환경에서 Prisma 마이그레이션 실행 및 앱 실행 (Dockerfile 및 docker-compose.yml에서 사용)
      "start:migrate:prod": "prisma migrate deploy && npm run start:prod"

      # 개발 모드로 Nest.js 앱 실행
      "start:dev": "nest start --watch",
    }
    ```

    </details>

    <br/>

4.  Dockerfile에 해당 환경에 맞는 환경 변수를 추가한다.

    <details markdown="block"><summary>Dockerfile 코드</summary>

    ```Dockerfile
    # 개발용 스테이지
    FROM node:18-alpine AS development

    WORKDIR /app

    # package.json과 package-lock.json 파일 복사
    COPY package\*.json ./

    # 종속성 설치
    RUN npm install

    # 생성된 Prisma 파일 복사
    COPY prisma ./prisma/

    # 환경 변수 복사
    COPY .env.production ./

    # tsconfig.json 파일 복사
    COPY tsconfig.json ./

    # 소스 코드 복사
    COPY . .

    # 서버를 포트 5000으로 실행하도록 설정
    EXPOSE 5000

    # 마이그레이션을 포함하여 시작 스크립트 실행
    CMD ["npm", "run", "start:migrate:prod"]
    ```

    </details>

## 📎 Related articles

| 이슈명                                          | 링크                                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------ |
| NestJs Course for Beginners - Create a REST API | [https://www.youtube.com/watch?v=GHTA143_b-s](https://www.youtube.com/watch?v=GHTA143_b-s) |
