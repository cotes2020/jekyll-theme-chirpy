---
title: 깃허브 액션으로 CI&CD대작전
date: 2023-08-17
categories: [troubleshooting]
tags: [CI/CD, docker, github-actions]
---

## 🤔 Problem

백엔드 api 서버와 DB와 https 인증까지 한번에 돌려야하는 상황이기에 **컨테이너**를 사용해야했다.

컨테이너로 올리기 위해 도커이미지 빌드를 진행했는데 m1 아이맥 기준, 1회 빌드하는데 200여초가 걸렸다.

사실 빌드가 되는 것도 대단했다.

가엾은 내 아이맥은 자신의 로컬에서 백엔드서버도 켜고, 프론트엔드 확인용으로 리액트 실행도 하고, postgresql db 도 켜면서, 웹브라우저로 검색까지 해주고 있었다.

조금만 힘들게 하면 vscode 와 연결된 터미널 세션이 자꾸 멈춰버렸다.

아무리 도커파일에서 멀티스테이징 빌드를 하게끔 최적화를 하고 nodejs 의존성을 alpine 버전으로 설치해도 빌드시간이 요지부동인걸 확인했고,

이 상태에선 프론트엔드 담당자들과 실시간으로 맞춰가며 개발을 할 수 없겠단 판단이 섰다.

그래서 **깃허브 actions** 를 사용해 빌드서버를 따로 만들었고, 빌드 시간이 총 200초에서 50초로 줄어들었다.

최근에 시스템디자인 서적을 읽으면서 왜 그냥 배포 컴퓨터에서 빌드하지 않고 굳이 빌드 서버를 따로 쓸까 궁금했던 경험이 있는데, 이번에 극락을 맛보고 나서 왜 따로 쓰는지 알 수 있게 됐다.

## 🌱 Solution

### CI (Continuos Integration)

<iframe width="560" height="315" src="https://www.youtube.com/embed/JS07npwL3Ps" title="Continuous Deployment using GitHub Actions | Auto Deploy MERN Stack | AWS EC2 | CICD Pipeline" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

1. main.yml 작성

   [https://github.com/fog-of-war/dev-be/commit/243505be92f9cf2cfb1bbd49ba49f700d42ed53e](https://github.com/fog-of-war/dev-be/commit/243505be92f9cf2cfb1bbd49ba49f700d42ed53e)

2. github secret 과 Dockerfile 연결

   [Dockerfile](https://github.com/fog-of-war/dev-be/blob/dev/Dockerfile)

3. ~~github actions runner~~

   ec2의 저장용량이 작아서 실패

### CD (Continuos Delivery)

<div markdown="block" style="width: 30%;">
![https://raw.githubusercontent.com/containrrr/watchtower/main/logo.png](https://raw.githubusercontent.com/containrrr/watchtower/main/logo.png)
</div>

#### watchtower

docker-compose.yml 하단에 watchtower 라는 이미지를 설치하여 사용했다.

프론트분들과 개발용으로 사용하는 머신엔 `WATCHTOWER_POLL_INTERVAL` 인터벌을 60초로 주어 깃허브 빌드서버에 최신 이미지가 올라갈때마다 바로바로 받아올 수 있게 하였다.

실제 배포용으로 사용하는 머신엔 12시간 인터벌을 주어 안정된 버전을 받아올 수 있게 하였다.

```yaml
watchtower:
  image: containrrr/watchtower
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
  environment:
    TZ: Asia/Seoul
    WATCHTOWER_CLEANUP: "true"
    WATCHTOWER_POLL_INTERVAL: 60
  restart: unless-stopped
```

<details markdown="block"><summary>개발 머신용 docker-compose.yml 전문</summary>
```yaml
version: "3.8"
services:
  web:
    image: shinyubin/fow-be
    container_name: fow-be
    restart: always
    labels:
      - "com.centurylinklabs.watchtower.enable=true"
      - "traefik.enable=true"
      - "traefik.http.routers.web.rule=Host(`api.yubinhome.com`)"
      - "traefik.http.routers.web.entrypoints=websecure"
      - "traefik.http.routers.web.tls.certresolver=myresolver"
    ports:
      - "5000:5000"
    volumes:
      - .:/usr/src/app
      - /usr/src/app/node_modules
    command: sh -c "npx prisma migrate dev && npm run start:dev"
    environment:
      - DATABASE_URL={DATABASE_URL}
      - JWT_SECRET={JWT_SECRET}
      - GOOGLE_OAUTH_ID={GOOGLE_OAUTH_ID}
      - GOOGLE_OAUTH_SECRET={GOOGLE_OAUTH_SECRET}
      - KAKAO_CLIENT_ID={KAKAO_CLIENT_ID}
      - NAVER_CLIENT_ID={NAVER_CLIENT_ID}
      - NAVER_CLIENT_PW={NAVER_CLIENT_PW}
      - GOOGLE_REDIRECT_URL={GOOGLE_REDIRECT_URL}
      - KAKAO_REDIRECT_URL={KAKAO_REDIRECT_URL}
      - NAVER_REDIRECT_URL={NAVER_REDIRECT_URL}
    networks:
      - freecodecamp
    depends_on:
      - dev-db
  dev-db:
    image: postgres:13
    ports:
      - 5432:5432
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: 123
      POSTGRES_DB: nest
    networks:
      - freecodecamp
  traefik:
    image: "traefik:v2.0"
    command:
      - "--api.insecure=false"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.httpchallenge=true"
      - "--certificatesresolvers.myresolver.acme.httpchallenge.entrypoint=web"
      - "--certificatesresolvers.myresolver.acme.email=fogofseoul@gmail.com"
      - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - "./letsencrypt:/letsencrypt"
      - "/var/run/docker.sock:/var/run/docker.sock"
    networks:
      - freecodecamp
  watchtower:
    image: containrrr/watchtower
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      TZ: Asia/Seoul
      WATCHTOWER_CLEANUP: "true"
      WATCHTOWER_POLL_INTERVAL: 60
    restart: unless-stopped
networks:
  freecodecamp:
```
</details>
<br/><br/>
> 23.08.27 추가
>
> 1분 마다 pull 해오는 것은 도커 규정에 어긋났다.
>
> 간격을 좀 더 늘려야겠다.
>
> 6시간동안 익명사용자 100회 제한, 1계정당 200회 제한
>
> `Many Requests. You have reached your pull rate limit.`
> 
> [https://www.docker.com/increase-rate-limits/](https://www.docker.com/increase-rate-limits/)

<br/><br/>

## 📎 Related articles

| 이슈명                                         | 링크                                                                                       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Continuous Deployment using GitHub Actions ... | [https://www.youtube.com/watch?v=JS07npwL3Ps](https://www.youtube.com/watch?v=JS07npwL3Ps) |
| watchtower                                     | [https://containrrr.dev/watchtower/](https://containrrr.dev/watchtower/)                   |
