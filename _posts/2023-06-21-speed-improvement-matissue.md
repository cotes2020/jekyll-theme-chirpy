---
title: 맛이슈 속도 개선 대작전
date: 2023-06-21
categories: [troubleshooting]
layout: post
tags: [fastapi, improvement, mongodb, redis]
---

## 🤔 Problem

맛이슈 사이트가 로그인, 댓글불러오기 등 여러방면에서 속도가 느린 것을 확인했다.

- 백엔드 서버와 레디스 서버를 도커를 하나의 머신에서 실행
  유저부분 로딩 속도를 개선하여 로그인시 기존 7초 가량 에서 3초전후로 **절반 이상의 시간을 절약**하였습니다.

---

- 몽고디비의 파이프라인 기능을 이용
  개선전엔 **최대 10초 이상의 시간이 걸렸지만,** 현재는 **193.40 밀리초**가 되어 **98.066% 의 성능 향상**을 이끌었습니다.
  - **원인**
    크롬 네트워크 탭에서 어느 api 요청이 느린지 확인하였더니 백엔드에서 댓글을 불러올때 속도가 느린 것을 확인했다. (전체 게시글을 불러오는 경우 최대 15초 소요까지 확인)
    기존 코드에선 댓글을 가져올때 게시물과 댓글을 각각 db에 직접 접근하는 방식을 사용하고 있었고, 전체 게시물을 가져올 땐 모든 게시물의 length 만큼 for문으로 db 접근을 반복하고 있었다.
    해당 기간에는 2가지 해결방안을 사용했는데,
    1. 백엔드 서버에서 데이터베이스에 접근하는 시간을 줄이는 것
    - 아틀라스 서버와 백엔드 배포 서버를 동일 리전으로 이주 (싱가폴 무료서버에서 gcp 서울 서버로 이사하는 것으로 해결)
    2. 데이터베이스 자체에 접근하는 횟수를 줄이는 것
    - MongoDB Atlas 에서 제공하는 **pipeline**으로 해결할 수 있었다.
      여러개의 작업을 중첩으로 처리하여 작업 처리량과 시간적 효율을 향상시킨다.
      백엔드 엔지니어로서 api의 호출을 줄일 수 있는 방법, db 접근 횟수를 줄일 수 있는 방법 등 여러 방식으로 성능 개선이 가능하다는 것을 알 수 있었다.
      추가로 리팩토링 시 **캐싱**을 적용해볼 예정이다.

## 🌱 Solution

1. BE ) 도커이미지로 **백엔드서버와 redis 서버**를 하나의 머신에서 실행

   - 상세
     도커 설치 공식홈페이지
     [https://docs.docker.com/desktop/install/linux-install/](https://docs.docker.com/desktop/install/linux-install/)
     설치 후 프로젝트 루트 폴더에 docker-compose.yml 을 위치시키고 실행합니다.

     ```bash
      $ cd dev-be
      $ sudo docker-compose up --build
     ```

     - docker-compose.yml

       ```bash
       version: "3.8"

       services:
         web:
           build: .
           labels:
             - "traefik.http.routers.web.rule=Host(`<<도메인>>`)"
             - "traefik.http.routers.web.entrypoints=websecure"
             - "traefik.http.routers.web.tls.certresolver=myresolver"
           depends_on:
             - redis
           environment:
             - MONGO_DB_URL=<<MONGO_DB_URL>>
             - MONGO_DB_NAME=FAST
             - REDIS_URL=redis://redis:6379
             - SMTP_SERVER=mail.gmx.com
             - SMTP_PORT=465
             - SENDER_EMAIL=matissue@gmx.com
             - SMTP_PASSWORD=<<SMTP_PASSWORD>>

         redis:
           image: "redis:latest"

         traefik:
           image: "traefik:v2.0"
           command:
             - "--api.insecure=false"
             - "--providers.docker=true"
             - "--entrypoints.web.address=:80"
             - "--entrypoints.websecure.address=:443"
             - "--certificatesresolvers.myresolver.acme=true"
             - "--certificatesresolvers.myresolver.acme.email=matissue@gmx.com"
             - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
             - "--certificatesresolvers.myresolver.acme.httpchallenge.entrypoint=web"
           ports:
             - "80:80"
             - "443:443"
           volumes:
             - "/var/run/docker.sock:/var/run/docker.sock"
             - "./letsencrypt:/letsencrypt"

       ```

2. BE ) **파이프 라인**을 사용하여 레시피를 가져오는 쿼리를 개선
   - 깃허브 코드
     기존 데이터베이스에는 여러번 접근하기도 하고 메서드가 끝날때마다 새로 함수를 시작했다면, 파이프라인으로 순차적으로 쿼리를 실행하게 작성하였습니다.
     ```bash
     async def get_all_recipes_with_comments(self, skip: int = 0, limit: int = 160):
             pipeline = [
                 {"$sort": {"created_at": -1}},
                 {"$skip": skip},
                 {"$limit": limit},
                 {
                     "$lookup": {
                         "from": "comments",
                         "localField": "recipe_id",
                         "foreignField": "comment_parent",
                         "as": "comments"
                     }
                 }
             ]
             result = await self.collection.aggregate(pipeline).to_list(length=None)
             return result
     ```
     **깃허브 링크**
     [🔗기존코드](https://github.com/YubinShin/matissue-BE/blob/8766a33fedae29baa51fd6de711bf41033d1fe74/dao/recipe_dao.py)
     [🔗속도개선코드](https://github.com/YubinShin/matissue-BE/blob/2b0e8c1a4e1dfa4032e8ea56b33f324bd5d33116/dao/recipe_dao.py)
3. FE ) **Next/image 태그**로 변경하기

   [Next/Image를 활용한 이미지 최적화 | 카카오엔터테인먼트 FE 기술블로그](https://fe-developers.kakaoent.com/2022/220714-next-image/)

## 📎 Related articles

| 이슈명                        | 링크                                                              |
| ----------------------------- | ----------------------------------------------------------------- |
| 우분투에 도커 설치하기        | https://velog.io/@osk3856/Docker-Ubuntu-22.04-Docker-Installation |
| 카카오 기술 블로그 next-image | https://fe-developers.kakaoent.com/2022/220714-next-image/        |
