---
title: CORS 에러 피하는 proxy server 만들기 (Feat. CORS Anywhere & Heroku)
author: Bean
date: 2022-04-12 16:32:00 +0800
categories: [Web frontend, React]
tags: [Catty]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: True
cover: assets/img/post_images/catty_cover2.png
---

## CORS란?
&nbsp;

프론트개발을 하다보면 CORS 에러를 정말정말 많이 접하게 된다. CORS는 cross-origin resource sharing 의 약자로 교차 출처 리소스 공유라고 표현된다.

이 CORS 문제가 발생하는 이유는 한마디로 **보안** 때문이다.

CORS는 내가 운영하는 웹사이트에서 (피싱 위험이 있는) 다른 웹사이트의 request가 넘어오게 되면, 응답을 받지 않고 에러가 발생되도록 한다.

아무런 CORS 설정을 하지 않으면 **동일 도메인** + **동일 포트**로 넘어온 request는 잘 받지만 그렇지 않은 경우 모두 요청을 거절하게 된다.

많이 쓰는 AWS S3도 사실 내 웹사이트에서 S3가 위치하는 외부 도메인으로 요청을 보내 그곳에 저장된 데이터를 받아오는 것이기 때문에 아무런 설정이 없으면 CORS 에러가 발생된다.

&nbsp;
## 이런 CORS 에러는 서버를 건드릴 수 있다면 서버를 수정하는 게 제일 간단하다.
&nbsp;

HTTP 응답헤더 Access-Control-Allow-Origin : * 혹은 Access-Control-Allow-Origin: 허용하고자 하는 도메인 설정해주면 해결이 되며, express에서는 이를 쉽게 해결해주는 미들웨어를 제공해준다.

```
const express = require('express')
const cors = require('cors')
const app = express()

app.use(cors())
```

언급한 S3 CORS 에러의 경우 S3 버킷 설정을 수정하여 해결할 수 있고 이 내용은 [AWS S3 쉽게 사용하자]()에서 다루고 있다.

그렇지만 이번에 CORS 에러를 해결해야 하는 경우는 Catty 웹앱에서 유저가 북마크한 웹사이트 주소의 html 응답을 받아와야 하는 경우라 서버 수정이 불가능한 경우였다. (크롬 익스텐션에서 웹사이트를 북마크하면 readable한 html이 함께 저장되지만-[크롬 익스텐션으로 웹사이트 북마크해서 read mode로 읽기]() 참고) 웹앱에서 주소를 복사-붙여넣기 하여 북마크를 저장한 경우에는 별도로 html 응답을 받아와야 했다.

&nbsp;
## 프론트에서는 프록시 서버를 구축하여 에러를 해결할 수 있다.
&nbsp;

그렇지만 프록시 서버를 직접 구축하는 것은 귀찮다. 또한 이미 사용되고 있는 프록시 서버도 웹에 검색하면 몇개 나오는데, 이런 것들은 내가 테스트해봤을 때 거의 작동하지 않았고 작동하더라도 됐다 안됐다 해서 배포 프로덕트에서 사용할 수는 없었다. 이럴 때 CORS Anywhere와 Heroku를 이용하여 빠르고 간단하게 프록시 서버를 구축할 수 있다.

### CORS Anywhere란?

[CORS Anywhere](https://github.com/Rob--W/cors-anywhere)는 프록시 된 요청에 CORS 헤더를 추가하는 NodeJS 프록시다. MIT 라이선스로 자유롭게 사용할 수 있다. whitelist, blacklist, rate limit 등의 다양한 설정도 간단하게 할 수 있다.

### Heroku로 CORS Anywhere 배포하기

원래 Vercel을 쓰고 있어서 Vercel로 배포할 까 했는 데 Vercel은 정책상 프록시 서버 배포를 막고 있는 듯했다. 그래서 별도로 Heroku 계정을 만들어서 사용하였다.

Heroku로 CORS Anywhere을 배포하는 방법은 [https://nhj12311.tistory.com/278](https://nhj12311.tistory.com/278) 이 블로그 글을 참고하였다. 배포 방법이 어렵진 않아서 블로그 글만 보고 따라하니 금방 배포할 수 있었다.

그 결과 이렇게 빌드가 잘되어 Proxy 서버가 잘 동작함을 확인할 수 있다!

<div style="text-align: left">
  <img src="/assets/img/post_images/proxy.png" width="100%"/>
  <p style="font-size: medium; text-align: center;">[빌드 로그]</p>
</div>

<div style="text-align: left">
  <img src="/assets/img/post_images/proxy1.png" width="100%"/>
  <p style="font-size: medium; text-align: center;">[완성된 proxy 서버에 들어가면 이렇게 나온다!]</p>
</div>

&nbsp;

***

참고 내용 출처 :
* [https://donggov.tistory.com/132](https://donggov.tistory.com/132)
* [https://velog.io/@kimtaeeeny/CORS-%EC%99%80-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95-express-proxy-FE-study8](https://velog.io/@kimtaeeeny/CORS-%EC%99%80-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95-express-proxy-FE-study8)