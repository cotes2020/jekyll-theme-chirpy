---
title: Bearer 토큰과 Passport.js로 떠나는 코드 여행
date: 2023-05-30 20:00:00 +0900
categories:
  - JavaScript
tags:
  - 토큰
  - PassportJS
---

안녕하세요, 여러분! 오늘은 우리 함께 Passport.js라는 자바스크립트 프레임워크를 이용해서 Bearer 토큰 기반 API 인증을 만들어 볼거에요. 좌석을 뒤로 잘 누르시고 안전벨트를 매세요! 여행을 떠나 봅시다. 🚀

## 비행 전 사전 준비 🛫

우선, 무거운 짐을 줄이기 위해서 Express.js 패키지를 설치해야 해요. npm을 이용해서 깔끔하게 설치를 해 볼까요? 콘솔에 다음과 같이 입력해주세요!

```sh
$ npm install express
```

좋아요, 이제 비행기(서버 앱)를 만들 차례에요! 하지만 이 비행기는 매우 특별해요. GET 요청이 오면 ‘Hello, World!’라고 대답하는 비행기에요! 😄 아래 코드를 확인해봅시다.

```javascript
import express from "express";
const app = express();
app.get("/", (req, res) => {
  res.send("Hello, World!");
});
app.listen(3000, () => {
  console.log("Server is listening on port 3000");
});
```

## Passport.js와 함께하는 특별한 여행 🌄

여기서 여행이 더 특별해지는 순간이에요! Passport.js를 설치해서 Bearer 토큰 기반 인증을 준비할 거에요. Passport.js는 우리에게 너무도 편리하게 인증 전략을 제공해요. 

```sh
$ npm i passport passport-http-bearer
```

이제 비밀번호 같은 인증 토큰을 받아 볼 차례에요! 이 토큰을 사용해서 우리만의 비밀 클럽에 들어갈 수 있는 문을 만들어 봅시다! 

```javascript
import passport from "passport";
import BearerStrategy from "passport-http-bearer";

passport.use(
  new BearerStrategy((token, done) => {
    if (token === "1234") {
      done(null, { email: "user@test.com" });
    } else {
      done(null, false);
    }
  })
);
```

"1234"라는 비밀번호를 알고 있는 사람만 우리 비밀 클럽에 들어올 수 있어요! 😎 (실제로 이 비밀번호는 너무나도 간단해서 사용하면 안돼요! 😅)

## 마지막 관문을 통과하자! 🚪

이제 마지막으로, 우리만의 비밀 클럽에 들어올 수 있는 문을 만들 차례에요. 이 문은 유효한 Bearer 토큰이 있는 사람만 통과시켜줘요! passport.authenticate() 함수를 이용해서 마법의 문을 만들어 볼까요? 

```javascript
app.get(
  "/secret",
  passport.authenticate("bearer", { session: false }),
  (req, res) => {
    res.send("Welcome to the secret club! 🥳");
  }
);
```

어때요? 이제 우리만의 비밀 클럽이 완성되었어요! 🎉 ‘1234’라는 특별한 토큰을 가지고 `/secret` 경로로 오면, 비밀 클럽에 들어올 수 있어요!

---

이렇게 Passport.js와 Bearer 토큰을 이용하여 특별한 여행을 떠나봤어요. 비록 간단한 여행이었지만, 재미있으셨죠? 실제 애플리케이션에서는 이 방법을 더 발전시켜서 더욱 튼튼한 보안 시스템을 만들 수 있어요!

다음 여행에서 또 만나요! 안녕! 👋
