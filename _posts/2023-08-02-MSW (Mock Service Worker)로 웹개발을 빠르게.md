---
title: MSW (Mock Service Worker)로 웹개발을 빠르게
date: 2023-08-02 20:00:00 +0900
categories:
  - React
tags:
  - MSW
---

안녕하세요, 여러분! 😄 오늘은 신나는 주제로 찾아왔어요. 그 주제는 바로 MSW, 즉 Mock Service Worker에 관한 이야기입니다. 이것은 웹 개발을 한층 더 쉽고 재미있게 만들어주는 마법 같은 도구입니다!

## MSW가 뭐에요? 🤔

MSW는 Mock Service Worker의 약자로, 웹 브라우저에게 가짜API를 제공합니다. 그리고 이 가짜 API는 우리가 작성한 프론트엔드 코드에 데이터를 넘겨줍니다. 진짜 API가 준비되기 전에 프론트엔드 개발을 빠르게 진행할 수 있게 해주는 것이죠.

## MSW의 놀라운 특징들 🌟

### 실제 API와 다를 게 없어요!

MSW의 가장 큰 장점은 네트워크 단에서 작동한다는 것입니다. 그래서 실제 API와 가짜 API를 교체하는 작업이 아주 쉽답니다. 그럼 이것은 뭐가 좋나요? 개발 생산성이 올라간다는 거죠! 🎉

### 유연한 디자인 🤸‍♀️

이 녀석은 굉장히 유연해요. 테스트 환경에서도, 브라우저에서도 동일한 요청 핸들러 코드를 공유해서 사용할 수 있습니다. 그러니까, 쓸모 있는 코드를 더 많이 작성할 수 있다는 거죠!

### 다양한 API 모킹 지원 🌈

REST API는 물론, GraphQL API도 모킹할 수 있어요! 

## MSW 실습: To-Do 앱 만들기 🛠️

실습으로 To-Do 앱을 만들어보려고 합니다. Create React App을 사용해서 간단하게 프로젝트를 생성할 거에요.

```bash
$ npx create-react-app our-msw
```

### MSW 설치와 설정 🔧

MSW 라이브러리를 설치합니다.

```bash
$ npm install msw --save
```

그 다음, MSW를 초기화합니다.

```bash
$ npx msw init public/ --save
```

### 요청 핸들러 작성 ✍️

이제 가짜 API를 구현할 차례입니다! `src/mocks/handlers.js`라는 파일을 만들어서 핸들러 코드를 작성하겠습니다.

```javascript
// src/mocks/handlers.js
import { rest } from "msw";

export const handlers = [
  rest.get("/todos", (req, res, ctx) => {
    return res(
      ctx.json([
        { id: 1, text: "빨래하기" },
        { id: 2, text: "코딩하기" },
        { id: 3, text: "운동하기" },
      ])
    );
  }),

  rest.post("/todos", (req, res, ctx) => {
    const newTodo = req.body;
    return res(ctx.json(newTodo));
  }),
];
```

이제 할 일 목록을 불러오거나 새로운 할 일을 추가할 수 있을 거에요!

## 마무리 🎉

이렇게 해서 MSW를 활용한 웹 개발의 새로운 세계를 열어보았습니다! 이 도구를 활용하면 프론트엔드 개발이 훨씬 더 빠르고 재밌어질 거에요! 다음에 또 신나는 주제로 찾아올게요! 😄
