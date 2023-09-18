---
title: 리액트 Styled Components로 전역 스타일 설정
date: 2023-08-21 20:00:00 +0900
categories:
  - React
tags:
  - StyledComponents
---

## 컴포넌트에 전역 스타일 적용!

안녕하세요! 오늘은 Styled Components로 무엇을 할까요? **전역 스타일**을 설정하는 방법을 알려 드릴 거예요! 아주아주 쉽게 말해서, 모든 페이지에 동일한 스타일을 주고 싶을 때 어떻게 해야 할지 알려 드리겠습니다. 그럼 시작해 볼까요? 🚀

## `createGlobalStyle()` 🚀

Styled Components에서는 전역 스타일을 설정하는 방법이 따로 있어요. 바로 `createGlobalStyle()` 함수를 사용하는 것이죠. 이 함수를 사용하면 애플리케이션 전체에 스타일을 적용할 수 있습니다. 대단하죠?

```jsx
// GlobalStyle.jsx
import { createGlobalStyle } from "styled-components";

const GlobalStyle = createGlobalStyle`
  *, *::before, *::after {
  box-sizing: border-box;
  }
  body {
  font-family: "Comic Sans MS", cursive, sans-serif;
  line-height: 1.5;
  }
`;

export default GlobalStyle;
```

## 최상위 컴포넌트에 얹기 🎂

이제 전역 스타일을 애플리케이션에 적용해보죠. 아주 쉽게, 최상위 컴포넌트에다가 끼워넣으면 돼요!

```jsx
// App.jsx
import GlobalStyle from "./GlobalStyle";

function App() {
  return (
    <>
    <GlobalStyle />
    {/* 다른 컴포넌트들 */}
    </>
  );
}

export default App;
```

이렇게 하면 모든 페이지, 모든 컴포넌트에 전역 스타일이 적용돼요. 신기방기하죠? 😲

## 별도의 전역 스타일을 적용하는 방법 🌈

기본적인 것 외에도 다양한 전역 스타일을 적용할 수 있어요. 예를 들어, 특정 HTML 엘리먼트에 전역 스타일을 적용하고 싶다면 다음과 같이 할 수 있습니다.

```jsx
// GlobalStyle.jsx
import { createGlobalStyle } from "styled-components";

const GlobalStyle = createGlobalStyle`
  // 기존 코드
  h1, h2 {
  color: royalblue;
  }
  p {
  color: crimson;
  }
`;

export default GlobalStyle;
```

## 마무리 🎉

오늘은 Styled Components로 전역 스타일을 정말 멋지게 설정하는 방법에 대해 알아봤어요. 다음에 또 만나요! 🌈🌟
