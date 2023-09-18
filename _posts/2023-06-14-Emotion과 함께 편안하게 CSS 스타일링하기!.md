---
title: Emotion과 함께 편안하게 CSS 스타일링하기!
date: 2023-06-14 20:00:00 +0900
categories:
  - JavaScript
tags:
  - CSS
  - Emotion
---

안녕하세요, 여러분! 오늘은 놀랍도록 강력하면서도 깜찍한 라이브러리, Emotion을 사용하여 React 컴포넌트를 아름답게 꾸미는 방법을 알아보겠습니다. 이제 손을 힘껏 털고, 키보드 위로 손을 가져가 보세요! 😁

## Emotion의 세계로 한 발짝! 🌈

우리는 여기서 'Emotion'이라는 라이브러리를 사용하여 마치 화가가 미술지에 색을 입히듯, 우리의 웹 사이트에 생기를 불어넣을 거에요. 이 라이브러리는 'CSS-in-JS' 방식을 사용해서 스타일을 입히는 거랍니다. 여러분도 무지개 같은 화려한 웹 페이지를 만들 준비가 되셨나요? 🌈

### 패키지 설치하기 🛠️

먼저, 우리는 Emotion을 우리의 프로젝트에 설치할 필요가 있어요. 이를 위해 npm이라는 패키지 매니저를 사용하면 금방 설치할 수 있어요! 여러분의 프로젝트가 React가 아니라면 걱정하지 마세요! Emotion은 다른 자바스크립트 프로젝트에서도 사용할 수 있답니다! 🌟

```javascript
// React 프로젝트에서는 이렇게 설치해요!
npm install @emotion/react @emotion/styled
```

### 기본 문법 익히기 📚

이제 기본 문법을 배워 볼 거에요. Emotion에서는 `css()` 함수를 사용해서 스타일을 정의할 수 있어요. 여기에 우리가 원하는 스타일을 넣어주면 된답니다! 이렇게 말이죠!

```javascript
/** @jsxImportSource @emotion/react */
import { css } from "@emotion/react";

function MyComponent() {
  return (
    <div
      css={css({
        backgroundColor: "yellow",
      })}
    >
      노란색 영역
    </div>
  );
}
```

그리고 이 스타일을 원하는 HTML 요소에 적용시킬 수 있어요! 🎨

### JSX Pragma와 친해지기 🤗

여러분이 코드를 보면 `/** @jsxImportSource @emotion/react */` 라는 뭔가 신기한 코드를 볼 수 있어요. 이건 Babel 트랜스파일러에게 "이 JSX를 Emotion의 `jsx()` 함수를 사용하여 변환해줘!"라고 알려주는 역할을 한답니다. 이렇게 하면 우리가 정의한 스타일이 제대로 적용된답니다! ✨

### 버튼 만들기에 도전! 🎮

이제 본격적으로 버튼을 만들어 볼 거에요! GitHub에서 보신 그런 멋진 버튼을 만들어볼텐데, 어렵게 생각하지 마세요! 아래 코드를 따라하면 금방 만들 수 있어요!

```javascript
/** @jsxImportSource @emotion/react */
function Button({ children }) {
  return (
    <button
      css={{
        borderRadius: "6px",
        border: "1px solid rgba(27, 31, 36, 0.15)",
        backgroundColor: "rgb(246, 248, 250)",
        color: "rgb(36, 41, 47)",
        fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
        fontWeight: "600",
        lineHeight: "20px",
        fontSize: "14px",
        padding: "5px 16px",
        textAlign: "center",
        cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}
```

마법같죠? 이렇게 우리의 React 앱에서 이 아름다운 버튼을 사용할 수 있게 되었어요! 😁

### 마무리 🌟

여러분! 여러분이 만든 아름다운 버튼은 어떤가요? 브라우저에서 소스를 확인해보면 Emotion이 자동으로 만들어준 클래스 이름이 있는 것을 확인할 수 있을 거에요! 자, 이제 여러분도 Emotion 마법사가 되었습니다! ✨

다음에도 더 많은 마법 같은 코딩 이야기로 찾아뵐게요! 그때 봐요! 🌈
