---
title: Emotion을 활용한 React 스타일링 마스터하기
date: 2023-08-01 20:00:00 +0900
categories:
  - React
tags:
  - Emotion
---

## Emotion과 Styled Components, 무엇이 다른가요? 🤔

Emotion과 Styled Components는 둘 다 CSS-in-JS라는 동일한 컨셉을 공유하고 있습니다. Emotion이 특별한 이유는 바로 성능과 확장성입니다! 🚀 

## Emotion 설치하기 📦

우선, Emotion을 설치하려면 터미널을 열고 다음 명령어를 입력하면 됩니다.

```bash
npm install @emotion/react @emotion/styled
```

단순하죠? 설치가 끝나면 이제 본격적으로 스타일링을 시작해 봅시다! 🎉

## Emotion 기초 문법 📚

### css() 함수의 매력 😍

`css()` 함수는 Emotion의 핵심입니다. 🌟 이 함수를 이용하면 객체나 문자열 형태로 CSS 스타일을 적용할 수 있습니다.

#### 객체 형태로 스타일링하기

```javascript
/** @jsxImportSource @emotion/react */
import { css } from '@emotion/react'

function MyComponent() {
  return (
    <div css={css({ backgroundColor: 'yellow' })}>
      노란색 배경
    </div>
  )
}
```

#### 문자 형태로 스타일링하기

```javascript
/** @jsxImportSource @emotion/react */
import { css } from '@emotion/react'

function MyComponent() {
  return (
    <div css={css`background-color: yellow;`}>
      노란색 배경
    </div>
  )
}
```

### JSX Pragma는 뭐에요? 🤷‍♀️

JSX Pragma는 Emotion의 `jsx()` 함수를 사용하도록 알려주는 신호입니다.

```javascript
/** @jsxImportSource @emotion/react */
```

이 신호를 빼먹으면 스타일이 제대로 반영되지 않아요. 주의, 주의! 🚨

## 실전! 버튼 컴포넌트 만들기 🎮

### 예제 코드 📝

```javascript
/** @jsxImportSource @emotion/react */
function Button({ children }) {
  return (
    <button
      css={{
        borderRadius: '6px',
        border: '1px solid rgba(27, 31, 36, 0.15)',
        backgroundColor: 'rgb(246, 248, 250)',
        color: 'rgb(36, 41, 47)'
      }}
    >
      {children}
    </button>
  )
}
```

이렇게 코드 몇 줄로 멋진 버튼을 만들어 냈습니다! 😎

## 결론 🎉

이상으로 Emotion을 활용한 React 스타일링의 세계를 함께 탐험한 것을 마무리하겠습니다. 이제 여러분도 스타일링 마스터가 되어 볼 시간입니다! 🎓 감사합니다! 다음에 또 만나요! 🙌
