---
title: React `forwardRef()` 깊게 알아보기
date: 2023-08-03 20:00:00 +0900
categories:
  - React
tags:
  - forwardRef
---

안녕하세요, React 선수단 여러분! 🚀 오늘은 React의 `forwardRef()` 함수에 대한 **꿀정보**를 제대로 털어보려고 해요. 준비는 되셨나요? 🤸‍♂️ 시작해 봅시다!

## 🌈 `ref`가 뭐에요? 🤔

일단 본론에 들어가기 전에 `ref`가 뭔지 알아볼까요? React에서 `ref`는 직접적인 HTML 엘리먼트 접근을 위해 쓰입니다. 아래는 예시입니다.

```javascript
import React, { useRef } from "react";

function MyComponent() {
  const myRef = useRef(null);

  function focusElement() {
    myRef.current.focus();
  }

  return (
    <>
      <input type="text" ref={myRef} />
      <button onClick={focusElement}>Focus!</button>
    </>
  );
}
```

## 😵‍💫 `forwardRef()` 소개

### 컴포넌트에서 `ref` 다루기

`forwardRef()`가 왜 필요한지 알려면 먼저 어떤 문제가 발생하는지 봐야겠죠. 아래의 코드를 봅시다.

```javascript
import React, { useRef } from "react";

function ChildComponent({ ref }) {
  return <input type="text" ref={ref} />;
}

function ParentComponent() {
  const inputRef = useRef(null);
  
  function focusInput() {
    inputRef.current.focus();
  }
  
  return (
    <>
      <ChildComponent ref={inputRef} />
      <button onClick={focusInput}>Focus!</button>
    </>
  );
}
```

위 코드는 실행하면 에러가 납니다. 왜냐하면 `ref`는 일반적인 `props`로 사용되지 않기 때문이죠. 그렇다면 어떻게 해야할까요? 🤷‍♂️

### 문제의 해결사, `forwardRef()` 등장! 🎉

`forwardRef()`를 사용하면, 이 문제를 쉽게 해결할 수 있습니다. 아래와 같이 코드를 수정해보죠.

```javascript
import React, { forwardRef, useRef } from "react";

const ChildComponent = forwardRef((props, ref) => {
  return <input type="text" ref={ref} />;
});

function ParentComponent() {
  const inputRef = useRef(null);
  
  function focusInput() {
    inputRef.current.focus();
  }
  
  return (
    <>
      <ChildComponent ref={inputRef} />
      <button onClick={focusInput}>Focus!</button>
    </>
  );
}
```

이렇게 하면 문제가 해결됩니다! `forwardRef()`를 사용하면 `ref`를 자식 컴포넌트로 전달할 수 있게 되는 거죠. 문제 해결 완료! 🎊

## 🎁 실제 예제: 오디오 플레이어 만들기

마지막으로 하나의 예제를 들어보겠습니다. 오디오 플레이어를 만들어볼게요.

```javascript
import React, { useRef, forwardRef } from "react";

const AudioComponent = forwardRef((props, ref) => {
  return <audio ref={ref} />;
});

function PlayerComponent() {
  const audioRef = useRef(null);

  function playAudio() {
    audioRef.current.play();
  }

  return (
    <>
      <AudioComponent ref={audioRef} />
      <button onClick={playAudio}>Play!</button>
    </>
  );
}
```

이렇게 하면 오디오를 제어할 수 있는 간단한 플레이어가 완성됩니다! 🎶

## 마무리 🎬

이제 `forwardRef()`에 대해서 아주 잘 알게 되셨죠? 다음 글에서 또 만나요! 🤩
