---
title: React에서 useRef 훅을 마스터하자!
date: 2023-08-13 20:00:00 +0900
categories:
  - React
tags:
  - useRef
---

안녕하세요, 여러분! 오늘은 React Hooks 중에서도 꼭 알아야 할 `useRef`에 대해 한 번 제대로 파헤쳐봅시다. 이녀석은 그냥 지나칠 수 없죠!

## useRef가 뭐에요? 🤷‍♀️

`useRef`는 React Hooks 중 하나로, 이녀석은 뭐든지 기억해낼 수 있는 능력이 있어요. 변덕스러운 변수값들을 다룰 때 정말로 유용하죠. 상태(state)가 변경될 때마다 컴포넌트를 재랜더링하는 React에서, 렌더링에 영향을 미치지 않으면서도 데이터를 유지할 수 있게 해줍니다.

## 예제로 알아보자! 🎯

### 카운터 버튼 예제 🖲️

변수를 이용해서 카운터를 만들어볼게요. 그냥 단순 클릭 카운터입니다.

```jsx
import React, { useState } from "react";

function Counter() {
  const [count, setCount] = useState(0);
  return (
    <>
      <p>{count}번 클릭하셨습니다.</p>
      <button onClick={() => setCount(count + 1)}>클릭</button>
    </>
  );
}
```

아주 쉽죠? 😎 그런데 만약 이 카운터를 자동으로 동작하게 만들려면 어떻게 해야 할까요? `setInterval`을 이용해서 자동 카운터를 만들어봅시다.

```jsx
import React, { useState, useEffect } from "react";

function AutoCounter() {
  const [count, setCount] = useState(0);
  useEffect(() => {
    const intervalId = setInterval(() => setCount(count + 1), 1000);
    return () => clearInterval(intervalId);
  }, []);
  return <p>자동 카운트: {count}</p>;
}
```

### useRef를 활용한 카운터 🎰

자, 이제 `useRef`의 무대입니다! 위에서 만든 자동 카운터를 시작/정지 버튼을 이용하여 제어해봅시다.

```jsx
import React, { useState, useEffect, useRef } from "react";

function SuperCounter() {
  const [count, setCount] = useState(0);
  const intervalId = useRef(null);

  const startCounter = () => {
    intervalId.current = setInterval(() => setCount(count + 1), 1000);
  };

  const stopCounter = () => {
    clearInterval(intervalId.current);
  };

  return (
    <>
      <p>🦸‍♀️ 슈퍼 카운터: {count}</p>
      <button onClick={startCounter}>🚀 시작</button>
      <button onClick={stopCounter}>🛑 정지</button>
    </>
  );
}
```

여기서 `useRef`가 도와준 건, `startCounter` 함수와 `stopCounter` 함수가 `intervalId`를 공유할 수 있게 해준 것이죠! 그래서 자유롭게 카운터를 시작하거나 멈출 수 있어요. 멋지죠? 😆

## 마무리 🎉

그래서 오늘 배운 것은 `useRef`가 뭐하는 녀석인지, 그리고 이 친구를 어떻게 활용할 수 있는지 알아봤어요. 기억하세요, 프로그래밍에서는 경험이 최고의 선생님이에요! 코드를 많이 작성해보면서 `useRef`의 놀라운 능력을 체험해보세요! 다음에 또 뵙겠습니다! 안녕! 👋
