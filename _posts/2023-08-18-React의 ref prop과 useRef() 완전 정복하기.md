---
title: React의 ref prop과 useRef() 완전 정복하기
date: 2023-08-18 20:00:00 +0900
categories:
  - React
tags:
  - prop
---

안녕하세요, 리액트 마스터가 되고 싶은 여러분! 🎉 오늘은 리액트에서 실제 HTML 엘리먼트에 접근하는 방법을 살펴볼 거예요. 그럼 이제 고고씽! 🚴‍♀️

## ref prop의 기본 개념 👨‍🏫

`ref prop`은 HTML 엘리먼트의 레퍼런스(주소 같은 거)를 변수에 담아줘요. 그래서 이 변수를 통해 HTML 엘리먼트를 마음대로 조종할 수 있답니다! 🎮

## useRef()는 무엇인가요? 🎣

`useRef()`는 리액트의 훅(hook) 중 하나예요. 함수형 컴포넌트에서 `ref prop`을 쓰려면 이 친구를 사용해야 해요. 코드를 살펴볼까요? 🧐

```javascript
import React, { useRef } from "react";

function AwesomeInput() {
  const inputRef = useRef(null);

  const handleClick = () => {
    inputRef.current.focus();
  }

  return (
    <>
      <input ref={inputRef} />
      <button onClick={handleClick}>포커스 주기</button>
    </>
  );
}
```

위 코드에서 보시다시피, `inputRef`라는 변수에 HTML 엘리먼트의 레퍼런스가 담기게 돼요. 그럼 이제 `inputRef.current`로 input 엘리먼트를 조종할 수 있어요! 🕹

## 다양한 엘리먼트 제어하기 🎭

`ref prop`은 input 엘리먼트 외에도 다양한 엘리먼트에 사용할 수 있어요. 예를 들어 오디오나 비디오 엘리먼트도 쉽게 조종할 수 있죠. 그러니까 복잡한 애플리케이션에서도 유용하게 사용할 수 있어요! 🌟

## 그래도 조심해야 할 점! ⚠️

마지막으로, 리액트는 선언형 프로그래밍을 기반으로 하기 때문에, 꼭 필요한 경우가 아니면 `ref prop`을 쓰지 않는 게 좋아요. 그래도 사용할 때는 이 포스트를 잘 참고하시면 된답니다! 😉

## 마무리 🎬

이렇게 하면 리액트에서 HTML 엘리먼트를 제어하는 `ref prop`과 `useRef()`에 대해 알게 되셨을 거에요. 그럼 저는 다음글에서 뵙겠습니다. 감사합니다!
