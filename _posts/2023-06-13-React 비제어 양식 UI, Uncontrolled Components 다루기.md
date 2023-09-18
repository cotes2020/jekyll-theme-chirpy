---
title: React 비제어 양식 UI, Uncontrolled Components 다루기
date: 2023-06-13 20:00:00 +0900
categories:
  - JavaScript
tags:
  - React
  - 비제어
  - Uncontrolled
  - Components
---

안녕하세요, 여러분! 😊 오늘은 아주 특별하고 중요한 주제로 찾아왔어요. 그 주제는 바로 "Uncontrolled Components"에 대한 이야기인데요, 이 주제를 잘 알면 여러분의 React 프로젝트가 더욱 더 풍부해질 거에요!

## Uncontrolled Components란 무엇인가요? 🤔

Uncontrolled Components라는 이름에서 알 수 있듯이, React에서 제어하지 않는 컴포넌트를 말해요. 그럼 "제어하지 않는다"는 게 무슨 뜻일까요? 간단히 말해 브라우저가 상태를 제어하도록 내버려 두는 컴포넌트를 말해요. 어렵게 느껴질 수 있으니, 이제부터 차근차근 알아볼게요!

## 첫 번째 방법: DOM API 사용하기 😎

React를 사용하지 않는 순수 자바스크립트처럼, Uncontrolled Components를 개발하는 첫 번째 방법은 DOM API를 사용하는 거에요. "제출" 이벤트의 `target.elements` 속성을 통해서 양식 내부의 HTML 요소 값을 쉽게 읽어올 수 있어요. 

하지만! 여기서 조심해야 할 점이 있어요. 각 HTML 요소의 초기 상태를 지정할 때, `value`와 `checked` 속성을 사용하면 값이 고정되어 버려서 사용자가 변경할 수 없어져요. 😱 그래서, React의 `defaultValue`와 `defaultChecked` prop을 사용해야 해요.

```javascript
function Form() {
  const handleSubmit = (event) => {
    // ... (코드 생략)
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* ... (코드 생략) */}
    </form>
  );
}

export default Form;
```

## 두 번째 방법: useState() 후크 사용하기 🎣

DOM API를 사용하지 않고, React 답게 해결하고 싶다면 `useState()` 후크를 사용할 수 있어요. 후크는 기능을 "빌려주는" 친구라고 생각하면 돼요!

`useState()` 후크를 호출해서 양식 관련 HTML 요소를 저장하기 위한 상태 변수와 상태를 변경하기 위한 함수를 만들어요. 그리고 `ref prop`을 통해 콜백 함수를 설정하면 됩니다.

```javascript
import { useState } from "react";

function Form() {
  const [input, setInput] = useState(null);
  // ... (코드 생략)
  
  const handleSubmit = (event) => {
    // ... (코드 생략)
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* ... (코드 생략) */}
    </form>
  );
}
```

이렇게 두 가지 방법을 활용하면 Uncontrolled Components를 잘 다룰 수 있어요! 아직 조금 어려우시다면 걱정하지 마세요. 여러분이 훌륭한 개발자이니까 어렵지 않게 꼭 익히실 수 있을 거에요! 🌟

다음 시간에도 유익한 내용으로 찾아올게요. 그때 봐요! 👋
