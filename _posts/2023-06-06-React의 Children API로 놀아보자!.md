---
title: React의 Children API로 놀아보자!
date: 2023-06-06 20:00:00 +0900
categories:
  - JavaScript
tags:
  - React
  - ChildrenAPI
---

안녕하세요 여러분! 오늘은 여러분과 함께 React의 Children API를 탐험할 예정이에요! 들어가기 전에, Children API는 무엇인지, 왜 필요한지 한번 간단히 알아볼까요? 🤔

## Children API가 뭐에요?

자, 여러분! React에서 Children라는 이름을 들어본 적이 있나요? 아니요? 그럼 오늘 처음 들어볼 날이네요! 😊 

Children API는 React에서 children prop을 다루기 위한 특별한 도구입니다. Children과 children, 두 친구는 조금 다른데요. 소문자로 시작하는 'children'은 컴포넌트의 자식 요소들을 나타내며, 대문자로 시작하는 'Children'은 그 자식 요소들을 더 잘 다루기 위한 API랍니다! 

Children API를 통해서 children prop을 좀 더 안전하고 효과적으로 다룰 수 있답니다. 어떻게 사용하는지 함께 알아볼까요? 😁

## 어떻게 접근하나요?

자, 첫 번째 방법은 React 패키지에서 `React.Children`로 접근하는 것이에요! 이렇게 사용해볼 수 있답니다.

```javascript
import React from "react";
function ReactChildren({ children }) {
  console.log(React.Children);
  return <>ReactChildren</>;
}
```

두 번째 방법은 `Children`을 바로 불러와서 사용하는 거에요! 이렇게요:

```javascript
import { Children } from "react";
function ReactChildren({ children }) {
  console.log(Children);
  return <>ReactChildren</>;
}
```

어떤 방법을 선택할지는 여러분의 선택이에요! 저는 개인적으로 첫 번째 방식을 선호하는데, 두 친구 'Children'과 'children'이 서로 너무 닮아서 헷갈리기 쉽거든요! 👀

## Children API의 주요 기능들

Children API에는 여러 가지 기능들이 있는데, 그 중 몇 가지를 뽑아 소개해드릴게요! 😊

### Children.map()

`Children.map()`는 아마도 가장 많이 사용되는 함수일 거에요. 배열의 `map()` 함수와 비슷하다고 생각하시면 돼요! 이 함수를 통해 자식 요소들 각각에 특별한 변화를 줄 수 있어요. 예를 들어, 다음과 같이 사용할 수 있어요:

```javascript
function Map({ children }) {
  return React.Children.map(children, (child, i) => 
    i % 2 === 0 ? <b>{child}</b> : <u>{child}</u>
  );
}
```

여기서는 홀수 번째(짝수 인덱스) 자식의 글씨를 굵게 하고, 짝수 번째(홀수 인덱스) 자식에 밑줄을 그어줬어요! 이렇게 편리하게 사용할 수 있답니다! 😁

## 이제 여러분도 Children API 마스터!

이제 여러분도 Children API를 사용해서 멋진 React 프로젝트를 만들어 볼 차례에요! Children API를 사용하면 복잡하고 어려운 children prop을 쉽고 안전하게 다룰 수 있다는 걸 꼭 기억하세요! 

이제 함께 React의 세계로 뛰어들 준비가 되셨나요? 그럼, 함께 놀러가요~ 🚀
