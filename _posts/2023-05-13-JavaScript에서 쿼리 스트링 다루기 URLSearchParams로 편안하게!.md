---
title: JavaScript에서 쿼리 스트링 다루기 URLSearchParams로 편안하게!
date: 2023-05-13 20:00:00 +0900
categories:
  - JavaScript
tags:
  - 자바스크립트
  - 쿼리스트링
  - URLSearchParams
---

안녕하세요, 여러분! 여러분은 웹 개발을 할 때 URL에 있는 쿼리 스트링(🤔 뭐지? 물음표(?) 뒤에 따라오는 그 긴 문자열이다!)을 다루는 법을 어떻게 알고 계신가요? 네, 네, 전 정확히 알고 있습니다. 그런 복잡하고 어려운 일들은 이제 그만! 자바스크립트의 URLSearchParams를 사용하면 그런 걱정은 없어져요! 자, 그럼 함께 본격적으로 시작해볼까요?

## URLSearchParams의 탄생 이야기

자바스크립트의 세계에서 새롭게 등장한 슈퍼 히어로 `URLSearchParams`는 URL에 있는 쿼리 스트링을 다루기 위해 나타났어요. 😎 이 친구를 사용하면 쿼리 스트링을 더 안전하게, 그리고 더 쉽게 다룰 수 있어요. 어떻게 사용하는지 함께 알아볼까요?

## 객체 생성 마법

🧙‍♂️마법의 시작은 객체 생성부터 시작된답니다. 우리는 `URLSearchParams` 객체를 다양한 방법으로 만들 수 있어요. 

```javascript
// 방법 1: 2차원 배열을 사용하는 방법
new URLSearchParams([
  ["mode", "dark"],
  ["page", 1],
  ["draft", false],
  ["sort", "email"],
  ["sort", "date"],
]);

// 방법 2: 쿼리 스트링 문자열을 사용하는 방법
new URLSearchParams("?mode=dark&page=1&draft=false&sort=email&sort=date");
// 혹은 ? 기호를 생략해도 됩니다!
new URLSearchParams("mode=dark&page=1&draft=false&sort=email&sort=date");
```

기억하세요! 우리는 이 마법을 사용하여 빈 객체를 만든 후에 나중에 파라미터를 추가할 수도 있다는 것을!

## 쿼리 스트링을 속속들이 알아보기

이제 쿼리 스트링 안을 들여다 볼 차례입니다! 😄 `URLSearchParams` 객체에는 `size` 속성이 있어요. 이 속성으로 쿼리 스트링에 얼마나 많은 매개변수가 있는지 셀 수 있답니다!

```javascript
const searchParams = new URLSearchParams("mode=dark&page=1&draft=false");
console.log(searchParams.size); // 출력: 3
```

하지만 여기서 주의! 동일한 키에 여러 값이 있는 경우, 값의 개수를 기준으로 `size` 속성이 계산된답니다!

```javascript
const searchParams = new URLSearchParams("sort=date&sort=email");
console.log(searchParams.size); // 출력: 2
```

아, 그리고 `Set`을 사용하면 유일한 키의 개수도 찾을 수 있어요!

```javascript
console.log([...new Set(searchParams.keys())].length); // 출력: 1
```

## URLSearchParams의 마법 스킬들

`URLSearchParams` 객체는 여러 가지 마법 스킬을 가지고 있답니다! 👍

### toString() 메서드: 문자열 변환기

`toString()` 메서드는 우리가 만든 쿼리 스트링을 다시 문자열로 바꾸는 데 사용된답니다. 아래 코드를 보세요!

```javascript
const searchParams = new URLSearchParams();
console.log(searchParams.toString()); // 출력: ''

const searchParams = new URLSearchParams([
  ["mode", "dark"],
  ["page", 1],
  ["draft", false],
]);
console.log(searchParams.toString()); // 출력: 'mode=dark&page=1&draft=false'
```

### append()와 set() 메서드: 파라미터 조종사

`append()` 메서드로 새로운 파라미터를 추가할 수 있어요. 그리고 `set()` 메서드를 사용하면 기존 파라미터의 값을 바꿀 수 있어요.

```javascript
const searchParams = new URLSearchParams();
searchParams.append("mode", "dark");
searchParams.append("page", 1);
// 더 많은 파라미터 추가
```

아마 여러분도 이제 `URLSearchParams`의 마법을 사용할 준비가 된 것 같네요! ✨ 그럼, 행복한 코딩 되세요!
