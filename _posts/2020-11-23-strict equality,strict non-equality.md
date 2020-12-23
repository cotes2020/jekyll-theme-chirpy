---
title: 일치연산자(===)와 동등연산자(==)
author: juyoung
date: 2020-11-23 18:28:00 +0800
categories: [javascript, syntax]
tags: [javascript]
---

1. 일치 연산자 ===를 사용하여 null과 undefined를 비교하면
두 값의 자료형이 다르기 때문에 일치 비교 시 거짓이 반환됩니다.
```
 alert( null === undefined ); // false
```

2. 동등 연산자 ==를 사용하여 null과 undefined를 비교
동등 연산자를 사용해 null과 undefined를 비교하면 특별한 규칙이 적용돼 true가 반환됩니다.  두 값은 자기들끼리는 잘 어울리지만 다른 값들과는 잘 어울리지 못하죠.

```
alert( null == undefined ); // true
```



