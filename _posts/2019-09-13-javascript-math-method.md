---
title: 자바스크립트 자주 사용하는 Math 함수
author: Poburi
date: 2020-09-13 19:34:00 +0800
categories: [Dev, JS]
tags: [js]
toc: false
pin: true

---

# Math 메서드

| 메서드          | 기능                                               | 매개변수 |
| :-------------- | :------------------------------------------------- | -------: |
| [trunc](#trunc) | 주어진 숫자의 정수부분                             |     숫자 |
| [floor](#floor) | 주어진 수 이하의 가장 큰 정수.                     |     숫자 |
| [ceil](#ceil)   | 주어진 숫자보다 크거나 같은 숫자 중 가장 작은 숫자 |     숫자 |

# trunc

함수는 주어진 값의 소수부분을 제거하고 숫자의 정수부분을 반환합니다. 

```javascript
Math.trunc(13.37);    // 13
Math.trunc(42.84);    // 42
Math.trunc(0.123);    //  0
Math.trunc(-0.123);   // -0
Math.trunc('-1.123'); // -1
Math.trunc(NaN);      // NaN
Math.trunc('foo');    // NaN
Math.trunc();         // NaN
```

# floor

함수는 주어진 숫자와 같거나 작은 정수 중에서 가장 큰 수를 반환합니다.
음의 값에서는 반올림을 하여 계산한다는 점을 주의해야합니다.

```javascript
Math.floor( 45.95); //  45
Math.floor( 45.05); //  45
Math.floor(  4   ); //   4
Math.floor(-45.05); // -46 
Math.floor(-45.95); // -46
```

# ceil

함수는 주어진 숫자보다 크거나 같은 숫자 중 가장 작은 숫자를 integer 로 반환합니다.
양의 값에서 올림을 하여 계산한다는 점을 주의해야합니다.

```javascript
Math.ceil(.95);    // 1
Math.ceil(4);      // 4
Math.ceil(7.004);  // 8
Math.ceil(-0.95);  // -0
Math.ceil(-4);     // -4
Math.ceil(-7.004); // -7
```