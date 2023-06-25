---
title: JavaScript 연산자
date: 2023-06-25 18:38:55 +0900
author: kkankkandev
categories: [Language, JavaScript]
tags: [web, javascript, js, gitblog, vscode, gitpage, js연산자, 연산자]     # TAG names should always be lowercase
comments: true
# image:
#   path: https://github.com/War-Oxi/Oxi/assets/72260110/3af8c7c9-cc3a-4fed-84d5-c736bad8ba53
---

# 연산자

> 연산자(operator)는 하나 이상의 표현식을 대상으로 산술, 할당, 비교, 논리, 타입 지수 연산(operation) 등을 수행하는 하나의 값을 만든다.
> 

# 1. 산술 연산자

```jsx
//산술 연산자
5 * 4 // -> 20

//문자열 연결 연산자
'My name is ' + 'Lee' // -> My name is Lee

//할당 연산자
color = 'red' // -> 'red'

//비교 연산자
3 > 5 // -> false

//논리 연산자
true && false // -> false

//타입 연산자
typeof 'Hi' // -> string
```

## 1.1 단항 산술 연산자

### 1.1.1 '+' 연산자

> 숫자 타입이 아닌 피연산자에 + 단항 연산자를 사용하면 피연산자를 숫자 타입으로 변환하여 반환한다.

* 피연산자가 숫자 타입으로 변경하는 것이 아닌 숫자 타입으로 변환한 값을 생성해서 반환
> 

```jsx
var x = '1';

console.log(+x); // 1(숫자타입) 반환

console.log(x); // "1"

x = true; //boolean 타입으로 변환 후 true 할당

console.log(+x); // 1

console.log(x); == true
```

- 주의할점
    - 문자열을 숫자로 타입을 반환할 수 없다. ⇒ (NaN) 반환

### 1.1.2 '-' 연산자

> 비연산자의 부호를 반전한 값을 반환
> 

## 1.2 문자열 연결 연산자

> +연산자는 피연산자 중 하나 이상이 문자열인 경우 문자열 연결 연산자로 동작한다. 
그 외의 경우는 산술 연산자로 동작한다
> 

```jsx
'1' + 2; // -> '12'
1 + '2'; // -> ''12

1 + 2; // -> 3

1 + true; // -> 2

1 + false; // -> 1

1 + null; // -> 1

//undefined는 숫자로 타입 변환되지 않는다.
+undefined; // -> NaN
1 + undefined; // -> NaN
```

*암묵적 타입 변환(implicit), 타입 강제 변환(type coercion)

- 위의 코드에서 개발자의 의도와 상관없이 javascript 엔진에 의해 암묵적으로 타입이 자동으로 변환되는 것.

# 2. 비교 연산자

> 비교 연산자(comparison operator)는 좌항과 우항의 피연산자를 비교한 다음 그 결과를 불리언 값으로 반환한다. 비교 연산자는 if문으나 for문과 같은 제어문의 조건식에서 주로 사용한다
> 

## 2.1 동등/일치 비교 연산자

| 비교 연산자 | 의미 | 사례 | 설명 | 부수 효과 |
| --- | --- | --- | --- | --- |
| == | 동등 비교 | x == y | x와 y의 값이 같은 | X |
| === | 일치 비교 | x === y | x와 y의 값과 타입이 같음 | X |
| != | 부동등 비교 | x != y | x와 y의 값이 다름 | X |
| !== | 불일치 비교 | x !== y | x와 y의 값과 타입이 다름 | X |
- 동등 비교(==) 연산자는 좌항과 우항의 피연산자를 비교할 때 먼저 암묵적 타입 변환을 통해 타입을 일치시킨 후 같은 값인지 비교한다.
    
    ***좌항과 우항의 피연산자가 다른 타입이더라도 암묵적 타입 변환 후 같은 값이면 true를 반환**
    
- NaN은 자신과 일치하지 않는 유일한 값이다. 따라서 숫자가 NaN인지 조사하려면 빌트인 함수 `Number.isNaN`을 사용한다

```jsx
Number.isNaN(NaN); // -> true
Number.isNaN(10); // -> false
Number.isNan(1 + undefined); // -> true
```

### 2.1.1 [Object.is](http://Object.is) 메서드.

<aside>
👨🏽‍🦯 ES6에서 도입된 [Object.is](http://Object.is) 메서드는 다음과 같이 예측 가능한 정확한 비교 결과를 반환한다.
그 외에는 일치 비교 연산자(===)와 동일하게 동작한다

</aside>

```jsx
-0 === +0;  // -> true
Object.is(-1, +0); // -> false;

NaN === NaN;  // -> false
Object.is(NaN, NaN); // -> true
```

# 3. 삼항 조건 연산자

> 삼항 조건 연산자(tenary operator)는 조건식의 평가 결과에 따라 반환할 값을 결정한다.
> 

```jsx
조건식 ? 조건식이 true일 때 반활할 값 : 조건식이 false일 때 반환할 값

var result = score >= 60 ? 'pass' : 'fail'; //score >= 60일 때 pass, 아닐 때 fail
```

# 4. typeof 연산자

> typeof 연산자는 피연산자의 데이터 타입을 문자열로 반환한다.
> 
- 테이터 타입의 종류
    - string
    - number
    - boolean
    - undefined
    - symbol
    - object
    - function
- typeof 연산자로 null 값은 연산해 보면 “null”이 아닌 “object”를 반환하는 버그가 있다.
    - 따라서 값이 null타입인지 확인할 때는 typeof 연산자를 사용하지 말고 일치 연산자(===)를 사용해야 한다.

# 5. 지수 연산자

> ES7에서 도입된 지수 연산자는 좌항의 피연산자를 밑으로, 우항의 피연산자를 지수로 거듭 제곱하여 숫자 값을 반환한다
> 

```jsx
// 지수 연산자 사용
2 ** 2; // -> 4
2 ** -2; // -> 0.25

// Math.pow 메서드 사용 -> 경우에 따라 지수 연산자를 사용하는 것보다 가독성이 좋음
Math.pow(2, 2); // -> 4
Math.pow(2, -2); // -> 0.25

// 음수를 거듭제곱의 밑으로 사용해 계산하려면 괄호로 묶어야 한다
-5 ** 2; // Error 발생

(-5) ** 2; // -> 25
```