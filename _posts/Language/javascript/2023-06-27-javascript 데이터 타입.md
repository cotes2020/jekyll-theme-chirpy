---
title: JavaScript 연산자
date: 2023-06-26 18:38:55 +0900
author: kkankkandev
categories: [Language, JavaScript]
tags: [web, javascript, js, gitblog, vscode, gitpage, datatype, javascript data type]     # TAG names should always be lowercase
comments: true
# image:
#   path: https://github.com/War-Oxi/Oxi/assets/72260110/3af8c7c9-cc3a-4fed-84d5-c736bad8ba53
---

# 데이터 타입

> 자바스크립트의 모든 값은 데이터 타입을 갖는다. ES6는 7개의 데이터 타입을 제공한다. 7개의 데이터 타입은 원시 타입(primitive type)과 객체 타입(object/reference type)으로 분류할 수 있다.
> 

| 구분 | 데이터 타입 | 설명 |
| --- | --- | --- |
|  | number 타입 | 숫자, 정수와 실수 구분 없이 하나의 숫자 타입만 존재 |
|  | string 타입 | 문자열 |
| 원시 타입 | boolean 타입 | 논리적 참과 거짓 |
|  | undefined 타입 |  var 키워드로 선언된 변수에 암묵적으로 할당되는 값 |
|  | null 타입 | 값이 없다는 것을 의도적으로 명시할 때 사용하는 값 |
|  | symbol 타입 | ES6에서 추가된 7번째 타입 |
| 객체 타입 |  | 객체, 함수, 배열 등 |

# 1. 숫자 타입

> C나 자바의 경우, 정수와 실수를 구분해서 int, long, float, double 등과 같은 다양한 숫자 타입을 제공한다. 하지만 자바스크립트는 독특하게 하나의 숫자 타입만 존재한다.
> 
- ECMAScript 사양에 따르면 숫자 타입의 값은 배정밀도 64비트 부동소수점 형식을 따른다.
    - 즉 **모든 수를 실수로 처리**하며, 정수만 표현하기 위한 데이터 타입이 별도로 존재하지 않는다.
- 자바스크립트는 2진수, 8진수, 16진수를 표현하기 위한 데이터 타입을 제공하지 않기 때문에 이들 값을 참조하면 모두 10진수로 해석된다.
    
    ```jsx
    var binary = 0b01000001 //2진수
    var octal = 0o101;      //8진수
    var hex = 0x41;         //16진수
    
    console.log(binary); //65
    console.log(octal);  //65
    console.log(hex);    //65
    console.log(binary === octal); //true
    console.log(octal === hex);    //true
    ```
    

# 2. 문자열 타입

> 문자열은 0개 이상의 16비트 유니코드 문자(UTF-16)의 집합으로 전 세계 대부분의 문자를 표한할 수 있다.
> 
- 문자열은 작은따옴표(’’), 큰따옴표(””) 또는 백틱(₩₩)으로 텍스트를 감싼다.
- 자바스크립트의 문자열은 원시 타입이며, 변경 불가능한 값(**immutable value**)이다.

# 3. 템플릿 리터럴(template literal)

> 멀티라인 문자열(multi-line string), 표현식 삽입(expression interpolation), 태그드 템플릿(tagged template) 등 편리한 문자열 처리 기능을 제공한다.
> 
- 템플릿 리터럴은 일반 문자열과 비슷해 보이지만 백틱(₩₩)을 사용해 표현한다.
    
    ```jsx
    var template = `Template literal`;
    ```
    
- 템플릿 리터럴 내에서는 표현식 삽입(expression interpolation)을 통해 간단히 문자열을 삽입할 수 있다.

```jsx
//템플릿 리터럴 미사용시.
var first = 'Ung-mo';
var last = 'Lee';

console.log('My name is ' + first + ' ' + last + '.'); //My name is Ung-mo Lee.

//템플릿 리터럴 사용시.
var first = 'Ung-mo';
var last = 'Lee';

console.log(`My name is ${first} ${last}.`); //My name is Ung-mo Lee.
```

# 4. 불리언 타입

> 논리적 참, 거짓을 나타내는 true와 false
> 

# 5. undefined 타입

> javascript 엔진이 변수를 초기화 할 때 사용하는 값.
> 
- 변수에 값이 없다는 것을 명시하고 싶을 때는 undefined를 할당하는 것이 아니라 null을 할당하는 것이 바람직함.

# 6. null 타입

> 변수에 값이 없다는 것을 의도적으로 명시할 때 사용
> 
- 변수에 null을 할당하는 것은 변수가 이전에 참조하던 값을 더 이상 참조하지 않겠다는 의미.
    - 이전에 할당되어 있던 값에 대한 참조를 명시적으로 제거 ⇒ 자바스크립트 엔진은 누구도 참조하지 않는 메모리 공간에 대해 가비지 콜렉션을 수행.
- 함수가 유효한 값을 반환할 수 없는 경우에도 명시적으로 null을 반환하기도 한다.

# 7. Symbol 타입

> 변경 불가능한 원시 타입의 값.
> 
- Symbol 값은 다른 값과 중복되지 않는 유일무이한 값이다.
- 주로 이름이 충돌할 위험이 없는 객체의 유일한 프로퍼티 키를 만들기 위해 사용
- Symbol은 Symbol 함수를 호출해 생성한다.
    - 이때 생성된 Symbol 값은 외부에 노출되지 않으며, 다른 값과 절대 중복되지 않는 유일무이한 값이다.
    
    ```jsx
    var key = Symbol('key');
    console.log(typeof key); //symbol
    
    var obj = {};
    
    obj[key] = 'value';
    console.log(obj[key]); //value
    ```
    

# 8. 객체(object) 타입

> 자바스크립트는 객체 기반의 언어이며, 자바스크립트를 이루고 있는 거의 모든 것이 객체이다.
> 

# 9. 데이터 타입의 필요성

1. 값을 저장할 때 확보해야 하는 메모리 공간의 크기를 결정하기 위해.
2. 값을 참조할 때 한 번에 읽어 들여야 할 메모리 공간의 크기를 결정하기 위해
3. 메모리에서 읽어 들인 2진수를 어떻게 해석할지 결정하기 위해

## 9.1 데이터 타입에 의한 메모리 공간의 확보와 참조

> 값은 메모리에 저장하고 참조할 수 있어야 한다. 메모리에 값을 저장하려면 먼저 확보해야 할 메모리 공간의 크기를 결정해야 한다. 다시 말해, 몇 바이트의 메모리 공간을 사용해야 낭비와 손실 없이 값을 저장할 수 있는지 알아야 한다.
> 
- 값을 참조하기 위해서는 한 번에 읽어 들여야 할 메모리 공간의 크기, 즉 메모리 셀의 개수(바이트 수)를 알아야 한다.
- 컴퓨터는 한 번에 읽어 들여야 할 메모리 셀의 크기를 어떻게 알 수 있는 것일까?
    - ⇒ 변수에는 데이터 타입의 값이 할당되어 있으므로 데이터 타입에 해당하는 메모리 공간에 저장된 값을 읽어 들인다.

*심벌 테이블

<aside>
👨🏽‍🦯 컴파일러 또는 인터프리너는 심벌 테이블이라고 부르는 자료 구조를 통해 식별자를 키로 바인딩된 값의 메모리 주소, 데이터 타입, 스코프 등을 관리

</aside>

# 10. 동적 타이핑(dynamic typing)

> 자바스크립트의 변수는 선언이 아닌 할당에 의해 타입이 결정(타입 추론(type inference))된다.
그리고 재할당에 의해 변수의 타입은 언제든지 동적으로 변할 수 있다.
> 
- **동적 타입 언어는 유연성은 높지만 신뢰성은 떨어진다**

### 변수를 사용할 때 주의할 사항

1. 변수는 꼭 필요한 경우에 한해 제한적으로 사용한다.
2. 변수의 유효 범위(Scope)는 최대한 좁게 만들어 변수의 부작용을 억제해야 한다.
3. 전역 변수는 최대한 사용하지 않도록 한다
4. 변수보다는 상수를 사용해 값의 변경을 억제한다.
5. 변수 이름은 변수의 목적이나 의미를 파악할 수 있도록 네이밍한다.