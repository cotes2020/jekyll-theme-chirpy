---
title: Javascript
date: 2025-01-01
categories: [Language, Javascript]
tags: [javascript, interpreter, 인터프리터]
---

## 웹의 브라우징
<hr>
본격적으로 JS에 대해 알아보기에 앞서 웹의 브라우징 과정 중 탐색, 응답을 제외한 나머지 과정에 대해 간략하게 알아보고자 한다.

1. 서버로부터 HTML을 받은 브라우저는 파서를 통해 **DOM(Document Object Model) 트리**로 변환한다.
    > 파싱 중간에 ```<script>``` 태그를 만나면 **파싱을 중단**하고 JS를 실행한 뒤 다시 이어간다. 
    {: .prompt-warning}
2. CSS도 마찬가지로 **CSSOM 트리**로 변환한다.
3. JS를 다운로드하고 이를 **추상 구문 트리(AST)**의 형태로 구문 분석이 된다.
    > CSS와 이미지는 html 파싱과 **병렬적인 처리**가 가능하지만 JS는 **동기적인 방식**으로 다운로드된다.
    {: .prompt-info}
4. DOM과 CSSOM을 합쳐 시각적으로 표시될 요소를 **랜더 트리**로 구성한 뒤 레이아웃을 계산하고 스타일을 적용한 뒤 브라우징한다.
5. 이후 랜더 트리를 기반으로 화면을 브라우징하며 사용자와 상호작용한다.

위의 과정에서 브라우징의 중요한 특징이 등장한다.

1. **어떤 대상과 상호작용하는 지 알 수 없다.**
2. **즉각적인 상호작용이 가능해야 한다.**

## 프로토타입
<hr>
MDN에서 Javascript는 프로토타입 기반의 언어[^1]라고 한다. 그렇다면 Javascript에서의 **프로토타입**이란 무엇일까?

프로토타입이란 **객체의 원형**을 의미한다. ES6부터 JS에도 *클래스*라는 개념이 도입되었으나 이 또한 프로토타입 기반에서 동작하기에 JS의 근본은 프로토타입이라 할 수 있다. 

<script src="https://gist.github.com/jjjung0921/40f56dd791eae92ee59c72842a0da1b6.js"></script>

위 코드를 확인하면 두 객체 모두 **같은 HTMLHeadingElement 프로토타입 객체**를 참조하고 있음을 알 수 있다.

<script src="https://gist.github.com/jjjung0921/c40d7d3560102c13df710dc89064de27.js"></script>

두 button 요소는 같은 프로토타입 객체를 참조하기에 하나의 프로토타입의 속성만 변경해도 둘 모두의 속성이 변경된다.


## 인터프리터
<hr>


[^1]: [MDN \| JavaScript](https://developer.mozilla.org/ko/docs/Web/JavaScript)
