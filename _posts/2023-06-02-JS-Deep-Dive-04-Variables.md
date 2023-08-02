---
title: 04 - Variables
author: wannastudyhardyeah
date: 2023-06-02 00:00:00 +0800
categories: [Javascript]
tags: [Javascript]

---
```js
10 + 20
```
<h2>1. 변수</h2>
<br>
계산(평가<span style="color: #808080;">evaluate</span>)<br>
기호(리터럴<span style="color: #808080;">literal</span>, 연산자<span style="color: #808080;">operator</span>)<br>
식(표현식<span style="color: #808080;">expression</span>)<br>
해석(파싱<span style="color: #808080;">parsing</span>)<br>
피연산자<span style="color: #808080;">operand</span><br>
<br>

변수<span style="color: #808080;">variable</span><br>
\: 하나의 값을 저장하기 위해 확보한 메모리 공간 자체<br>
또는<br>
그 메모리 공간을 식별하기 위해 붙인 이름

- 변수 이름(변수명)<br>
\: 메모리 공간에 저장된 값을 식별할 수 있는 고유한 이름

★ 변수에 여러 개의 값 저장하는 방법<br>
&nbsp; 여러 개의 값을 저장하기 위해선 여러 개의 변수를 사용해야 하지만,
배열이나 객체 등의 자료구조를 통해 여러 개의 값을 그룹화해서 하나의 값처럼 사용 가능.

```js
var userId = 5;
var userName = 'Kim';

var user = { id: 5, name: 'Kim' };

var users = [
    { id: 5, name: 'Kim' },
    { id: 10, name: 'Park' }
]
```

- 변수 값<br>
\: 변수에 저장된 값

- 할당<span style="color: #808080;">assignment</span>(대입, 저장)<br>
\: 변수에 값을 저장하는 것

- 참조<span style="color: #808080;">reference</span>
\: 변수에 저장된 값을 읽어 들이는 것


<h2>2. 식별자</h2>
식별자<span style="color: #808080;">identifier</span><br>
: \: 어떤 값을 구별해서 식별할 수 있는 고유한 이름<br>
변수 이름을 식별자라고도 한다.

★ 식별자는 값이 아니라 메모리 주소를 기억함.

<h2>3. 변수 선언</h2>
변수 선언<span style="color: #808080;">variable declaration</span><br>
: \: 변수를 생성하는 것.<br>
값을 저장하기 위한 메모리 공간을 확보<span style="color: #808080;">allocate</span>,<br>
변수 이름과 확보된 메모리 공간의 주소를 연결<span style="color: #808080;">name binding</span>해서<br>
값을 저장할 수 있게 준비하는 것.<br>
<br>
변수 사용하려면 반드시 선언이 필요함.
<br>
변수를 선언할 때는 <code class="language-plaintext highlighter-rouge" style="color: #a626a4;"><span class="kd" >var</span></code>, <code class="language-plaintext highlighter-rouge" style="color: #a626a4;"><span class="kd" >let</span></code>, <code class="language-plaintext highlighter-rouge" style="color: #a626a4;"><span class="kd" >const</span></code> 키워드를 사용함.<br>
(<code class="language-plaintext highlighter-rouge" style="color: #a626a4;"><span class="kd" >let</span></code>과 <code class="language-plaintext highlighter-rouge" style="color: #a626a4;"><span class="kd" >const</span></code>는 ES6에서 도입됨)

<h3>&nbsp;&nbsp;3.1. var 키워드로 변수 선언</h3>
```js
var score;
```
<br>
<b><span  style="color: red;">&nbsp;&nbsp;※키워드</span><span style="color: #808080;">keyword</span><br></b>
&nbsp;&nbsp;\: 자바스크립트 엔진은 키워드를 만나면 자신이 수행해야 할 약속된 동작을 수행한다.<br>

<b><span  style="color: red;">&nbsp;&nbsp;※undefined</span><br></b>
&nbsp;&nbsp;\: 자바스크립트에서 제공하는 원시 타입의 값<span style="color: #808080;">primitive value</span>.<br>
&nbsp;변수를 선언하고 값은 할당하지 않았으면, 이 메모리 공간은 비어 있는 것이 아니라<br>
&nbsp;자바스크립트 엔진에 의해 ``undefined``라는 값이 암묵적으로 할당되어 초기화된다.<br>

<b><span  style="color: red;">&nbsp;&nbsp;※변수 이름이 등록되는 곳</span></b><br>
&nbsp;&nbsp;\: 변수 이름을 비롯한 모든 식별자는 실행 컨텍스트<span style="color: #808080;">execution context</span>에 등록됨.<br>

<h3>&nbsp;&nbsp;3.2. ReferenceError(참조 에러)</h3>

식별자를 통해 값을 참조하려 했지만, 자바스크립트 엔진이 등록된 식별자를 찾을 수 없을 때 발생.<br>
(선언하지 않은 식별자에 접근 시 발생.)


<h2>4. 변수 선언의 실행 시점과 변수 호이스팅</h2>

```js
console.log(score);

var score;
```

```js
var hi;
```


참조 에러가 아닌, ``undefined``가 출력되는 이유<br>
\: 변수 선언이 런타임<span style="color: #808080;">runtime</span>(소스코드가 한 줄씩 순차 실행되는 시점)이 아니라 그 이전 단계에서 먼저 실행되기 때문.
