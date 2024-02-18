---
title : Javascript Syntax 1
date: 2022-04-28 22:00 +0900
categories: [Programming, Javascript]
tags: [Javascript Syntax]
---

## 참고
<hr style="border-top: 1px solid;"><br>

Link
: 바닐라 자바스크립트(저자 고승원)
: <a href="https://www.programiz.com/javascript" target="_blank">www.programiz.com/javascript</a>
: <a href="https://boycoding.tistory.com/category/자바스크립트%20이야기" target="_blank">boycoding.tistory.com/category/자바스크립트%20이야기</a>
: <a href="https://www.scaler.com/topics/javascript/" target="_blank">https://www.scaler.com/topics/javascript/</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 변수
<hr style="border-top: 1px solid;"><br>

변수에는 let, var이 있다. 두 변수의 차이점은 아래와 같다.

![image](https://user-images.githubusercontent.com/52172169/165768873-e546367e-7d73-40c4-912d-bfbf65887774.png)

<br>

var로 선언된 변수는 같은 변수명으로 재선언 하는 것이 가능하나, let으로 선언된 변수는 불가능하다.

let이 더 최신 버전이어서 let을 사용하는 것이 좋다.

<br>

변수를 초기화하지 않으면 ```undefined``` 값을 갖게 된다.

변수명은 영문자, underline```(_)```, ```$```로 시작해야 하고 숫자로는 시작할 수 없다. 또한 키워드가 변수명이 될 수 없다. 

자바스크립트에서는 대소문자를 구분하고 있어서 ```let y```와 ```let Y```는 서로 다른 변수이다.

<br>

c, c++에도 ```const``` 가 있듯이 자바스크립트에도 있다.

```const``` 변수는 초기화 한 후에는 다른 값으로 초기화 할 수 없고, 선언과 동시에 초기화를 시켜줘야 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Data Type
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/165772010-ca02b029-be57-42c0-ae9b-6f44d3b57806.png)

<br>

Symbol에 대해서는 아래 내장 객체에서 정리를 해놨고, Symbol은 자바스크립트에서 절대 충돌이 일어나지 않는 유일한 key 값을 만들어준다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 64비트 부동소수점
<hr style="border-top: 1px solid;"><br>

자바스크립트는 숫자를 64비트 부동소수점으로 저장한다.

그래서 예를 들어 ```0.1 + 0.2 = 0.3```이 아니라 ```0.1 + 0.2 = 0.30000000000004```와 같이 나온다.

<br>

![image](https://user-images.githubusercontent.com/52172169/187865230-0e4a3930-93da-4ecd-b363-4b112bdb27da.png)

<br>

64비트 부동소수점은 위와 같은 구조로 되어 있으며, 이 구조를 토대로 0.1과 0.2를 더해서 0.3000000004가 나오는 것이다.

그래서 이러한 부분을 해결하기 위해서 소수점 이하 두 자리에서 다섯 자리를 넘어가지 않게 사용해주면 된다.

즉, 결과 값을 소수점 이하 5자리를 기준으로 잘라내주면 원하는 값인 0.3이 나온다.

```Math.pow()``` 함수를 이용하면 된다.

<br>

또한 숫자에는 사용 가능한 범위가 있기 때문에, overflow를 방지하기 위해서 개발된 오픈 소스 라이브러리가 있다.
: Big.js, BigNumber.js, Decimal.js

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 조건문
<hr style="border-top: 1px solid;"><br>

조건문에는 ```if - else if - else``` 가 있다.

조건식을 거짓으로 취급하는 값에는 ```false, undefined, null, 0, NaN, 비어 있는 문자열("")```가 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 반복문
<hr style="border-top: 1px solid;"><br>

반복문에는 ```for-loop, for-in, for-of, while```가 있다.

<br>

```javascript
// for loop == c언어의 for문과 동일

for (statement 1; statement 2; statement 3) {
  // code block
}

// statement 1 : 코드 블록이 실행되기 전에 한 번 실행됨
// statement 2 : 코드 블록을 실행키실 조건 정의
// statement 3 : 코드 블록을 실행한 후 실행됨
```

<br>

```javascript
// for-in

for (const key in object) { code block }

/*

for-in 문은 배열뿐만 아니라 Object에도 사용할 수 있는 반복문이다.

Object는 데이터를 저장할 때 키를 사용하여 저장하고, 키를 이용해서 읽는다.

for-in 문은 Object 내에 정의된 키 값의 수만큼 코드 블록을 실행한다.

배열에 사용할 때는 배열에 저장된 데이터 개수만큼 반복문을 실행한다.

*/

let example = { a: "test1", b: "test2", c: "test3" };

for (const k in example) 
{
  console.log(example[k])
}
```

<br>

```javascript
// for-of

for (const element of iterable)  { code block; }

/*

for-of 문은 Array, Map, String 등 itreable(반복 가능한) 
객체에서 사용 가능한 반복문이다.

*/
```

<br>

```javascript
// while, do-while

while (condition) { code block; }

do { code block } while (condition) 

// do while문은 코드 블럭을 무조건 한 번 실행 후 조건식을 확인한다.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 함수
<hr style="border-top: 1px solid;"><br>

```javascript
function FuncName(argv1, argv2, ..., ) 
{
  code block;
  return value;
}
```

<br>

함수에 대한 주석을 처리하는 방법에는 함수 바로 윗줄에 ```/**```라고 입력하면 ```/** */``` 코드 가이드가 나오며, 엔터 키를 입력하면 함수의 파라미터 개수와 return 포함 여부에 따라 자동 완성된다.

이걸 알면 고수가 된다.

<br>

함수를 선언하는 다른 방식으로 함수 표현식이 있다.

함수 표현식은 변수에 함수를 할당해서 사용하는 방식이다.
: ```let sum = function(p1,p2) { return p1+p2; }```

<br>

이렇게 선언하면 함수를 호출할 때 변수명을 사용하면 된다.
: ```let sum1 = sum(21,24)```

<br>

두 방식의 차이점은 실행 방식에 있다.

일반적으로 함수를 선언하는 방식은 함수를 선언하는 코드가 함수를 호출하는 코드보다 아래에 있더라도 정상적으로 실행된다.

그 이유는 자바스크립트 해석 엔진이 일반적인 함수 방식으로 선언되는 함수의 경우 먼저 해석을 하기 때문이다.

<br>

하지만, 변수에 함수를 할당하는 방식은 작성된 코드 순서대로 실행된다.

따라서 함수를 호출하는 변수가 함수를 할당하는 변수보다 위에 있는 경우, 에러가 발생한다.

<br>

또 다른 방식으로 Function 생성자 함수를 사용하는 것이다.
: ```let sum = new Function("p1", "p2", "return p1 + p2;");```

<br>

자바스크립트의 내장 함수인 Function 함수에 파라미터와 코드 블록을 문자열로 순서대로 전달하여 생성한다.

전달할 파라미터가 없다면 실핼할 코드 블록만 문자열로 전달하면 된다.

이러한 방식을 사용하는 경우는 코드 블록을 동적으로 생성하고 싶을 때 유리하다.

예를 들어, 계산기는 사용자가 숫자와 연산자를 선택한 후 연산에 대한 결과를 호출해야 한다.

각 연산자를 적용하기 위해서 사용자가 선택한 연산자가 무엇인지 조건문을 사용해 확인해야 하지만, Function 생성자를 통해서 함수를 생성하면 효율적으로 작성할 수 있다.

<br>

함수 중에 callback 함수가 있다.

callback 함수는 배열의 각 요소를 시험할 함수이다.

시험을 통과하면 요소를 그대로 유지하고, false라면 버리게 된다.

즉, true인 요소만 찾아낸다.

<br>

```callback(element[, index[, array]])```
: ```element 는 처리할 현재 요소```
: ```index(optional) 는 처리할 현재 요소의 인덱스```
: ```array(optional) 는 배열 전체```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 내장 객체
<hr style="border-top: 1px solid;"><br>

내장 객체가 가지가 있는 다양한 함수 기능에 대해 이해하고 있어야 한다.

<br>

+ Object 객체
  + Object 객체는 모든 자바스크립트 객체의 루트(최상위) 객체이다.
  + 모든 객체의 객체이다. 

<br>

+ String 객체
  + 문자열을 다루기 위한 객체이다.
  + 다양한 문자열 프로퍼티와 함수를 제공한다.

<br>

+ Number 객체
  + 숫자를 다루기 위한 객체이다.
  + 다양한 프로퍼티와 함수를 제공하는 wrapper 객체이다.
  + wrapper 객체란 원시 타입의 값을 감싸는 형태의 객체이다.
  + 자바스크립트에서는 정수 실수 구분이 없으며, 모든 수를 실수 하나로 표현하며 모든 숫자는 64비트 부동 소수점 수로 저장된다.

<br>

+ Array 객체

<br>

+ Date 객체
  + 날짜와 시간을 다루는 객체이다.
  + 사용자 브라우저의 타임존을 기준으로 날짜와 시간을 보여준다.

<br>

+ Set 객체
  + 배열처럼 값들의 집합이다.
  + 배열처럼 데이터 타입에 상관없이 값을 추가할 수 있다.
  + 차이점은 중복된 값을 허용하지 않는다는 것으로, 유일한 값을 보장한다.

<br>

+ Map 객체
  + Object와 매우 유사한데, 키와 값을 맵핑시켜서 값을 저장한다.
  + 저장된 순서대로 각 요소에 접근할 수 있다.

<br>

+ Math 객체
  + 수학적인 상수와 내장 함수를 가진 객체이다.
  + 다른 객체와 달리 Math는 생성자가 아니다.
  + 숫자 자료형만 지원한다.

<br>

+ JSON 객체
  + 데이터를 저장하거나 전송할 때 많이 사용되는 객체이다.
  + JSON은 데이터 포맷일 뿐, 특정 통신 방법도 프로그래밍 문법도 아닌, 단순히 데이터를 표시하는 방법이다.
  + 서버와 클라이언트 간의 데이터 전송 시 많이 사용된다.
  + 데이터를 서버로 전송하기 위해서는 데이터 형태를 문자열 형태로 변환해야 한다. 
  + 자바스크립트의 Object 객체 표기법과 매우 유사하다.
  + JSON 데이터는 자바스크립트 JSON 객체의 ```parse()``` 함수를 이용하면 자바스크립트 Object 객체로 변환해서 사용할 수 있다.
  + key-value 형태로 key는 반드시 쌍따옴표를 이용해서 표기해야 한다.
  + 2개의 중요한 내장 함수가 있다.
    + ```JSON.stringify```는 데이터 형태를 문자열 형태로 변환한다.
    + ```JSON.parse```는 서버로부터 응답받은 데이터(문자열 형태)를 자바스크립트 Object 객체로 변환해준다. 

<br>

+ Window 객체
  + window 객체는 global 객체이다.

<br>

+ Symbol 객체
  + 자바스크립트에서 절대 충돌이 일어나지 않는 유일한 key 값을 만들어준다.
    + 만약 내가 생성한 오브젝트에서 등록해둔 키가 어느 날 오브젝트의 기본 내장 함수로 추가가 된다면 의도치 않은 충돌이 발생하고, 원하는 결과를 얻지 못할 수 있다.
  + Symbol로 등록을 해주면 원래는 for-in문으로 출력했을 때, 키 값으로 나오던 값이 Symbol이 되면 나오지 않게 된다. 
  + ```let getFullName = Symbol("getFullName");```
  
<br><br>
<hr style="border: 2px solid;">
<br><br>

## 고급 문법

### this 키워드
<hr style="border-top: 1px solid;"><br>

this 키워드는 사용되는 위치에 따라 this 키워드에 바인드 된 객체가 달라진다.

<br>

+ ```<script>``` 태그 내에서 사용되면 window 객체가 된다.

<br>

+ HTML DOM 요소에서 on 이벤트가 발생할 때 호출하는 함수의 파라미터로 this를 전달하면 HTML DOM 그 자체가 된다.
  + ```<button type"button" onclick="callFunc(this);">Click</button>```에서 this는 ```<button>``` 태그 그 자체가 된다.

<br>

+ Object 내에 정의된 다른 키에 접근 시 this로 접근 가능하다.
  + ```fullName: function() { return this.firstName + " " + this.lastName; }``` 

<br><br>

### Default Function Parameter
<hr style="border-top: 1px solid;"><br>

함수를 호출할 때 함수의 기본 파라미터를 지정할 수 있다.

<br>

```javascript
function say(message) { code block; }

function say(message="No Value") { code block; }
// 파라미터 값이 안오면 미리 설정한 값으로 전달
```

<br><br>

### Rest Parameter
<hr style="border-top: 1px solid;"><br>

예를 들어, sum 함수가 있을 때, 2개의 값을 더할 때의 함수와 4개의 값을 더할 때의 함수를 각각 정의해줘야 하므로 비효율적이다.

Rest Parameter는 몇 개의 파라미터가 전달될 지 모를 경우 매우 유용하다.

<br>

```javascript
function sum(...args) // args 라는 배열에 파라미터가 저장됨
{ 
  for (let x of args) { console.log(x); }
}

sum(1,2);
sum(1,2,3,4,5);
```

<br><br>

### Arrow Function
<hr style="border-top: 1px solid;"><br>

Arrow Function은 함수를 정의하는 새로운 방법이다.

화살표 함수를 사용해서 함수를 정의하면 함수 선언식이나 함수 표현식에 비해 구문이 짧아진다는 이점이 있다.

화살표 함수 내의 this는 언제나 상위 스코프의 this를 가리킨다.

<br>

```javascript
function hello(name) 
{
  return "Hello" + name;
}

// or 

const hello2 = function(name) { return "Hello" + name; };

const hello3 = new Function("name", "return 'Hello' + name;");

/* 원래 방식은 위의 3가지 */

const hello4 = (name) => {return "Hello" + name} ;
// 위의 일반적인 함수 표현을 Arrow Function으로 나타낸 것

const hello4 = name => {return "Hello" + name};
// 인자가 하나라면 괄호 생략 가능

const hello4 = name => "Hello" + Name;
// 함수의 유일한 문장이 return이면 return과 중괄호 생략 가능
```

<br><br>

### Template Literals
<hr style="border-top: 1px solid;"><br>

변수에 할당된 문자열을 하나의 문자열로 병합할 때, 더하기를 사용하지 않고 하나의 문자열로 만들 수 있도록 해준다.

문자열에 백틱을 사용하며 문자열 안에서 변수 값에 ```${변수명}```을 사용하면 더하기 기호 없이 바로 적용할 수 있다.

<br>

```javascript
let name = "John";
console.log(`Hello ${name}`);
```

<br><br>

### Object Literal Syntax Extension
<hr style="border-top: 1px solid;"><br>

Object에서는 변수에 할당된 값을 키로 치환해서 사용할 수 없지만, Object Literal Syntax Extension 문법을 사용하면 Object의 키로 변수에 할당된 문자열 값을 사용할 수 있다.

이 때, 키를 대괄호(```[]```) 안에 넣어주면 이때의 키는 할당된 변수 값을 의미하게 된다.

<br>

```javascript
let type = "student";

let score = {
  [type]: "Jonh",
  score: 95
};
```

<br><br>


### Spread Operator
<hr style="border-top: 1px solid;"><br>

Spread Operator는 배열, 문자열과 같이 iterable한 형태의 데이터를 요소 하나하나로 모두 분해해서 사용할 수 있게 해준다.

<br>

```javascript
let arr1 = [1,2,3]
let arr2 = [4,5,6]

let arr3 = [...arr1, ...arr2]
console.log(arr3) // [1,2,3,4,5,6]

let cd = "CD";
let alphabet = ['a','b', ...cd];

console.log(alphabet) // ['a','b','C','D']
```

<br><br>

### Destructuring
<hr style="border-top: 1px solid;"><br>

Object나 Array에 저장된 데이터를 분해해서 사용할 수 있다.

<br>

```javascript
function getPerson() {
  return {
    firstName: "John",
    lastName: "Doe",
    age: 37
  };
}

let {firstName, lastName} = getPerson();

console.log(firstName); // John
console.log(lastName); // Doe
```

<br>

```javascript
function getScores() { return [7,8,9]; }

let scores = getScores();
// scores[0] = 7, scores[1] = 8, scores[2] = 9

let [x,y,z] = getScores();
// x=7, y=8, z=9

// 반환하는 데이터가 많은 경우
let [x,y, ...args] = getScores();
// x=7, y=8, args = 9, 10, 11, ..., N
```

<br><br>

### Fetch API
<hr style="border-top: 1px solid;"><br>

Fetch API는 네트워크 통신을 포함한 리소스 취득을 위한 인터페이스가 정의되어 있다.

Fetch API는 Promise 방식으로 구현되어 있다. (비동기 방식)

<br>

+ 데이터 요청 (GET)
  + fetch 함수를 통해 데이터를 요청하고, 응답이 이루어지면 응답 결과를 then 함수의 인수로 전달받게 된다.
  + ```javascript
    fetch('http://this_is_url')
     .then( (response) => response.json() )
     .then( (json) => console.log(json) )
    ```

<br>

+ 데이터 생성 (POST)
  + method를 POST로 설정해주고 header 값을 추가해줘야 한다.
  + ```javascript
    fetch('http://this_is_url', {
     method: 'POST',
     body: JSON.stringify( {
       title: 'foo',
       body: 'bar',
       userId: 1,
     }),
     headers: {
       'Content-Type': 'application/json; charset=UTF-8',
     },
    }).then( (response) => response.json())
     .then( (json) => console.log(json) );
    ```

<br>

+ 파일 업로드
  + FormData 객체를 사용, HTML의 form 태그에 해당하는 form 필드와 그 값을 나타내는 일련의 키-값 쌍을 쉽게 생성해주는 객체이다.
  + 일반적인 텍스트 데이터뿐만 아니라 파일을 서버로 전송할 수 있게 해준다.
  + ```javascript
    let formData = new FormData();
    let fileField = document.querySelector('input[type="file"]');
    
    formData.append('username','abc123'); // 텍스트 데이터
    formData.append('attachment', fileField.files[0]); // 파일
    
    fetch('url', {
     method: 'POST', 
     body: formData
    })
    .then(response => response.json())
    .catch(error => console.error('Error', error))
    .then(response => console.log('Success:', JSON.stringify(response)));
    ```
  + 두 개 이상의 파일도 보낼 수 있다.
  + ```javascript
    let formData = new FormData();
    let photos = document.querySelector('input[type="file"][multiple]'); // 다중 파일 선택 HTML 요소
    
    formData.append('title','My photos');
    for (let i = 0; i < photos.files.length; i++) {
     formData.append('photos', photos.files[i]); // 선택한 파일 수 만큼 반복문으로 FormData에 삽입
    }
    
    fetch('url', {
     method: 'POST',
     body: formData
    })
    .then(response => response.json())
    .catch(error => console.error('Error', error))
    .then(response => console.log('Success:', JSON.stringify(response)))
    ```

<br><br>

### Promise
<hr style="border-top: 1px solid;"><br>

Promise는 자바스크립트에서 비동기 처리에 사용되는 객체이다.

비동기 처리란 특정 코드의 실행이 완료될 때까지 기다리지 않고 다음 코드를 실행할 수 있게 해주는 방식이다.

웹은 원래 요청에 대한 반응이 순차적으로 이루어지고, 먼저 실행된 코드가 실행이 완료되어야 다음 코드를 실행한다.

그러나 자바스크립트에 ajax가 추가되면서 XMLHTTPRequest 통신이 가능해지고, 서버로 요청을 보낸 후 응답을 받을 때까지 기다릴 필요 없이 다음 코드를 실행할 수 있게 되었다.

<br>

요청에 대한 응답이 성공적으로 오면 resolve 함수에 결과를 전달하고, 실패하면 reject 함수에 에러를 전달한다.

<br>

```javascript
const promise = new Promise((resolve, reject) => {
 if(/* 처리 성공 */) {
  resolve('결과 데이터');
 }
 else {
  reject(new Error('에러'));
 }
});
```

<br>

XMLHttpRequest 객체도 비동기 통신이므로 요청에 대한 응답이 오기 전이어도 다음 코드를 실행해준다.

그래서 서버로부터 응답을 받은 데이터를 이용해 구현해아 하는 코드가 있는 경우 문제가 발생할 수 있으므로, 이 때 Promise 객체를 사용하여 응답이 완료된 후 호출되는 then 함수를 통해 결과를 받고 나서 구현해야 할 코드를 작성해서 사용할 수 있다.

즉, 비동기 함수의 실행이 완료되면 then 함수를 통해서 그 결과에 대한 코드를 실행할 수 있다.

<br>

```javascript
function getData() {
 return new Promise((resolve, reject) => {
  const xhr = new XMLHttpRequest();
  xhr.open('GET', url);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send();
  
  xhr.onload = () => {
   if (xhr.status == 200) {
    const res = JSON.parse(xhr.response);
    resolve(res);
   }
   else {
    console.error(xhr.status, xhr.statusText);
    reject(new Error(xhr.status));
   }
  };
 });
}

getData().then((res) => {
  console.log(res);
  console.log("Next Code Execute");
});
```

<br><br>

### Async/Await
<hr style="border-top: 1px solid;"><br>

Promise와 동일한 목적으로 사용한다.

비동기 함수를 호출할 때 함수 앞에 await을 정의하면 비동기 함수가 실행되고, 서버로부터 응답을 받을 때까지 대기(await) 한 후 결과를 받으면 실행되도록 해준다.

<br>

```javascript
async function test() {
 const res1 = await fetch('http://jsonplaceholder.typicode.com/posts/1');
 const res1Json = await res1.json();
 console.log(res1JSon);
 
 const res2 = await fetch(
   'http://jsonplaceholder.typicode.com/posts/1',
   {
     method: 'PUT',
     body: JSON.stringify({
      id: 1,
      title: "foo",
      body: 'bar',
      userId: 1,
     });
     headers: {
      'Content-Type': 'application/json; charset=UTF-8';
     },
   }
 );
 const res2Json = await res2.json();
 console.log(res2Json);
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

<br>

Javascript Syntax 2
: <a href="https://ind2x.github.io/posts/javascript_syntax2" target="_blank">ind2x.github.io/posts/javascript_syntax2</a>

<br><br>
