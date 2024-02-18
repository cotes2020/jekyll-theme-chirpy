---
title: Javascript Syntax 2
date: 2022-09-07 10:56  +0900
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

## Class
<hr style="border-top: 1px solid;"><br>

```javascript
class Name {
  constructor(parameter) { initial code; } // 생성자
}

let testFunc = new Name(); // new 키워드로 객체 생성
```

<br>

상속은 extends 키워드로 받을 수 있다.

<br>

```javascript
class Child extends Parent {
  constructor(parameter) {
    super(parameter);
  }
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Error
<hr style="border-top: 1px solid;"><br>

오류는 try catch문으로 관리한다.

try 블록에 작성된 코드를 실행하는 중에 예외가 발생하면 catch 블록이 실행되고, 어떤 에러가 발생했는지를 알 수 있다.

finally 블록은 try 블록과 catch 블록 실행을 마친 후 항상 실행되는 블록이다.

throw 블록은 의도적으로 오류를 발생시켜 catch 블록을 실행해야 할 때 사용하는 구문이다.

<br>

발생되는 에러의 종류는 다음과 같다.

<br>

+ EvalError 
  + 전역 함수 eval에서 발생

+ RangeError
  + 숫자 변수나 배열처럼 유효한 범위를 벗어났을 때 발생  

+ ReferenceError
  + 정의되지 않은 함수, 변수 등 잘못된 참조를 했을 때 발생

+ SyntaxError

+ TypeError
  + 변수나 매개변수가 유효한 자료형이 아닌 경우 발생

+ URIError
  + encodeURI나 decodeURI 함수에 적절하지 않은 신수를 제공했을 경우 발생

+ AggregateError
  + 하나의 동작이 여러 개의 오류를 발생시키는 경우 발생

+ InternalError
  + 자바스크립트 엔진 내부에서 오류가 난 경우 발생 

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 정규표현식
<hr style="border-top: 1px solid;"><br>

정규식을 만드는 방법으로 두 가지가 있는데, 정규식 리터럴을 사용하는 것과 RegExp 객체의 생성자 함수를 사용하는 방법이 있다.
: ```const regexp = /World/;```
: ```const regexp = new RegExp('World');```

<br>

정규식에서 제공하는 내장함수는 다음과 같다.

+ exec
  + 정규식을 통해 찾고자 하는 문자열 패턴을 찾고 배열을 반환.
  + 대응되는 문자열을 찾지 못하면 null을 반환

<br>

+ test
  + 대응되는 문자열이 있는지 검사하고 있으면 true, 없으면 false 반환

<br>

+ match
  + 대응되는 문자열을 찾아 배열로 반환하는 String 객체 내장 함수
  + 대응되는 문자열이 없으면 null 반환
  + exec와 동일한 기능

<br>

+ search
  + 대응되는 문자열이 있는지 검사
  + 대응되는 첫 번째 인덱스 반환, 없으면 -1 반환

<br>

+ replace
  + 대응되는 문자열을 찾아 다른 문자열로 치환하는 String 객체 내장 함수

<br>

+ split
  + 대응되는 문자열을 찾고, 찾은 문자열을 기준으로 나누어서 배열로 반환하는 String 객체 내장 함수

<br>

정규식 플래그를 사용해서 전여 검색, 대소문자 구분 없는 검색 등을 수행할 수 있다.
: ```g - 전역 검색 (대응되는 문자 전부 검색)```
: ```i - 대소문자 구분 없는 검색```
: ```m - 다중 행 검색```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## HTML DOM
### DOM 요소 접근
<hr style="border-top: 1px solid;"><br>

+ id 속성을 통한 요소에 대한 접근
  + ```document.getElementById('tagIdName')```
  + ```let element = document.getElementById('userId');```

<br>

+ tag명을 통한 요소에 대한 접근
  + ```document.getElementByTagName('tagName')```
  + ```let element = document.getElementByTagName('p');```

<br>

+ 클래스명을 통한 요소에 대한 접근
  + ```document.getElementByClassName('className')```
  + ```let element = document.getElementByClassName('para');``` 

<br>

+ CSS 선택자를 통한 요소에 대한 접근
  + ```document.querySelectorAll(css 선택자)```
  + ```const element = document.querySelectorAll('p.para');```

<br><br>

### DOM 속성 접근
<hr style="border-top: 1px solid;"><br>

위에서 요소에 접근을 한 다음 DOM 요소가 가지고 있는 속성 정보를 가져오거나, 속성 정보의 값을 변경할 수 있다.

사용자가 브라우저에 입력하는 데이터는 DOM 요소의 value 속성에 저장된다.

예를 들면, input 태그와 select 태그는 입력 값이 value 속성에 저장된다.

아래는 DOM 요소가 가지고 있는 속성을 읽고 변경하는 방법들이다.

<br>

+ getAttribute()
  + DOM 요소가 가지고 있는 속성 정보를 가져온다.

+ setAttribute()
  + DOM 요소가 가지고 있는 속성 정보를 설정한다.

+ hasAttribute()
  + DOM 요소에 특정 속성이 있는지 확인한다. 

+ removeAttribute()
  + DOM 요소의 특정 속성을 삭제한다.

<br><br>

### HTML 내용 변경
<hr style="border-top: 1px solid;"><br>

+ innerHTML
  + HTML의 특정 위치에 새로운 HTML을 삽입할 때 사용
  + ```javascript
    let sel = document.getElementById('sel');
    sel.innerHTML = '<option value=''>Select</option>';
    ```
    
<br>

+ innerText
  + innerHTML과 달리 텍스트 내용만 삽입이 가능하다.
  + ```document.getElementById('title').innerText = 'Hello';``` 

<br>

innerHTML과 innerText는 현재 접근한 DOM 요소 안에 자식 노드로 텍스트 혹은 HTML을 삽입하는 방법이다.

이렇게 하면 자식 노드가 모두 교체된다.

단순히 자식 노드의 제일 앞 혹은 제일 뒤에 새로운 HTML을 삽입해야 하는 경우엔 ```insertAdjacentHTML()```를 사용한다.

2개의 파라미터를 사용하는데, 첫 번째는 DOM 요소를 삽입할 위치이고 두 번째는 삽입할 DOM 요소에 대한 문자열이다.

<br>

DOM의 위치는 4개를 사용할 수 있다.

+ afterbegin
  + 접근한 DOM 요소의 자식 노드의 제일 첫 번재 노드로 삽입 

+ afterend
  + 접근한 DOM 요소 바로 다음 노드로 삽입

+ beforebegin
  + 접근한 DOM 요소 바로 직전 노드로 삽입

+ beforeend
  + 접근한 DOM 요소의 자식 노드의 제일 마지막 노드로 삽입 

<br>

```html
<h1>h1 area</h1>
<!-- beforebegin -->
<ul id='myUL'>
  <!-- afterbegin -->
  <li>A</li>
  <li>B</li>
  <li>C</li>
  <!-- beforeend -->
</ul>
<!-- afterend -->
<h2>h2 area</h2>
```

<br>

마찬가지로 insertAdjacentText()가 있으며, 텍스트만 삽입가능하다.

<br>

HTML 요소를 삭제할 땐, remove() 함수를 사용한다.
: ```let obj = document.getElementById('id'); obj.remove();```

<br><br>
<hr style="border: 2px solid;">
<br><br>
