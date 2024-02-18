---
title : Google XSS Game
categories: [Wargame, Google XSS Game]
tags : [Google XSS Game]
---

## Google XSS Game 
<hr style="border-top: 1px solid;">

<a href="https://xss-game.appspot.com/?utm_source=webopsweekly&utm_medium=email" target="_blank">Google XSS Game</a>

<br>
<br>
<hr style="border: 2px solid;">
<br>
<br>

## Lv.1 Hello, world of XSS
<hr style="border-top: 1px solid;">


입력값을 이스케이프 처리를 하지 않는다고 함. 

문제 클리어 조건은 alert()를 실행하면 됨.

<br>

```js
<script>alert()</script>
```

<br>
<br>
<hr style="border: 2px solid;">
<br>
<br>

## Lv.2 Persistence is key
<hr style="border-top: 1px solid;">

```<script>```를 입력해보면 입력이 안됨.

우선 ```defaultmessage```를 보면 html tag를 사용 가능함을 알 수 있음.

html tag에는 error 발생 시 javascript를 실행시킬 수 있는 ```onerror```라는 속성이 있음.

<br>

```
Example

Execute a JavaScript if an error occurs when loading an image:
<img src="image.gif" onerror="myFunction()">
```

<br>

따라서 ```onerror```를 이용해 ```alert```를 해주면 됨.

<br>

```html
<img src="image.gif" onerror=alert()>
```

<br>
<br>
<hr style="border: 2px solid;">
<br>
<br>

## Lv.3 That sinking feeling...
<hr style="border-top: 1px solid;">

```html += "<img src='/static/level3/cloud" + num + ".jpg' />";```

<br>

이 코드를 보면 ```num```이 있는데 이곳이 입력값이 들어갈 부분임.  

이미지를 클릭해보면 url은 ```/frame/#1``` 이렇게 됨.  

<br>

위 코드에서 img 태그를 닫아버릴꺼임.

```'/>```로 닫아버리면 이미지 url을 보면 아래처럼 되서 에러 발생.

<br>

+ The requested URL /static/level3/cloud1 was not found on this server.

<br>

또한 이미지가 나타나던 부분에는 다음처럼 보여질꺼임.

<br>

```
Image 1
.jpg' />
```

<br>

따라서 img 태그를 닫아버리고 alert 스크립트를 추가해주면 됨.

<br>

```sql
/frame/#1'/><script>alert()</script>
```

<br>
<br>
<hr style="border: 2px solid;">
<br>
<br>

## Lv.4 Context matters
<hr style="border-top: 1px solid;">

```<script>alert()</script>```를 입력하면 문제 설명에도 있는 것처럼 이스케이프 처리가 되어버림. ```onload``` 속성에서 여러 개를 주면 됨.

+ ```onload="alert('1'); alert('2')"``` -> 1 뜨고 2 뜸.

<br>

```html
<img src="/static/loading.gif" onload="startTimer('{{ timer }}');" />
```

위 코드를 보면 ```timer``` 부분에 값이 들어가므로 여기에 alert를 주면 됨.

<br>

```html
<img src="/static/loading.gif" onload="startTimer(''); alert('1');" />
```

<br>

위에 처럼 하면 함수 실행 후 ```alert('1')```이 될꺼임. ```?timer='); alert('```

<br>

근데 안뜸 -> urlencode를 해봄. ```?timer=')%3b alert(' -> 성공```

<br>
<br>
<hr style="border: 2px solid;">
<br>
<br>

## Lv.5 Breaking protocol
<hr style="border-top: 1px solid;">

url javascript를 이용하면 됨. 

```<a href="{{ next }}">Next >></a>``` 부분이 있는데 url을 보면 ```?next=confirm```으로 되어있을꺼임. 이 값을 바꾸면 됨.

<br>

**input : ```?next=javascript:alert()```**

<br>
<br>
<hr style="border: 2px solid;">
<br>
<br>

## Lv.6 Follow the 🐇
<hr style="border-top: 1px solid;">

힌트에서 ```google.com/jsapi?callback=foo```라고 되어있음. 아마도 함수를 호출해주는 건가 봄.

근데 문제에서 ```https://```는 필터링함. 하지만 대소문자 구분은 하지 않음.

처음엔 ```:```를 두 번 썼는데 안되어서 ```Https```로 해주니 성공.

<br>

**input : ```Https://www.google.com/jsapi?callback=alert```**

<br>
<br>
