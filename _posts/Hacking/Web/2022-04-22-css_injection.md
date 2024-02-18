---
title : CSS Injection
date: 2022-04-22 22:37 +0900
categories: [Hacking, Web]
tags: [CSS Injection]
---

## CSS Injection
<hr style="border-top: 1px solid;"><br>

CSS Injection이란, 표현에 사용될 임의의 CSS 코드를 주입시켜 의도하지 않은 속성이 정의되는 것이라고 한다.

CSS Injection은 XSS와 비슷하게 웹 페이지 로딩 시 악의적인 문자열을 삽입하여 악의적인 동작을 이끄는 공격으로, 공격자가 임의의 CSS 속성을 삽입해 웹페이지의 UI를 변조하거나 CSS 속성의 다양한 기능을 통해 웹 페이지내의 데이터를 외부로 훔칠 수 있다.

데이터의 예로는 CSRF Token, 피해자의 API Key등 웹 페이지에 직접적으로 보여지는 값처럼 CSS 선택자를 통해 표현이 가능해야 한다.    
그래서 CSS 선택자로 표현이 불가능한 ```script``` 태그 내 데이터들은 탈취할 수 없다. 

CSP를 우회해야하거나 CSP로 자바스크립트를 실행할 수 없을 때 등 다양한 상황에서 사용한다.

<br>

데이터를 외부로 탈취하기 위해서는 공격자의 서버로 요청을 보낼 수 있어야 한다.

CSS는 외부 리소스를 불러오는 기능을 제공한다. 예를 들어, 다른 사이트의 이미지나 폰트 등이 있다.

다양한 방법이 있다.
: <a href="https://github.com/cure53/HTTPLeaks/blob/main/leak.html#L266" target="_blank">github.com/cure53/HTTPLeaks/blob/main/leak.html#L266</a>

<br>

![image](https://user-images.githubusercontent.com/52172169/164727977-58a9caee-ebdb-4795-932f-55e6ed26a721.png)

<br>

데이터 탈취를 위한 방법으로는 CSS Attribute Selector (특성 선택자)를 이용하는 방법이 있다.

특성 선택자 
: <a href="https://developer.mozilla.org/ko/docs/Web/CSS/Attribute_selectors" target="_blank">developer.mozilla.org/ko/docs/Web/CSS/Attribute_selectors</a>

<br>

![image](https://user-images.githubusercontent.com/52172169/164728885-d3ae5bc6-363d-4e23-97a2-ce9d560decbb.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 출처
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://dreamhack.io/lecture/courses/327" target="_blank">Exploit Tech: CSS Injection</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
