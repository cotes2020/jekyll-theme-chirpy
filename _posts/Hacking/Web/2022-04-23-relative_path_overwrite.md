---
title : Relative Path Overwrite (RPO)
date: 2022-04-23 17:05 +0900
categories: [Hacking, Web]
tags: [Relative Path Overwrite]
---

## Relative Path Overwrite
<hr style="border-top: 1px solid;"><br>

일부 웹 애플리케이션은 편의를 위해 url rewrite 기능을 사용한다고 한다.

url rewrite 기능을 사용하면 ```https://host/search_user_by_name.php?name=alice``` 와 같은 형태의 url을 ```https://host/search/alice/``` 와 같은 형태로 훨씬 기억하기 쉽고, 간결하게 만드는 것이 가능하다고 한다..

하지만 이 때 브라우저의 입장에서는 URL의 경로 중 어디부터가 파라미터인지 구별할 수 없다.

만약 이용자가 입력한 파라미터를 브라우저가 경로로 인식하여 해당 경로로부터 자원을 불러오게 되면 의도치 않은 동작이 발생할 수 있다.

<br>

**같이 읽기!!!!!!!!!!!!** 
: <a href="https://blog.rubiya.kr/index.php/2019/04/17/relative-path-overwrite/" target="_blank">blog.rubiya.kr/index.php/2019/04/17/relative-path-overwrite/</a>

<br>

Relative Path Overwrite (RPO)는 일반적으로 서버와 브라우저가 상대 경로를 해석하는 과정에서 발생하는 차이점을 이용한 공격이다.

즉, **브라우저와 서버가 상대경로를 해석하는 과정에서의 동작 차이를 악용한 공격 기법**이다.

<br>

+ 경로
  + 절대경로는 프로토콜과 도메인 이름을 포함한 목적지 주소의 전체 URL을 의미한다.

  + 상대경로는 목적지의 프로토콜이나 도메인을 특정하지 않는다. 

<br>

예를 들어 아래와 같은 코드가 있다고 가정.

```javascript
<script src="/app/main.js"></script>
<script src="app/main.js"></script>
```

<br>

첫 번째 줄은 앞에 존재하는 ```/```에 의해 스크립트 로드 시 최상위 경로부터 시작하여 탐색 (절대 경로)하고 로드한다.

반면에 두 번째 줄의 스크립트는 현재 경로에서 시작하여 탐색 (상대 경로)하고 로드한다.

test.php 파일이 있을 때, ```http://host/test.php```로 접근하면 같은 경로에 있는 ```/app/main.js```를 로드한다.

하지만, ```http://host/test.php/```로 접근하면 첫 번째 스크립트는 최상위 경로에서 탐색하는 반면, 두 번째 스크립트는 현재 경로가 ```/test.php/``` 이므로 ```/test.php/app/main.js```에 존재하는 파일에 대해 요청하게 된다. 

즉, test.php의 페이지 내용을 자바스크립트의 내용으로 활용 가능하게 되는 것이다.

<br>

RPO는 일반적으로 자바스크립트나 스타일시트 코드를 로드하는 과정에서 경로 해석의 문제로 인해 발생한다.

따라서 임포트하는 페이지의 내용을 조작시킬 수 있다면 공격자가 의도한 자바스크립트, 스타일시트 코드를 로드시킬 수 있다.

<br>

예를 들어, 자바스크립트의 경우에 ```<script src="/static/script.js">``` 라는 코드가 있다고 가정했을 때, 사용자의 경로 조작으로 인해 ```/userinput/static/script.js```라는 경로로 스크립트 코드가 로드 될 수 있다.

만약 userinput 값이 ```index.php/;alert(1);//static/script.js```가 된다면 alert가 실행될 것이다.

주의점은 자바스크립트는 로드하는 코드 내에서 에러가 발생할 경우, 전체 코드를 실행하지 않기 때문에 문법 에러가 발생하지 않도록 신경써야 한다고 한다.

<br>

스타일시트를 로드하는 경우에도 ```<link href="style.css" rel="stylesheet" type="text/css" />```란 코드가 있을 때 ```/test.php``` 페이지를 불러온다고 가정.

그럼 link 태그를 통해 ```/test.php/style.css```를 불러오게 된다. 

만약 url rewrite 기능으로 인해 ```/test.php/path/``` 파일이 호출된다면 **서버에서는 test.php 파일을 호출하겠지만, 브라우저는 ```/test.php/path```를 디렉터리 경로로 인식하게 되어서 ```/test.php/path/style.css```를 호출하게 된다.** 

그러나 서버에서 응답한 파일의 내용은 test.php가 되므로 test.php의 html 코드를 css로 import 하게 되는 것이라고 한다.

**스타일시트의 특징 중 하나로 올바르지 않은 문법을 만나면 무시하고 올바른 문법이 나올 때까지 다음 문법으로 넘어간다는 특징이 존재한다고 한다.**

이로 인해, 해당 html 코드 내에 정상적인 css 코드가 있다고 한다면, 해당 페이지 내 일부분에 유효한 CSS 문법을 삽입할 수 있다면 이를 CSS Injection으로 연계할 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 출처
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://dreamhack.io/lecture/courses/328" target="_blank">Exploit Tech: Relative Path Overwrite</a>
: <a href="https://blog.rubiya.kr/index.php/2019/04/17/relative-path-overwrite/" target="_blank">blog.rubiya.kr/index.php/2019/04/17/relative-path-overwrite/</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
