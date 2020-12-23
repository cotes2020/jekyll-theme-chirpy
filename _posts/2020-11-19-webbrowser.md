---
title: node란 무엇인가
author: juyoung
date: 2020-11-19 16:08:00 +0800
categories: [html, syntax]
tags: [html]
---

![](https://s3.ap-northeast-2.amazonaws.com/opentutorials-user-file/module/904/2234.png)
  
  ## BOM(Browser Object Model): 
  웹브라우저의 창이나 프래임을 추상화해서 프로그래밍적으로 제어할 수 있도록 제공하는 수단
<br />  

  ## DOM(Document Object Model): 
웹페이지를 자바스크립트로 제어하기 위한 객체 모델
<br />
Window 객체가 창을 의미한다면 Document 객체는 윈도우에 로드된 문서<br />
(ex: chrome browser가 window객체, document는 보여지는 각각의 탭-'생활코딩탭', 'react api reference탭' 등등)
<br />
DOM은 HTML만을 제어하기 위한 모델이 아니다. HTML이나 XML, SVG, XUL과 같이 마크업 형태의 언어를 제어하기 위한 규격이기 때문에 Element는 마크업 언어의 일반적인 규격에 대한 속성을 정의하고 있고,  
 각각의 구체적인 언어(HTML,XML,SVG)를 위한 기능은 HTMLElement, SVGElement, XULElement와 같은 객체를 통해서 추가해서 사용하고 있다.<br />
<br />

# Node객체
엘리먼트들의 관계를 보여주는 객체  

각각의 Node가 다른 Node와 연결된 정보를 보여주는 API를 통해서 문서를 프로그래밍적으로 탐색할 수 있다.
<br />

- - -

불과 2개월 전까지는 봐도 이걸 왜 알아야하는지 몰랐는데 React공부를 하면서 이런 기초 지식이 없어 공식문서를 이해하지 못하자 찾아보게 됐다.
<br />
생활코딩의 [웹브라우저와 JavaScript](https://opentutorials.org/course/1375/6619)에 자료가 잘 정리되어 있어 큰 도움을 받았다.<br />