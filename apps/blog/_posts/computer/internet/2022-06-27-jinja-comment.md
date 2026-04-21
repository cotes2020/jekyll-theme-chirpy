---
title: "HTML에서 Jinja 주석 처리 할 때"
# description: ""
categories: [컴퓨터, 인터넷]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2022-06-27. 07:45
# last_modified_at: 2022-06-27. 07:45
---

## 💫

---

HTML 파일에서 Jinja 코드를 주석처리 할 때

> &#60;!--&#123;% Jinja Code %} -->

같이 HTML 주석안에 Jinja 코드를 넣으면 주석처리 되지 않는다.

> &#123;# &#123;% Jinja Code %} #}

처럼 Jinja 주석안에 넣어줘야 한다.

외부 코드(HTML)와 내부 코드를 분리시키는 Jinja 같은 템플릿 엔진이,  
문서에서 읽을 수 있는 모든 데이터를 변환하고자 하기 때문에 생기는 문제

이 글 같은 경우에는, HTML Entity를 활용하여 코드를 표현하였다.
