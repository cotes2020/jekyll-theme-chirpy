---
title: RestDocs + Swagger, 2개 같이 사용하기
author: jimin
date: 2022-12-11 00:00:00 +0900
categories: [Spring, Test]
tags: [Test,Swagger,RestDocs,RestAPI]
pin: false
---

# RestDocs와 Swagger 비교

|           | RestDocs        | Swagger|
|-----------|:-----------------|:--------|
|명세서 자동생성| <span style="color:#008000">O</span>| <span style="color:#008000">O</span>    |
| 웹 테스트   | <span style="color:red">X</span>| <span style="color:#008000">O</span>     |
| 테스트 강제| <span style="color:#008000">O</span> , 신뢰도 좋음 | <span style="color:red">X</span> , 신뢰도 별로|
| 난이도        | <span style="color:red">어려움</span>| <span style="color:#008000">쉬움</span>  |
| 코드 가독성       |<span style="color:#008000">좋음</span>            |<span style="color:red">안좋음</span>


# 첫 시도
 - 그냥 둘 다 한 프로젝트에 적용해봤다.
 - 오류 천국 -> 그냥은 사용 할 수 없었다.


<!-- ```java
import java.io.*;
import java.lang.reflect.Array;
import java.util.*;
``` -->

# 참고

블로그
 - [https://jwkim96.tistory.com/274](https://jwkim96.tistory.com/274)