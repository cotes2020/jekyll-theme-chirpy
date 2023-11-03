---
title: 구글 맵 폴리곤으로 서울만 보이게 하기(동균, 유빈) (1)
date: 2023-07-19
categories: [troubleshooting]
layout: post
tags: [google-map-api, react, kml, polygon]
---

![seoul](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-google-map-polygon/seoul.png)

<iframe width="100%" height="400" src="//jsfiddle.net/dL5gaouy/13/embedded/js,html,css,result/dark/" allowfullscreen="allowfullscreen" allowpaymentrequest frameborder="0"></iframe>

### 서울 결과물

[https://jsfiddle.net/dL5gaouy/13/](https://jsfiddle.net/dL5gaouy/13/)

### 밀라노 예제

[https://jsfiddle.net/j5oncw06/](https://jsfiddle.net/j5oncw06/)

### 지도 개발 예시 모음 사이트

[https://openlayers.org/](https://openlayers.org/)

### 밀라노 반전하고 싶은 외국인의 스택오버플로우

키워드 : innerBoundaryIs, outerBoundaryIs

[KML invert country border](https://stackoverflow.com/questions/35242944/kml-invert-country-border)

[How to display a specific city on Google Maps API?](https://stackoverflow.com/questions/54637798/how-to-display-a-specific-city-on-google-maps-api)

### kml 파일 다운받는곳

[GADM](https://gadm.org/download_country.html)

GADM 이란 사이트에서 세계의 GeoJSON 과 kml 을 받을 수 있다.

<div markdown="block" style="width: 80%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-google-map-polygon/gadm.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-google-map-polygon/gadm.png)
</div>

level 에 따라서 데이터의 단위가 바뀐다.

- level_0 국가단위
- level_1 도(서울)단위
- level_3 구단위

### KML 이란 무엇인가

[KML 가이드 / Keyhole Markup Language Google for Developers](https://developers.google.com/kml/documentation/kml_tut?hl=ko)
