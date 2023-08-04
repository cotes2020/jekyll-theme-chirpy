---
title : XS-Search (Cross-Site Search)
date: 2022-04-24 22:21 +0900
categories: [Hacking, Web]
tags: [Cross-Site Search, XS-Search]
---

## XS-Search (Cross-Site Search)
<hr style="border-top: 1px solid;"><br>

웹 브라우저의 SOP 정책 때문에 일반적으로 다른 오리진에 요청을 보내고 그 응답 값을 받아오는 것은 불가능하다.

이를 우회하기 위한 대표적인 공격이 바로 Cross-Site Script (XSS) 공격이다.

대신 XSS 공격은 공격 대상 오리진에 반드시 공격자의 입력이 들어가 스크립트를 삽입해야하고, 피해자가 해당 페이지를 방문해야한다는 전제조건이 존재한다.

Cross-Site Search (XS-Search) 공격은 공격 대상 오리진에 스크립트를 삽입하거나 해당 페이지를 직접 방문하지 않고, **피해자가 공격자가 의도한 서버에만 접속하는 것만으로 SOP를 우회하여 공격 대상 오리진의 비밀 정보를 유출할 수 있는 강력한 공격기법이다.**

<br>

Cross-Site Search (XS-Search)는 쿼리 기반 시스템을 이용해 이용자의 비밀 정보를 유출하는 공격 기법이다.

XS-Search는 부채널 공격의 일종으로 Blind SQL Injection과 비슷한 형태로 공격이 이루어진다.

XS-Search 공격은 **SOP 정책을 위반하지 않는 선에서 다른 오리진에 요청을 보내고 요청에 소요된 시간, 응답 코드, 창의 프레임 개수 등의 요소를 활용해 비밀 정보를 유출**한다.

<br>

브라우저는 SOP에 구애 받지 않고 외부 출처에 대한 접근을 허용해주는 경우가 존재한다.

**이미지나 자바스크립트, CSS 등의 리소스를 불러오는 ```<img>, <style>, <script>, <iframe>``` 등의 태그는 SOP의 영향을 받지 않는다.**

XS-Search는 이와 같이 SOP에 구애받지 않는 태그를 활용하여 공격한다.

자세한 공격 방법은 아래에서 확인.
: <a href="https://learn.dreamhack.io/330#5" target="_blank">learn.dreamhack.io/330#5</a>
: <a href="https://defenit.kr/2019/10/06/Web/ㄴ%20Research/XS-SEARCH__ATTACK/" target="_blank">defenit.kr/2019/10/06/Web/ㄴ%20Research/XS-SEARCH__ATTACK/</a> 

<br>

XS-Search 공격을 할 때에는 SameSite Cookie를 유의해야 한다.
: <a href="https://web.dev/i18n/ko/samesite-cookies-explained/" target="_blank">web.dev/i18n/ko/samesite-cookies-explained/</a>
: <a href="https://seob.dev/posts/브라우저-쿠키와-SameSite-속성/" target="_blank">seob.dev/posts/브라우저-쿠키와-SameSite-속성/</a>

<br>

SameSite Cookie의 기본 값은 Lax 이기 때문에 다른 오리진에 GET 요청을 보낼 때에는 쿠키를 함께 전송하지만, POST 요청에는 포함되지 않는다.

또한 SameSite Cookie가 Strict로 설정되어있는 경우에는 GET 요청에도 쿠키가 함께 전송되지 않기 때문에 공격을 수행할 때 유의해야 합니다.

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>

## 출처
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://dreamhack.io/lecture/courses/330" target="_blank">Dreamhack - Exploit Tech: XS-Search</a> (대부분)

<br><br>
<hr style="border: 2px solid;"><br>
<br><br>
