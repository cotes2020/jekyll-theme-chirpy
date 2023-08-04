---
title : URL Encoding
categories: [Hacking, Web]
date: 2022-04-24 16:57 +0900
tags: [url encoding, 예약어]
---

## URI와 URL
<hr style="border-top: 1px solid;"><br>

URI, URL (출처)
: <a href="https://www.charlezz.com/?p=44767" target="_blank">www.charlezz.com/?p=44767</a>

정리하면 URI는 식별자, URL은 주소 즉, 위치를 알려주는 것이다. URI가 더 큰 개념.

URI는 식별하고, URL은 위치를 가르켜준다.

![URI](https://user-images.githubusercontent.com/52172169/164971963-49654c3f-c8f4-43ca-ab51-3ffc88b29d20.png)

<br>

참고하기!!
: <a href="https://learn.dreamhack.io/6#9" target="_blank">Dreamhack - Introduction of Webhacking</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## URL Encoding
<hr style="border-top: 1px solid;"><br>

URL 인코딩
: <a href="https://developers.google.com/maps/url-encoding?hl=ko" target="_blank">developers.google.com/maps/url-encoding?hl=ko</a>

Percent Encoding
: <a href="https://ko.wikipedia.org/wiki/퍼센트_인코딩" target="_blank">ko.wikipedia.org/wiki/퍼센트_인코딩</a>

<br>

URL에 특수문자 등이 포함될 수 있고 이 경우 브라우저는 이러한 문자를 전송하기 내부적으로 다른 인코딩으로 변환해야 한다.

마찬가지로 UTF-8 입력을 생성하거나 수락하는 코드는 UTF-8 문자가 있는 URL을 '유효한' 것으로 취급할 수 있지만 그러한 문자를 웹 서버로 보내기 전에 변환해야 한다.

이 과정을 URL 인코딩 또는 퍼센트 인코딩이라고 한다.

<br>

**URL에는 ASCII 문자 중 일부 특수한 문자(예: 익히 알고 있는 영숫자 기호, URL 내에서 제어 문자로 사용하기 위해 예약된 일부 문자)만 포함되어야 한다.**

아래는 유효한 문자 요약이다.

![유효한 문자](https://user-images.githubusercontent.com/52172169/164972351-49da2303-43c4-4e46-aea9-a28635018fc4.png)

<br>

URL에서 중요하게 사용되는 **예약(reserved) 문자**가 있고, 또한 인코딩이 필요하지 않은 **비예약(unreserved) 문자**가 존재한다.

**예약어는 URI 구조내에서 문법적으로 중요한 의미를 가지고 있기 때문에 문법적으로 사용하지 않을 경우에는 반드시 인코딩해 사용해야 한다.**

<br>

유효한 URL 작성 시 위의 표 안에 있는 유효한 문자들로 작성해야 하는데 이때 두 가지 문제점이 발생할 수 있다.

+ 처리하려는 문자가 위의 집합에 없는 경우
  + 외국어 등의 경우는 위의 표에 있는 문자들로 인코딩.
  + 관례적으로 URL 내에서 허용되지 않는 공백은 더하기 문자('+')를 사용해서 표현하기도 한다. 

+ 예약된 문자로 위 집합에 포함된 문자를 문자 그대로 사용해야 하는 경우
  + **문자 그대로 사용하고자 하면 인코딩을 해줘야 한다.** 예를 들어, 물음표를 사용하고자 하면 인코딩 해줘야 한다. 

<br><br>

+ 인코딩이 필요한 일반 문자는 아래와 같다.

![image](https://user-images.githubusercontent.com/52172169/164972496-ab19df63-ec2d-4235-a15d-d1ebbfcd5906.png)

<br>

URL의 길이는 최대 8,192자(영문 기준)이다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 출처
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://www.charlezz.com/?p=44767" target="_blank">www.charlezz.com/?p=44767</a>
: <a href="https://developers.google.com/maps/url-encoding?hl=ko" target="_blank">developers.google.com/maps/url-encoding?hl=ko</a>
: <a href="https://ko.wikipedia.org/wiki/퍼센트_인코딩" target="_blank">ko.wikipedia.org/wiki/퍼센트_인코딩</a>
: <a href="https://learn.dreamhack.io/6#9" target="_blank">Dreamhack - Introduction of Webhacking</a>


<br><br>
<hr style="border: 2px solid;">
<br><br>
