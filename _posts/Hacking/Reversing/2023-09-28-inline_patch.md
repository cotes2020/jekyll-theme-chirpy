---
title : 인라인 패치
categories : [Hacking, Reversing]
tags : [Reversecore, 인라인 패치]
---

## 인라인 패치
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://maple19out.tistory.com/31" target="_blank">maple19out.tistory.com/31</a>
: <a href="https://chive0402.tistory.com/8" target="_blank">chive0402.tistory.com/8</a>

<br>

인라인 패치는 실행 압축되거나 암호화된 파일을 패치할 때 자주 사용되는 기법이다.

이 기법은 원하는 코드를 직접 수정하기 어려울 때 간단히 코드 세이브라고 하는 패치 코드를 삽입한 후 실행해 프로그램을 패치시키는 기법이다.

주로 대상 프로그램이 실행 압축 혹은 암호화 되어 있어서 파일을 직접 수정하기 어려운 경우 많이 사용한다고 한다.

인라인 패치를 어디에 설치해야 하는가? 여기에는 세 가지 정도의 방법이 있다고 한다.

<br>

1. 파일의 빈 영역에 설치
2. 마지막 섹션을 확장한 후 설치
3. 새로운 섹션을 추가한 후 설치

<br>

보통 패치 코드의 크기가 작은 경우 1번 방법을 사용하고, 나머지 경우 2번 또는 3번 방법을 사용하면 된다고 한다.

<br><br>
<hr style="border: 2px solid;">
<br><br>