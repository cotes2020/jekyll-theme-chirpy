---
title: Javascript toLowerCase(), toUpperCase() Logical Bypass
date: 2022-08-09 01:23  +0900
categories: [Hacking, Web]
tags: [javascript toLowerCase, toLowerCase unicode, javascript toUpperCase, toUpperCase unicode]
---

## Javascript toLowerCase(), toUpperCase()
<hr style="border-top: 1px solid;"><br>

자바스크립트의 ```toLowerCase()``` 메소드는 문자열의 대문자를 소문자로 변환시켜주는 문자열 메소드이다.

즉, Unicode 범위의 A(65)부터 Z(90)까지의 값을 소문자로 변환시켜준다.

마찬가지로 ```toUpperCase()``` 메소드 또한 유니코드 범위 내의 알파벳 소문자를 대문자로 변환시켜주는 메소드이다.

<br>

여기서 몇 몇 유니코드 값을 ```toLowerCase``` 메소드로 변환시켜줄 때, 아스키 코드로 바뀌는 값도 있다.

반대로 대문자로 변환시켜줄 때도 마찬가지다.

아래는 그러한 값들을 일부 모아둔 것이다.

<br>

```
[223] ß (%C3%9F).toUpperCase() => SS (%53%53)
[304] İ (%C4%B0).toLowerCase() => i̇ (%69%307)
[305] ı (%C4%B1).toUpperCase() => I (%49)
[329] ŉ (%C5%89).toUpperCase() => ʼN (%2bc%4e)
[383] ſ (%C5%BF).toUpperCase() => S (%53)
[496] ǰ (%C7%B0).toUpperCase() => J̌ (%4a%30c)
[7830] ẖ (%E1%BA%96).toUpperCase() => H̱ (%48%331)
[7831] ẗ (%E1%BA%97).toUpperCase() => T̈ (%54%308)
[7832] ẘ (%E1%BA%98).toUpperCase() => W̊ (%57%30a)
[7833] ẙ (%E1%BA%99).toUpperCase() => Y̊ (%59%30a)
[7834] ẚ (%E1%BA%9A).toUpperCase() => Aʾ (%41%2be)
[8490] K (%E2%84%AA).toLowerCase() => k (%6b)
[64256] ﬀ (%EF%AC%80).toUpperCase() => FF (%46%46)
[64257] ﬁ (%EF%AC%81).toUpperCase() => FI (%46%49)
[64258] ﬂ (%EF%AC%82).toUpperCase() => FL (%46%4c)
[64259] ﬃ (%EF%AC%83).toUpperCase() => FFI (%46%46%49)
[64260] ﬄ (%EF%AC%84).toUpperCase() => FFL (%46%46%4c)
[64261] ﬅ (%EF%AC%85).toUpperCase() => ST (%53%54)
[64262] ﬆ (%EF%AC%86).toUpperCase() => ST (%53%54)

출처 : https://blog.p6.is/hacktm-ctf-quals-2020/
```

<br>

그러한 값들을 모아둔 사이트들은 아래와 같다.

<br>

Link
: <a href="https://domdomi22.github.io/unicode-pentester-cheatsheet/" target="_blank">domdomi22.github.io/unicode-pentester-cheatsheet/</a>
: <a href="https://blog.p6.is/hacktm-ctf-quals-2020/" target="_blank">blog.p6.is/hacktm-ctf-quals-2020/</a>

<br>

이러한 점을 이용해 인증을 우회를 할 수 있다던가.. 등 이용할 수 있다..!

<br><br>
<hr style="border: 2px solid;">
<br><br>
