---
title : "Wechall - Regex 1~4"
categories : [Wargame, Wechall]
tags: [regexp]
---

## 참고
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://choseongho93.tistory.com/130" target="_blank">choseongho93.tistory.com/130</a>  
: <a href="https://kimdoky.github.io/tech/2017/05/06/regular/" target="_blank">kimdoky.github.io/tech/2017/05/06/regular/</a>  
: <a href="https://hamait.tistory.com/342" target="_blank">hamait.tistory.com/342</a>  

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 1
<hr style="border-top: 1px solid;"><br>

```
Your objective in this challenge is to learn the regex syntax.
Regular Expressions are a powerful tool in your way to master programming, 
so you should be able to solve this challenge, at least!

The solution to every task is 
always the shortest regular expression pattern possible.
Also note that you have to submit delimiters in the patterns too.

Example pattern: /joe/i. The delimiter has to be /

Your first lesson is easy: 
submit the regular expression the matches an empty string, and only an empty string.
```

<br><br>

### Solution
<hr style="border-top: 1px solid;"><br>

empty string이라 함은 공백을 뜻하는 것이 아닌 빈 문자열임에 주의해야 함.

**공백은 ' ' 을 뜻하는 문자이고 빈 문자열은 ""을 뜻하는 것임.** 따라서 답은 ```/^$/```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 2
<hr style="border-top: 1px solid;"><br>

```
Easy enough. Your next task is to submit a regular expression 
that matches only the string 'wechall' without quotes.
```

<br><br>

### Solution
<hr style="border-top: 1px solid;"><br>

wechall만 가져오도록 코드를 짜야함. 따라서 답은 ```/^wechall$/```

<br><br>
<hr style="border: 2px solid;">
<br><br>

### Level 3
<hr style="border-top: 1px solid;"><br>

```
Ok, matching static strings is not the main goal of regular expressions.
Your next task is to submit an expression 
that matches valid filenames for certain images.

Your pattern shall match all images with the name wechall.ext or wechall4.ext 
and a valid image extension.

Valid image extensions are .jpg, .gif, .tiff, .bmp and .png.

Here are some examples for valid filenames:
wechall4.tiff, wechall.png, wechall4.jpg, wechall.bmp
```

<br><br>

### Solution
<hr style="border-top: 1px solid;"><br>

```
x? : x문자가 존재 할 수도, 않을 수도 있음을 나타냄.

x|y : OR을 나타내며, x문자 또는 y문자가 존재여부(선택지의 단락) 

() : 서브 패턴 감싸기

non-capturing group : 그룹화 하지 않고 여러 토큰을 모아둠.
```

처음 제출한 답은 ```/^wechall4?\.(jpg|gif|tiff|bmp|png)$/``` 하지만 non-capturing group을 사용하라 뜸.
: ```Your pattern would capture a string, but this is not wanted. Please use a non capturing group.```

따라서 답은 ```/^wechall4?\.(?:jpg|gif|tiff|bmp|png)$/```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Level 4
<hr style="border-top: 1px solid;"><br>

```
It is nice that we have valid images now, 
but could you please capture the filename, without extension, too?

As an example: wechall4.jpg should capture/return wechall4 in your pattern now.
```

<br><br>

### Solution
<hr style="border-top: 1px solid;"><br>


3번의 연장선으로 확장자를 빼고 제목만 추출하라고 함.

따라서 답은 ```/^(wechall4?)\.(?=jpg|gif|tiff|bmp|png)$/```

<br><br>
<hr style="border: 2px solid;">
<br><br>
