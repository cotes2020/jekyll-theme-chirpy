---
title: 깃허브 블로그 github.io Blog 만들기
author: juyoung
date: 2019-08-09 20:55:00 +0800
categories: [diary, github.io]
tags: [github.io]
---


## 오늘부터 <font color=skyblue>깃허브 블로그</font>를 시작하기로 했다.

1. 우선 git에 repository를 만들고 
2. jekyll에서 마음에 드는 theme을 선택한다.

<br /> 
문제는 ruby를 설치하고 jekyll server를 실행하는데 tzinfo 에러와 sass가 발생한 것ㅜㅠ
<br /> 

## 우선 tzinfo 오류는 

<http://callmejaden.github.io/github_pages/2019/12/30/GP-04-tzinfo>

 이곳에서 하라는 대로 Gemfile에 들어가 해당명렁어를 붙여넣기했다~  

```
# Windows does not include zoneinfo files, so bundle the tzinfo-data gem
gem 'tzinfo'
gem 'tzinfo-data', platforms: [:mingw, :mswin, :x64_mingw]
```
<br />    
## sass 오류의 경우 
윈도우에 저장된 사용자 계정 이름이 한글로 되어 있는 것이 문제여서  

원격 저장소를 C드라이브에서 다시 clone하여 문제를 해결했다. 
<https://tothefullest08.github.io/til/2019/02/11/TIL/>
 
<br /> 



멀고 먼 길을 돌아 4시간만에 blog에 이렇게 글을 올리니 정말 기쁘다~ 오늘부터 github blog를 시작한다.


친절하게 설명해주신 여러 blog 포스터 분들께 감사의 인사를 드립니다. 덕분에 해냈어요~
<br />
<br />
> 내가 본 깃허브 블로그 만들기 글 중 이해가 쉬웠던 곳을 남겨본다.  

> <https://recoveryman.tistory.com/323?category=635733>  

> <https://brunch.co.kr/@maemi/28>

> 그리고 앞으로 blog 만들기에 도움이 될 것 같은 곳:)  

> <https://devinlife.com/howto/>


<br />
구직활동금으로 시작한 front-end 개발자 공부가 벌써 4개월째지만  
아직도 할 게 많아 마음은 조급하다  
그럴수록 천천히 차근차근 무엇보다 꾸준히 가겠다
우선 blog 개시와 함께  
 오늘부터 nomade coder의 react 클론코딩 강의를 시작하기