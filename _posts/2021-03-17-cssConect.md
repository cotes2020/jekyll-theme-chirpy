```
title: HTML과 CSS파일의 연결
author: dongee_seo
date: 2021-03-17
categories: [Blogging, Tutorial]
tags: [google analytics, pageviews]
```

## CSS?

= HTML tag에 디자인을 입혀주는 것

## HTML과 CSS파일의 연결:

### 1. HTML document에 CSS파일 불러오기

> `<link href=“css파일경로” rel=“stylesheet” type=“text/css”>`

- `rel`: HTML file과 CSS file과의 관계를 설명

### 2. HTML에 CSS적용는 방법(in CSS):

~> Selector: CSS 적용대상

1. 개별 적용
   -tag=> 단순히 태그 이름만 기재
   -class=> ‘.classname’
   -id=> #idname

> 또는 `<style>` 태그를 활용해서 html문서 내에서 직접 적용하는 방법도 있지만. 나중에 문서를 수정하거나 CSS적용을 고려한다면 CSS요소는 CSS파일에서 따로 적용하는게 좋다.

2. 모든 tag에 적용: **"\*"** selector

\*{
box-sizing: border-box;
}

3. 동일한 스타일을 각기 다른 태그와 클래스 등에 적용할 경우
   : 각자의 selector에 모두 똑같은 스타일을 적기보다는
   한꺼번에 스타일을 지정=> **‘,’** 사용

.what-is-blockquote, span {
color: green;
}

4. 엄청 길게도 작성할 수 있다...
   .a div .b .pre span {
   background-color: yellow;
   }
   => a class 안의 div tag안의 b class 안의 pre class안의 span에 배경색을 노랑으로,,

```null
<div class="a">
  <div>
    <header class="b">
      <h1 class="pre">
        <span>제목! 노란색 배경 나옴!</span>
        <span class="title">이것도 나옴!</span>
      </h1>
      <span>이건 적용안 됨</span>
    </header>
  </div>
  <span>이건 적용안 됨</span>
</div>
```

> CSS Specificity: Selector 우선순위
> : tag <<<<< class <<<< id <<<<<<inline
> css거의 대부분의 요소에 class를 부여해주고,
> class를 selector로 styling해주기 때문에,
> 최대한 중복을 피할 수 있도록 작성합니다.

> /+ 추가내용
> `<div>` 태그는 만들고 나면 class나 id를 부여하여 각각의 스타일을 적용하게 된다. 그러나 id를 적용하면 우선순위가 아주 높다보니 다른 스타일 속성이 쉽게 무력화된다.
> 따라서 id를 범용하면 home, banner, nav a 와 같이 길고 복잡한 selector를 사용해야 하므로, id 대신 class를 더 자주 쓴다.

2. 밑줄귿는 방법
   1.

- text-decoration: overline;
- text-decoration: underline;

2. border 이용하기

3) font-family:폰트 스타일 지정하는 속성

주의) "Times New Roman"만 ""(쌍따옴표)로 감싸져 있는데, 폰트 이름에 띄워쓰기가 되어있으면 ""(쌍따옴표)를 사용해야합니다.
사용자가 어떤 브라우저를 사용할지 모르기 때문에 font-family 값에는 보통 여러가지 폰트를 나열합니다.

4. text-align: 글씨 정렬

> 다만 `<span>`은 inline-element이므로, "span의 오른쪽 정렬"이라는 text만큼 영역을 차지하고 있어 정렬이 되지 않는다.

5. text-indent: css 들여쓰기
6. blockquote
   : 인용구문을 넣을 때 쓰는 태그입니다.
   : 브라우저는 blockquote태그에 양쪽 여백을 넣는 기본 스타일을 자동으로 적용됨.

> Q. 문장 사이사이에 스페이스를 추가하고 싶을 때?
> : 스페이스를 의미하는 `&nbsp;` 을 넣어주면 됩니당.

/+ 추가내용
CSS selector를 표기하는 방법중에

해당 태그의 첫 번째 순서인지,

마지막 순서인인지,

홀수/짝수 인지 알 수 있는 selector 표기법!

```null
 selector: first-child
 selector: last-child
 selector: nth-child(odd) //홀수
 selector: nth-child(even) //짝수
```

## Layout의 모든것

---

> layout의 핵심은 블록 레벨 요소들을 원하는 위치에 배열하는 것
> 웹사이트를 구성하는 요소들을 배치할 공간을 분할하고 정렬하는 것이다. 공간을 분할할 때는 먼저 행을 구분한 후, 행 내부 요소를 분리하는 것이 일반적이다.

### 1. Position

:: html 코드는 작성한 순서대로 페이지에 나타난다.
이때 CSS의 `position` property를 사용하면, html 코드와 상관없이 그리고 싶은 어느 위치에나 요소를 배치시킬 수 있다.

- `position: relative;`
- `position: relative;`
- `position: absolute;` // position이 static이 아닌 parent를 기준으로 고정
- `position: fixed;` //해당 브라우저의 페이지를 기준으로 고정

### 2. inline과 inline-block, block

> Block element vs. Inline element

1. 의의

(1) Block element

-대부분의 HTML element(요소)는 block요소에 해당하는 태그들이다.
ex. `<header>, <footer>, <p>, <li>, <table>, <div>, <h1>`등

-이 요소들은 '바로 옆(좌우측)에 다른 요소를 붙여 넣을 수 없다.'는 뜻이다.

(2) Inline element
ex. `<span>, <a>, <img>` -요소끼리 서로 한 줄에, 바로 옆에 위치 할 수 있다.

2. `display`와 `float`
   > Block 요소의 성질을 가진 태그일지라도 CSS를 사용하여 Inline 스타일로 바꾸도록 하는 CSS property로 `display`와 `float` 가 있다.

```null
.block {
  float: left;
}

.block {
  display: inline-block;
}
```

해당 property에 위와 같은 값을 부여하면,
요소 옆에 위치하는 inline성질로 변하게 된다.

> 반대로 원래의 inline성질을 block으로 바꾸게 할 수도 있다.

```null
 .inline{
   display: block;
 }
```

> display: none; 이라는 값을 주면, 해당 요소는 화면에서 보이지 않는다. 이는 JavaScript와 결합하여 큰그림을 그릴때 사용한다.
> 요소를 보이게/안보이게 하는 효과랄까

### 3. float

\_페이지 전체의 레이아웃을 잡을 때에 정말 중요했던,,, 도구

- float 속성: `left, right, none` 등

\_float 속성을 가진 요소는 부모가 높이를 인지할 수 없어서 부모를 벗어난다.
이 문제를 해결하기 위해

`clear` property를 사용한다.
`<br class="clear">`
