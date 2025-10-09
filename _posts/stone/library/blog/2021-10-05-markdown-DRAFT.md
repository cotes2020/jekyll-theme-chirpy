---
layout: #_layouts 에 있는 layout
title: "Test Page"
excerpt: "포스트 목록에서 보이는 미리보기 글"
date: 1999-08-26 00:00:00 +0000
# category:
categories: 
tags: 
    - Test
toc: true # table of content
# toc_table: # toc의 이름
# toc_icon: # toc의 아이콘
toc_sticky: true # toc의 고정 유무
last_modified_at: # 말 그대로
# permalink:
# published: # 말 그대로
# custom: "Custom Variable 도 만들 수 있음"

#collection: pets
#entries_layout: grid
#classes: wide
#taxonomy: Edge Case

header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: /assets/img/profile/ProfileImage_Transparent.png
  actions:
    - label: "Download"
      url: "https://github.com/mmistakes/minimal-mistakes/"
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
intro: 
  - excerpt: 'Nullam suscipit et nam, tellus velit pellentesque at malesuada, enim eaque. Quis nulla, netus tempor in diam gravida tincidunt, *proin faucibus* voluptate felis id sollicitudin. Centered with `type="center"`'
---

[참고 링크 1](https://gist.github.com/ihoneymon/652be052a0727ad59601)  
[참고 링크 2](https://velog.io/@eona1301/Github-Blog-Jekyll-minimal-mistakes)  
[참고 링크 3](https://ansohxxn.github.io/blog/posting/)  
[참고 링크 4](https://mmistakes.github.io/minimal-mistakes/docs/utility-classes/)
[참고 링크 5](https://ansohxxn.github.io/blog/markdown/)
[참고 링크 6](https://www.markdownguide.org/extended-syntax/#fnref:bignote)

---

<!-- # This is an H1 -->

## This is an H2

### This is a H3

#### This is a H4

##### This is a H5

###### This is a H6

<!-- This is also an H1
===
-->

<!--
This is also an H2
---
-->

<!-- 줄바꿈 : 2 Space + Ender-->

<!-- 문자 블록 -->
`문자 블록`  

<!-- 리스트 : (혼합해서 사용가능) 숫자. * - + -->
<!-- MD004 : 문자의 경우, 일관되게 쓸 것 -->
<!-- MD030 : 옆으로도 쓸 수 있지만, 부적절하게 렌더링 될 수도 있음 *  *  * -->

> This is a first blockquote. 블럭인용문자
> > This is a second blockquote. 블럭인용문자
> > > This is a third blockquote.  블럭인용문자  

_  

> This is a first blockquote. 블럭인용문자
>> This is a second blockquote. 블럭인용문자
>>> This is a third blockquote.  블럭인용문자

1. One
    1. One
        1. One Hundred
        2. Ten
        3. One
2. Two
3. Three
   - Sans

Definition List Title
:   Definition list division.

<!-- 코드 블럭 : ``` ~~~ 들여쓰기 <pre><code> gist -->

```cs
public int i;
// ```(언어) (코드 내용) ```
// ~~~(언어) (코드 내용) ~~~
// 들여쓰기 (코드 내용)
```
{: file='파일_이름'}

<pre><code>
public int i;
// 적용은 되지만, 테마 때문인지 헤더가 제대로 보이지 않음.
</code></pre>

<script src="https://gist.github.com/mascari4615/b1e61891bbc6934fdf5eb77580ace000.js"></script>

<!-- 수평선 : ***, * * *, ---, - - -, ___, _ _ _, <hr/> -->
<!-- MD035 : 일관되게 쓸 것 -->
<!-- MD033 : Inline HTML 쓰지 말 것 -->
---

<!-- 강조 : *A* **A** _A_ __A__ ~~ ~~-->
*Single Asterisks*  
**Double Asterisks**  
***Triple Asterisks***  
~~Double Tildes~~  

> 문법 앞에 \ 를 붙이면 원래대로
\*Single Asterisks\*

<!-- HTML Tag -->
<!-- MD033 : Inline HTML 쓰지 말 것 -->
<u>U Tag</u>  
<span style="color:black">Span Color</span>  
<cite>인용 공부하는 식빵맘</cite> --- <https://ansohxxn.github.io/blog/markdown/>  
{: .small}

<!-- 인라인 링크, 자동 링크, 새 창에서 열기 -->
<https://google.com>  
<https://www.naver.com/>{: target="_blank"}  
<mascari46154444@gmail.com>  

<!-- 외부 링크 -->
Link: [외부 링크](https://google.com, "외부 링크")  
Link: [외부 링크](https://google.com "외부 링크")  

<!-- 참조 링크 -->
Link: [참조 링크][googlelink]  

[googlelink]: https://google.com "참조 링크"  

<!-- 문단(헤더) 링크 -->
<!-- 문단주소 : #로 시작, 특수문자 X, 소문자, 공백 - 로 치환 -->
Link: [문단(헤더) 링크](#테스트)  

<!-- 이미지 링크 -->
[![Alt text](/assets/img/profile/4444.gif)](https://google.com)  

<!-- 이미지 -->
![Alt text](/assets/img/profile/4444.gif)  
![Alt text](/assets/img/profile/4444.gif "Optional title")  
![4444](https://user-images.githubusercontent.com/55438621/131108421-96105b44-351b-4b3b-8db5-616ccd7d6848.gif "Github any repo > Issues > New issue > Drag Image > Image Code!")  
<img src="/assets/img/profile/4444.gif" width="200px" height="100px" title="크기 px 설정" alt="크기 px 설정">  
<img src="/assets/img/profile/4444.gif" width="20%" height="10%" title="크기 비율 설정" alt="크기 비율 설정">  

<!-- 문자 정렬 -->
왼쪽 정렬 (Default)
{: .text-left}
<div style="text-align: left"> 왼쪽 정렬 </div>

중앙 정렬
{: .text-center}
<center> 중앙 정렬 </center>  

오른쪽 정렬
{: .text-right}
<div style="text-align: right"> 오른쪽 정렬 </div>

<!-- 이미지 정렬 -->

![image](/assets/img/profile/4444.gif)
{: .align-left}

왼쪽 정렬. 동해 물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세. 무궁화 삼천리 화려 강산 대한 사람, 대한으로 길이 보전하세.  

![image](/assets/img/profile/4444.gif)
{: .align-center}

중앙 정렬. 동해 물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세. 무궁화 삼천리 화려 강산 대한 사람, 대한으로 길이 보전하세.  

![image](/assets/img/profile/4444.gif)
{: .align-right}

오른쪽 정렬. 동해 물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세. 무궁화 삼천리 화려 강산 대한 사람, 대한으로 길이 보전하세.  

![image](/assets/img/profile/4444.gif)
{: .full}

<!-- 문자 박스 -->

그냥 텍스트는 안됨
{: .notice}

**줄 안바꾸면**
{: .notice}

**Default**
{: .notice}

**Primary**
{: .notice--primary}

**Info**
{: .notice--info}

**Warning**
{: .notice--warning}

**Success**
{: .notice--success}

**Danger**
{: .notice--danger}

## 표

| Header1                                           | Header2 | Header2 | Header3 |         |
| :------------------------------------------------ | ------- | ------- | :-----: | ------: |
| cell1                                             | cell2   | cell3   |  cell4  | *cell5* |
| cell6                                             | cell7   | cell8   |  cell9  |         |
| ------------------------------------------------- |
| cell10                                            | cell11  | cell12  | cell13  |       - |
| ================================================= |
| 왼쪽                                              | 기본    | 기본    | 가운데  |  오른쪽 |

태그도 다 붙는 것 같음

## 태그?

Test<sup>\<sup\></sup>  
Test<sub>\<sub\></sub>  
<acronym title = "acronym">Test\<acronym\></acronym>  
<abbr title = "abbr">Test\<abbr\></abbr>

## 체크 박스

---

- `[ ] 체크 안됨`
- `[x] or [X] 체크 됨`

## 토글 리스트 (접기/펼치기)

---

마크다운에선 지원하지 않고 HTML의 details 태그로 사용 가능하다. div markdown=”1” 은 jekyll에서 html사이에 markdown을 인식 하기 위한 코드이다.  

- <https://ansohxxn.github.io/blog/markdown/>
- <https://inasie.github.io/it%EC%9D%BC%EB%B0%98/%EB%A7%88%ED%81%AC%EB%8B%A4%EC%9A%B4-expander-control/>

<details>
<summary>여기를 눌러주세요</summary>
<div markdown="1">

😎숨겨진 내용😎

</div>
</details>

`<a href="#" class="btn--success">Success Button</a>`  
<a href="#" class="btn--success">Success Button</a>  

`[Default Button](#테스트){: .btn .btn--primary }`  
[Default Button](#테스트){: .btn .btn--primary }  

## 테스트

---

### Address Tag

<address>
  1 Infinite Loop<br /> Cupertino, CA 95014<br /> United States
</address>

### Abbreviation Tag

The abbreviation CSS stands for "Cascading Style Sheets".

*[CSS]: Cascading Style Sheets

### Cite Tag

"Code is poetry." ---<cite>Automattic</cite>

### Strike Tag

This tag will let you <strike>strikeout text</strike>.

### Insert Tag

This tag should denote <ins>inserted</ins> text.

### Keyboard Tag

This scarcely known tag emulates <kbd>keyboard text</kbd>, which is usually styled like the `<code>` tag.

### Preformatted Tag

This tag styles large blocks of code.

<pre>
.post-title {
  margin: 0 0 5px;
  font-weight: bold;
  font-size: 38px;
  line-height: 1.2;
  and here's a line of some really, really, really, really long text, just to see how the PRE tag handles it and to find out how it overflows;
}
</pre>

### Quote Tag

<q>Developers, developers, developers&#8230;</q> &#8211;Steve Ballmer

### Variable Tag

This allows you to denote <var>variables</var>.

<https://docs.github.com/ko/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github>  

> [!NOTE]  
> Highlights information that users should take into account, even when skimming.

> [!TIP]
> Optional information to help a user be more successful.

> [!IMPORTANT]  
> Crucial information necessary for users to succeed.

> [!WARNING]  
> Critical content demanding immediate user attention due to potential risks.

> [!CAUTION]
> Negative potential consequences of an action.

- discord: `-# ⓘ`
