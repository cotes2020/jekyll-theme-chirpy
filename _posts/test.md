---
title: Chirpy 테마 커스터마이징
date: 2021-12-24 12:15:33 +0900
categories: [jekyll, chirpy]
tags: [jekyll, chirpy, 커스터마이징, chirpy custormizing ]
description: jekyll, chirpy, 커스터마이징, irgroup, 하얀눈길
toc: true
toc_sticky: true
toc_label: 목차
math: true
mermaid: true
#image:
#  src: https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Markdown-mark.svg/1200px-Markdown-mark.svg.png
#  width: 1000   # in pixels
#  height: 400   # in pixels
#  alt: image alternative text
---

> 지난 포스팅에서 Chirpy테마를 사용하여 설치하는 기본적인 절차를 정리하였습니다.\
> 이번은 주로 많이 수정하는 부분에 대한 것을 정리해 보겠습니다\

> 관련글 :
* [Jekyll Chirpy 테마 사용하여 블로그 만들기](https://www.irgroup.org/posts/jekyll-chirpy/){:target="_blank"}
* [Jekyll 테마에 utterances 댓글 연동하기](https://www.irgroup.org/posts/utternace-comments-system/){:target="_blank"}


<br>
<div class="card">
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-8993100314477491"
     crossorigin="anonymous"></script>
<ins class="adsbygoogle"
     style="display:block; text-align:center;"
     data-ad-layout="in-article"
     data-ad-format="fluid"
     data-ad-client="ca-pub-8993100314477491"
     data-ad-slot="6115278830"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
</div>
<br>

## _config.yml 수정 
블로그 환경 설정을 위한 중요한 설정 파일입나디. [여기](https://www.irgroup.org/posts/jekyll-chirpy/#_configyml-%EC%88%98%EC%A0%95){:target="_blank"}를 참고 하시기 바랍니다.  
> local에서 `jekyll servce`로 구동했을 때, 이 파일이 수정되면 반드시 재구동해 줘야 합니다.  

<br>

## 언어 설정하기
블로그에 기본적으로 정의되어 있는 단어들이 있습니다. 예를 들어 사이드바의 탭메뉴멍 HOME, CATEGORIES 등, 오른쪽의 Recent Update, Trending Tags 등이 이에 해당합니다. 이런 기본적인 단어들을 원하는 언어 설정에 맞게 수정할 수 있습니다. 그러나 언어별로 모든 단어가 준비되어 있지 않습니다. 아래 내용대로 따라하면 모든 명칭을 한글로 바꿀 수 있습니다.  

언어설정은 `_data/locales`에 모두 들어 있습니다.   
처음 Chirpy를 설치하면 아래와 같이 3개의 파일밖에 없습니다.

```bash
-rw-r--r--  1 a2021054  staff  2059 12 20 11:22 en.yml
-rw-r--r--  1 a2021054  staff  2133 12 20 11:22 id-ID.yml
-rw-r--r--  1 a2021054  staff  2015 12 20 11:22 zh-CN.yml
```

기본값은 `en.yml`입니다. 영어를 말합니다. 만일 없는 언어를 설정한다면 이 파일이 기본으로 적용됩니다.  
`en.yml` 파일을 열어보면, 블로그의 각 위치별 명칭이 정의되어 있는 것을 확인할 수 있습니다.  
`en.yml`을 복사해서 `ko.yml`로 만들고 값을 한글로 바꿔 봅시다.  

저는 아래와 같이 사이브바의 탭메뉴명을 바꿔 보았습니다.  

```yml
# The tabs of sidebar
tabs:
  # format: <filename_without_extension>: <value>
  home: 홈
  categories: 카테고리
  tags: 태그
```

그리고나서, `_config.yml`을 열어서 `lang: en` 으로 되어 있는 부분을 `lang: ko`로 변경합니다.  
페이지를 열어보면 아래와 같이 변경된 것을 보실 수 있을 겁니다.  
(😓 한글 메뉴명이 안 이쁘네요... 다시 영어로.. 🤣)   


![한글 탭명으로 변경](/assets/img/Chirpy_change_tab_menu_language.jpg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }`

> local에서 `jekyll servce`로 구동했을 때, 이 파일이 수정되면 반드시 재구동해 줘야 합니다.

<br>


## 이미지 업로드 하기
외부 url을 입력해도 되지만, 나의 블로그에 이미지를 업로드 하여 설정하는 것도 좋은 방법입니다.  
이미지가 들어 있는 경로는 `assets/img/`입니다. 여기에 필요한 이미지를 넣어 두면 됩니다.

## 아바타 바꾸기
_config.yml의 `avatar` 항목에 이미지를 업로드 한 후 경로를 입력합니다.  
경로는 `assets/img/[아바타이미지]` 이런 식으로 넣어 주면 됩니다.



## 블로그 타이틀과 서브타이틀 폰트/색상 바꾸기
`_sass/addon/commons.scss`에 타이틀과 서브타이틀에 대한 설정이 있습니다.   
아래를 참고해서 바꿉니다.  
저는 배경이 어두워서 흰색계열로 폰트 색상을 변경하였습니다.   
(css에 대해 궁금하면 구글링~ 😅)

```css
  /* 타이틀에 대한 색상 크기 등을 변경할 수 있습니다. */
  .site-title {
    a {
      @extend %clickable-transition;

      font-weight: 900;
      font-size: 1.5rem;
      letter-spacing: 0.5px;
      color: rgba(254, 254, 254, 99%);
    }
  }

  /* 서브 타이틀에 대한 색상 크기 등을 변경할 수 있습니다. */
  .site-subtitle {
    font-size: 95%;
    /*
    color: var(--sidebar-muted-color);
    */
    color: rgba(254, 254, 254, 99%);
    line-height: 1.2rem;
    word-spacing: 1px;
    margin: 0.5rem 1.5rem 0.5rem 1.5rem;
    min-height: 3rem; // avoid vertical shifting in multi-line words
    user-select: none;
  }
```
<br>

## 왼쪽 사이드바 배경 이미지 넣기
`_sass/addon/commons.scss`에 사이드바 배경에 대한 설정이 있습니다.  
아래와 같이 변경 합니다.

```css
#sidebar {
  ...
  /* 원본 내용
  background: var(--sidebar-bg);
  */

  /*
  아래 3줄 추가
  이미지는 assets/img/ 디렉토리에 넣어 주세요. 
  */
  background: url('/assets/img/일출.jpg');
  background-size: 100% 100%;
  background-position: center;
```
<br>

## 사이드바 내 페이스북 아이콘 및 링크 넣기
링크대한 정의는 `_data/contact.yml`에 들어 있습니다.   
아래와 같이 내용을 추가합니다. [이곳](https://github.com/focuschange/focuschange.github.io/blob/main/_data/contact.yml){:target="_blank"}을 참고하세요

```yml
-
  type: facebook
  icon: 'fab fa-facebook'
  url: 'https://www.facebook.com/focuschange'
```

icon은 반드시 위와 같이 넣어주세요.   
url은 당연히 본인의 폐북 홈 URL로...   
<br>

사이드바의 하단의 아이콘들은 상당히 깔끔합니다. 헌데, 이것이 오픈소스인가 봅니다. 무료로 아무나 쓸 수 있는 것 같습니다.   
[fontawesome](https://fontawesome.com/){:target="_blank"}에 가 보시면 이쁜 폰트들을 사용할 수 있습니다.   
chirpy가 이 폰트들을 기본으로 사용하고 있습니다.    
<br>

## Footer 수정 하기
`_includes/footer.html`에 아래 그림처럼 하단의 내용이 정의되어 있습니다.
![하단 html](/assets/img/Chirpy_footer.jpg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

"Powered by ..."이 눈에 거슬리는군요. 삭제해 보겠습니다.
파일을 열어서 아래 부분을 주석처리 합니다. 삭제해도 무방합니다.   
저는 주석처리를 해 두었습니다. [여기](https://github.com/focuschange/focuschange.github.io/blob/main/_includes/footer.html){:target="_blank"}를 참고하세요~.  

```html
    <div class="footer-right">
      <p class="mb-0">
        {% raw %}
        {% capture _platform %}
          <a href="https://jekyllrb.com" target="_blank" rel="noopener">Jekyll</a>
        {% endcapture %}

        {% capture _theme %}
          <a href="https://github.com/cotes2020/jekyll-theme-chirpy" target="_blank" rel="noopener">Chirpy</a>
        {% endcapture %}

        {{ site.data.locales[lang].meta
          | default: 'Powered by :PLATFORM with :THEME theme.'
          | replace: ':PLATFORM', _platform | replace: ':THEME', _theme
        }}
        {% endraw %}
      </p>
    </div>
```
주석은 시작과 끝을 `<!--`, `-->`로 막아주면 됩니다.  
<br>

footer의 왼쪽 부분에 `Some rights reserved.`라고 되어 있네요. 이걸 바꾸려면 `_data/locales/en.yml` 파일을 엽니다.
그 안에, `copyright:` 아래 `brief:` 부분을 수정하면 됩니다.   
하는 김에, `license:` 아래 `template:`을 지워버립시다. 그러면, 포스팅된 글의 하단에 `This post is licensed under CC BY 4.0 by the author.` 부분이 사라집니다.

<br>

## Utterances 댓글 붙이기
Chirpy는 댓글 시스템으로 [Disqus](){:target="_blank"}를 지원합니다. 한동안 잘 썼었는데, 어느때 부터인가 몇몇 기능들이 유료로 바뀌고, UI가 너무 난잡(?) 합니다.   
그래서 [Utterances](https://utteranc.es/){:target="_blank"}를 적용해 보도록 하겠습니다.  
자세한 내용은 아래 포스트에서 확인하면 됩니다.  
* [Jekyll 테마에 utterances 댓글 연동하기](https://www.irgroup.org/posts/utternace-comments-system/){:target="_blank"}

<br>

## About 편집하기
`_tabs` 디렉토리의 내용을 보면 아래와 같습니다.  
```bash
-rw-r--r--   1 a2021054  staff   93 12 20 11:23 about.md
-rw-r--r--   1 a2021054  staff   72 12 20 11:23 archives.md
-rw-r--r--   1 a2021054  staff   74 12 20 11:23 categories.md
-rw-r--r--   1 a2021054  staff   59 12 20 11:23 tags.md
```
이 중에 `about.md` 파일을 수정하면 됩니다.   
마크다운 문법은 [여기](https://www.irgroup.org/posts/usage-markdown/){:target="_blank"}를 참고하세요~ 
<br>
같은 방법으로, `archives.md`, `categories.md`, `tags.md` 파일들을 수정하면, 왼쪽 탭메뉴에 클릭 시 나오는 페이지를 수정할 수 있습니다.

<BR>
## favicon 변경하기
브라우저 탭에 나오는 favicon 이미지를 변경하는 방법입니다.   
favicon에 대해 더 알고 싶으시면 [블로그 브랜딩, 파비콘(favicon)만드는 방법](https://do-son.tistory.com/74){:target="_blank"} 이런 글이 있네요..   
Chirpy 테마는 favicon이 기본값 아래 이미지로 되어 있습니다.

![chirpy 아이콘](/assets/img/Chirpy-default-favicon.jpg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;"}

자신의 이미지로 변경하고 싶으면, 적당한 크기의 이미지를 만든 후, [여기](https://www.favicon-generator.org/){:target="_blank"}에서 favicons을 만듭니다.   
다른 좋은 사이트도 많은 것 같은데.. 그냥 단순해서 여기서 만들어 봤습니다.   
생성하게 되면 꽤 많은 파일들이 만들어지게 됩니다.  
이 중에, `favicon`으로 시작하는 파일 4개를 `assets/img/favicon` 디렉토리로 복사해 줍니다. 기존 파일을 엎어 버립시다.   

![favicon 수정 파일들](/assets/img/Chirpy-modify-favicon.jpg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;"}

<br>
## post 상단 이미지 style 적용하기
`_layouts/post.html` 파일을 열어보면 아래와 같은 내용이 있습니다.  

```html
{% raw %}
{% if page.image.src %}
  {% capture bg %}
    {% unless page.image.no_bg %}{{ 'bg' }}{% endunless %}
  {% endcapture %}
  <img src="{{ page.image.src }}" class="preview-img {{ bg | strip }}"
      alt="{{ page.image.alt | default: 'Preview Image' }}"

      {% if page.image.width %}
        width="{{ page.image.width }}"
      {% elsif page.image.w %}
        width="{{ page.image.w }}"
      {% endif %}

      {% if page.image.height %}
        height="{{ page.image.height }}"
      {% elsif page.image.h %}
        height="{{ page.image.h }}"
      {% endif %}>
{% endif %}
{% endraw %}
```

여기에서 7번째 라인에 원하는 스타일을 적용해 주면 됩니다.  
> 예) 회색으로 이미지 외곽선 추가 : style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;"

<br>
## 사이트맵 설정
sitemap.xml 파일을 생성해서 구글 검색엔진에 노출되는 글을 제어할 수 있습니다.  
아래를 참고하세요~
* [Jekyll 테마에서 sitemap.xml 설정하기](){:target="_blank"}


<br>
## 구글 Analytics 연동하기
내 블로그에 얼마나 많은 사람들이 들어왔는지 보려면 구글 애널리틱스를 붙이는 것이 아주 빠른 방법입니다. 물론 사용성도 좋고, 많은 정보를 얻을 수 있습니다.   
(업무용으로 쓰신다면 비추! 더 좋은 다른 툴이 있습니다.)   
아래를 참고하세요~  
* [Jekyll 테마에 구글 애널리틱스 연동하기](){:target="_blank"}

<br>
## 구글 애드센스 붙이기
블로그를 하다보면 광고를 붙이는 경우가 참 많습니다. 저도 한번 붙여 봤는데요. 광고 수익이 올라오는 것을 보면 신기합니다. 이런 글도 읽어 주는 구나.. 😅   
헌데, 사이트가 너무 지저분해 집니다.  모바일에서는 탭메뉴 사이에 광고가.... 😡   
아무튼, 광고 적용은 아래를 참조 하세요~   
* [Jekyll 테마에 구글 애드센스 붙이기](){:target="_blank"}


<br>
<div class="card">
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-8993100314477491"
     crossorigin="anonymous"></script>
<!-- 디스플레이광고-수평형 -->
<ins class="adsbygoogle"
     style="display:block"
     data-ad-client="ca-pub-8993100314477491"
     data-ad-slot="9549119208"
     data-ad-format="auto"
     data-full-width-responsive="true"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
</div>
<br>