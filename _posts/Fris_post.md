layout: post
title: Jekyll 테마에서 구글 검색 노출 등록하기
date: 2022-01-13 15:11:33 +0900
categories: [jekyll]
tags: [jekyll, chirpy, google, google search, search engine, 검색노출, 구글검색, sitemap.xml, sitemap, 사이트맵]
description: 우리나라는 naver의 영향력이 막강하지만, 외부의 블로그나 사이트들은 그래도 구글을 통한 인입이 강력한 편입니다. jekyll 테마에서 구글 검색엔진에 포스트 글들이 검색되도록 설정해 보겠습니다.
toc: true
math: true
mermaid: true
image:
  src: /assets/img/Google-search.jpg
  width: 1000   # in pixels
  height: 400   # in pixels
  alt: 구글 검색엔진
---
> 구글에 검색을 통해 사이트로 인입되는 비율은 엄청납니다. \
> 우리나라는 naver의 영향력이 막강하지만, 외부의 블로그나 사이트들은 그래도 구글을 통한 인입이 강력한 편입니다. \
> jekyll 테마에서 구글 검색엔진에 포스트 글들이 검색되도록 설정해 보겠습니다.\
> chirpy 기준이기는 하나 기본이 같기 때문에 쉽게 적용할 수 있을 겁니다.
> [참고한 사이트](https://imweb.me/faq?mode=view&category=29&category2=35&idx=15573){:target="_blank"}가 너무 잘 정리되어 있으나, 끝부분이 저와 환경이 달라 재정리 합니다.\
> 앞부분은 거의 똑같습니다.(카피라고 뭐라한다면... 음.. 내리겠습니다... 😭)
<!-- 상단 광고 -->
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
<!-- start post -->
# 준비하기
---
우선 구글 계정이 필요합니다. 없다면 아래 링크를 통해 구글 계정을 새로 만듭니다.
> [구글 계정](https://accounts.google.com/){:target="_blank"}
# 구글 웹마스터 도구 등록하기
---
## 시작하기
[구글 웹마스터 도구(Search Console)](https://accounts.google.com/){:target="_blank"}에 접속합니다. 그러면 아래와 같이 화면이 나타나고, `시작하기` 버튼을 클릭합니다.
<br>
![구글검색 등록하기 #1](/assets/img/jekyll_google_search_1.jpg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
<br>
<br>
## URL 입력하기
속성유형 선택하는 화면에서 URL접두어를 사용하여 사이트 주소를 입력하고 `계속` 버튼을 클릭합니다. 
<br>
![구글검색 등록하기 #2](/assets/img/jekyll_google_search_2.jpg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
<br>
## 다른 확인 방법 선택하기